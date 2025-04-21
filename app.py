import re
import secrets
import logging
import os
import tempfile
import json
import glob
import time
from datetime import datetime
import base64
from functools import wraps
from dotenv import load_dotenv
from flask import Flask, render_template, request, jsonify, session, redirect, url_for, send_from_directory
import firebase_admin
from firebase_admin import credentials, auth, firestore, storage
from google_vision import extract_text_from_image
from gemini_processor import process_text_with_gemini
from error_gemini_processor import generate_error_poem
from symptom_mapping import symptom_specialty_map, update_mapping_with_gemini, determine_specialty
from langchain_openai import ChatOpenAI
from langchain.prompts import PromptTemplate
from openai import OpenAI
from gtts import gTTS
import httpx

logging.basicConfig(level=logging.DEBUG, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

load_dotenv(dotenv_path=os.path.join(os.getcwd(), '.env'))

logger.debug(f"Current directory: {os.getcwd()}")
logger.debug(f"Found .env file: {os.path.exists('.env')}")
logger.debug(f"Loaded environment variables: {dict(os.environ)}")
openai_api_key = os.getenv("OPENAI_API_KEY")
logger.debug(f"Loaded API key: {openai_api_key}")
if not openai_api_key:
    logger.error("OPENAI_API_KEY not found in .env")
    raise ValueError("OPENAI_API_KEY not found in .env file")

# Validate OpenAI API key at startup
try:
    test_client = OpenAI(api_key=openai_api_key)
    test_client.models.list()
    logger.info("✅ OpenAI API key validated successfully")
except Exception as e:
    logger.error(f"Invalid OpenAI API key: {str(e)}")
    raise ValueError(f"Invalid OpenAI API key: {str(e)}")

client = OpenAI(api_key=openai_api_key)

app = Flask(__name__)
app.secret_key = os.environ.get("FLASK_SECRET_KEY", secrets.token_hex(32))

cred_path = './Cred/Firebase/Med App/med-labs-42f13-firebase-adminsdk-fbsvc-43b78dfef7.json'
if not firebase_admin._apps:
    cred = credentials.Certificate(cred_path)
    firebase_admin.initialize_app(cred, {
        'storageBucket': 'med-labs-42f13'
    })
    logger.info("✅ Firebase initialized successfully with bucket: med-labs-42f13. SDK Version: %s", firebase_admin.__version__)

db = firestore.client()
bucket = storage.bucket()

try:
    bucket.get_blob('test-check')
    logger.info("Bucket exists and is accessible")
except Exception as e:
    logger.error(f"Bucket validation failed: {str(e)}. Please ensure the bucket 'med-labs-42f13' exists.")
    raise

def calculate_age(dob_str):
    try:
        dob = datetime.strptime(dob_str, "%d-%m-%Y")
        today = datetime(2025, 4, 20)
        age = today.year - dob.year - ((today.month, today.day) < (dob.month, dob.day))
        return age
    except ValueError as e:
        logger.error(f"Failed to parse DOB '{dob_str}': {str(e)}")
        return None

# Conversation LLM (GPT-4o-mini)
conversation_llm = ChatOpenAI(
    model="gpt-4o-mini",
    openai_api_key=openai_api_key,
    temperature=0.7
)

# JSON LLM (GPT-3.5-turbo)
json_llm = ChatOpenAI(
    model="gpt-3.5-turbo",
    openai_api_key=openai_api_key,
    temperature=0.0
)

conversation_prompt_template = PromptTemplate(
    input_variables=["history", "input", "medical_data"],
    template=(
        "You are a professional healthcare assistant. Your task is to guide the patient through a conversation to gather their symptoms, severity, duration, and triggers to recommend a doctor from the following specialties: Dermatology, ENT, Nephrology, Endocrinology, Gastroenterology, Urology, Psychiatry, General Medicine, General Surgery, Gynecology, Oncology, Pediatrics, Cardiology, Neurology, Orthopedics.\n\n"
        "**Current Medical Data:**\n{medical_data}\n\n"
        "**Conversation History:**\n{history}\n\n"
        "**Patient’s Latest Input:**\n{input}\n\n"
        "**Instructions:**\n"
        "- Interpret symptoms accurately:\n"
        "  - 'sadness' or 'heavy sadness' as 'depression' (Psychiatry).\n"
        "  - 'anxiety' or 'heart racing' as 'anxiety' (Psychiatry).\n"
        "  - 'focus issues' or 'feeling stuck' as 'difficulty concentrating' (Psychiatry).\n"
        "  - 'irritability' as 'irritability' (Psychiatry).\n"
        "  - 'fatigue' as 'chronic fatigue' if persistent or severe (Oncology).\n"
        "  - 'weight loss' as 'unexplained weight loss' if unintentional (Oncology).\n"
        "  - 'lump' or 'lumps' as 'lump' (Oncology).\n"
        "  - 'night sweats' as 'night sweats' (Oncology).\n"
        "  - 'tight band sensation around head' as 'headache' (Neurology).\n"
        "  - 'hand shaking' or 'tremors' as 'tremors' (Neurology).\n"
        "- Extract symptoms, severity, duration, and triggers from input and history.\n"
        "- Severity should be categorized as 'mild', 'moderate', or 'severe'.\n"
        "- Treat 'random situations' or similar phrases as equivalent to 'unknown' for triggers to avoid unnecessary clarification.\n"
        "- Update 'medical_data' with:\n"
        "  - 'symptoms': list of symptom names (e.g., ['depression', 'anxiety']).\n"
        "  - 'severity': list of severities (e.g., ['moderate', 'mild']).\n"
        "  - 'duration': list of durations (e.g., ['2 weeks', '1 month']).\n"
        "  - 'triggers': list of triggers (e.g., ['stress', 'unknown']).\n"
        "- If severity, duration, or triggers are missing, ask **one question at a time** for a single symptom and field. Prioritize:\n"
        "  1. Severity for the first incomplete symptom.\n"
        "  2. Duration for the same symptom once severity is provided.\n"
        "  3. Triggers for the same symptom once duration is provided.\n"
        "  - Example: 'How severe is your depression? Is it mild, moderate, or severe?'\n"
        "  - Move to the next symptom only after all fields are complete for the current symptom.\n"
        "- If a field is ambiguous (e.g., severity 'mild to moderate'), use the higher value (e.g., 'moderate').\n"
        "- When all details (symptoms, severity, duration, triggers) are complete for all symptoms, ask: 'May I recommend a doctor for you?' unless the patient has already requested a doctor.\n"
        "- If the patient agrees or requests a doctor (e.g., 'Can you help me figure out who to see?'), set 'assign_doctor' to 'true'.\n"
        "- Return:\n"
        "  - Response: [Text response with one question or doctor assignment confirmation]\n"
        "  - Updated Medical Data: [JSON object with 'symptoms', 'severity', 'duration', 'triggers' as lists]\n"
        "  - Assign Doctor: [true/false]\n"
        "  - Check Availability: [false]\n"
        "  - Condition: [null]\n"
        "- Ensure 'medical_data' is a JSON object, not a string, with lists for all fields.\n"
    )
)

json_prompt_template = PromptTemplate(
    input_variables=["conversation_output"],
    template=(
        "You are a JSON formatting assistant. Your task is to take the output from a healthcare conversation and format it into a JSON object.\n\n"
        "**Conversation Output:**\n{conversation_output}\n\n"
        "**Instructions:**\n"
        "- Parse the conversation output in this format:\n"
        "  - Response: [Text]\n"
        "  - Updated Medical Data: [JSON object]\n"
        "  - Assign Doctor: [true/false]\n"
        "  - Check Availability: [true/false]\n"
        "  - Condition: [condition or null]\n"
        "- Return a JSON object with keys: \"response\", \"medical_data\", \"assign_doctor\", \"check_availability\", \"condition\".\n"
        "- Ensure 'medical_data' is a parsed JSON object with 'symptoms', 'severity', 'duration', and 'triggers' as lists, not a string.\n"
        "- Use double quotes for all property names and string values.\n"
    )
)

conversation_chain = conversation_prompt_template | conversation_llm
json_chain = json_prompt_template | json_llm

def token_required(f):
    @wraps(f)
    def decorated(*args, **kwargs):
        auth_header = request.headers.get('Authorization')
        logger.debug(f"Authorization header: {auth_header if auth_header else 'None'}")
        token = None
        if auth_header and auth_header.startswith("Bearer "):
            token = auth_header.split(" ")[1]
            logger.debug(f"Using token from Authorization header: {token[:20]}... (Length: {len(token)})")
        elif 'idToken' in session:
            token = session['idToken']
            logger.debug(f"Falling back to session token: {token[:20]}... (Length: {len(token)})")
        else:
            logger.error("No Authorization header or session token provided")
            error_poem = generate_error_poem("token_missing")
            return render_template('errors/token_missing.html', error_poem=error_poem), 401

        try:
            decoded_token = auth.verify_id_token(token)
            request.user = decoded_token
            logger.info(f"✅ Token verified for: {decoded_token.get('email')}, UID: {decoded_token.get('uid')}")
        except auth.InvalidIdTokenError as e:
            logger.error(f"❌ Invalid ID token: {str(e)} with token: {token[:50]}...")
            error_poem = generate_error_poem("authentication")
            return render_template('errors/authentication_error.html', error_poem=error_poem), 401
        except auth.ExpiredIdTokenError as e:
            logger.error(f"❌ Expired ID token: {str(e)} with token: {token[:50]}...")
            error_poem = generate_error_poem("authentication")
            return render_template('errors/authentication_error.html', error_poem=error_poem), 401
        except Exception as e:
            logger.error(f"❌ Token verification error: {str(e)} with token: {token[:50]}...")
            error_poem = generate_error_poem("authentication")
            return render_template('errors/authentication_error.html', error_poem=error_poem), 401
        return f(*args, **kwargs)
    return decorated

def generate_consultant_id():
    counter_ref = db.collection('counters').document('consultant_id')
    @firestore.transactional
    def transactional_generate(transaction):
        snapshot = counter_ref.get(transaction=transaction)
        next_id = snapshot.get('next_id') if snapshot.exists else 1
        transaction.set(counter_ref, {'next_id': next_id + 1})
        return next_id
    transaction = db.transaction()
    next_id = transactional_generate(transaction)
    return f'DR{next_id:04d}'

# Static dictionary of doctors (replacing Firebase dependency)
DOCTORS = [
    {"consultant_id": "DR0006", "full_name": "Dr. Ravi Deshmukh", "specialty": "dermatology", "availability": True},
    {"consultant_id": "DR0007", "full_name": "Dr. Meera Kapoor", "specialty": "ent", "availability": True},
    {"consultant_id": "DR0008", "full_name": "Dr. Sanjay Rao", "specialty": "nephrology", "availability": True},
    {"consultant_id": "DR0009", "full_name": "Dr. Latha Nair", "specialty": "endocrinology", "availability": True},
    {"consultant_id": "DR0010", "full_name": "Dr. Ashok Iyer", "specialty": "gastroenterology", "availability": True},
    {"consultant_id": "DR0011", "full_name": "Dr. Nisha Bhatt", "specialty": "urology", "availability": True},
    {"consultant_id": "DR0012", "full_name": "Dr. Karan Malhotra", "specialty": "psychiatry", "availability": True},
    {"consultant_id": "DR0013", "full_name": "Dr. Alka Verma", "specialty": "general medicine", "availability": True},
    {"consultant_id": "DR0014", "full_name": "Dr. Rajesh Rana", "specialty": "general surgery", "availability": True},
    {"consultant_id": "DR0015", "full_name": "Dr. Shalini Joshi", "specialty": "gynecology", "availability": True},
    {"consultant_id": "DR0016", "full_name": "Dr. Vikas Sharma", "specialty": "oncology", "availability": True},
    {"consultant_id": "DR0017", "full_name": "Dr. Nandita Rao", "specialty": "pediatrics", "availability": True},
    {"consultant_id": "DR0018", "full_name": "Dr. Manoj Mehta", "specialty": "cardiology", "availability": True},
    {"consultant_id": "DR0019", "full_name": "Dr. Supriya Sen", "specialty": "neurology", "availability": True},
    {"consultant_id": "DR0020", "full_name": "Dr. Anand Kulkarni", "specialty": "orthopedics", "availability": True}
]

def fetch_available_doctors(specialty):
    try:
        specialty = specialty.lower()
        logger.debug(f"Fetching doctors for specialty: {specialty}")
        doctors_list = [
            {"consultant_id": doc["consultant_id"], "full_name": doc["full_name"], "specialty": doc["specialty"]}
            for doc in DOCTORS if doc["specialty"] == specialty and doc["availability"]
        ]
        if not doctors_list:
            logger.warning(f"No doctors found for specialty: {specialty} with availability: True")
        return doctors_list
    except Exception as e:
        logger.error(f"Error fetching available doctors: {str(e)}")
        return []

def assign_doctor(medical_data):
    symptoms = medical_data.get('symptoms', [])
    if not symptoms:
        logger.warning("No symptoms provided for doctor assignment")
        return [], "general medicine"  # Default to general medicine if no symptoms

    # Determine specialty using symptom_mapping.py's determine_specialty
    specialty = determine_specialty(symptoms)
    logger.debug(f"Determined specialty: {specialty} for symptoms: {symptoms}")

    # Fetch available doctors from static dictionary
    doctors_list = fetch_available_doctors(specialty)
    if not doctors_list:
        logger.warning(f"No doctors available for specialty: {specialty}. Falling back to general medicine.")
        specialty = "general medicine"
        doctors_list = fetch_available_doctors(specialty)

    return doctors_list, specialty

def process_conversation(audio_path=None, transcript=None, history=""):
    try:
        language = "en"
        medical_data = session.get('medical_data', {"symptoms": [], "severity": [], "duration": [], "triggers": []})
        if not medical_data.get("symptoms"):
            medical_data = {"symptoms": [], "severity": [], "duration": [], "triggers": []}

        asked_questions = session.get('asked_questions', {})
        if not asked_questions:
            asked_questions = {}
            for symptom in medical_data.get("symptoms", []):
                asked_questions[symptom] = {
                    "severity": False,
                    "duration": False,
                    "triggers": False
                }
        session['asked_questions'] = asked_questions

        # Track current symptom and field being queried
        current_query = session.get('current_query', {"symptom": None, "field": None})

        # Initialize contradiction tracker
        contradictions = session.get('contradictions', [])
        history = session.get('conversation_history', '')

        if audio_path:
            transcription = transcribe_audio(audio_path, language)
            if "failed" in transcription.lower():
                return {"response": "Audio issue detected. Please try again.", "medical_data": medical_data, "audio_url": None}, 400
        elif transcript:
            transcription = transcript
        else:
            intro = "Hello, I am your healthcare assistant. What symptoms are you experiencing?"
            audio_path = synthesize_audio(intro, language)
            return {"response": intro, "medical_data": medical_data, "audio_url": f"/static/{os.path.basename(audio_path)}"}, 200

        input_vars = {
            "input": transcription,
            "history": history,
            "medical_data": json.dumps(medical_data)
        }
        logger.debug(f"Input variables for conversation_chain.invoke: {input_vars}")

        conversation_response = conversation_chain.invoke(input_vars)
        conversation_output = conversation_response.content
        logger.debug(f"Conversation LLM output: {conversation_output}")

        json_response = json_chain.invoke({"conversation_output": conversation_output})
        logger.debug(f"JSON LLM output: {json_response}")

        if isinstance(json_response, dict):
            data = json_response
        elif hasattr(json_response, "content"):
            if isinstance(json_response.content, str):
                try:
                    data = json.loads(json_response.content)
                except json.JSONDecodeError as e:
                    logger.error(f"Failed to parse JSON LLM output: {json_response.content}, Error: {str(e)}")
                    return {
                        "response": "I encountered an error processing your response. Could you please repeat that?",
                        "medical_data": medical_data,
                        "audio_url": None
                    }, 500
            else:
                data = json_response.content
        else:
            logger.error(f"Unexpected JSON LLM output format: {json_response}")
            return {
                "response": "I encountered an error processing your response. Could you please repeat that?",
                "medical_data": medical_data,
                "audio_url": None
            }, 500

        expected_keys = ["response", "medical_data", "assign_doctor", "check_availability", "condition"]
        if not all(key in data for key in expected_keys):
            logger.error(f"JSON LLM output missing required keys: {data}")
            return {
                "response": "I encountered an error processing your response. Could you please repeat that?",
                "medical_data": medical_data,
                "audio_url": None
            }, 500

        response_text = data["response"]
        raw_medical_data = data["medical_data"]
        if not isinstance(raw_medical_data, dict):
            logger.error(f"medical_data is not a dictionary: {raw_medical_data}")
            try:
                raw_medical_data = json.loads(raw_medical_data)
            except (json.JSONDecodeError, TypeError):
                raw_medical_data = {"symptoms": [], "severity": [], "duration": [], "triggers": []}

        # Validate medical_data structure
        required_fields = ["symptoms", "severity", "duration", "triggers"]
        for field in required_fields:
            if field not in raw_medical_data or not isinstance(raw_medical_data[field], list):
                logger.warning(f"medical_data[{field}] is invalid or not a list: {raw_medical_data.get(field)}. Resetting to empty list.")
                raw_medical_data[field] = []

        updated_symptoms = raw_medical_data.get("symptoms", [])
        updated_severities = raw_medical_data.get("severity", [])
        updated_durations = raw_medical_data.get("duration", [])
        updated_triggers = raw_medical_data.get("triggers", [])

        # Normalize triggers to prevent contradiction loops
        trigger_synonyms = ["random situations", "out of nowhere", "unpredictable", "no specific triggers"]
        updated_triggers = [
            "unknown" if trigger and any(synonym in trigger.lower() for synonym in trigger_synonyms) else trigger
            for trigger in updated_triggers
        ]

        # Check if user responded with "unknown" or similar to a trigger question
        if current_query.get("field") == "triggers" and current_query.get("symptom"):
            unknown_responses = ["unknown", "not sure", "don't know", "unsure", "no idea", "i'm not sure"]
            if any(phrase in transcription.lower() for phrase in unknown_responses):
                symptom = current_query.get("symptom")
                if symptom in updated_symptoms:
                    idx = updated_symptoms.index(symptom)
                    updated_triggers[idx] = "unknown"
                    if symptom in asked_questions:
                        asked_questions[symptom]["triggers"] = True
                        logger.debug(f"Marked trigger question as answered for symptom: {symptom}")
                        current_query = {"symptom": None, "field": None}  # Clear current_query to move forward

        # Ensure lists are of equal length
        max_length = max(len(updated_symptoms), len(updated_severities), len(updated_durations), len(updated_triggers))
        updated_symptoms = updated_symptoms + [''] * (max_length - len(updated_symptoms))
        updated_severities = updated_severities + [''] * (max_length - len(updated_severities))
        updated_durations = updated_durations + [''] * (max_length - len(updated_durations))
        updated_triggers = updated_triggers + ['unknown'] * (max_length - len(updated_triggers))

        medical_data = {
            "symptoms": updated_symptoms,
            "severity": updated_severities,
            "duration": updated_durations,
            "triggers": updated_triggers
        }

        # Update asked_questions based on provided answers
        for idx, symptom in enumerate(updated_symptoms):
            if not symptom:
                continue
            if symptom not in asked_questions:
                asked_questions[symptom] = {
                    "severity": False,
                    "duration": False,
                    "triggers": False
                }
            # Update based on valid responses
            if updated_severities[idx] in ["mild", "moderate", "severe"]:
                asked_questions[symptom]["severity"] = True
            if updated_durations[idx] and updated_durations[idx].strip():
                asked_questions[symptom]["duration"] = True
            if updated_triggers[idx]:  # Accept any trigger response, including "unknown"
                asked_questions[symptom]["triggers"] = True
        session['asked_questions'] = asked_questions

        # Simplified contradiction detection
        existing_symptoms = session.get('medical_data', {}).get("symptoms", [])
        existing_severities = session.get('medical_data', {}).get("severity", [])
        existing_durations = session.get('medical_data', {}).get("duration", [])
        contradictions = []
        for idx, symptom in enumerate(updated_symptoms):
            if not symptom:
                continue
            existing_idx = existing_symptoms.index(symptom) if symptom in existing_symptoms else -1
            if existing_idx != -1:
                fields = [
                    ("severity", existing_severities, updated_severities),
                    ("duration", existing_durations, updated_durations)
                ]
                for field_name, existing_list, updated_list in fields:
                    existing_value = existing_list[existing_idx] if existing_idx < len(existing_list) else ""
                    new_value = updated_list[idx] if idx < len(updated_list) else ""
                    if new_value and existing_value and new_value.lower() != existing_value.lower():
                        contradictions.append({
                            "key": f"{symptom}_{field_name}",
                            "symptom": symptom,
                            "field": field_name,
                            "existing_value": existing_value,
                            "new_value": new_value
                        })

        if contradictions:
            contradiction = contradictions[0]
            symptom_name = contradiction["symptom"]
            field = contradiction["field"]
            existing_value = contradiction["existing_value"]
            new_value = contradiction["new_value"]
            if new_value.lower() in transcription.lower():
                medical_data[field][updated_symptoms.index(symptom_name)] = new_value
                asked_questions[symptom_name][field] = True
                session['asked_questions'] = asked_questions
            else:
                response_text = f"I noticed you previously mentioned your {symptom_name}'s {field} as '{existing_value}', but now you’ve said '{new_value}'. Could you please clarify this for me?"
                current_query = {"symptom": symptom_name, "field": field}
                session['current_query'] = current_query
                audio_path = synthesize_audio(response_text, language)
                session['medical_data'] = medical_data
                return {
                    "response": response_text,
                    "medical_data": medical_data,
                    "audio_url": f"/static/{os.path.basename(audio_path)}" if audio_path else None
                }, 200
        session['contradictions'] = []

        # Determine next question or proceed to doctor assignment
        if not data.get("assign_doctor"):
            next_question_found = False
            for idx, symptom in enumerate(medical_data["symptoms"]):
                if not symptom:
                    continue
                if not asked_questions[symptom]["severity"]:
                    response_text = f"How severe is your {symptom}? Is it mild, moderate, or severe?"
                    current_query = {"symptom": symptom, "field": "severity"}
                    next_question_found = True
                    break
                elif not asked_questions[symptom]["duration"]:
                    response_text = f"How long have you been experiencing {symptom}?"
                    current_query = {"symptom": symptom, "field": "duration"}
                    next_question_found = True
                    break
                elif not asked_questions[symptom]["triggers"]:
                    response_text = f"What triggers your {symptom}? If you're unsure, you can say 'unknown'."
                    current_query = {"symptom": symptom, "field": "triggers"}
                    next_question_found = True
                    break
            if next_question_found:
                session['current_query'] = current_query
                audio_path = synthesize_audio(response_text, language)
                session['medical_data'] = medical_data
                return {
                    "response": response_text,
                    "medical_data": medical_data,
                    "audio_url": f"/static/{os.path.basename(audio_path)}" if audio_path else None
                }, 200
            else:
                # No more questions to ask, proceed to doctor assignment
                data["assign_doctor"] = True

        # Handle doctor assignment
        all_details_complete = all(
            asked_questions[symptom]["severity"] and asked_questions[symptom]["duration"] and asked_questions[symptom]["triggers"]
            for symptom in medical_data["symptoms"] if symptom
        )
        if not all_details_complete:
            logger.warning("Not all symptom details are complete; cannot assign doctor yet.")
            next_question_found = False
            for symptom in medical_data["symptoms"]:
                if not symptom:
                    continue
                if not asked_questions[symptom]["severity"]:
                    response_text = f"How severe is your {symptom}? Is it mild, moderate, or severe?"
                    current_query = {"symptom": symptom, "field": "severity"}
                    next_question_found = True
                    break
                elif not asked_questions[symptom]["duration"]:
                    response_text = f"How long have you been experiencing {symptom}?"
                    current_query = {"symptom": symptom, "field": "duration"}
                    next_question_found = True
                    break
                elif not asked_questions[symptom]["triggers"]:
                    response_text = f"What triggers your {symptom}? If you're unsure, you can say 'unknown'."
                    current_query = {"symptom": symptom, "field": "triggers"}
                    next_question_found = True
                    break
            if next_question_found:
                session['current_query'] = current_query
                audio_path = synthesize_audio(response_text, language)
                session['medical_data'] = medical_data
                return {
                    "response": response_text,
                    "medical_data": medical_data,
                    "audio_url": f"/static/{os.path.basename(audio_path)}" if audio_path else None
                }, 200

        # If we've reached here, we're ready to assign a doctor
        doctors_list, specialty = assign_doctor(medical_data)
        if not doctors_list:
            response_text = f"No doctors available for {specialty or 'unknown specialty'}. Please try again later or contact support."
        else:
            doctor = doctors_list[0]
            consultant_id = doctor["consultant_id"]
            uid = request.user.get("uid")
            patient_name = session.get('user_info', {}).get('full_name', uid)

            # Update initial_screenings
            doc_ref = db.collection('initial_screenings').document(f'initial_screening_{uid}')
            firestore_data = {
                "symptoms": ', '.join([s for s in medical_data["symptoms"] if s]),
                "severity": ', '.join([s for s in medical_data["severity"] if s]),
                "duration": ', '.join([d for d in medical_data["duration"] if d]),
                "triggers": ', '.join([t for t in medical_data["triggers"] if t]),
                "consultant_id": consultant_id,
                "specialty": specialty,
                "patient_name": patient_name,
                "uid": uid,
                "timestamp": firestore.SERVER_TIMESTAMP
            }
            try:
                doc_ref.set(firestore_data, merge=True)
            except Exception as e:
                logger.error(f"Failed to update Firestore: {str(e)}")
                response_text = "Error saving your data. Please try again later."

            # Update patient_registrations
            patient_ref = db.collection('patient_registrations').document(uid)
            patient_snap = patient_ref.get()
            if patient_snap.exists:
                patient_ref.set({'consultant_id': consultant_id}, merge=True)
            else:
                patient_data = {
                    'uid': uid,
                    'email': session.get('user_info', {}).get('email', ''),
                    'full_name': patient_name,
                    'consultant_id': consultant_id,
                    'role': 'patient'
                }
                patient_ref.set(patient_data)

            response_text = f"You have been assigned to {doctor['full_name']} (specializing in {specialty.capitalize()}). Please log in to view details."
            session.pop('asked_questions', None)
            session.pop('medical_data', None)
            session.pop('contradictions', None)
            session.pop('current_query', None)

        audio_path = synthesize_audio(response_text, language)
        session['medical_data'] = medical_data
        session['conversation_history'] = f"{history}\nPatient: {transcription}\nAgent: {response_text}" if transcription else f"{history}\nAgent: {response_text}"

        return {
            "response": response_text,
            "medical_data": medical_data,
            "audio_url": f"/static/{os.path.basename(audio_path)}" if audio_path else None
        }, 200

    except Exception as e:
        logger.error(f"Error: {str(e)}")
        return {"response": f"Error: {str(e)}", "medical_data": {"symptoms": []}, "audio_url": None}, 500

def cleanup_old_tts_files(static_dir="static", max_age=300):
    try:
        now = time.time()
        for mp3_file in glob.glob(os.path.join(static_dir, "tmp*.mp3")):
            file_age = now - os.path.getmtime(mp3_file)
            if file_age > max_age:
                os.remove(mp3_file)
                logger.debug(f"Deleted old TTS file: {mp3_file} (age: {file_age}s)")
    except Exception as e:
        logger.warning(f"Failed to clean up old TTS files: {str(e)}")

def transcribe_audio(audio_path, language="en", retries=3):
    attempt = 0
    while attempt < retries:
        try:
            logger.debug(f"Transcribing audio from file: {audio_path} with language: {language} (Attempt {attempt + 1})")
            with open(audio_path, "rb") as audio_file:
                audio_size = os.path.getsize(audio_path)
                logger.debug(f"Audio file size: {audio_size} bytes")
                if audio_size < 1024:
                    return "Audio file too small or corrupted"
                result = client.audio.transcriptions.create(
                    model="whisper-1",
                    file=audio_file,
                    language="en"
                )
                transcribed_text = result.text
                logger.debug(f"Raw transcribed text: {transcribed_text}")
                return transcribed_text
        except httpx.HTTPStatusError as e:
            if e.response.status_code in [429, 500, 502, 503]:
                attempt += 1
                time.sleep(2 ** attempt)
                continue
            error_message = f"Transcription failed with HTTP {e.response.status_code}: {e.response.text}"
            logger.error(error_message)
            return error_message
        except Exception as e:
            error_message = f"Transcription failed due to an error: {str(e)}"
            logger.error(error_message)
            return error_message
    return "Transcription failed after multiple attempts"

def synthesize_audio(text, language="en"):
    try:
        logger.debug(f"Attempting to synthesize audio with input language: {language}, text: {text[:50]}...")
        if not text or not text.strip():
            logger.warning("TTS input is empty or whitespace only")
            return None
        tts = gTTS(text=text, lang="en", slow=False)
        temp_file = tempfile.NamedTemporaryFile(delete=False, suffix=".mp3", dir="static")
        tts.write_to_fp(temp_file.file)
        temp_file.file.close()
        audio_size = os.path.getsize(temp_file.name)
        if audio_size < 1024:
            logger.error(f"Generated audio file {temp_file.name} is too small: {audio_size} bytes")
            os.remove(temp_file.name)
            return None
        logger.info(f"Saved audio to {temp_file.name} with size {audio_size} bytes")
        return temp_file.name
    except Exception as e:
        logger.error(f"Audio synthesis error: {str(e)}")
        if 'temp_file' in locals() and os.path.exists(temp_file.name):
            os.remove(temp_file.name)
        return None

@app.before_request
def before_request_cleanup():
    cleanup_old_tts_files()

@app.route('/', methods=['GET'])
def index():
    try:
        return render_template('index.html')
    except Exception as e:
        logger.error("index.html not found: " + str(e))
        return "Homepage not found. Please create templates/index.html", 404

@app.route('/login', methods=['GET', 'POST'])
def login():
    if request.method == 'GET':
        try:
            if 'user_info' in session:
                logger.debug("User already logged in, rendering login page to re-authenticate")
                id_token = session.get('idToken')
                if id_token:
                    try:
                        decoded_token = auth.verify_id_token(id_token)
                        email = decoded_token.get('email')
                        logger.debug(f"Session token valid for email: {email}")
                    except auth.InvalidIdTokenError as e:
                        logger.error(f"Invalid session token: {str(e)}. Clearing session.")
                        session.pop('user_info', None)
                        session.pop('idToken', None)
            return render_template('login.html')
        except Exception as e:
            logger.error("login.html not found: " + str(e))
            return "Login page not found. Please create templates/login.html", 404
    try:
        id_token = request.headers.get('Authorization', '').replace('Bearer ', '') or request.form.get('idToken')
        logger.debug(f"Received ID token: {id_token[:20]}... (Full length: {len(id_token)})")

        if not id_token:
            logger.error("No ID token provided")
            return jsonify({"error": "No ID token provided"}), 400

        decoded_token = auth.verify_id_token(id_token)
        email = decoded_token.get('email')
        uid = decoded_token.get('uid')
        logger.debug(f"Decoded ID token for email: {email}, uid: {uid}")

        token_parts = id_token.split('.')
        if len(token_parts) == 3:
            header = json.loads(base64.b64decode(token_parts[0] + '==').decode('utf-8'))
            logger.debug(f"Token header: {header}")
        else:
            logger.error("Invalid token format")

        request_data = request.get_json() or {}
        request_uid = request_data.get('uid')
        if request_uid and request_uid != uid:
            logger.error(f"UID mismatch: Request UID {request_uid} does not match token UID {uid}")
            return jsonify({"error": "UID mismatch", "details": f"Expected: {uid}, Got: {request_uid}"}), 401

        role = request_data.get('role')
        if not role:
            logger.error("Role not provided in request")
            return jsonify({"error": "Role not provided"}), 400

        role_mapping = {
            'patient': 'patient_registrations',
            'doctor': 'consultant_registrations',
            'assistant': 'assistant_registrations'
        }
        if role not in role_mapping:
            logger.error(f"Invalid role provided: {role}")
            return jsonify({"error": "Invalid role", "details": f"Role {role} not recognized"}), 400

        user_data = None
        doc_id = None
        collection_name = role_mapping[role]
        query = db.collection(collection_name).where('email', '==', email).limit(1).get()
        for doc in query:
            user_data = doc.to_dict()
            doc_id = doc.id
            break

        if not user_data:
            logger.error(f"User not found in {collection_name} for email: {email} or uid: {uid}")
            return jsonify({"error": f"User not found in {role} collection", "details": f"Checked UID: {uid}, Email: {email}"}), 401
        elif not user_data.get('email') or user_data.get('email') != email:
            logger.error(f"Email mismatch for uid: {uid}, expected {email}, got {user_data.get('email')}")
            return jsonify({"error": "Email mismatch", "details": f"Expected: {email}, Got: {user_data.get('email')}"}), 401

        full_name = user_data.get('full_name', uid)
        logger.info(f"User found with role: {role}, email: {email}, uid: {uid}, full_name: {full_name}")

        session['user_info'] = {
            'email': email,
            'role': role,
            'uid': doc_id if role == 'doctor' else uid,
            'full_name': full_name
        }
        session["idToken"] = id_token
        logger.info(f"🔑 User logged in: {email} with role: {role}, UID: {doc_id if role == 'doctor' else uid}, Full Name: {full_name}, Stored ID Token: {id_token[:20]}...")

        session_user_ref = db.collection(collection_name).document(doc_id if role == 'doctor' else uid)
        session_user_snap = session_user_ref.get()
        if session_user_snap.exists:
            session_user_data = session_user_snap.to_dict()
            if session_user_data.get('email') != email or session_user_data.get('role') != role:
                logger.error("Session data mismatch with Firestore")
                session.pop('user_info', None)
                session.pop('idToken', None)
                return jsonify({"error": "Session data mismatch", "details": "Please log in again"}), 401

        return jsonify({"success": True, "redirect": "/dashboard"})
    except auth.InvalidIdTokenError as e:
        logger.error(f"Invalid ID token error: {str(e)} with token: {id_token[:50]}...")
        return jsonify({"error": "Invalid ID token", "details": str(e)}), 401
    except Exception as e:
        logger.exception(f"Unexpected error during login: {str(e)} with token: {id_token[:50]}...")
        return jsonify({"error": "Login failed", "details": str(e)}), 500

@app.route('/register', methods=['GET', 'POST'])
def register():
    try:
        if request.method == 'POST':
            logger.debug(f"Received data: {request.get_json() or request.form.to_dict()}")
            user_data = request.get_json() if request.is_json else request.form.to_dict()
            email = user_data.get('email')
            password = user_data.get('password')
            role = user_data.get('role')

            if not email or not password or not role:
                logger.error(f"Missing required fields: email={email}, password={password}, role={role}")
                return jsonify({"error": "Email, password, and role are required"}), 400

            required_fields = ["full_name", "email", "phone", "dob", "location", "role"]
            if role == 'patient':
                required_fields.extend(["age"])
            elif role == 'doctor':
                required_fields.extend(["specialty"])
            elif role == 'assistant':
                required_fields.extend(["department"])
                if 'assigned_doctors' in user_data and user_data['assigned_doctors']:
                    required_fields.append("assigned_doctors")
            missing_fields = [field for field in required_fields if field not in user_data or not user_data[field]]
            if missing_fields:
                logger.error(f"Missing fields in registration: {', '.join(missing_fields)}")
                return jsonify({"error": f"Missing fields: {', '.join(missing_fields)}"}), 400

            user = auth.create_user(email=email, password=password, uid=None)
            uid = user.uid
            logger.info(f"✅ Created new user in Firebase Auth: {email}, UID: {uid}")

            if role == 'doctor':
                consultant_id = generate_consultant_id()
                user_data["consultant_id"] = consultant_id
                user_data["doctor_id"] = consultant_id
                user_ref = db.collection('consultant_registrations').document(consultant_id)
            elif role == 'assistant':
                user_data["assistant_id"] = f"ASST{uid[-8:]}"
                user_data["lab_id"] = user_data.get("department", "LAB") + "_" + uid[-4:]
                user_ref = db.collection('assistant_registrations').document(uid)
            else:
                user_ref = db.collection('patient_registrations').document(uid)

            user_data['uid'] = uid
            user_data['firebase_id'] = uid
            user_ref.set(user_data)
            custom_token = auth.create_custom_token(uid).decode('utf-8')
            session['user_info'] = {
                'email': user_data['email'],
                'role': role,
                'uid': consultant_id if role == 'doctor' else uid,
                'full_name': user_data.get('full_name', '')
            }
            session['idToken'] = custom_token
            logger.info(f"✅ Registered new {role} with UID: {uid}, Document ID: {consultant_id if role == 'doctor' else uid}")
            logger.debug(f"Stored custom token in session: {custom_token[:20]}...")

            return jsonify({
                'success': True,
                'customToken': custom_token,
                'redirect': '/further_patient_registration' if role == 'patient' else '/dashboard'
            })
        logger.debug("Rendering registration page (registration.html)")
        return render_template('registration.html')
    except Exception as e:
        logger.exception(f"Registration error: {e}")
        return jsonify({"error": str(e)}), 500

@app.route('/further_patient_registration', methods=['GET'])
@token_required
def further_patient_registration():
    try:
        user = request.user
        uid = user.get('uid')
        session_info = session.get('user_info', {})
        patient_name = session_info.get('full_name', uid)

        doc_id = f'initial_screening_{uid}'
        doc_ref = db.collection('initial_screenings').document(doc_id)

        doc_snap = doc_ref.get()
        if not doc_snap.exists:
            medical_data = {
                'uid': uid,
                'patient_name': patient_name,
                'symptoms': '',
                'severity': '',
                'duration': '',
                'triggers': '',
                'consultant_id': None,
                'timestamp': firestore.SERVER_TIMESTAMP
            }
            doc_ref.set(medical_data)
            logger.info(f"Initialized medical_data for UID: {uid} with doc_id: {doc_id}")
        else:
            logger.debug(f"Initial screening document already exists for UID: {uid}, doc_id: {doc_id}")

        logger.info(f"Loading further patient registration for UID: {uid}, Name: {patient_name}")
        return render_template('further_patient_registration.html', user_info={'uid': uid, 'patient_name': patient_name})
    except Exception as e:
        logger.exception(f"Further patient registration error: {e}")
        return jsonify({"error": "Error loading further patient registration", "details": str(e)}), 500

@app.route('/transcribe', methods=['POST'])
@token_required
def transcribe():
    try:
        language = "en"
        if 'audio' not in request.files:
            logger.error("No audio file provided in the request")
            return jsonify({"error": "No audio file provided"}), 400

        audio_file = request.files['audio']
        if audio_file.filename == '':
            logger.error("No audio file selected")
            return jsonify({"error": "No audio file selected"}), 400

        temp_dir = tempfile.mkdtemp()
        audio_path = os.path.join(temp_dir, f"temp_audio_{request.user.get('uid')}.webm")
        audio_file.save(audio_path)
        logger.debug(f"Audio saved to: {audio_path}")

        transcription = transcribe_audio(audio_path, language=language)
        if "failed" in transcription.lower() or "too small" in transcription.lower():
            return jsonify({"error": transcription}), 400

        logger.debug(f"Transcribed text: {transcription}")
        return jsonify({"success": True, "transcript": transcription})
    except Exception as e:
        logger.exception(f"Transcription error: {e}")
        return jsonify({"error": str(e)}), 500
    finally:
        if 'temp_dir' in locals() and os.path.exists(temp_dir):
            try:
                os.remove(audio_path)
                os.rmdir(temp_dir)
                logger.debug(f"Cleaned up temporary files in: {temp_dir}")
            except Exception as e:
                logger.warning(f"Failed to clean up temp files: {str(e)}")

@app.route('/start_conversation', methods=['POST'])
@token_required
def start_conversation():
    try:
        uid = request.user.get('uid')
        history = session.get('conversation_history', '')

        if request.is_json:
            data = request.get_json()
            transcript = data.get('transcript', '')
            logger.debug(f"Received transcript for UID {uid}: {transcript}")
            processed_data, status_code = process_conversation(transcript=transcript, history=history)

            if status_code != 200:
                return jsonify(processed_data), status_code

            medical_data = processed_data.get('medical_data')
            if not isinstance(medical_data, dict):
                logger.error(f"medical_data is not a dictionary: {medical_data}")
                medical_data = {
                    "symptoms": [],
                    "severity": [],
                    "duration": [],
                    "triggers": []
                }
            expected_keys = ["symptoms", "severity", "duration", "triggers"]
            for key in expected_keys:
                if key not in medical_data or not isinstance(medical_data[key], list):
                    logger.warning(f"medical_data[{key}] is not a list, resetting: {medical_data.get(key)}")
                    medical_data[key] = []

            session['conversation_history'] = f"{history}\nPatient: {transcript}\nAgent: {processed_data['response']}" if transcript else f"{history}\nAgent: {processed_data['response']}"

            return jsonify({
                'success': True,
                'response': processed_data['response'],
                'medical_data': medical_data,
                'audio_url': processed_data.get('audio_url'),
                'redirect': processed_data.get('redirect', '/dashboard')
            })
        elif 'audio' in request.files:
            audio_file = request.files['audio']
            if audio_file.filename == '':
                logger.error("No audio file selected")
                return jsonify({"error": "No audio file selected"}), 400

            temp_dir = tempfile.mkdtemp()
            audio_path = os.path.join(temp_dir, f"temp_audio_{uid}.mp3")
            audio_file.save(audio_path)
            logger.debug(f"Audio saved to: {audio_path}")

            processed_data, status_code = process_conversation(audio_path, history)

            if status_code != 200:
                return jsonify(processed_data), status_code

            medical_data = processed_data.get('medical_data')
            if not isinstance(medical_data, dict):
                logger.error(f"medical_data is not a dictionary: {medical_data}")
                medical_data = {
                    "symptoms": [],
                    "severity": [],
                    "duration": [],
                    "triggers": []
                }
            expected_keys = ["symptoms", "severity", "duration", "triggers"]
            for key in expected_keys:
                if key not in medical_data or not isinstance(medical_data[key], list):
                    logger.warning(f"medical_data[{key}] is not a list, resetting: {medical_data.get(key)}")
                    medical_data[key] = []

            session['conversation_history'] = f"{history}\nPatient: {audio_file.filename.split('.')[0]}\nAgent: {processed_data['response']}"

            return jsonify({
                'success': True,
                'response': processed_data['response'],
                'medical_data': medical_data,
                'audio_url': processed_data.get('audio_url'),
                'redirect': processed_data.get('redirect', '/dashboard')
            })
        else:
            logger.error("Invalid request format: neither JSON nor audio")
            return jsonify({"error": "Invalid request format"}), 400
    except Exception as e:
        logger.exception(f"Start conversation error: {e}")
        return jsonify({"error": str(e)}), 500
    finally:
        if 'temp_dir' in locals() and os.path.exists(temp_dir):
            try:
                os.remove(audio_path)
                os.rmdir(temp_dir)
                logger.debug(f"Cleaned up temporary files in: {temp_dir}")
            except Exception as e:
                logger.warning(f"Failed to clean up temp files: {str(e)}")

@app.route('/static/<path:path>')
def send_static(path):
    logger.debug(f"Serving static file: {path} from {os.path.join('static', path)}")
    return send_from_directory('static', path)

@app.route('/upload-image', methods=['POST'])
@token_required
def upload_image():
    try:
        uid = request.user.get('uid')
        if 'file' not in request.files:
            logger.error("No file part in the request")
            return jsonify({"error": "No file part in the request"}), 400
        
        file = request.files['file']
        if file.filename == '':
            logger.error("No file selected or empty filename")
            return jsonify({"error": "No file selected"}), 400
        
        category = request.form.get('category')
        if not category or category not in ['prescriptions', 'lab_records']:
            logger.error(f"Invalid or missing category: {category}")
            return jsonify({"error": "Invalid or missing category"}), 400
        
        file_name = file.filename
        storage_path = f"{category}/{uid}/raw_uploads/{file_name}"
        logger.debug(f"Attempting to upload file to GCS path: {storage_path}")
        
        blob = bucket.blob(storage_path)
        try:
            blob.upload_from_file(file.stream, content_type=file.content_type)
            logger.info(f"File successfully uploaded to GCS: {storage_path}")
        except Exception as e:
            logger.error(f"Failed to upload to GCS: {str(e)}", exc_info=True)
            return jsonify({"error": f"Failed to upload to GCS: {str(e)}"}), 500
        
        file_url = blob.public_url
        logger.info(f"File available at: {file_url}")
        
        return jsonify({"success": True, "filePath": storage_path})
    except Exception as e:
        logger.exception(f"Upload error: {e}")
        return jsonify({"error": str(e)}), 500

@app.route('/process-upload', methods=['POST'])
def process_upload():
    try:
        language_text = request.form.get('languageText')
        file_path = request.form.get('filePath')
        category = request.form.get('category')
        uid = request.form.get('uid')
        consultant_id = request.form.get('consultantId')

        logger.debug(f"Processing upload: language={language_text}, file_path={file_path}, category={category}, uid={uid}, consultant_id={consultant_id}")

        blob = bucket.blob(file_path)
        temp_file = os.path.join(tempfile.gettempdir(), f'temp_{uid}.jpg')
        os.makedirs(os.path.dirname(temp_file), exist_ok=True)
        blob.download_to_filename(temp_file)
        logger.debug(f"Image downloaded to: {temp_file}")

        extracted_text = extract_text_from_image(temp_file)
        logger.debug(f"Extracted English text: {extracted_text[:100]}... (length: {len(extracted_text)})")

        existing_text = None
        if category == 'prescriptions':
            existing_summaries = db.collection('prescriptions').where('uid', '==', uid).order_by('timestamp', direction=firestore.Query.DESCENDING).limit(1).get()
            if existing_summaries:
                existing_text = existing_summaries[0].to_dict().get('summary', '')

        patient_ref = db.collection('patient_registrations').document(uid).get()
        patient_name = patient_ref.to_dict().get('full_name', 'ರೋಗಿ') if patient_ref.exists else 'ರೋಗಿ'

        result = process_text_with_gemini(
            extracted_text=extracted_text,
            category=category,
            language=language_text,
            patient_name=patient_name,
            existing_text=existing_text,
            uid=uid
        )

        doc_ref = db.collection(category).document()
        doc_ref.set({
            'uid': uid,
            'consultant_id': consultant_id,
            'patient_name': patient_name,
            'summary': result['regional_summary'],
            'professional_summary': result['professional_summary'],
            'language': result['language'],
            'timestamp': firestore.SERVER_TIMESTAMP
        })
        logger.info(f"Saved {category} summary for UID: {uid} with language: {result['language']}")

        return jsonify({'success': True, 'language': result['language']})
    except Exception as e:
        logger.error(f"Gemini processing failed: {str(e)}")
        return jsonify({'success': False, 'error': str(e)}), 500

@app.route('/process-medical-history', methods=['POST'])
@token_required
def process_medical_history():
    try:
        data = request.get_json()
        uid = data.get('uid')

        if not uid:
            logger.error("Missing UID in request")
            return jsonify({'success': False, 'error': 'Missing UID'}), 400

        logger.debug(f"Processing medical history for UID: {uid}")

        from gemini_processor import generate_medical_history
        summary = generate_medical_history(uid, db)

        logger.info(f"Generated medical history summary for UID: {uid}")
        return jsonify({'success': True, 'summary': summary})
    except Exception as e:
        logger.error(f"Error processing medical history: {str(e)}")
        return jsonify({'success': False, 'error': str(e)}), 500

@app.route('/dashboard', methods=['GET'])
@token_required
def dashboard():
    try:
        user = request.user
        email = user.get('email')
        session_info = session.get('user_info', {})
        role = session_info.get('role')
        uid = session_info.get('uid')

        if not role:
            logger.error("No valid role found in session")
            raise ValueError("Role is required to render dashboard")

        collection_map = {
            'patient': 'patient_registrations',
            'doctor': 'consultant_registrations',
            'assistant': 'assistant_registrations'
        }
        collection_name = collection_map.get(role)
        user_query = db.collection(collection_name).where('email', '==', email).limit(1).get()
        user_snap = None
        for doc in user_query:
            user_snap = doc
            break

        if not user_snap:
            logger.error(f"No user document found for email: {email}")
            session.pop('user_info', None)
            session.pop('idToken', None)
            return jsonify({"error": "User not found, please log in again"}), 401

        user_data = user_snap.to_dict()
        if user_data.get('email') != email:
            logger.error(f"Email mismatch for document, expected {email}, got {user_data.get('email')}")
            session.pop('user_info', None)
            session.pop('idToken', None)
            return jsonify({"error": "Email mismatch, please log in again"}), 401

        if role == 'doctor':
            consultant_id = user_snap.id
            if not consultant_id.startswith('DR'):
                logger.warning(f"Unexpected document ID for doctor: {consultant_id}, falling back to uid")
                consultant_id = user_data.get('consultant_id', uid)
            session['user_info']['uid'] = consultant_id
            uid = consultant_id
            logger.info(f"Updated session uid to consultant_id: {uid} for doctor (document ID: {user_snap.id})")

        logger.info(f"Loading dashboard for {email} with role: {role}, uid: {uid}")
        dashboards = {
            'patient': 'patient_dashboard.html',
            'doctor': 'consultantDashboard.html',
            'assistant': 'assistant_dashboard.html'
        }
        template = dashboards.get(role)
        if not template:
            logger.error(f"Invalid role {role} for dashboard")
            raise ValueError(f"No template found for role {role}")

        logger.debug(f"Rendering dashboard template: {template} with uid: {uid}")
        try:
            rendered = render_template(template, user_info={'uid': uid, 'email': email, 'role': role}, login_success=True)
            logger.debug("Template rendered successfully with consultantId: " + uid)
            return rendered
        except Exception as render_error:
            logger.error(f"Failed to render template {template}: {render_error}")
            raise
    except Exception as e:
        logger.exception(f"Dashboard error: {e}")
        return jsonify({"error": "Error rendering dashboard", "details": str(e)}), 500

@app.route('/logout')
def logout():
    logger.info("User logging out")
    if request.args.get('confirm') != 'yes':
        return "Are you sure you want to log out? <a href='/logout?confirm=yes'>Confirm</a> | <a href='/dashboard'>Cancel</a>", 200
    session.pop('user_info', None)
    session.pop('idToken', None)
    session.pop('preferred_language', None)
    session.pop('pending_doctor_selection', None)
    session.pop('pending_specialty', None)
    session.pop('pending_medical_data', None)
    session.pop('pending_uid', None)
    logger.info("Session cleared, redirecting to /login")
    return redirect('/login')

@app.errorhandler(404)
def page_not_found(e):
    static_poem = "Blank space resonates\n404 hums creation\nNew worlds now beckon"
    return render_template('errors/404.html', error_poem=static_poem), 404

if __name__ == '__main__':
    port = int(os.environ.get('PORT', 5000))
    logger.info(f"Starting Flask app on port {port}")
    app.run(debug=True, port=port)