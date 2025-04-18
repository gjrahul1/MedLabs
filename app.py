import re
import secrets
import logging
import os
import tempfile
import json
import glob
import time
import random
import base64
from functools import wraps
from dotenv import load_dotenv
from flask import Flask, render_template, request, jsonify, session, redirect, url_for, send_from_directory
import firebase_admin
from firebase_admin import credentials, auth, firestore, storage
from google_vision import extract_text_from_image
from gemini_processor import process_text_with_gemini
from error_gemini_processor import generate_error_poem
from langchain_openai import ChatOpenAI
from langchain.prompts import PromptTemplate
from langchain.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnableSequence
from langchain_core.runnables import RunnablePassthrough, RunnableLambda
from openai import OpenAI
from gtts import gTTS
from datetime import datetime
import traceback
import httpx
import os.path

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

llm = ChatOpenAI(
    model="gpt-4o-mini",
    openai_api_key=openai_api_key,
    temperature=0.7
)

prompt_template = PromptTemplate(
    input_variables=["history", "input"],
    template=(
        "You are a professional healthcare agent assisting a patient. "
        "The patient provides input in English. Respond formally in English. "
        "Ask only one question at a time based on the conversation history to gather medical information: symptoms, severity (mild/moderate/severe), duration (days/weeks), or triggers. "
        "Use respectful and formal language, avoiding casual words. "
        "Questions must relate to symptoms, severity, duration, or triggers—do not ask about rest or unrelated topics. "
        "If the patient's response is unclear or unrelated, request clarification using: 'I’m sorry, your response is not clear. Could you please clarify your symptoms?' "
        "Recognize symptoms directly in English. For example: "
        "- If the input is 'I can’t sleep for 2-3 days', identify the symptom as 'inability to sleep'. "
        "- If the input is 'headache', identify the symptom as 'headache'. "
        "When asking about symptoms, use 'What symptoms are you experiencing?' "
        "When asking about severity, use 'How severe is your [symptom]? (mild/moderate/severe)' replacing [symptom] with the symptom. "
        "When asking about duration, use 'How long have you had your [symptom]? (days/weeks)' replacing [symptom] with the symptom. "
        "When asking about triggers, tailor the question to the symptom using the following examples:\n"
        "- For 'headache': 'Do you feel your headache is caused by any specific factors (e.g., stress, noise, lack of sleep)? If you don’t know, please say so.'\n"
        "- For 'inability to sleep': 'Do you feel your inability to sleep is caused by any specific factors (e.g., stress, caffeine, environment)? If you don’t know, please say so.'\n"
        "- For 'rash': 'Do you feel your rash is caused by any specific factors (e.g., allergies, irritants, infections)? If you don’t know, please say so.'\n"
        "- For 'fever': 'Do you feel your fever is caused by any specific factors (e.g., infection, weather, fatigue)? If you don’t know, please say so.'\n"
        "- For any other symptom: 'Do you feel your [symptom] is caused by any specific factors? If you don’t know, please say so.'\n"
        "Be flexible with patient responses: accept multi-word answers, phrases, or sentences. For example, if the patient says 'I don’t know' for triggers, accept it as a valid response ('unknown') and move to the next step. "
        "Do not repeat a question if the patient has already provided an answer for that category (e.g., if duration is answered as 'three days', do not ask for duration again). "
        "End the conversation if all four categories (symptoms, severity, duration, triggers) are answered. When all categories are answered, ask the patient if you can recommend a doctor for them using: 'May I recommend a doctor for you?' The patient can respond with 'proceed with consultation' or express their opinion in a generic manner such as a simple 'okay' or 'yes'. "
        "ALWAYS return a valid JSON object with 'response' (string, containing only the conversational text) and 'medical_data' (object with fields symptoms, severity, duration, triggers — null if not provided), formatted as: {{'response': '<your response>', 'medical_data': {{'symptoms': null, 'severity': null, 'duration': null, 'triggers': null}}}}. "
        "Do not include the JSON object itself in the 'response' string. Ensure proper JSON formatting with no extra text.\n\n"
        "Conversation history:\n{history}\n\n"
        "Patient: {input}\n"
        "Agent:"
    )
)

chain = RunnablePassthrough.assign(history=RunnableLambda(lambda x: "" if not x.get("history") else x["history"])) | prompt_template | llm

# Define the prompt for interpreting user responses using ChatPromptTemplate
interpret_response_prompt = PromptTemplate(
    input_variables=["history", "input", "current_question"],
    template=(
        "You are a professional healthcare agent assisting a patient in a conversational flow. Your task is to interpret the patient's response to determine the category and value of their input, or to identify if the response indicates agreement to proceed with a doctor recommendation. The conversation may contain noise (e.g., irrelevant words, typos, or misheard phrases), so focus on extracting the intent and key information.\n\n"
        "The patient provides input in English. The current question is: {current_question}\n\n"
        "Categories to identify:\n"
        "- symptoms (e.g., 'fever', 'headache', 'inability to sleep')\n"
        "- severity (e.g., 'mild', 'moderate', 'severe')\n"
        "- duration (e.g., '2 days', 'a few weeks')\n"
        "- triggers (e.g., 'weather', 'infection', 'fatigue')\n"
        "- agreement to proceed (e.g., 'yes', 'ok', 'sure', 'proceed', 'assign me a doctor')\n\n"
        "Rules:\n"
        "1. **Extract Key Information Despite Noise**:\n"
        "   - Ignore irrelevant words or noise (e.g., 'Prasad Weather' should be interpreted as 'weather' for triggers).\n"
        "   - Recognize synonyms or variations (e.g., 'slight fever' or 'slightly bigger' should map to 'fever' with severity 'mild').\n"
        "   - For triggers, accept any reasonable cause (e.g., 'weather', 'stress', 'infection') even if the response is noisy.\n"
        "2. **Identify Agreement to Proceed**:\n"
        "   - If the input contains phrases like 'yes', 'ok', 'okay', 'sure', 'proceed', 'assign me a doctor', or similar, set the category to 'proceed' and value to 'yes'.\n"
        "3. **Handle Unclear Responses**:\n"
        "   - If the response is ambiguous or lacks clear information for the current question, set 'needs_clarification' to true.\n"
        "   - For triggers, if the response is unclear after clarification, default to 'unknown'.\n"
        "4. **Return Structured Output**:\n"
        "   - Always return a JSON object with 'category' (string or null), 'value' (string or null), and 'needs_clarification' (boolean).\n"
        "   - If the response matches the current question, set 'category' to the question type (e.g., 'symptoms', 'severity'), 'value' to the extracted value, and 'needs_clarification' to false.\n"
        "   - If the response indicates agreement to proceed, set 'category' to 'proceed', 'value' to 'yes', and 'needs_clarification' to false.\n\n"
        "Conversation history:\n{history}\n\n"
        "Patient: {input}\n"
        "Agent:"
    )
)

# Define the LLM
llm = ChatOpenAI(model_name="gpt-4o-mini", temperature=0.7)

# Create a parser to ensure JSON output
def parse_json_output(output):
    try:
        return json.loads(output.content)
    except json.JSONDecodeError as e:
        return {"category": None, "value": None, "needs_clarification": True}

# Create the Runnable Sequence for interpreting responses
interpret_response_chain = RunnableSequence(
    interpret_response_prompt,  # Step 1: Format the prompt with input variables
    llm,                       # Step 2: Pass the formatted prompt to the LLM
    parse_json_output          # Step 3: Parse the LLM's output as JSON
)

# Decorator to require Bearer Token
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
            # Generate a poem for token missing error
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

symptom_specialty_map = {
    'rash': 'Dermatology',
    'itchiness': 'Dermatology',
    'skin irritation': 'Dermatology',
    'sore throat': 'ENT',
    'ear pain': 'ENT',
    'nasal congestion': 'ENT',
    'swelling': 'Nephrology',
    'urination issues': 'Nephrology',
    'kidney pain': 'Nephrology',
    'fatigue': 'Endocrinology',
    'weight changes': 'Endocrinology',
    'thirst': 'Endocrinology',
    'stomach pain': 'Gastroenterology',
    'nausea': 'Gastroenterology',
    'diarrhea': 'Gastroenterology',
    'urinary tract infection': 'Urology',
    'lower back pain': 'Urology',
    'difficulty urinating': 'Urology',
    'anxiety': 'Psychiatry',
    'depression': 'Psychiatry',
    'sleep issues': 'Psychiatry',
    'inability to sleep': 'Psychiatry',
    'general weakness': 'General Medicine',
    'fever': 'General Medicine',
    'body aches': 'General Medicine',
    'abdominal pain': 'General Surgery',
    'injury': 'General Surgery',
    'swelling (surgical)': 'General Surgery',
    'menstrual issues': 'Gynecology',
    'pelvic pain': 'Gynecology',
    'irregular periods': 'Gynecology',
    'unexplained weight loss': 'Oncology',
    'lump': 'Oncology',
    'chronic fatigue': 'Oncology',
    'child fever': 'Pediatrics',
    'child cough': 'Pediatrics',
    'child rash': 'Pediatrics',
    'chest pain': 'Cardiology',
    'shortness of breath': 'Cardiology',
    'palpitations': 'Cardiology',
    'headache': 'Neurology',
    'dizziness': 'Neurology',
    'seizures': 'Neurology',
    'joint pain': 'Orthopedics',
    'muscle pain': 'Orthopedics',
    'fracture': 'Orthopedics',
    'unknown': 'General Medicine'
}

def fetch_available_doctors(specialty):
    try:
        # Log the specialty being queried
        logger.debug(f"Fetching doctors for specialty: {specialty}")
        
        # Perform a query to find available doctors
        doctors_ref = db.collection('consultant_registrations').where(
            filter=firestore.FieldFilter('specialty', '==', specialty)
        ).where(
            filter=firestore.FieldFilter('availability', '==', True)
        )
        docs = doctors_ref.get()
        
        doctors_list = []
        for doc in docs:
            doc_data = doc.to_dict()
            doctors_list.append({
                'id': doc.id,
                'full_name': doc_data.get('full_name', 'Unknown Consultant')
            })
            logger.debug(f"Found doctor: {doc_data.get('full_name')} (ID: {doc.id}, Specialty: {doc_data.get('specialty')}, Availability: {doc_data.get('availability')})")
        
        if not doctors_list:
            logger.warning(f"No doctors found for specialty: {specialty} with availability: True")
            
            # Additional debugging: fetch all doctors to see what's in the collection
            all_docs = db.collection('consultant_registrations').get()
            for doc in all_docs:
                doc_data = doc.to_dict()
                logger.debug(f"Available in collection: ID: {doc.id}, Specialty: {doc_data.get('specialty')}, Availability: {doc_data.get('availability')}")
        
        return doctors_list
    except Exception as e:
        logger.error(f"Error fetching available doctors: {str(e)}")
        return []

# Symptom to specialty mapping
symptom_specialty_map = {
    'headache': 'Neurology',
    'inability to sleep': 'Psychiatry',
    'fatigue': 'General Medicine',
    'fever': 'General Medicine',
    'rash': 'Dermatology'
}

# Symptom to specialty mapping
symptom_specialty_map = {
    'headache': 'Neurology',
    'inability to sleep': 'Psychiatry',
    'fatigue': 'General Medicine',
    'fever': 'General Medicine',
    'rash': 'Dermatology'
}

def assign_doctor(medical_data):
    if not medical_data or not isinstance(medical_data, dict) or 'symptoms' not in medical_data or not medical_data['symptoms']:
        logger.error(f"Invalid or missing medical_data: {medical_data}. Using default empty dict.")
        medical_data = {"symptoms": "", "severity": "", "duration": "", "triggers": ""}
    
    symptoms = medical_data.get('symptoms', '').lower()
    severity = medical_data.get('severity', '').lower()
    specialty = 'unknown'

    logger.debug(f"Processing symptoms: {symptoms}, severity: {severity}")

    # Override specialty if specified (e.g., for general physician fallback)
    if "specialty_override" in medical_data:
        specialty = medical_data["specialty_override"]
        logger.debug(f"Specialty overridden to: {specialty}")
    else:
        # Special handling for headaches based on severity
        if symptoms == 'headache':
            if severity == 'mild':
                specialty = 'General Medicine'
                logger.debug(f"Assigned specialty: {specialty} for mild headache")
            else:  # moderate, severe, or unspecified severity
                specialty = 'Neurology'
                logger.debug(f"Assigned specialty: {specialty} for headache (severity: {severity or 'unspecified'})")
        # Fallback to general symptom mapping using the full phrase
        elif symptoms in symptom_specialty_map:
            specialty = symptom_specialty_map[symptoms]
            logger.debug(f"Assigned specialty: {specialty} for symptom: {symptoms}")
        else:
            logger.warning(f"Symptom '{symptoms}' not found in symptom_specialty_map. Defaulting to specialty: {specialty}")
    
    logger.debug(f"Final assigned specialty: {specialty}")
    
    doctors_ref = db.collection('consultant_registrations').where(filter=firestore.FieldFilter('specialty', '==', specialty)).where(filter=firestore.FieldFilter('availability', '==', True))
    docs = doctors_ref.get()
    for doc in docs:
        consultant_id = doc.id
        logger.info(f"Assigned doctor with consultant_id: {consultant_id} for specialty: {specialty}")
        return consultant_id, specialty
    
    logger.warning(f"No available doctor found for specialty: {specialty}")
    return None, specialty

def detect_language(transcription, history=""):
    # Since we're only supporting English, always return "en"
    session['preferred_language'] = "en"
    return "en"

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

def process_conversation(audio_path=None, transcript=None, history=""):
    try:
        language = detect_language(transcript, history)
        logger.debug(f"Detected language: {language}")

        # Define medical_data early to avoid undefined variable issues
        medical_data = session.get('medical_data', {
            "symptoms": None, "severity": None, "duration": None, "triggers": None,
            "symptoms_english": None, "severity_english": None, "duration_english": None, "triggers_english": None
        })

        # Define transcription early to avoid undefined variable issues
        transcription = ""
        if audio_path:
            transcription = transcribe_audio(audio_path, language=language)
            if "too small" in transcription or "failed" in transcription:
                return {
                    "response": f"Audio issue detected: {transcription}. Please re-record.",
                    "medical_data": medical_data,
                    "audio_url": None
                }, 400
            language = detect_language(transcription, history)
            logger.debug(f"Re-detected language after transcription: {language}")
        elif transcript:
            transcription = transcript
        else:
            intro = "Hello, I am your healthcare assistant. I am here to know more about your health conditions and assign you with the consultant."
            audio_path = synthesize_audio(intro, language)
            logger.debug(f"Initial greeting generated: {intro}, audio_path: {audio_path}")
            return {
                "response": intro,
                "medical_data": medical_data,
                "audio_url": f"/static/{os.path.basename(audio_path)}" if audio_path else None
            }, 200

        transcription_lower = transcription.lower() if transcription else ""

        # Use a consistent document ID for initial_screenings based on UID
        uid = request.user.get("uid")
        doc_id = f'initial_screening_{uid}'
        doc_ref = db.collection('initial_screenings').document(doc_id)

        if 'pending_doctor_selection' in session and transcript:
            doctors_list = session.get('pending_doctor_selection', [])
            specialty = session.get('pending_specialty', 'unknown')
            medical_data = session.get('pending_medical_data', {
                "symptoms": "", "severity": "", "duration": "", "triggers": ""
            })
            uid = session.get('pending_uid')
            retries = session.get('doctor_selection_retries', 0)

            if not doctors_list or not uid:
                logger.error("Missing session data for doctor selection")
                return {
                    "response": "An error occurred. Please restart the conversation.",
                    "medical_data": medical_data,
                    "audio_url": None
                }, 500

            selected_doctor = None

            if "anyone" in transcription_lower:
                selected_doctor = random.choice(doctors_list)
                logger.debug(f"Randomly selected doctor: {selected_doctor['full_name']} (ID: {selected_doctor['id']})")
            else:
                for doctor in doctors_list:
                    doctor_name_lower = doctor['full_name'].lower()
                    if doctor_name_lower in transcription_lower or doctor_name_lower.replace("dr.", "").strip() in transcription_lower:
                        selected_doctor = doctor
                        logger.debug(f"Selected doctor by name: {selected_doctor['full_name']} (ID: {selected_doctor['id']})")
                        break

            if selected_doctor:
                consultant_id = selected_doctor['id']
                medical_data['consultant_id'] = consultant_id
                medical_data['consultant_name'] = selected_doctor['full_name']

                # Update initial_screenings with the full medical_data, including consultant_id
                updated_data = {
                    'consultant_id': consultant_id,
                    'symptoms': medical_data.get('symptoms', ''),
                    'severity': medical_data.get('severity', ''),
                    'duration': medical_data.get('duration', ''),
                    'triggers': medical_data.get('triggers', ''),
                    'patient_name': session.get('user_info', {}).get('full_name', uid),
                    'uid': uid,
                    'timestamp': firestore.SERVER_TIMESTAMP
                }
                doc_ref.set(updated_data, merge=True)
                logger.info(f"Updated initial_screenings with full medical_data for UID: {uid}")

                patient_ref = db.collection('patient_registrations').document(uid)
                patient_snap = patient_ref.get()
                if patient_snap.exists:
                    patient_data = patient_snap.to_dict()
                    logger.debug(f"Before update, patient_registrations for UID {uid}: {patient_data}")
                    try:
                        patient_ref.set({'consultant_id': consultant_id}, merge=True)
                        logger.info(f"Updated patient_registrations with consultant_id: {consultant_id} for UID: {uid}")
                    except Exception as e:
                        logger.error(f"Failed to update patient_registrations: {str(e)}")
                        raise
                else:
                    patient_data = {
                        'uid': uid,
                        'email': session.get('user_info', {}).get('email', ''),
                        'full_name': session.get('user_info', {}).get('full_name', uid),
                        'consultant_id': consultant_id,
                        'role': 'patient'
                    }
                    try:
                        patient_ref.set(patient_data)
                        logger.info(f"Created new patient_registrations document with consultant_id: {consultant_id} for UID: {uid}")
                    except Exception as e:
                        logger.error(f"Failed to create patient_registrations: {str(e)}")
                        raise

                session.pop('pending_doctor_selection', None)
                session.pop('pending_specialty', None)
                session.pop('pending_medical_data', None)
                session.pop('pending_uid', None)
                session.pop('doctor_selection_retries', None)

                response_text = f"You have been assigned to {selected_doctor['full_name']}. Please log in to access your dashboard and view further details."
                
                audio_path = synthesize_audio(response_text, language)
                return {
                    "response": response_text,
                    "medical_data": medical_data,
                    "audio_url": f"/static/{os.path.basename(audio_path)}" if audio_path else None,
                    "redirect": "/login"
                }, 200
            else:
                retries += 1
                session['doctor_selection_retries'] = retries

                if retries >= 3:
                    response_text = "I'm sorry, I couldn't understand your doctor preference. Please try again later or contact support."
                    
                    audio_path = synthesize_audio(response_text, language)
                    session.pop('pending_doctor_selection', None)
                    session.pop('pending_specialty', None)
                    session.pop('pending_medical_data', None)
                    session.pop('pending_uid', None)
                    session.pop('doctor_selection_retries', None)
                    return {
                        "response": response_text,
                        "medical_data": medical_data,
                        "audio_url": f"/static/{os.path.basename(audio_path)}" if audio_path else None
                    }, 200

                clarify_text = "I’m sorry, I didn’t understand. Please specify the doctor’s name, such as " + ", ".join([doctor['full_name'] for doctor in doctors_list[:-1]]) + (" or " if len(doctors_list) > 1 else "") + (doctors_list[-1]['full_name'] if doctors_list else "") + ", or say 'anyone' to proceed with any available consultant."
                
                audio_path = synthesize_audio(clarify_text, language)
                return {
                    "response": clarify_text,
                    "medical_data": medical_data,
                    "audio_url": f"/static/{os.path.basename(audio_path)}" if audio_path else None
                }, 200

        if 'pending_general_physician' in session and transcript:
            # Handle user response to general physician question
            transcript_lower = transcription_lower.strip().rstrip('.')
            if any(phrase in transcript_lower for phrase in ["yes", "sure", "okay", "proceed", "ok", "assign me a doctor", "assign", "doctor"]):
                # User agreed to proceed with a general physician
                medical_data = session.get('pending_medical_data', {
                    "symptoms": "", "severity": "", "duration": "", "triggers": ""
                })
                # Reassign specialty to General Medicine
                consultant_id, _ = assign_doctor({"symptoms": medical_data.get("symptoms_english", ""), 
                                                "severity": medical_data.get("severity_english", ""), 
                                                "duration": medical_data.get("duration_english", ""), 
                                                "triggers": medical_data.get("triggers_english", ""), 
                                                "specialty_override": "General Medicine"})
                if not consultant_id:
                    response_text = "I'm sorry, no general physicians are currently available either. Please try again later or contact support."
                    
                    audio_path = synthesize_audio(response_text, language)
                    session.pop('pending_general_physician', None)
                    session.pop('pending_medical_data', None)
                    return {
                        "response": response_text,
                        "medical_data": medical_data,
                        "audio_url": f"/static/{os.path.basename(audio_path)}" if audio_path else None
                    }, 200

                session['pending_doctor_selection'] = [{"id": consultant_id, "full_name": "Dr. Alka Verma"}]  # Adjust based on Firestore data
                session['pending_specialty'] = "General Medicine"
                session['pending_medical_data'] = medical_data
                session['pending_uid'] = request.user.get('uid')
                session.pop('pending_general_physician', None)

                # Proceed with assignment as usual
                doctors_list = session.get('pending_doctor_selection', [])
                selected_doctor = doctors_list[0]  # Since we know there's only one
                consultant_id = selected_doctor['id']
                medical_data['consultant_id'] = consultant_id
                medical_data['consultant_name'] = selected_doctor['full_name']

                # Update initial_screenings with the full medical_data, including consultant_id
                updated_data = {
                    'consultant_id': consultant_id,
                    'symptoms': medical_data.get('symptoms', ''),
                    'severity': medical_data.get('severity', ''),
                    'duration': medical_data.get('duration', ''),
                    'triggers': medical_data.get('triggers', ''),
                    'patient_name': session.get('user_info', {}).get('full_name', request.user.get('uid')),
                    'uid': request.user.get('uid'),
                    'timestamp': firestore.SERVER_TIMESTAMP
                }
                doc_ref.set(updated_data, merge=True)
                logger.info(f"Updated initial_screenings with full medical_data for UID: {request.user.get('uid')}")

                patient_ref = db.collection('patient_registrations').document(request.user.get('uid'))
                patient_snap = patient_ref.get()
                if patient_snap.exists:
                    patient_data = patient_snap.to_dict()
                    logger.debug(f"Before update, patient_registrations for UID {request.user.get('uid')}: {patient_data}")
                    try:
                        patient_ref.set({'consultant_id': consultant_id}, merge=True)
                        logger.info(f"Updated patient_registrations with consultant_id: {consultant_id} for UID: {request.user.get('uid')}")
                    except Exception as e:
                        logger.error(f"Failed to update patient_registrations: {str(e)}")
                        raise
                else:
                    patient_data = {
                        'uid': request.user.get('uid'),
                        'email': request.user.get('email', ''),
                        'full_name': session.get('user_info', {}).get('full_name', request.user.get('uid')),
                        'consultant_id': consultant_id,
                        'role': 'patient'
                    }
                    try:
                        patient_ref.set(patient_data)
                        logger.info(f"Created new patient_registrations document with consultant_id: {consultant_id} for UID: {request.user.get('uid')}")
                    except Exception as e:
                        logger.error(f"Failed to create patient_registrations: {str(e)}")
                        raise

                session.pop('pending_doctor_selection', None)
                session.pop('pending_specialty', None)
                session.pop('pending_medical_data', None)
                session.pop('pending_uid', None)

                response_text = f"You have been assigned to {selected_doctor['full_name']}. Please log in to access your dashboard and view further details."
                
                audio_path = synthesize_audio(response_text, language)
                return {
                    "response": response_text,
                    "medical_data": medical_data,
                    "audio_url": f"/static/{os.path.basename(audio_path)}" if audio_path else None,
                    "redirect": "/login"
                }, 200
            else:
                # User declined to proceed with a general physician
                response_text = "I understand. Please try again later when a specialist is available, or contact support for further assistance."
                
                audio_path = synthesize_audio(response_text, language)
                session.pop('pending_general_physician', None)
                session.pop('pending_medical_data', None)
                return {
                    "response": response_text,
                    "medical_data": medical_data,
                    "audio_url": f"/static/{os.path.basename(audio_path)}" if audio_path else None
                }, 200

        proceed_with_doctor_assignment = False
        clarification_attempts = session.get('clarification_attempts', 0)

        if transcription:
            current_question = None
            if not medical_data.get("symptoms"):
                current_question = "symptoms"
            elif not medical_data.get("severity"):
                current_question = "severity"
            elif not medical_data.get("duration"):
                current_question = "duration"
            elif not medical_data.get("triggers"):
                current_question = "triggers"

            logger.debug(f"Current question: {current_question}, User response: {transcription}")

            current_question_text = f"Question about {current_question}"

            try:
                interpretation = interpret_response_chain.invoke({
                    "history": history,
                    "input": transcription,
                    "current_question": current_question_text
                })
                logger.debug(f"LLM interpretation: {interpretation}")

                if interpretation.get("category") == "proceed" and interpretation.get("value") == "yes":
                    proceed_with_doctor_assignment = True
                    session['clarification_attempts'] = 0  # Reset clarification attempts
                    logger.debug("User agreed to proceed with doctor assignment via LLM interpretation")
                elif interpretation.get("needs_clarification", True) and current_question != "triggers":
                    clarification_attempts += 1
                    session['clarification_attempts'] = clarification_attempts
                    logger.debug(f"LLM indicates response needs clarification. Attempt {clarification_attempts}.")

                    if clarification_attempts >= 2:
                        # After 2 clarification attempts, set a default value and proceed
                        if current_question == "severity":
                            medical_data['severity'] = "mild"
                            medical_data['severity_english'] = "mild"
                            logger.debug("Set default severity to 'mild' after max clarification attempts.")
                        elif current_question == "duration":
                            medical_data['duration'] = "unspecified"
                            medical_data['duration_english'] = "unspecified"
                            logger.debug("Set default duration to 'unspecified' after max clarification attempts.")
                        session['clarification_attempts'] = 0  # Reset attempts
                    else:
                        # Ask for clarification
                        response_text = f"I’m sorry, your response is not clear. Could you please clarify your {current_question}?"
                        audio_path = synthesize_audio(response_text, language)
                        return {
                            "response": response_text,
                            "medical_data": medical_data,
                            "audio_url": f"/static/{os.path.basename(audio_path)}" if audio_path else None
                        }, 200
                else:
                    session['clarification_attempts'] = 0  # Reset attempts on successful interpretation
                    category = interpretation.get("category")
                    value = interpretation.get("value")
                    if category and value:
                        medical_data[category] = value if category in ["symptoms", "severity", "duration", "triggers"] else transcription
                        medical_data[f"{category}_english"] = value
                        session['medical_data'] = medical_data
                        logger.debug(f"Updated medical_data from LLM: {medical_data}")
                    elif current_question == "triggers":
                        medical_data['triggers'] = "unknown"
                        medical_data['triggers_english'] = "unknown"
                        session['medical_data'] = medical_data
                        logger.debug(f"Set triggers to 'unknown' due to unclear response: {transcription}")
            except Exception as e:
                logger.warning(f"Failed to interpret response with LLM: {str(e)}. Falling back to manual extraction.")

                if current_question == "symptoms":
                    if 'fever' in transcription_lower or 'slight fever' in transcription_lower or 'slightly bigger' in transcription_lower:
                        medical_data['symptoms'] = 'fever'
                        medical_data['symptoms_english'] = 'fever'
                        if 'slight' in transcription_lower or 'slightly' in transcription_lower:
                            medical_data['severity'] = 'mild'
                            medical_data['severity_english'] = 'mild'
                    elif 'headache' in transcription_lower:
                        medical_data['symptoms'] = 'headache'
                        medical_data['symptoms_english'] = 'headache'
                    elif 'sleep' in transcription_lower and 'not' in transcription_lower:
                        medical_data['symptoms'] = 'inability to sleep'
                        medical_data['symptoms_english'] = 'inability to sleep'
                    elif 'fatigue' in transcription_lower:
                        medical_data['symptoms'] = 'fatigue'
                        medical_data['symptoms_english'] = 'fatigue'
                elif current_question == "severity":
                    if any(word in transcription_lower for word in ['mild', 'slight']):
                        medical_data['severity'] = 'mild'
                        medical_data['severity_english'] = 'mild'
                    elif 'moderate' in transcription_lower:
                        medical_data['severity'] = 'moderate'
                        medical_data['severity_english'] = 'moderate'
                    elif any(word in transcription_lower for word in ['severe', 'bad']):
                        medical_data['severity'] = 'severe'
                        medical_data['severity_english'] = 'severe'
                    else:
                        # If severity is not clear, increment clarification attempts
                        clarification_attempts += 1
                        session['clarification_attempts'] = clarification_attempts
                        logger.debug(f"Severity unclear. Attempt {clarification_attempts}.")
                        if clarification_attempts >= 2:
                            medical_data['severity'] = "mild"
                            medical_data['severity_english'] = "mild"
                            logger.debug("Set default severity to 'mild' after max clarification attempts.")
                            session['clarification_attempts'] = 0
                        else:
                            response_text = "I’m sorry, your response is not clear. Could you please specify the severity of your fever (mild, moderate, or severe)?"
                            audio_path = synthesize_audio(response_text, language)
                            return {
                                "response": response_text,
                                "medical_data": medical_data,
                                "audio_url": f"/static/{os.path.basename(audio_path)}" if audio_path else None
                            }, 200
                elif current_question == "duration":
                    duration_value = None
                    if 'days' in transcription_lower:
                        # Extract number of days, including ranges like "2-3"
                        words = transcription_lower.split()
                        for word in words:
                            if word.isdigit() or '-' in word or 'few' in word or 'some' in word:
                                duration_value = word if word.isdigit() or '-' in word else ("a few" if 'few' in word else "some")
                                medical_data['duration'] = f"{duration_value} days"
                                medical_data['duration_english'] = f"{duration_value} days"
                                break
                    elif 'weeks' in transcription_lower:
                        # Extract number of weeks
                        words = transcription_lower.split()
                        for word in words:
                            if word.isdigit() or '-' in word or 'few' in word or 'some' in word:
                                duration_value = word if word.isdigit() or '-' in word else ("a few" if 'few' in word else "some")
                                medical_data['duration'] = f"{duration_value} weeks"
                                medical_data['duration_english'] = f"{duration_value} weeks"
                                break
                    if not duration_value:
                        # If duration is vague (e.g., just "weeks"), increment clarification attempts
                        clarification_attempts += 1
                        session['clarification_attempts'] = clarification_attempts
                        logger.debug(f"Duration unclear or vague. Attempt {clarification_attempts}.")
                        if clarification_attempts >= 2:
                            medical_data['duration'] = "unspecified"
                            medical_data['duration_english'] = "unspecified"
                            logger.debug("Set default duration to 'unspecified' after max clarification attempts.")
                            session['clarification_attempts'] = 0
                        else:
                            response_text = "I’m sorry, your response is not clear. Could you please specify how many days or weeks you have had your symptoms? For example, '2 days' or 'a few weeks'."
                            audio_path = synthesize_audio(response_text, language)
                            return {
                                "response": response_text,
                                "medical_data": medical_data,
                                "audio_url": f"/static/{os.path.basename(audio_path)}" if audio_path else None
                            }, 200
                    # Fallback if duration is not specified
                    if not medical_data.get("duration"):
                        medical_data['duration'] = "unspecified"
                        medical_data['duration_english'] = "unspecified"
                elif current_question == "triggers":
                    # Flexible trigger extraction
                    trigger_words = ['weather', 'infection', 'fatigue', 'stress', 'noise', 'lack of sleep', 'environment', 'caffeine', 'allergies', 'irritants']
                    found_trigger = None
                    for trigger in trigger_words:
                        if trigger in transcription_lower:
                            found_trigger = trigger
                            break
                    if found_trigger:
                        medical_data['triggers'] = found_trigger
                        medical_data['triggers_english'] = found_trigger
                        logger.debug(f"Extracted trigger: {found_trigger}")
                    elif any(phrase in transcription_lower for phrase in ['don\'t know', 'not sure', 'unsure']) or not transcription:
                        medical_data['triggers'] = "unknown"
                        medical_data['triggers_english'] = "unknown"
                        logger.debug("Set triggers to 'unknown' due to user uncertainty")
                    else:
                        # Increment clarification attempts for unclear triggers
                        clarification_attempts += 1
                        session['clarification_attempts'] = clarification_attempts
                        logger.debug(f"Trigger unclear. Attempt {clarification_attempts}.")
                        if clarification_attempts >= 2:
                            medical_data['triggers'] = "unknown"
                            medical_data['triggers_english'] = "unknown"
                            logger.debug("Set default trigger to 'unknown' after max clarification attempts.")
                            session['clarification_attempts'] = 0
                        else:
                            response_text = "I’m sorry, your response is not clear. Could you please clarify if you feel your fever is caused by any specific factors (e.g., infection, weather, fatigue)? If you don’t know, please say so."
                            audio_path = synthesize_audio(response_text, language)
                            return {
                                "response": response_text,
                                "medical_data": medical_data,
                                "audio_url": f"/static/{os.path.basename(audio_path)}" if audio_path else None
                            }, 200

        session['medical_data'] = medical_data
        logger.debug(f"Updated medical_data: {medical_data}")

        # Incrementally update initial_screenings with the current medical_data
        updated_data = {
            'symptoms': medical_data.get('symptoms', ''),
            'severity': medical_data.get('severity', ''),
            'duration': medical_data.get('duration', ''),
            'triggers': medical_data.get('triggers', ''),
            'patient_name': session.get('user_info', {}).get('full_name', uid),
            'uid': uid,
            'timestamp': firestore.SERVER_TIMESTAMP
        }
        doc_ref.set(updated_data, merge=True)
        logger.info(f"Incrementally updated initial_screenings for UID: {uid} with data: {updated_data}")

        history_lines = history.split('\n')
        user_has_agreed = False
        for line in history_lines:
            if line.startswith('Patient:'):
                patient_response = line[len('Patient:'):].strip()
                patient_response_lower = patient_response.lower()
                if any(phrase in patient_response_lower for phrase in [
                    "proceed with consultation", "okay", "yes", "sure", "ok", "assign me a doctor", "assign", "doctor"
                ]):
                    user_has_agreed = True
                    break

        if any(phrase in transcription_lower for phrase in [
            "proceed with consultation", "okay", "yes", "sure", "ok", "assign me a doctor", "assign", "doctor"
        ]):
            user_has_agreed = True
            proceed_with_doctor_assignment = True
            logger.debug("User agreed to proceed with doctor assignment via manual check")

        if user_has_agreed and all(medical_data.get(key) for key in ["symptoms", "severity", "duration", "triggers"]):
            english_medical_data = {
                "symptoms": medical_data.get("symptoms_english", ""),
                "severity": medical_data.get("severity_english", ""),
                "duration": medical_data.get("duration_english", ""),
                "triggers": medical_data.get("triggers_english", "")
            }
            consultant_id, specialty = assign_doctor(english_medical_data)
            logger.debug(f"Assigned consultant_id: {consultant_id}, specialty: {specialty}")

            if not consultant_id:
                response_text = f"I'm sorry, no doctors are currently available for the specialty '{specialty}'. Would you like to proceed with a general physician instead?"
                
                audio_path = synthesize_audio(response_text, language)
                session['pending_general_physician'] = True
                session['pending_medical_data'] = medical_data
                return {
                    "response": response_text,
                    "medical_data": medical_data,
                    "audio_url": f"/static/{os.path.basename(audio_path)}" if audio_path else None
                }, 200

            session['pending_doctor_selection'] = [{"id": consultant_id, "full_name": "Dr. Priya Sharma"}]  # Adjust based on Firestore data
            session['pending_specialty'] = specialty
            session['pending_medical_data'] = medical_data
            session['pending_uid'] = request.user.get('uid')

            if len(session['pending_doctor_selection']) == 1:
                consultant_id = session['pending_doctor_selection'][0]['id']
                medical_data['consultant_id'] = consultant_id
                medical_data['consultant_name'] = session['pending_doctor_selection'][0]['full_name']
                logger.debug(f"Assigned consultant ID: {consultant_id}")

                # Update initial_screenings with the full medical_data, including consultant_id
                updated_data = {
                    'consultant_id': consultant_id,
                    'symptoms': medical_data.get('symptoms', ''),
                    'severity': medical_data.get('severity', ''),
                    'duration': medical_data.get('duration', ''),
                    'triggers': medical_data.get('triggers', ''),
                    'patient_name': session.get('user_info', {}).get('full_name', request.user.get('uid')),
                    'uid': request.user.get('uid'),
                    'timestamp': firestore.SERVER_TIMESTAMP
                }
                doc_ref.set(updated_data, merge=True)
                logger.info(f"Updated initial_screenings with full medical_data for UID: {request.user.get('uid')}")

                patient_ref = db.collection('patient_registrations').document(request.user.get('uid'))
                patient_snap = patient_ref.get()
                if patient_snap.exists:
                    patient_data = patient_snap.to_dict()
                    logger.debug(f"Before update, patient_registrations for UID {request.user.get('uid')}: {patient_data}")
                    try:
                        patient_ref.set({'consultant_id': consultant_id}, merge=True)
                        logger.info(f"Updated patient_registrations with consultant_id: {consultant_id} for UID: {request.user.get('uid')}")
                    except Exception as e:
                        logger.error(f"Failed to update patient_registrations: {str(e)}")
                        raise
                else:
                    patient_data = {
                        'uid': request.user.get('uid'),
                        'email': request.user.get('email', ''),
                        'full_name': session.get('user_info', {}).get('full_name', request.user.get('uid')),
                        'consultant_id': consultant_id,
                        'role': 'patient'
                    }
                    try:
                        patient_ref.set(patient_data)
                        logger.info(f"Created new patient_registrations document with consultant_id: {consultant_id} for UID: {request.user.get('uid')}")
                    except Exception as e:
                        logger.error(f"Failed to create patient_registrations: {str(e)}")
                        raise

                session.pop('pending_doctor_selection', None)
                session.pop('pending_specialty', None)
                session.pop('pending_medical_data', None)
                session.pop('pending_uid', None)

                response_text = f"You have been assigned to {medical_data['consultant_name']}. Please log in to access your dashboard and view further details."
                
                audio_path = synthesize_audio(response_text, language)
                return {
                    "response": response_text,
                    "medical_data": medical_data,
                    "audio_url": f"/static/{os.path.basename(audio_path)}" if audio_path else None,
                    "redirect": "/login"
                }, 200

            doctor_names = [doctor['full_name'] for doctor in session['pending_doctor_selection']]
            response_text = f"With whom would you be comfortable? For example, you can choose {', '.join(doctor_names[:-1]) + (' or ' if len(doctor_names) > 1 else '') + (doctor_names[-1] if doctor_names else '')}. Alternatively, you can say 'anyone' to proceed with any available consultant."
            
            audio_path = synthesize_audio(response_text, language)
            return {
                "response": response_text,
                "medical_data": medical_data,
                "audio_url": f"/static/{os.path.basename(audio_path)}" if audio_path else None
            }, 200

        logger.debug(f"Invoking chain with: 'input': {transcription}, 'history': {history}")
        response = chain.invoke({
            "input": transcription,
            "history": history
        })

        data = {}
        response_text = ""
        if hasattr(response, "content"):
            content = response.content
            logger.debug(f"Raw LLM content: {content}")
            try:
                data = json.loads(content) if isinstance(content, str) else content
                if not isinstance(data, dict) or "response" not in data:
                    logger.warning(f"LLM response content is not a valid JSON object with 'response' key: {data}")
                    response_text = content
                    data = {
                        "response": response_text,
                        "medical_data": medical_data
                    }
                else:
                    response_text = data["response"]
                    if "medical_data" in data and isinstance(data["medical_data"], dict):
                        for key in ['symptoms', 'severity', 'duration', 'triggers']:
                            if data["medical_data"].get(key) is not None:
                                medical_data[key] = data["medical_data"][key]
                                medical_data[f"{key}_english"] = data["medical_data"][key]
                        session['medical_data'] = medical_data
                    data["medical_data"] = medical_data
            except json.JSONDecodeError:
                logger.warning(f"LLM content is not JSON, treating as plain text: {content}")
                response_text = content
                data = {
                    "response": response_text,
                    "medical_data": medical_data
                }
        elif hasattr(response, "text"):
            content = response.text
            logger.debug(f"Raw LLM text: {content}")
            try:
                data = json.loads(content) if isinstance(content, str) else content
                if not isinstance(data, dict) or "response" not in data:
                    logger.warning(f"LLM response text is not a valid JSON object with 'response' key: {data}")
                    response_text = content
                    data = {
                        "response": response_text,
                        "medical_data": medical_data
                    }
                else:
                    response_text = data["response"]
                    if "medical_data" in data and isinstance(data["medical_data"], dict):
                        for key in ['symptoms', 'severity', 'duration', 'triggers']:
                            if data["medical_data"].get(key) is not None:
                                medical_data[key] = data["medical_data"][key]
                                medical_data[f"{key}_english"] = data["medical_data"][key]
                        session['medical_data'] = medical_data
                    data["medical_data"] = medical_data
            except json.JSONDecodeError:
                logger.warning(f"LLM text is not JSON, treating as plain text: {content}")
                response_text = content
                data = {
                    "response": response_text,
                    "medical_data": medical_data
                }
        else:
            logger.error(f"Unexpected LLM response format: {response}")
            response_text = "Error processing response. Please try again."
            data = {
                "response": response_text,
                "medical_data": medical_data
            }

        # Clean up any JSON leakage in response_text
        if "{'response'" in response_text:
            response_text = response_text.split("{'response'")[0].strip()
            if not response_text:
                response_text = data.get("response", "Error processing response. Please try again.")
            logger.debug(f"Cleaned up JSON leakage from response_text: {response_text}")

        if "healthcare assistant" in response_text.lower():
            logger.debug("LLM response already contains a greeting, skipping prepend.")
        else:
            if not history:
                intro = "Hello, I am your healthcare assistant. I am here to know more about your health conditions and assign you with the consultant."
                response_text = intro + " " + response_text

        # Update the data dictionary with the corrected response text
        data["response"] = response_text

        response_audio_path = synthesize_audio(response_text, language)
        audio_url = f"/static/{os.path.basename(response_audio_path)}" if response_audio_path else None
        data["audio_url"] = audio_url

        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        json_filename = f"medical_data_{timestamp}.json"
        json_path = os.path.join("C:", "Users", "gjrah", "Documents", "Major Project", "Voice_Demo_Project", json_filename)
        os.makedirs(os.path.dirname(json_path), exist_ok=True)
        with open(json_path, "w", encoding="utf-8") as f:
            json.dump(data, f, ensure_ascii=False, indent=2)

        return data, 200
    except json.JSONDecodeError as e:
        logger.error(f"Failed to parse LLM response as JSON: {str(e)}\nResponse: {response if 'response' in locals() else 'N/A'}")
        return {
            "response": "Error processing response. Please try again.",
            "medical_data": {
                "symptoms": None, "severity": None, "duration": None, "triggers": None,
                "symptoms_english": None, "severity_english": None, "duration_english": None, "triggers_english": None
            },
            "audio_url": None
        }, 500
    except Exception as e:
        logger.error(f"Request error: {str(e)}\n{traceback.format_exc()}")
        return {
            "response": f"Request processing failed due to an error: {str(e)}",
            "medical_data": {
                "symptoms": None, "severity": None, "duration": None, "triggers": None,
                "symptoms_english": None, "severity_english": None, "duration_english": None, "triggers_english": None
            },
            "audio_url": None
        }, 500
        
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

# --------------------------------------------------------------------
# LOGIN ROUTE (GET/POST) - MODIFIED TO SUPPORT GET
# --------------------------------------------------------------------
@app.route('/login', methods=['GET', 'POST'])
def login():
    if request.method == 'GET':
        try:
            if 'user_info' in session:
                logger.debug("User already logged in, rendering login page to re-authenticate")
                # Check if the session token is valid
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

        # Verify the ID token
        decoded_token = auth.verify_id_token(id_token)
        email = decoded_token.get('email')
        uid = decoded_token.get('uid')
        logger.debug(f"Decoded ID token for email: {email}, uid: {uid}")

        # Log the token type for debugging
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
            else:  # Default to patient
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

        # Use a consistent document ID based on UID
        doc_id = f'initial_screening_{uid}'
        doc_ref = db.collection('initial_screenings').document(doc_id)

        # Check if the document already exists
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
        language = "en"  # Fixed to English
        if 'audio' not in request.files:
            logger.error("No audio file provided in the request")
            return jsonify({"error": "No audio file provided"}), 400

        audio_file = request.files['audio']
        if audio_file.filename == '':
            logger.error("No audio file selected")
            return jsonify({"error": "No audio file selected"}), 400

        # Save the audio file temporarily
        temp_dir = tempfile.mkdtemp()
        audio_path = os.path.join(temp_dir, f"temp_audio_{request.user.get('uid')}.webm")
        audio_file.save(audio_path)
        logger.debug(f"Audio saved to: {audio_path}")

        # Transcribe the audio using Whisper
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
            # Handle JSON transcript
            data = request.get_json()
            transcript = data.get('transcript', '')  # Allow empty transcript
            logger.debug(f"Received transcript for UID {uid}: {transcript}")
            processed_data, status_code = process_conversation(transcript=transcript, history=history)

            if status_code != 200:
                return jsonify(processed_data), status_code

            # Ensure medical_data is valid
            medical_data = processed_data.get('medical_data', {"symptoms": "", "severity": "", "duration": "", "triggers": ""})
            logger.debug(f"Extracted medical_data: {medical_data}")
            if medical_data is None:
                logger.error("medical_data is None. Cannot assign doctor.")
                return jsonify({'error': 'Failed to extract medical data.'}), 400

            # Update conversation history
            session['conversation_history'] = f"{history}\nPatient: {transcript}\nAgent: {processed_data['response']}" if transcript else f"{history}\nAgent: {processed_data['response']}"

            return jsonify({
                'success': True,
                'response': processed_data['response'],
                'medical_data': medical_data,
                'audio_url': processed_data.get('audio_url'),
                'redirect': processed_data.get('redirect', '/dashboard')
            })
        elif 'audio' in request.files:
            # Handle audio file
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

            # Ensure medical_data is valid
            medical_data = processed_data.get('medical_data', {"symptoms": "", "severity": "", "duration": "", "triggers": ""})
            logger.debug(f"Extracted medical_data: {medical_data}")
            if medical_data is None:
                logger.error("medical_data is None. Cannot assign doctor.")
                return jsonify({'error': 'Failed to extract medical data.'}), 400

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

        # Download the image
        blob = bucket.blob(file_path)
        temp_file = os.path.join(tempfile.gettempdir(), f'temp_{uid}.jpg')
        os.makedirs(os.path.dirname(temp_file), exist_ok=True)
        blob.download_to_filename(temp_file)
        logger.debug(f"Image downloaded to: {temp_file}")

        # Extract text using Google Vision
        extracted_text = extract_text_from_image(temp_file)
        logger.debug(f"Extracted English text: {extracted_text[:100]}... (length: {len(extracted_text)})")

        # Fetch existing summaries for prescriptions
        existing_text = None
        if category == 'prescriptions':
            existing_summaries = db.collection('prescriptions').where('uid', '==', uid).order_by('timestamp', direction=firestore.Query.DESCENDING).limit(1).get()
            if existing_summaries:
                existing_text = existing_summaries[0].to_dict().get('summary', '')

        # Fetch patient name (for this project, we'll use a placeholder)
        patient_ref = db.collection('patient_registrations').document(uid).get()
        patient_name = patient_ref.to_dict().get('full_name', 'ರೋಗಿ') if patient_ref.exists else 'ರೋಗಿ'

        # Process with Gemini
        result = process_text_with_gemini(
            extracted_text=extracted_text,
            category=category,
            language=language_text,
            patient_name=patient_name,
            existing_text=existing_text,
            uid=uid
        )

        # Save summaries to Firestore with language metadata
        doc_ref = db.collection(category).document()
        doc_ref.set({
            'uid': uid,
            'consultant_id': consultant_id,
            'patient_name': patient_name,
            'summary': result['regional_summary'],
            'professional_summary': result['professional_summary'],
            'language': result['language'],  # Save the language used for the regional summary
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

        # Use the existing generate_medical_history function from gemini_processor.py
        from gemini_processor import generate_medical_history
        summary = generate_medical_history(uid, db)

        logger.info(f"Generated medical history summary for UID: {uid}")
        return jsonify({'success': True, 'summary': summary})
    except Exception as e:
        logger.error(f"Error processing medical history: {str(e)}")
        return jsonify({'success': False, 'error': str(e)}), 500

# --------------------------------------------------------------------
# DASHBOARD ROUTE (GET)
# --------------------------------------------------------------------
@app.route('/dashboard', methods=['GET'])
@token_required
def dashboard():
    try:
        user = request.user  # Access user from request.user set by @firebase_token_required
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
        return render_template('confirm_logout.html', message="Are you sure you want to log out?")
    session.pop('user_info', None)
    session.pop('idToken', None)
    session.pop('preferred_language', None)
    session.pop('pending_doctor_selection', None)
    session.pop('pending_specialty', None)
    session.pop('pending_medical_data', None)
    session.pop('pending_uid', None)
    logger.info("Session cleared, redirecting to /login")
    return redirect('/login')

# Custom 404 error handler
# Custom 404 error handler (temporary for testing)
@app.errorhandler(404)
def page_not_found(e):
    static_poem = "Blank space resonates\n404 hums creation\nNew worlds now beckon"
    return render_template('errors/404.html', error_poem=static_poem), 404

if __name__ == '__main__':
    port = int(os.environ.get('PORT', 5000))
    logger.info(f"Starting Flask app on port {port}")
    app.run(debug=True, port=port)