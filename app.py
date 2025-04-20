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

def calculate_age(dob_str):
    try:
        # Parse DOB in DD-MM-YYYY format
        dob = datetime.strptime(dob_str, "%d-%m-%Y")
        today = datetime(2025, 4, 20)  # Current date as per your system
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
    temperature=0.0  # Low temperature for precise JSON output
)

conversation_prompt_template = PromptTemplate(
    input_variables=["history", "input", "medical_data"],
    template=(
        "You are a professional healthcare assistant. Your task is to guide the patient through a conversation to gather their symptoms (severity, duration, triggers) and recommend a doctor when ready. The patient may provide unexpected or off-topic inputs, and you must handle these gracefully.\n\n"
        "**Current Medical Data:**\n{medical_data}\n\n"
        "**Conversation History:**\n{history}\n\n"
        "**Patient’s Latest Input:**\n{input}\n\n"
        "**Instructions:**\n"
        "- Start fresh with each session unless history indicates otherwise.\n"
        "- From the patient's input and conversation history, identify and extract all relevant information:\n"
        "  - Symptoms (e.g., 'fever', 'sore throat', 'kidney stones', 'anxiety', 'inability to sleep').\n"
        "  - Severity (e.g., 'mild', 'moderate', 'severe').\n"
        "  - Duration (e.g., '2 days', 'a few weeks').\n"
        "  - Triggers (e.g., 'stress', 'infection', 'unknown').\n"
        "- Add new symptoms to the 'symptoms' list in 'medical_data' if not already present, unless the input explicitly denies the symptom (e.g., 'I don’t have a fever' means 'fever' should not be added).\n"
        "- If the input denies a symptom (e.g., 'I don’t have X', 'No X'), acknowledge it (e.g., 'I understand you’re not experiencing X.') and do not add that symptom to 'medical_data'.\n"
        "- Update the details for each symptom (severity, duration, triggers) in 'medical_data' based on the input and history. The 'medical_data' structure is a dictionary with the following keys:\n"
        "  - 'symptoms': a list of symptom names (e.g., ['sore throat', 'fever']).\n"
        "  - 'severity': a list of severities corresponding to each symptom (e.g., ['mild', 'mild']).\n"
        "  - 'duration': a list of durations corresponding to each symptom (e.g., ['two to three weeks', '']).\n"
        "  - 'triggers': a list of triggers corresponding to each symptom (e.g., ['infection', '']).\n"
        "- **Avoid repeating questions for details already provided**: Before asking for a detail (severity, duration, triggers), check 'medical_data' to see if the corresponding list already has a non-empty value for that symptom. For example, if 'severity' for 'sore throat' is already 'mild', do not ask for the severity of the sore throat again unless the patient's response contradicts previous information.\n"
        "- **Handle ambiguous severity inputs**: If the patient provides an ambiguous severity (e.g., 'mild to moderate', 'sometimes mild, sometimes moderate'), interpret it as the higher severity (e.g., 'moderate') and update 'medical_data' accordingly. Acknowledge the ambiguity by saying: 'I appreciate your clarification about the [symptom] being [interpreted severity].' Then proceed to the next missing detail.\n"
        "- **Handle 'I don't know' responses for triggers**: If the patient responds with 'I don't know', 'I'm not sure', or similar phrases when asked about triggers (e.g., 'Do you feel your [symptom] is caused by any specific factors?'), interpret this as a valid response meaning the triggers are unknown. Update the corresponding 'triggers' entry in 'medical_data' to 'unknown' and treat this as a complete detail. Do not re-ask for the triggers of that symptom; proceed to the next missing detail or to doctor recommendation if all details are complete.\n"
        "- **Handle contradictions**: If the patient's latest input provides a detail (e.g., triggers) that contradicts a previous response (e.g., 'maybe infection' vs. 'unknown'), respond with: 'I noticed you previously mentioned your [symptom] might be caused by [previous detail], but now you’ve said [new detail]. Could you please clarify this for me?' and update 'medical_data' only after clarification.\n"
        "- **Interpret typos and frustration**: If the patient uses phrases like 'I’ve told you before', 'I already said', or 'I said I don’t know', recognize that they are frustrated about repeating information. Acknowledge this with: 'I’m sorry for asking again. I want to ensure I have all the details correct.' Then, use the history and 'medical_data' to confirm the detail and proceed without repeating the question. Also, interpret common typos like 'mind' or 'mine' as 'mild' when asking about severity.\n"
        "- **Accurately attribute triggers to specific symptoms**: When asking about triggers, ensure you specify which symptom you are referring to (e.g., 'Do you feel your sore throat is caused by any specific factors?'). When summarizing, only attribute a trigger to the specific symptom it applies to. For example, if the patient mentions 'dehydration' as a trigger for their sore throat, do not attribute it to other symptoms like cough or fever in the summary unless explicitly stated.\n"
        "- If any details are missing for a symptom, ask for them one at a time in this order: severity, duration, triggers.\n"
        "  - For severity: 'How severe is your [symptom]? (mild/moderate/severe)'\n"
        "  - For duration: 'How long have you had your [symptom]? (e.g., days/weeks)'\n"
        "  - For triggers: 'Do you feel your [symptom] is caused by any specific factors (e.g., stress, infection)? If you don’t know, please say so.'\n"
        "- If the input is unclear or off-topic (e.g., 'What’s the weather like?', 'I need to reschedule'), and it does not match expected responses like 'I don’t know' for triggers, respond with: 'I’m sorry, I’m here to assist with your health concerns. Could you please tell me about any symptoms you’re experiencing?' and continue the conversation.\n"
        "- If the input asks about doctor availability (e.g., 'Is there a doctor available for kidney stones?'), recognize the condition (e.g., 'kidney stones'), map it to a specialty (use the mapping below), and respond with: 'Let me check if a doctor is available for [condition]. First, I need to gather more details about your symptoms. What symptoms are you experiencing related to [condition]?' Then, set 'check_availability' to 'true' and 'condition' to the identified condition (e.g., 'kidney stones').\n"
        "**Condition to Specialty Mapping:**\n"
        "  - kidney stones: Urology\n"
        "  - fever: General Medicine\n"
        "  - sore throat: ENT\n"
        "  - headache: Neurology\n"
        "  - rash: Dermatology\n"
        "  - fatigue: General Medicine\n"
        "  - anxiety: Psychiatry\n"
        "  - inability to sleep: Psychiatry\n"
        "  - dry cough: ENT\n"
        "  - (default for unknown conditions): General Medicine\n"
        "- When all symptom details are complete for all symptoms (i.e., severity, duration, and triggers are non-empty for each symptom, where 'unknown' is considered a valid value for triggers), ask: 'May I recommend a doctor for you?'\n"
        "- If the patient agrees (e.g., 'yes', 'proceed with doctor'), set 'assign_doctor' to 'true'.\n"
        "- Return a plain text response with the following format:\n"
        "  - Response: [Your response text]\n"
        "  - Updated Medical Data: [Updated medical_data as a JSON string]\n"
        "  - Assign Doctor: [true/false]\n"
        "  - Check Availability: [true/false]\n"
        "  - Condition: [condition name or null]\n"
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
        "  - Updated Medical Data: [JSON string]\n"
        "  - Assign Doctor: [true/false]\n"
        "  - Check Availability: [true/false]\n"
        "  - Condition: [condition or null]\n"
        "- Return a JSON object with keys: \"response\", \"medical_data\", \"assign_doctor\", \"check_availability\", \"condition\".\n"
        "- Use double quotes for all property names and string values.\n"
    )
)

# Define the chains
conversation_chain = conversation_prompt_template | conversation_llm
json_chain = json_prompt_template | json_llm

# Define the LLM
llm = ChatOpenAI(model_name="gpt-4o-mini", temperature=0.7)

# Create a parser to ensure JSON output
def parse_json_output(output):
    try:
        return json.loads(output.content)
    except json.JSONDecodeError as e:
        return {"category": None, "value": None, "needs_clarification": True}

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
        medical_data = {"symptoms": [], "severity": [], "duration": [], "triggers": []}
    
    symptoms = medical_data.get('symptoms', [])
    severities = medical_data.get('severity', [])
    durations = medical_data.get('duration', [])
    specialty = 'General Medicine'  # Default specialty if no symptoms match

    # Get user data from session (assumes user data is stored in session)
    user_info = session.get('user_info', {})
    provided_age = user_info.get('age', None)
    dob = user_info.get('dob', None)
    calculated_age = None

    # Calculate age from DOB if provided
    if dob:
        calculated_age = calculate_age(dob)
        if calculated_age is not None:
            logger.debug(f"Calculated age from DOB '{dob}': {calculated_age}")
            if provided_age and int(provided_age) != calculated_age:
                logger.warning(f"Age mismatch: provided age is '{provided_age}', but calculated age from DOB '{dob}' is '{calculated_age}'")
            age_to_use = calculated_age
        else:
            age_to_use = int(provided_age) if provided_age else None
    else:
        age_to_use = int(provided_age) if provided_age else None

    logger.debug(f"Processing symptoms: {symptoms}, Age: {age_to_use}")

    # Override specialty if specified (e.g., for general physician fallback)
    if "specialty_override" in medical_data:
        specialty = medical_data["specialty_override"]
        logger.debug(f"Specialty overridden to: {specialty}")
    else:
        # Determine specialty based on symptoms
        # Prioritize based on severity if available
        max_severity_score = -1  # 2 for severe, 1 for moderate, 0 for mild
        priority_symptom = None
        priority_index = None
        for idx, symptom in enumerate(symptoms):
            symptom_lower = symptom.lower()
            severity = severities[idx].lower() if idx < len(severities) else 'mild'
            severity_score = 2 if severity == 'severe' else 1 if severity == 'moderate' else 0
            if severity_score > max_severity_score:
                max_severity_score = severity_score
                priority_symptom = symptom_lower
                priority_index = idx

        if priority_symptom:
            # Adjust specialty for pediatric patients (under 18)
            if age_to_use is not None and age_to_use < 18 and priority_symptom in ['fever', 'sore throat', 'rash']:
                specialty = 'Pediatrics'
                logger.debug(f"Assigned specialty: {specialty} for symptom '{priority_symptom}' (pediatric patient, age {age_to_use})")
            else:
                # Special handling for headaches based on severity
                if priority_symptom == 'headache':
                    if max_severity_score == 0:  # mild
                        specialty = 'General Medicine'
                        logger.debug(f"Assigned specialty: {specialty} for mild headache")
                    else:  # moderate or severe
                        specialty = 'Neurology'
                        logger.debug(f"Assigned specialty: {specialty} for headache (severity score: {max_severity_score})")
                # Fallback to general symptom mapping
                elif priority_symptom in symptom_specialty_map:
                    specialty = symptom_specialty_map[priority_symptom]
                    logger.debug(f"Assigned specialty: {specialty} for symptom: {priority_symptom}")
                else:
                    logger.warning(f"Symptom '{priority_symptom}' not found in symptom_specialty_map. Defaulting to specialty: {specialty}")
        else:
            logger.warning("No priority symptom identified. Defaulting to General Medicine.")
    
    logger.debug(f"Final assigned specialty: {specialty}")
    
    # Dynamically fetch available doctors
    doctors_list = fetch_available_doctors(specialty)
    if doctors_list:
        # Return the list of doctors for selection
        return doctors_list, specialty
    
    logger.warning(f"No available doctors found for specialty: {specialty}")
    return [], specialty

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
        language = "en"  # Fixed for now
        medical_data = session.get('medical_data', {"symptoms": [], "severity": [], "duration": [], "triggers": []})
        if not medical_data.get("symptoms"):
            medical_data = {"symptoms": [], "severity": [], "duration": [], "triggers": []}

        # Initialize a tracker for asked questions (stored in session)
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

        # Log initial state for debugging
        logger.debug(f"Initial medical_data: {medical_data}")
        logger.debug(f"Initial asked_questions: {asked_questions}")

        # Transcribe audio if provided
        if audio_path:
            transcription = transcribe_audio(audio_path, language)
            if "failed" in transcription.lower():
                return {"response": "Audio issue detected. Please try again.", "medical_data": medical_data, "audio_url": None}, 400
        elif transcript:
            transcription = transcript
        else:
            # Initial greeting
            intro = "Hello, I am your healthcare assistant. What symptoms are you experiencing?"
            audio_path = synthesize_audio(intro, language)
            return {"response": intro, "medical_data": medical_data, "audio_url": f"/static/{os.path.basename(audio_path)}"}, 200

        # Prepare input variables for the conversation chain
        input_vars = {
            "input": transcription,
            "history": history,
            "medical_data": json.dumps(medical_data)
        }
        logger.debug(f"Input variables for conversation_chain.invoke: {input_vars}")

        # Step 1: Use the conversation LLM to generate the response and updates
        conversation_response = conversation_chain.invoke(input_vars)
        conversation_output = conversation_response.content
        logger.debug(f"Conversation LLM output: {conversation_output}")

        # Step 2: Use the JSON LLM to format the conversation output into JSON
        json_response = json_chain.invoke({"conversation_output": conversation_output})
        logger.debug(f"JSON LLM output: {json_response}")

        # Check if json_response is already a dictionary or has a .content attribute
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

        # Validate expected keys
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
            raw_medical_data = {"symptoms": [], "severity": [], "duration": [], "triggers": []}

        # Detect contradictions by comparing with existing medical_data
        updated_symptoms = raw_medical_data.get("symptoms", [])
        updated_severities = raw_medical_data.get("severity", [])
        updated_durations = raw_medical_data.get("duration", [])
        updated_triggers = raw_medical_data.get("triggers", [])
        existing_symptoms = medical_data.get("symptoms", [])
        existing_severities = medical_data.get("severity", [])
        existing_durations = medical_data.get("duration", [])
        existing_triggers = medical_data.get("triggers", [])
        contradictions = []

        for idx, symptom in enumerate(updated_symptoms):
            try:
                existing_idx = existing_symptoms.index(symptom)
            except ValueError:
                existing_idx = -1

            if existing_idx != -1:
                fields = [
                    ("severity", existing_severities, updated_severities),
                    ("duration", existing_durations, updated_durations),
                    ("triggers", existing_triggers, updated_triggers)
                ]
                for field_name, existing_list, updated_list in fields:
                    existing_value = existing_list[existing_idx] if existing_idx < len(existing_list) else ""
                    new_value = updated_list[idx] if idx < len(updated_list) else ""
                    if not new_value:
                        continue
                    if field_name == "triggers" and existing_value and new_value:
                        existing_value = existing_value.lower().replace("maybe ", "")
                        new_value = new_value.lower().replace("maybe ", "")
                    if existing_value and new_value and existing_value != new_value:
                        contradictions.append({
                            "symptom": symptom,
                            "field": field_name,
                            "existing_value": existing_value,
                            "new_value": new_value
                        })

        # If there are contradictions, ask for clarification
        if contradictions:
            contradiction = contradictions[0]
            symptom_name = contradiction["symptom"]
            field = contradiction["field"]
            existing_value = contradiction["existing_value"]
            new_value = contradiction["new_value"]
            response_text = f"I noticed you previously mentioned your {symptom_name}'s {field} as '{existing_value}', but now you’ve said '{new_value}'. Could you please clarify this for me?"
            audio_path = synthesize_audio(response_text, language)
            return {
                "response": response_text,
                "medical_data": medical_data,
                "audio_url": f"/static/{os.path.basename(audio_path)}" if audio_path else None
            }, 200

        # Update asked_questions tracker based on the response
        for idx, symptom in enumerate(updated_symptoms):
            if symptom not in asked_questions:
                asked_questions[symptom] = {
                    "severity": False,
                    "duration": False,
                    "triggers": False
                }
            # Mark fields as asked if they are present and non-empty in the updated data
            if idx < len(updated_severities) and updated_severities[idx]:
                asked_questions[symptom]["severity"] = True
            if idx < len(updated_durations) and updated_durations[idx]:
                asked_questions[symptom]["duration"] = True
            if idx < len(updated_triggers) and updated_triggers[idx]:
                asked_questions[symptom]["triggers"] = True
        session['asked_questions'] = asked_questions
        logger.debug(f"Updated asked_questions: {asked_questions}")

        # Update medical_data with the transformed structure
        medical_data = {
            "symptoms": updated_symptoms,
            "severity": updated_severities,
            "duration": updated_durations,
            "triggers": updated_triggers
        }
        should_assign_doctor = data["assign_doctor"]
        check_availability = data["check_availability"]
        condition = data["condition"]

        # Update Firestore with the transformed structure in initial_screenings
        uid = request.user.get("uid")
        doc_ref = db.collection('initial_screenings').document(f'initial_screening_{uid}')
        doc_ref.set(medical_data, merge=True)

        # Handle doctor availability check
        if check_availability and condition:
            condition_specialty_map = {
                "kidney stones": "Urology",
                "fever": "General Medicine",
                "sore throat": "ENT",
                "headache": "Neurology",
                "rash": "Dermatology",
                "fatigue": "General Medicine",
                "anxiety": "Psychiatry",
                "inability to sleep": "Psychiatry",
                "dry cough": "ENT"
            }
            specialty = condition_specialty_map.get(condition.lower(), "General Medicine")
            doctors_list, _ = assign_doctor(medical_data)
            if doctors_list:
                response_text = f"Yes, a doctor is available for {condition}. Let's gather more details about your symptoms related to {condition}. {response_text}"
            else:
                response_text = f"I'm sorry, no doctors are currently available for {condition}. Let's gather more details about your symptoms, and I can recommend a general physician if needed. {response_text}"

        # Handle doctor assignment
        if should_assign_doctor:
            # Check if all details are complete for all symptoms (treat "unknown" as a valid trigger)
            all_details_complete = True
            for idx, symptom in enumerate(medical_data["symptoms"]):
                if not (medical_data["severity"][idx] and medical_data["duration"][idx]):
                    all_details_complete = False
                    break
                # Ensure triggers is non-empty (including "unknown")
                if idx >= len(medical_data["triggers"]) or not medical_data["triggers"][idx]:
                    all_details_complete = False
                    break
            if not all_details_complete:
                logger.warning("Not all symptom details are complete; cannot assign doctor yet.")
                should_assign_doctor = False
            else:
                doctors_list, specialty = assign_doctor(medical_data)
                logger.debug(f"Doctor assignment result: Specialty={specialty}, Doctors={doctors_list}")
                if not doctors_list:
                    response_text = f"No doctors available for {specialty}. Please try again later or contact support."
                    logger.warning(f"Doctor assignment failed: No doctors available for specialty '{specialty}'")
                elif len(doctors_list) == 1:
                    doctor = doctors_list[0]
                    consultant_id = doctor["id"]
                    medical_data["consultant_id"] = consultant_id
                    # Update initial_screenings with the consultant_id and specialty
                    doc_ref.set({
                        "consultant_id": consultant_id,
                        "specialty": specialty
                    }, merge=True)
                    # Update patient_registrations with the consultant_id
                    patient_ref = db.collection('patient_registrations').document(uid)
                    patient_snap = patient_ref.get()
                    if patient_snap.exists:
                        patient_ref.set({'consultant_id': consultant_id}, merge=True)
                        logger.info(f"Updated patient_registrations with consultant_id: {consultant_id} for UID: {uid}")
                    else:
                        patient_data = {
                            'uid': uid,
                            'email': session.get('user_info', {}).get('email', ''),
                            'full_name': session.get('user_info', {}).get('full_name', uid),
                            'consultant_id': consultant_id,
                            'role': 'patient'
                        }
                        patient_ref.set(patient_data)
                        logger.info(f"Created new patient_registrations document with consultant_id: {consultant_id} for UID: {uid}")
                    response_text = f"You have been assigned to {doctor['full_name']} (specializing in {specialty}). Please log in to view details."
                    logger.info(f"Assigned doctor: {doctor['full_name']} (ID: {consultant_id}) for specialty '{specialty}'")
                    # Clear session data after assignment
                    session.pop('asked_questions', None)
                    session.pop('medical_data', None)
                else:
                    response_text = f"Multiple doctors available for {specialty}: {', '.join([d['full_name'] for d in doctors_list])}. Please choose one."

        # Generate audio response
        audio_path = synthesize_audio(response_text, language)
        session['medical_data'] = medical_data  # Save updated medical_data to session
        return {
            "response": response_text,
            "medical_data": medical_data,
            "audio_url": f"/static/{os.path.basename(audio_path)}" if audio_path else None
        }, 200

    except Exception as e:
        logger.error(f"Error: {str(e)}")
        return {"response": f"Error: {str(e)}", "medical_data": {"symptoms": []}, "audio_url": None}, 500
        
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

            # Ensure medical_data is a dictionary with lists
            medical_data = processed_data.get('medical_data')
            if not isinstance(medical_data, dict):
                logger.error(f"medical_data is not a dictionary: {medical_data}")
                medical_data = {
                    "symptoms": [],
                    "severity": [],
                    "duration": [],
                    "triggers": []
                }
            # Validate the structure of medical_data
            expected_keys = ["symptoms", "severity", "duration", "triggers"]
            for key in expected_keys:
                if key not in medical_data or not isinstance(medical_data[key], list):
                    logger.warning(f"medical_data[{key}] is not a list, resetting: {medical_data.get(key)}")
                    medical_data[key] = []

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

            # Ensure medical_data is a dictionary with lists
            medical_data = processed_data.get('medical_data')
            if not isinstance(medical_data, dict):
                logger.error(f"medical_data is not a dictionary: {medical_data}")
                medical_data = {
                    "symptoms": [],
                    "severity": [],
                    "duration": [],
                    "triggers": []
                }
            # Validate the structure of medical_data
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