import re
import secrets
import logging
import os
import tempfile
import json
import whisper
import glob
import time
from functools import wraps
from dotenv import load_dotenv
import firebase_admin
from firebase_admin import credentials, auth, firestore, storage

# Setup logging
logging.basicConfig(level=logging.DEBUG, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Load Firebase credentials
cred_path = './Cred/Firebase/Med App/med-labs-42f13-firebase-adminsdk-fbsvc-43b78dfef7.json'

# Initialize Firebase before any imports that use Firestore
if not firebase_admin._apps:
    cred = credentials.Certificate(cred_path)
    firebase_admin.initialize_app(cred, {
        'storageBucket': 'med-labs-new-bucket-2025'
    })
    logger.info("✅ Firebase initialized successfully with bucket: med-labs-new-bucket-2025. SDK Version: %s", firebase_admin.__version__)

db = firestore.client()

# Load environment variables
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
from openai import OpenAI
try:
    test_client = OpenAI(api_key=openai_api_key)
    test_client.models.list()
    logger.info("✅ OpenAI API key validated successfully")
except Exception as e:
    logger.error(f"Invalid OpenAI API key: {str(e)}")
    raise ValueError(f"Invalid OpenAI API key: {str(e)}")

client = OpenAI(api_key=openai_api_key)

# Load the same service account credentials for GCS using google-auth
from google.auth import load_credentials_from_file
gcs_credentials, _ = load_credentials_from_file(cred_path)

# Initialize GCS client with the new project, using the loaded credentials
from google.cloud import storage as gcs
gcs_client = gcs.Client(project='med-labs-new-2025', credentials=gcs_credentials)
bucket = gcs_client.bucket('med-labs-new-bucket-2025')

# Define a retry predicate for 400 errors
from google.api_core import retry, exceptions
def if_bad_request(exception):
    return isinstance(exception, exceptions.BadRequest)

# Validate bucket access with retry logic
@retry.Retry(predicate=if_bad_request, initial=1, maximum=10, multiplier=2, deadline=60)
def validate_bucket(bucket_name):
    bucket_obj = gcs_client.lookup_bucket(bucket_name)
    if bucket_obj is None:
        raise ValueError(f"Bucket {bucket_name} does not exist or is inaccessible.")
    return bucket_obj

# Try the primary bucket, fall back to a different bucket if needed
bucket_name = 'med-labs-new-bucket-2025'
try:
    bucket_obj = validate_bucket(bucket_name)
    logger.info(f"Bucket '{bucket_name}' exists and is accessible. Project ID: {gcs_client.project}")
    bucket = gcs_client.bucket(bucket_name)
except Exception as e:
    logger.warning(f"Failed to access bucket '{bucket_name}': {str(e)}. Proceeding without bucket validation. GCS operations may fail.")
    fallback_bucket_name = 'med-labs-42f13'
    logger.info(f"Attempting to use fallback bucket '{fallback_bucket_name}'...")
    try:
        bucket_obj = validate_bucket(fallback_bucket_name)
        bucket = gcs_client.bucket(fallback_bucket_name)
        logger.info(f"Fallback bucket '{fallback_bucket_name}' exists and is accessible. Project ID: {gcs_client.project}")
    except Exception as e2:
        logger.error(f"Bucket validation failed for both '{bucket_name}' and '{fallback_bucket_name}': {str(e2)}. Please ensure the bucket exists, the project has billing enabled, the Storage API is enabled, and the service account has the necessary permissions.")
        raise

# Now import modules that depend on Firebase
from flask import Flask, render_template, request, jsonify, session, redirect, url_for, send_from_directory
from google_vision import extract_text_from_image
from gemini_processor import process_text_with_gemini, generate_medical_history
from error_gemini_processor import generate_error_poem
from langchain_openai import ChatOpenAI
from langchain.prompts import PromptTemplate
from langchain.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnableSequence
from langchain_core.runnables import RunnablePassthrough, RunnableLambda
from gtts import gTTS
from datetime import datetime
import traceback
import httpx
from pydub import AudioSegment
import asyncio
import hashlib
from func_timeout import FunctionTimedOut
from symptom_mapping import determine_specialty, normalize_symptom, symptom_specialty_map

app = Flask(__name__)
app.secret_key = os.environ.get("FLASK_SECRET_KEY", secrets.token_hex(32))

# LLM Setup
conversation_llm = ChatOpenAI(
    model="gpt-4o-mini",
    openai_api_key=openai_api_key,
    temperature=0.7
)

json_llm = ChatOpenAI(
    model="gpt-3.5-turbo",
    openai_api_key=openai_api_key,
    temperature=0.0
)

symptom_extraction_llm = ChatOpenAI(
    model="gpt-4o-mini",
    openai_api_key=openai_api_key,
    temperature=0.5
)

doctor_mapping_validation_llm = ChatOpenAI(
    model="gpt-4o-mini",
    openai_api_key=openai_api_key,
    temperature=0.3
)

conversation_prompt_template = PromptTemplate(
    input_variables=["history", "input", "medical_data", "current_state"],
    template=(
        "You are a professional healthcare assistant. Your task is to guide the patient through a conversation to gather their symptoms, severity, duration, and triggers to recommend a doctor from the following specialties: Dermatology, ENT, Nephrology, Endocrinology, Gastroenterology, Urology, Psychiatry, General Medicine, General Surgery, Gynecology, Oncology, Pediatrics, Cardiology, Neurology, Orthopedics.\n\n"
        "**Current State:** {current_state}\n"
        "**Current Medical Data:**\n{medical_data}\n\n"
        "**Conversation History:**\n{history}\n\n"
        "**Patient’s Latest Input:**\n{input}\n\n"
        "**Instructions:**\n"
        "- Symptoms are preprocessed to standard terms using a symptom mapping database.\n"
        "- For state INITIAL: Update medical_data['symptoms'], ask for severity ('How severe is your {{symptoms}}? Is it mild, moderate, or severe?'), set next_state to SEVERITY.\n"
        "- For state SEVERITY: Extract severity (mild/moderate/severe). If numerical (e.g., '7'), map 1-3 as 'mild', 4-6 as 'moderate', 7-10 as 'severe', ask for confirmation ('You said your {{symptoms}} severity is {{input}}, which I interpret as {{severity}}. Is that correct?'), set next_state to CONFIRM_SEVERITY. Otherwise, update medical_data['severity'], ask for duration ('How long have you been experiencing {{symptoms}}?'), set next_state to DURATION.\n"
        "- For state CONFIRM_SEVERITY: If confirmed (e.g., 'yes'), update medical_data['severity'], ask for duration, set next_state to DURATION. If denied (e.g., 'no'), ask for severity again, set next_state to SEVERITY. If unclear, ask for clarification, keep next_state as CONFIRM_SEVERITY.\n"
        "- For state DURATION: Extract duration, update medical_data['duration'], ask for triggers ('What triggers your {{symptoms}}, if anything? Say \"unknown\" if unsure.'), set next_state to TRIGGERS.\n"
        "- For state TRIGGERS: Extract triggers, update medical_data['triggers'], ask to assign doctor ('May I recommend a doctor for you?'), set next_state to CONFIRM.\n"
        "- For state CONFIRM: If confirmed (e.g., 'yes', 'sure'), set assign_doctor to true, set next_state to COMPLETE; otherwise, ask again.\n"
        "- If input is unclear, ask for clarification and do not advance state.\n"
        "- Return a JSON object with: 'response', 'medical_data' (symptoms, severity, duration, triggers as lists), 'next_state', 'assign_doctor'."
    )
)

json_prompt_template = PromptTemplate(
    input_variables=["conversation_output"],
    template=(
        "You are a JSON formatting assistant. Parse the conversation output into a JSON object with keys: 'response', 'medical_data' (symptoms, severity, duration, triggers as lists), 'next_state', 'assign_doctor'.\n\n"
        "**Conversation Output:**\n{conversation_output}\n\n"
        "Return only the JSON string."
    )
)

symptom_extraction_prompt_template = PromptTemplate(
    input_variables=["input", "history", "current_state"],
    template=(
        "You are a medical assistant extracting symptoms from patient input, which may include descriptive or vague terms (e.g., 'my stomach is upset and I keep puking'). Map symptoms to standard terms using a Firebase Firestore symptom mapping database. Handle complex explanations by identifying all relevant symptoms.\n\n"
        "**Current State:** {current_state}\n"
        "**Conversation History:**\n{history}\n\n"
        "**Patient's Latest Input:**\n{input}\n\n"
        "**Instructions:**\n"
        "- Identify all symptoms, even from complex inputs (e.g., 'my head feels heavy and I’m puking' → ['headache', 'vomiting']).\n"
        "- Normalize to standard terms using the symptom mapping database (e.g., 'puking' → 'vomiting').\n"
        "- Flag unmatched terms as 'unknown'.\n"
        "- Use context to disambiguate (e.g., 'burning when I pee' with 'blood in urine' → 'urinary tract infection').\n"
        "- Return a JSON object with: 'symptoms', 'needs_clarification', 'clarification_message', 'mapped_symptoms' (list of {{'original': 'original_term', 'mapped': 'mapped_term'}}).\n"
        "- Example: Input: 'My stomach is upset and I keep puking'\n"
        "  Output: {{'symptoms': ['vomiting'], 'needs_clarification': false, 'clarification_message': '', 'mapped_symptoms': [{{'original': 'puking', 'mapped': 'vomiting'}}]}}\n"
        "Return only the JSON string."
    )
)

doctor_mapping_validation_prompt_template = PromptTemplate(
    input_variables=["symptoms", "assigned_specialty"],
    template=(
        "You are a medical assistant validating symptom-to-specialty mapping. Available specialties: Dermatology, ENT, Nephrology, Endocrinology, Gastroenterology, Urology, Psychiatry, General Medicine, General Surgery, Gynecology, Oncology, Pediatrics, Cardiology, Neurology, Orthopedics.\n\n"
        "**Patient's Symptoms:**\n{symptoms}\n\n"
        "**Assigned Specialty:**\n{assigned_specialty}\n\n"
        "**Instructions:**\n"
        "- Validate if the assigned specialty is appropriate.\n"
        "- If incorrect, suggest the correct specialty with reasoning.\n"
        "- Return a JSON object with: 'is_correct', 'correct_specialty', 'reasoning'.\n"
        "Return only the JSON string."
    )
)

conversation_chain = conversation_prompt_template | conversation_llm
json_chain = json_prompt_template | json_llm
symptom_extraction_chain = symptom_extraction_prompt_template | symptom_extraction_llm
doctor_mapping_validation_chain = doctor_mapping_validation_prompt_template | doctor_mapping_validation_llm

# Static Doctors List
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

def assign_doctor(medical_data):
    """
    Assign a doctor based on medical data symptoms.
    Returns a tuple: (list of matching doctors, assigned specialty).
    """
    try:
        symptoms = medical_data.get('symptoms', [])
        if not symptoms:
            logger.warning("No symptoms provided for doctor assignment")
            return [], "general medicine"

        normalized_symptoms = [normalize_symptom(symptom) for symptom in symptoms]
        specialty = determine_specialty(normalized_symptoms)
        if not specialty:
            specialty = "general medicine"

        matching_doctors = [
            doctor for doctor in DOCTORS
            if doctor['specialty'].lower() == specialty.lower() and doctor['availability']
        ]

        if not matching_doctors:
            logger.warning(f"No available doctors found for specialty: {specialty}")
            return [], specialty

        logger.info(f"Assigned specialty: {specialty}, Found {len(matching_doctors)} doctors")
        return matching_doctors, specialty
    except Exception as e:
        logger.error(f"Error in assign_doctor: {str(e)}")
        return [], "general medicine"
    
def update_conversation_state(session, medical_data, current_state, next_state, asked_questions, current_query, conversation_id):
    """
    Update the conversation state in the session.
    """
    session['medical_data'] = medical_data
    session['current_state'] = next_state
    session['asked_questions'] = asked_questions
    session['current_query'] = current_query
    session['conversation_id'] = conversation_id
    logger.debug(f"Updated session state: current_state={next_state}, medical_data={medical_data}")

def generate_consultant_id():
    """
    Generate a unique consultant ID using a hash of timestamp and random string.
    """
    try:
        timestamp = str(firestore.SERVER_TIMESTAMP)
        random_str = secrets.token_hex(8)
        consultant_id = hashlib.sha256((timestamp + random_str).encode()).hexdigest()[:16]
        logger.debug(f"Generated consultant ID: {consultant_id}")
        return consultant_id
    except Exception as e:
        logger.error(f"Error generating consultant ID: {str(e)}")
        return secrets.token_hex(8)[:16]

def token_required(f):
    @wraps(f)
    def decorated(*args, **kwargs):
        auth_header = request.headers.get('Authorization')
        token = None
        if auth_header and auth_header.startswith("Bearer "):
            token = auth_header.split(" ")[1]
            logger.debug(f"Using token from Authorization header: {token[:20]}...")
        elif 'idToken' in session:
            token = session['idToken']
            logger.debug(f"Using token from session: {token[:20]}...")
        else:
            logger.error("No Authorization header or session token provided")
            return render_template('errors/token_missing.html', error_poem="Token missing error"), 401

        try:
            decoded_token = auth.verify_id_token(token, clock_skew_seconds=60)
            current_time = int(time.time())
            token_iat = decoded_token.get('iat')
            logger.debug(f"Server time: {current_time}, Token iat: {token_iat}, Skew: {token_iat - current_time} seconds")
            request.user = decoded_token
            logger.info(f"✅ Token verified for: {decoded_token.get('email')}, UID: {decoded_token.get('uid')}")
        except auth.InvalidIdTokenError as e:
            if "Token used too early" in str(e):
                logger.error(f"Clock skew error during token verification: {str(e)}")
                return render_template('errors/authentication_error.html', error_poem="Token used too early. Please ensure your device's clock is synchronized."), 401
            logger.error(f"Invalid ID token: {str(e)}")
            return render_template('errors/authentication_error.html', error_poem="Invalid ID token error"), 401
        except auth.ExpiredIdTokenError as e:
            logger.error(f"Expired ID token: {str(e)}")
            return render_template('errors/authentication_error.html', error_poem="Expired ID token error"), 401
        except Exception as e:
            logger.error(f"Token verification error: {str(e)}")
            return render_template('errors/authentication_error.html', error_poem="Authentication error"), 401
        return f(*args, **kwargs)
    return decorated

def cleanup_old_tts_files(static_dir="static", max_age=3600):
    try:
        now = time.time()
        for mp3_file in glob.glob(os.path.join(static_dir, "temp_tts_*.mp3")):
            file_age = now - os.path.getmtime(mp3_file)
            if file_age > max_age:
                os.remove(mp3_file)
                logger.debug(f"Deleted old temp TTS file: {mp3_file} (age: {file_age}s)")
    except Exception as e:
        logger.warning(f"Failed to clean up old TTS files: {str(e)}")

def transcribe_audio(audio_path, language="en", retries=3):
    attempt = 0
    while attempt < retries:
        try:
            logger.debug(f"Transcribing audio: {audio_path} (language: {language}, attempt: {attempt + 1})")
            audio_size = os.path.getsize(audio_path)
            logger.debug(f"Audio file size: {audio_size} bytes")
            if audio_size < 1024:
                logger.error(f"Audio file too small: {audio_size} bytes")
                return "Audio file too small or corrupted"
            with open(audio_path, "rb") as audio_file:
                result = client.audio.transcriptions.create(
                    model="whisper-1",
                    file=audio_file,
                    language=language,
                    response_format="text"
                )
                transcribed_text = result.text
                logger.debug(f"Transcribed text: {transcribed_text}")
                return transcribed_text
        except httpx.HTTPStatusError as e:
            logger.error(f"Transcription HTTP error {e.response.status_code}: {e.response.text}", exc_info=True)
            if e.response.status_code in [429, 500, 502, 503]:
                attempt += 1
                time.sleep(2 ** attempt)
                continue
            return f"Transcription failed: HTTP {e.response.status_code} - {e.response.text}"
        except Exception as e:
            logger.error(f"Transcription error: {str(e)}", exc_info=True)
            attempt += 1
            if attempt < retries:
                time.sleep(2 ** attempt)
            else:
                return f"Transcription failed: {str(e)}"
    return "Transcription failed after multiple attempts"

def synthesize_audio(text, language="en", session_id=None):
    try:
        logger.debug(f"Synthesizing audio: {text[:50]}... (language: {language})")
        if not text or not text.strip():
            logger.warning("TTS input is empty")
            return None

        # Generate cache key
        cache_key = hashlib.md5((text + language).encode()).hexdigest()
        doc_ref = db.collection('tts_cache').document(cache_key)
        
        # Check Firestore cache (synchronous)
        logger.debug("Checking Firestore cache for cache_key: %s", cache_key)
        doc = doc_ref.get()
        if doc.exists:
            cached = doc.to_dict()
            gcs_url = cached.get('gcs_url')
            logger.info(f"Retrieved cached TTS for: {text[:50]}")
            return gcs_url

        # Generate TTS
        logger.debug("Generating TTS audio")
        tts = gTTS(text=text, lang=language, slow=False)
        temp_file = os.path.join(tempfile.gettempdir(), f"temp_tts_{cache_key}.mp3")
        tts.save(temp_file)
        audio_size = os.path.getsize(temp_file)
        if audio_size < 1024:
            logger.error(f"Generated audio {temp_file} too small: {audio_size} bytes")
            os.remove(temp_file)
            raise ValueError(f"Generated audio file is too small: {audio_size} bytes")

        # Upload to GCS (bucket is publicly readable via IAM)
        logger.debug("Uploading audio to GCS")
        gcs_path = f"tts/{cache_key}.mp3"
        blob = bucket.blob(gcs_path)
        blob.upload_from_filename(temp_file)
        logger.debug("Audio uploaded to GCS at path: %s", gcs_path)
        
        # Construct public URL manually without accessing metadata
        gcs_url = f"https://storage.googleapis.com/{bucket.name}/{gcs_path}"
        logger.debug(f"Constructed public URL: {gcs_url}")
        
        os.remove(temp_file)

        # Cache metadata in Firestore (synchronous)
        logger.debug("Caching URL in Firestore")
        doc_ref.set({
            'text': text,
            'language': language,
            'gcs_url': gcs_url,
            'timestamp': firestore.SERVER_TIMESTAMP
        })
        logger.info(f"Uploaded TTS to GCS and cached in Firestore: {gcs_url}")
        return gcs_url
    except Exception as e:
        logger.error(f"Audio synthesis error: {str(e)}")
        if 'temp_file' in locals() and os.path.exists(temp_file):
            os.remove(temp_file)
        return None

def reset_conversation_state(session, clear_firestore=False, uid=None):
    """
    Reset all conversation-related session data to start a new conversation.
    Optionally clear Firestore data to prevent stale symptoms.
    """
    # Clear all possible session keys except user authentication data
    keys_to_remove = [key for key in session.keys() if key not in ('user_info', 'idToken')]
    for key in keys_to_remove:
        session.pop(key, None)
    # Reinitialize with default values
    session['medical_data'] = {"symptoms": [], "severity": [], "duration": [], "triggers": []}
    session['current_state'] = "INITIAL"
    session['asked_questions'] = {}
    session['current_query'] = {"symptom": None, "field": None, "interpreted_severity": None, "raw_severity": None}
    session['conversation_id'] = secrets.token_hex(16)
    session['conversation_history'] = ""
    session['last_session_id'] = None
    logger.debug("Fully reset conversation state in session: %s", session)

    # Optionally clear Firestore initial_screenings document
    if clear_firestore and uid:
        try:
            doc_id = f'initial_screening_{uid}'
            doc_ref = db.collection('initial_screenings').document(doc_id)
            doc_ref.set({
                'uid': uid,
                'patient_name': session.get('user_info', {}).get('full_name', uid),
                'symptoms': '',
                'severity': '',
                'duration': '',
                'triggers': '',
                'consultant_id': None,
                'timestamp': firestore.SERVER_TIMESTAMP
            }, merge=True)
            logger.debug("Cleared Firestore initial_screenings for user: %s", uid)
        except Exception as e:
            logger.error(f"Failed to clear Firestore initial_screenings for user {uid}: {str(e)}")

def process_conversation(audio_path=None, transcript=None, history="", session_id=None, current_state="INITIAL"):
    try:
        language = "en"
        # Reset session data for initial request to avoid stale symptoms
        if transcript is None or (transcript == "" and current_state == "INITIAL"):
            reset_conversation_state(session, clear_firestore=True, uid=request.user.get('uid'))
            logger.debug("After reset in process_conversation, session state: %s", session)

        # Ensure medical_data is fresh for INITIAL state
        medical_data = session.get('medical_data', {"symptoms": [], "severity": [], "duration": [], "triggers": []})
        if current_state == "INITIAL" and (transcript is None or transcript == ""):
            medical_data = {"symptoms": [], "severity": [], "duration": [], "triggers": []}
            session['medical_data'] = medical_data
            logger.debug("Forced fresh medical_data for INITIAL state: %s", medical_data)
        elif not medical_data.get("symptoms"):
            medical_data = {"symptoms": [], "severity": [], "duration": [], "triggers": []}
            session['medical_data'] = medical_data
        else:
            logger.debug("medical_data after session retrieval: %s", medical_data)

        asked_questions = session.get('asked_questions', {})
        if not asked_questions:
            asked_questions = {symptom: {"severity": False, "duration": False, "triggers": False} 
                               for symptom in medical_data.get("symptoms", [])}

        current_query = session.get('current_query', {"symptom": None, "field": None, "interpreted_severity": None, "raw_severity": None})
        conversation_id = session.get('conversation_id', secrets.token_hex(16))

        logger.debug(f"Processing conversation: state={current_state}, transcript={transcript}, session_id={session_id}, medical_data={medical_data}")

        # Handle initial conversation start
        if transcript is None or (transcript == "" and current_state == "INITIAL"):
            intro = "What symptoms are you experiencing?"
            audio_url = synthesize_audio(intro, language, session_id)
            update_conversation_state(session, medical_data, current_state, "INITIAL", asked_questions, current_query, conversation_id)
            db.collection('conversations').add({
                'conversation_id': conversation_id,
                'uid': hashlib.sha256(request.user.get('uid').encode()).hexdigest(),
                'session_id': session_id,
                'transcription': '',
                'normalized_transcription': '',
                'state': "INITIAL",
                'response': intro,
                'symptoms': [],
                'next_state': "INITIAL",
                'timestamp': firestore.SERVER_TIMESTAMP,
                'is_repeat': False,
                'reward': 0
            })
            return {"success": True, "response": intro, "medical_data": medical_data, "audio_url": audio_url}, 200

        # Handle transcript processing
        transcription = transcript
        normalized_transcription = normalize_symptom(transcript) if transcript else ""

        if audio_path:
            uid = request.user.get('uid')
            gcs_path = f"voice_inputs/{uid}/{session_id}.webm"
            blob = bucket.blob(gcs_path)
            blob.upload_from_filename(audio_path)
            gcs_uri = f"https://storage.googleapis.com/{bucket.name}/{gcs_path}"
            logger.info(f"Uploaded audio to GCS: {gcs_uri}")

            transcription = transcribe_audio(audio_path, language)
            if "failed" in transcription.lower() or "too small" in transcription.lower():
                logger.error(f"Transcription failed: {transcription}")
                return {"success": False, "response": "Audio issue detected. Please try again.", "medical_data": medical_data, "audio_url": None}, 400
            normalized_transcription = normalize_symptom(transcription)

        # Extract symptoms
        symptom_extraction_input = {
            "input": transcription,
            "history": history,
            "current_state": current_state
        }
        try:
            symptom_extraction_response = symptom_extraction_chain.invoke(symptom_extraction_input)
        except FunctionTimedOut:
            return {"success": False, "response": "Symptom extraction timed out.", "medical_data": medical_data, "audio_url": None}, 500
        raw_response = symptom_extraction_response.content
        cleaned_response = raw_response.strip().lstrip("```json").rstrip("```")
        symptom_extraction_data = json.loads(cleaned_response)

        extracted_symptoms = symptom_extraction_data.get("symptoms", [])
        needs_clarification = symptom_extraction_data.get("needs_clarification", False)
        clarification_message = symptom_extraction_data.get("clarification_message", "Could you please clarify your symptoms?")

        reward = 0
        if needs_clarification:
            reward = 0
        elif extracted_symptoms and current_state == "INITIAL":
            reward = 2 if len(extracted_symptoms) > 1 else 1

        if needs_clarification:
            audio_url = synthesize_audio(clarification_message, language, session_id)
            update_conversation_state(session, medical_data, current_state, current_state, asked_questions, current_query, conversation_id)
            db.collection('conversations').add({
                'conversation_id': conversation_id,
                'uid': hashlib.sha256(request.user.get('uid').encode()).hexdigest(),
                'session_id': session_id,
                'transcription': transcription,
                'normalized_transcription': normalized_transcription,
                'state': current_state,
                'response': clarification_message,
                'symptoms': extracted_symptoms,
                'next_state': current_state,
                'timestamp': firestore.SERVER_TIMESTAMP,
                'is_repeat': False,
                'reward': reward
            })
            return {"success": True, "response": clarification_message, "medical_data": medical_data, "audio_url": audio_url}, 200

        # State machine transitions
        response_text = ""
        next_state = current_state
        is_repeat = False
        updated_symptoms = extracted_symptoms

        if current_state == "INITIAL":
            if not extracted_symptoms:
                response_text = "I couldn't identify any symptoms. Could you please tell me more?"
                next_state = "INITIAL"
                reward = -1
            else:
                medical_data["symptoms"] = extracted_symptoms
                medical_data["severity"] = [''] * len(extracted_symptoms)
                medical_data["duration"] = [''] * len(extracted_symptoms)
                medical_data["triggers"] = ['unknown'] * len(extracted_symptoms)
                session['medical_data'] = medical_data
                response_text = f"How severe is your {', '.join(extracted_symptoms)}? Is it mild, moderate, or severe?"
                next_state = "SEVERITY"
                reward = 1

        elif current_state == "SEVERITY":
            transcription_lower = transcription.lower()
            severity = "moderate"  # Default severity
            if "mild" in transcription_lower:
                severity = "mild"
            elif "moderate" in transcription_lower:
                severity = "moderate"
            elif "severe" in transcription_lower:
                severity = "severe"
            else:
                num_match = re.search(r'\b([1-9]|10)\b', transcription_lower)
                if num_match:
                    num = int(num_match.group(1))
                    severity = "mild" if num <= 3 else "moderate" if num <= 6 else "severe"
                    current_query["raw_severity"] = num_match.group(1)
                    current_query["interpreted_severity"] = severity
                    response_text = f"You said your {', '.join(medical_data['symptoms'])} severity is {severity}. Is that correct?"
                    next_state = "CONFIRM_SEVERITY"
                else:
                    response_text = "Please specify the severity: mild, moderate, or severe."
                    next_state = "SEVERITY"
                    is_repeat = True
                    reward = -1
            if not is_repeat:
                medical_data["severity"] = [severity] * len(medical_data["symptoms"])
                response_text = f"How long have you been experiencing {', '.join(medical_data['symptoms'])}?"
                next_state = "DURATION"
                reward = 1
            for symptom in medical_data["symptoms"]:
                if symptom not in asked_questions:
                    asked_questions[symptom] = {"severity": False, "duration": False, "triggers": False}
                asked_questions[symptom]["severity"] = True

        elif current_state == "CONFIRM_SEVERITY":
            transcription_lower = transcription.lower()
            symptom = current_query.get("symptom")
            interpreted_severity = current_query.get("interpreted_severity")
            if any(x in transcription_lower for x in ["yes", "correct", "right", "okay"]):
                for idx, s in enumerate(medical_data["symptoms"]):
                    if s == symptom:
                        medical_data["severity"][idx] = interpreted_severity
                        break
                response_text = f"How long have you been experiencing {', '.join(medical_data['symptoms'])}?"
                current_query = {"symptom": symptom, "field": "duration"}
                next_state = "DURATION"
                reward = 1
            elif any(x in transcription_lower for x in ["no", "incorrect", "wrong"]):
                response_text = f"Please specify the severity of your {', '.join(medical_data['symptoms'])}. Is it mild, moderate, or severe?"
                current_query = {"symptom": symptom, "field": "severity"}
                next_state = "SEVERITY"
                reward = -1
            else:
                response_text = "Could you please confirm if this is correct? Please say 'yes' or 'no'."
                next_state = "CONFIRM_SEVERITY"
                reward = 0

        elif current_state == "DURATION":
            transcription_lower = transcription.lower()
            duration_match = re.search(r'(\d+\s*(to\s*\d+\s*)?(month|week|day))', transcription_lower)
            duration = duration_match.group(0) if duration_match else "unknown"
            medical_data["duration"] = [duration] * len(medical_data["symptoms"])
            response_text = f"What triggers your {', '.join(medical_data['symptoms'])}, if anything? Say 'unknown' if unsure."
            next_state = "TRIGGERS"
            reward = 1
            for symptom in medical_data["symptoms"]:
                asked_questions[symptom]["duration"] = True

        elif current_state == "TRIGGERS":
            transcription_lower = transcription.lower()
            triggers = "unknown" if any(phrase in transcription_lower for phrase in ["i'm not sure", "it's unknown", "i don't know"]) else transcription_lower
            medical_data["triggers"] = [triggers] * len(medical_data["symptoms"])
            response_text = "May I recommend a doctor for you?"
            next_state = "CONFIRM"
            reward = 1
            for symptom in medical_data["symptoms"]:
                asked_questions[symptom]["triggers"] = True

        elif current_state == "CONFIRM":
            transcription_lower = transcription.lower()
            yes_synonyms = ['yes', 'sure', 'okay', 'please', 'ok', 'recommend', 'doctor']
            logger.debug(f"CONFIRM state: transcription_lower='{transcription_lower}', checking against yes_synonyms={yes_synonyms}")
            if any(x in transcription_lower for x in yes_synonyms):
                logger.debug("Affirmative response detected, transitioning to COMPLETE")
                doctors_list, specialty = assign_doctor(medical_data)
                if not doctors_list:
                    response_text = f"No doctors available for {specialty}. Please try again later."
                    audio_url = synthesize_audio(response_text, language, session_id)
                    update_conversation_state(session, medical_data, current_state, current_state, asked_questions, current_query, conversation_id)
                    db.collection('conversations').add({
                        'conversation_id': conversation_id,
                        'uid': hashlib.sha256(request.user.get('uid').encode()).hexdigest(),
                        'session_id': session_id,
                        'transcription': transcription,
                        'normalized_transcription': normalized_transcription,
                        'state': current_state,
                        'response': response_text,
                        'symptoms': medical_data["symptoms"],
                        'next_state': current_state,
                        'timestamp': firestore.SERVER_TIMESTAMP,
                        'is_repeat': False,
                        'reward': -1
                    })
                    return {
                        "success": False,
                        "response": response_text,
                        "medical_data": medical_data,
                        "audio_url": audio_url,
                        "redirect": None,
                        "conversationComplete": False
                    }, 200
                else:
                    doctor = doctors_list[0]
                    consultant_id = doctor["consultant_id"]
                    uid = request.user.get("uid")
                    patient_name = session.get('user_info', {}).get('full_name', uid)

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
                    save_success = True
                    try:
                        doc_ref.set(firestore_data, merge=True)
                    except Exception as e:
                        logger.error(f"Failed to save initial screening: {str(e)}")
                        save_success = False

                    patient_ref = db.collection('patient_registrations').document(uid)
                    patient_snap = patient_ref.get()
                    try:
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
                    except Exception as e:
                        logger.error(f"Failed to update patient registration: {str(e)}")
                        save_success = False

                    response_text = f"You have been assigned to {doctor['full_name']} ({specialty.capitalize()}). Redirecting to dashboard."
                    if not save_success:
                        response_text += " Note: Data saving issue occurred."
                    audio_url = synthesize_audio(response_text, language, session_id)
                    next_state = "COMPLETE"
                    reward = 2
                    update_conversation_state(session, medical_data, current_state, next_state, asked_questions, current_query, conversation_id)
                    db.collection('conversations').add({
                        'conversation_id': conversation_id,
                        'uid': hashlib.sha256(request.user.get('uid').encode()).hexdigest(),
                        'session_id': session_id,
                        'transcription': transcription,
                        'normalized_transcription': normalized_transcription,
                        'state': current_state,
                        'response': response_text,
                        'symptoms': medical_data["symptoms"],
                        'next_state': next_state,
                        'timestamp': firestore.SERVER_TIMESTAMP,
                        'is_repeat': False,
                        'reward': reward
                    })
                    return {
                        "success": True,
                        "response": response_text,
                        "medical_data": medical_data,
                        "audio_url": audio_url,
                        "redirect": "/dashboard",
                        "conversationComplete": True
                    }, 200
            else:
                logger.debug("No affirmative response detected, staying in CONFIRM state")
                response_text = "Please confirm if you'd like me to recommend a doctor. Say 'yes' or 'no'."
                next_state = "CONFIRM"
                is_repeat = True
                reward = 0

        audio_url = synthesize_audio(response_text, language, session_id)
        update_conversation_state(session, medical_data, current_state, next_state, asked_questions, current_query, conversation_id)
        db.collection('conversations').add({
            'conversation_id': conversation_id,
            'uid': hashlib.sha256(request.user.get('uid').encode()).hexdigest(),
            'session_id': session_id,
            'transcription': transcription,
            'normalized_transcription': normalized_transcription,
            'state': current_state,
            'response': response_text,
            'symptoms': extracted_symptoms,
            'next_state': next_state,
            'timestamp': firestore.SERVER_TIMESTAMP,
            'is_repeat': is_repeat,
            'reward': reward
        })
        return {"success": True, "response": response_text, "medical_data": medical_data, "audio_url": audio_url}, 200

    except Exception as e:
        logger.error(f"Error in process_conversation: {str(e)}", exc_info=True)
        return {
            "success": False,
            "response": f"Error: {str(e)}. Please try again.",
            "medical_data": {"symptoms": []},
            "audio_url": None,
            "redirect": None,
            "conversationComplete": False
        }, 500

@app.before_request
def before_request_cleanup():
    cleanup_old_tts_files()

@app.route('/', methods=['GET'])
def index():
    try:
        return render_template('index.html')
    except Exception as e:
        logger.error(f"index.html not found: {str(e)}")
        return "Homepage not found.", 404

@app.route('/register', methods=['GET', 'POST'])
def register():
    try:
        if request.method == 'POST':
            user_data = request.get_json() or request.form.to_dict()
            email = user_data.get('email')
            password = user_data.get('password')
            role = user_data.get('role')

            if not email or not password or not role:
                return jsonify({"error": "Email, password, and role required"}), 400

            required_fields = ["full_name", "email", "phone", "dob", "location", "role"]
            if role == 'patient':
                required_fields.append("age")
            elif role == 'doctor':
                required_fields.append("specialty")
            elif role == 'assistant':
                required_fields.append("department")
            missing_fields = [field for field in required_fields if field not in user_data or not user_data[field]]
            if missing_fields:
                return jsonify({"error": f"Missing fields: {', '.join(missing_fields)}"}), 400

            user = auth.create_user(email=email, password=password, uid=None)
            uid = user.uid

            if role == 'doctor':
                consultant_id = generate_consultant_id()
                user_data["consultant_id"] = consultant_id
                user_data["doctor_id"] = consultant_id
                user_ref = db.collection('consultant_registrations').document(consultant_id)
            elif role == 'assistant':
                user_data["assistant_id"] = f"ASST{uid[-8:]}"
                user_ref = db.collection('assistant_registrations').document(uid)
            else:
                user_ref = db.collection('patient_registrations').document(uid)

            user_data['uid'] = uid
            user_ref.set(user_data)
            return jsonify({'success': True, 'message': 'Registration successful.', 'redirect': '/login'})
        return render_template('registration.html')
    except Exception as e:
        return jsonify({"error": str(e)}), 500

@app.route('/login', methods=['GET', 'POST'])
def login():
    if request.method == 'GET':
        try:
            if 'user_info' in session:
                id_token = session.get('idToken')
                if id_token:
                    decoded_token = auth.verify_id_token(id_token, clock_skew_seconds=60)
                    return redirect('/dashboard')
            return render_template('login.html')
        except Exception as e:
            return "Login page not found.", 404
    try:
        id_token = request.headers.get('Authorization', '').replace('Bearer ', '') or request.form.get('idToken')
        if not id_token:
            return jsonify({"error": "No ID token provided"}), 400

        decoded_token = auth.verify_id_token(id_token, clock_skew_seconds=60)
        email = decoded_token.get('email')
        uid = decoded_token.get('uid')

        data = request.get_json() or {}
        request_uid = data.get('uid')
        if request_uid and request_uid != uid:
            return jsonify({"error": "UID mismatch"}), 401

        role = data.get('role')
        if not role:
            return jsonify({"error": "Role not provided"}), 400

        role_mapping = {
            'patient': 'patient_registrations',
            'doctor': 'consultant_registrations',
            'assistant': 'assistant_registrations'
        }
        if role not in role_mapping:
            return jsonify({"error": "Invalid role"}), 400

        collection_name = role_mapping[role]
        query = db.collection(collection_name).where('email', '==', email).limit(1).get()
        user_data = None
        doc_id = None
        for doc in query:
            user_data = doc.to_dict()
            doc_id = doc.id
            break

        if not user_data:
            return jsonify({"error": f"User not found in {role} collection"}), 401

        full_name = user_data.get('full_name', uid)
        session['user_info'] = {
            'email': email,
            'role': role,
            'uid': doc_id if role == 'doctor' else uid,
            'full_name': full_name
        }
        session["idToken"] = id_token
        # Reset conversation state on login to clear stale data
        reset_conversation_state(session)
        logger.debug("Conversation state reset after login for user: %s", email)
        return jsonify({"success": True, "redirect": "/dashboard"})
    except Exception as e:
        return jsonify({"error": "Login failed", "details": str(e)}), 500

@app.route('/check-auth', methods=['GET'])
def check_auth():
    auth_header = request.headers.get('Authorization')
    token = None
    if auth_header and auth_header.startswith("Bearer "):
        token = auth_header.split(" ")[1]
    elif 'idToken' in session:
        token = session['idToken']
    else:
        return jsonify({"authenticated": False}), 401

    try:
        decoded_token = auth.verify_id_token(token, clock_skew_seconds=60)
        return jsonify({"authenticated": True})
    except Exception:
        return jsonify({"authenticated": False}), 401

@app.route('/further_patient_registration', methods=['GET'])
@token_required
def further_patient_registration():
    try:
        user = request.user
        uid = user.get('uid')
        session_info = session.get('user_info', {})
        patient_name = session_info.get('full_name', uid)

        # Reset conversation state on page load/refresh, including Firestore
        reset_conversation_state(session, clear_firestore=True, uid=uid)
        logger.debug("Conversation state reset in further_patient_registration for user: %s", uid)

        return render_template('further_patient_registration.html', user_info={'uid': uid, 'patient_name': patient_name})
    except Exception as e:
        return jsonify({"error": "Error loading further patient registration", "details": str(e)}), 500

@app.route('/transcribe', methods=['POST'])
@token_required
def transcribe():
    try:
        language = "en"
        if 'audio' not in request.files:
            logger.error("No audio file provided")
            return jsonify({"error": "No audio file provided"}), 400

        audio_file = request.files['audio']
        if audio_file.filename == '':
            logger.error("No audio file selected")
            return jsonify({"error": "No audio file selected"}), 400

        temp_dir = tempfile.mkdtemp()
        audio_path = os.path.join(temp_dir, f"temp_audio_{request.user.get('uid')}.webm")
        audio_file.save(audio_path)

        transcription = transcribe_audio(audio_path, language)
        if "failed" in transcription.lower() or "too small" in transcription.lower():
            logger.error(f"Transcription error: {transcription}")
            return jsonify({"error": transcription}), 400

        logger.debug(f"Transcribed text: {transcription}")
        return jsonify({"success": True, "transcript": transcription})
    except Exception as e:
        logger.error(f"Transcription error: {str(e)}")
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
        if not uid:
            logger.error("No authenticated user found")
            return jsonify({"error": "No authenticated user found"}), 401

        data = request.get_json()
        if not data:
            logger.error("No JSON payload provided")
            return jsonify({"error": "No JSON payload provided"}), 400

        session_id = data.get('sessionId')
        transcript = data.get('transcript', '').strip()
        medical_data = data.get('medicalData', {})
        current_state = data.get('currentState', 'INITIAL')

        # Reset session for new session_id to avoid stale data
        if 'last_session_id' not in session or session['last_session_id'] != session_id:
            reset_conversation_state(session, clear_firestore=True, uid=uid)
            session['last_session_id'] = session_id
            logger.debug("New session started with session_id: %s", session_id)
            current_state = "INITIAL"  # Force INITIAL state for new session

        if not session_id:
            logger.error("Missing sessionId")
            return jsonify({"error": "Missing sessionId"}), 400

        logger.debug(f"Received start_conversation payload: {data}, using session state: {current_state}")

        # Proceed with conversation
        history = session.get('conversation_history', '')
        processed_data, status_code = process_conversation(
            transcript=transcript, history=history, session_id=session_id, current_state=current_state
        )
        if status_code != 200:
            logger.error(f"Conversation processing failed: {processed_data}")
            return jsonify(processed_data), status_code

        # Update session history
        session['conversation_history'] = f"{history}\nPatient: {transcript}\nAgent: {processed_data['response']}"

        return jsonify({
            "success": True,
            "response": processed_data['response'],
            "medical_data": processed_data.get('medical_data', {"symptoms": []}),
            "audio_url": processed_data.get('audio_url'),
            "nextState": session.get('current_state', 'INITIAL')
        })

    except Exception as e:
        logger.error(f"Error in start_conversation: {str(e)}", exc_info=True)
        return jsonify({"error": str(e)}), 500

@app.route('/add_symptom', methods=['POST'])
@token_required
def add_symptom():
    try:
        data = request.get_json()
        symptom = data.get('symptom').lower()
        synonyms = data.get('synonyms', [])
        specialty = data.get('specialty', 'general medicine').lower()

        valid_specialties = [
            'dermatology', 'ent', 'nephrology', 'urology', 'endocrinology', 'general medicine',
            'gastroenterology', 'general surgery', 'psychiatry', 'gynecology', 'oncology',
            'pediatrics', 'cardiology', 'neurology', 'orthopedics', 'none'
        ]
        if specialty not in valid_specialties:
            return jsonify({"error": f"Invalid specialty. Must be one of: {', '.join(valid_specialties)}"}), 400

        doc_ref = db.collection('symptom_mappings').document(symptom)
        doc_ref.set({
            'standard_term': symptom,
            'synonyms': synonyms,
            'specialty': specialty,
            'timestamp': firestore.SERVER_TIMESTAMP
        })

        symptom_specialty_map[symptom] = {
            'standard_term': symptom,
            'synonyms': synonyms,
            'specialty': specialty
        }
        logger.info(f"Added symptom {symptom} to Firestore")
        return jsonify({"message": f"Added {symptom} to symptom mappings"})
    except Exception as e:
        return jsonify({"error": str(e)}), 500

@app.route('/generate_prescription', methods=['POST'])
@token_required
def generate_prescription():
    try:
        data = request.get_json()
        symptom = data.get('symptom').lower()
        uid = request.user.get('uid')

        doc_ref = db.collection('prescriptions_cache').document(symptom)
        doc = doc_ref.get()
        if doc.exists:
            return jsonify({"symptom": symptom, "prescription": doc.to_dict().get('prescription'), "source": "cache"})

        prescription = process_text_with_gemini(
            extracted_text=f"Generate a prescription for {symptom}",
            category="prescriptions",
            language="en",
            patient_name="Patient",
            existing_text=None,
            uid=uid
        )['professional_summary']

        doc_ref.set({
            'symptom': symptom,
            'prescription': prescription,
            'timestamp': firestore.SERVER_TIMESTAMP
        })

        return jsonify({"symptom": symptom, "prescription": prescription, "source": "generated"})
    except Exception as e:
        return jsonify({"error": str(e)}), 500

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
        blob = bucket.blob(storage_path)
        blob.upload_from_file(file.stream, content_type=file.content_type)
        file_url = blob.public_url
        
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

        if not all([language_text, file_path, category, uid, consultant_id]):
            missing = [key for key, value in request.form.items() if not value]
            return jsonify({'success': False, 'error': f"Missing fields: {missing}"}), 400

        doc_ref = db.collection('lab_records_cache').document(f"{uid}_{hash(file_path)}")
        doc = doc_ref.get()
        if doc.exists:
            return jsonify({'success': True, **doc.to_dict()})

        blob = bucket.blob(file_path)
        temp_file = os.path.join(tempfile.gettempdir(), f'temp_{uid}.jpg')
        os.makedirs(os.path.dirname(temp_file), exist_ok=True)
        blob.download_to_filename(temp_file)

        extracted_text = extract_text_from_image(temp_file)
        os.remove(temp_file)

        patient_ref = db.collection('patient_registrations').document(uid)
        patient_snap = patient_ref.get()
        patient_name = patient_snap.to_dict().get('full_name', 'Patient') if patient_snap.exists else 'Patient'

        result = process_text_with_gemini(
            extracted_text=extracted_text,
            category=category,
            language=language_text,
            patient_name=patient_name,
            existing_text=None,
            uid=uid
        )

        doc_ref.set({
            'uid': uid,
            'file_path': file_path,
            'language': result['language'],
            'regional_summary': result['regional_summary'],
            'professional_summary': result['professional_summary'],
            'english_patient_summary': result['english_patient_summary'],
            'timestamp': firestore.SERVER_TIMESTAMP
        })

        doc_ref = db.collection(category).document()
        doc_ref.set({
            'uid': uid,
            'consultant_id': consultant_id,
            'patient_name': patient_name,
            'summary': result['regional_summary'],
            'professional_summary': result['professional_summary'],
            'english_patient_summary': result['english_patient_summary'],
            'language': result['language'],
            'timestamp': firestore.SERVER_TIMESTAMP
        })

        return jsonify({'success': True, **result})
    except Exception as e:
        return jsonify({'success': False, 'error': str(e)}), 500

@app.route('/process-medical-history', methods=['POST'])
@token_required
def process_medical_history():
    try:
        data = request.get_json()
        uid = data.get('uid')
        if not uid:
            return jsonify({'success': False, 'error': 'Missing UID'}), 400

        summary = generate_medical_history(uid, db)
        return jsonify({'success': True, 'summary': summary})
    except Exception as e:
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
            raise ValueError("Role is required")

        collection_map = {
            'patient': 'patient_registrations',
            'doctor': 'consultant_registrations',
            'assistant': 'assistant_registrations'
        }
        collection_name = collection_map.get(role)
        query = db.collection(collection_name).where('email', '==', email).limit(1).get()
        user_snap = next(iter(query), None)

        if not user_snap:
            session.pop('user_info', None)
            session.pop('idToken', None)
            return jsonify({"error": "User not found, please log in again"}), 401

        user_data = user_snap.to_dict()
        if user_data.get('email') != email:
            session.pop('user_info', None)
            session.pop('idToken', None)
            return jsonify({"error": "Email mismatch, please log in again"}), 401

        if role == 'doctor':
            consultant_id = user_snap.id
            if not consultant_id.startswith('DR'):
                consultant_id = user_data.get('consultant_id', uid)
            session['user_info']['uid'] = consultant_id
            uid = consultant_id

        dashboards = {
            'patient': 'patient_dashboard.html',
            'doctor': 'consultantDashboard.html',
            'assistant': 'assistant_dashboard.html'
        }
        template = dashboards.get(role)
        if not template:
            raise ValueError(f"No template for role {role}")

        return render_template(template, user_info={'uid': uid, 'email': email, 'role': role}, login_success=True)
    except Exception as e:
        return jsonify({"error": "Error rendering dashboard", "details": str(e)}), 500

@app.route('/logout')
def logout():
    if request.args.get('confirm') != 'yes':
        return "Are you sure you want to log out? <a href='/logout?confirm=yes'>Confirm</a> | <a href='/dashboard'>Cancel</a>", 200
    session.clear()
    return redirect('/login')

@app.errorhandler(404)
def page_not_found(e):
    return render_template('errors/404.html', error_poem="Blank space resonates\n404 hums creation\nNew worlds now beckon"), 404

@app.route('/debug_static', methods=['GET'])
def debug_static():
    """
    Debug route to list files in the static directory.
    """
    try:
        static_dir = os.path.join(app.root_path, 'static')
        files = os.listdir(static_dir)
        logger.debug(f"Static directory contents: {files}")
        return jsonify({"static_files": files})
    except Exception as e:
        logger.error(f"Error listing static files: {str(e)}")
        return jsonify({"error": str(e)}), 500

if __name__ == '__main__':
    import hypercorn.asyncio
    config = hypercorn.Config()
    config.bind = [f"127.0.0.1:{os.environ.get('PORT', 5000)}"]
    asyncio.run(hypercorn.asyncio.serve(app, config))