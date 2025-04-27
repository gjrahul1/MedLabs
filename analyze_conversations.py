import firebase_admin
from firebase_admin import credentials, firestore
from google.cloud import storage
import hashlib
from collections import Counter
from sentence_transformers import SentenceTransformer, util
import os
import logging
import json
from datetime import datetime, timedelta
from dotenv import load_dotenv
import re
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.prompts import PromptTemplate
from google.auth import load_credentials_from_file

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Initialize Firebase
cred_path = './Cred/Firebase/Med App/med-labs-42f13-firebase-adminsdk-fbsvc-43b78dfef7.json'
cred = credentials.Certificate(cred_path)
firebase_admin.initialize_app(cred)
db = firestore.Client(project='med-labs-42f13')

# Load credentials for GCS using google-auth
gcs_credentials, _ = load_credentials_from_file(cred_path)
gcs_client = storage.Client(project='med-labs-new-2025', credentials=gcs_credentials)
gcs_bucket = gcs_client.bucket('med-labs-new-bucket-2025')

# Load environment variables from .env file
load_dotenv(dotenv_path=os.path.join(os.getcwd(), '.env'))

# Load sentence transformer model for clustering
model = SentenceTransformer('all-MiniLM-L6-v2')

# Load Gemini API key for LLM normalization
gemini_api_key = os.getenv('GEMINI_API_KEY')
if not gemini_api_key:
    logger.error("GEMINI_API_KEY not found in environment")
    raise ValueError("GEMINI_API_KEY not found in environment")

# LLM for symptom normalization using Gemini-1.5-flash
normalization_llm = ChatGoogleGenerativeAI(
    model="gemini-2.0-flash",  # Note: "gemini-2.0-flash" is not a valid model; using "gemini-1.5-flash"
    google_api_key=gemini_api_key,
    temperature=0.0,
    max_tokens=None,
    timeout=None,
    max_retries=2
)

normalization_prompt_template = PromptTemplate(
    input_variables=["text"],
    template=(
        "You are a medical assistant tasked with normalizing transcriptions to identify standard medical symptoms. "
        "Given the following transcription, extract and list the standard symptom terms (e.g., 'headache', 'dizziness', 'vomiting'). "
        "Focus on medical symptoms only, ignoring non-symptom phrases (e.g., 'I am having', 'feeling'). "
        "If no symptoms are identified, return 'unknown'. "
        "Return a comma-separated list of symptoms in lowercase, or 'unknown' if none are found.\n\n"
        "Transcription: {text}\n\n"
        "Output only the symptom list or 'unknown'."
    )
)

normalization_chain = normalization_prompt_template | normalization_llm | (lambda x: x.content.strip())

def normalize_transcription_regex(text):
    """
    Normalize transcriptions using regex to fix common STT formatting errors.
    
    Args:
        text (str): The raw transcription (e.g., "I am having . head.ache").
    
    Returns:
        str: Normalized transcription (e.g., "i am having a headache").
    """
    # Convert to lowercase for consistency
    text = text.lower()
    
    # Remove periods within words (e.g., "head.ache" → "headache")
    text = re.sub(r'(\w)\.(\w)', r'\1\2', text)
    
    # Replace multiple spaces with a single space
    text = re.sub(r'\s+', ' ', text)
    
    # Trim leading/trailing spaces
    text = text.strip()
    
    # Replace common STT artifacts
    text = text.replace('a head ache', 'headache')
    text = text.replace('head ache', 'headache')
    
    return text

def extract_duration(text):
    """
    Extract duration phrases from text using regex and keyword mapping.
    
    Args:
        text (str): The raw transcription (e.g., "a couple of weeks").
    
    Returns:
        str: Extracted duration phrase or 'unknown'.
    """
    # Map common phrases to standard forms
    duration_phrases = {
        'a couple of weeks': 'two weeks',
        'a couple of days': 'two days',
        'a few weeks': 'three weeks',
        'a few days': 'three days',
        'about a week': 'one week',
        'about a month': 'one month',
        'around a week': 'one week',
        'around a month': 'one month'
    }
    
    # First, check for known phrases
    text_lower = text.lower()
    for phrase, standard in duration_phrases.items():
        if phrase in text_lower:
            return standard
    
    # Then, match patterns like "X weeks", "X days", "X months", etc.
    duration_pattern = r'(\d+\.?\d*\s*(?:week|day|month|year)s?)'
    match = re.search(duration_pattern, text_lower)
    if match:
        return match.group(0).strip()
    
    return "unknown"

def normalize_transcription_llm(text):
    """
    Normalize transcriptions using Gemini-1.5-flash to map to standard symptom terms.
    
    Args:
        text (str): The transcription to normalize (e.g., "I’m feelin’ head pain").
    
    Returns:
        str: Comma-separated list of symptoms or 'unknown' (e.g., "headache" or "headache,dizziness").
    """
    try:
        normalized = normalization_chain.invoke({"text": text})
        return normalized
    except Exception as e:
        logger.error(f"Error normalizing transcription with Gemini: {str(e)}")
        return "unknown"

def load_yes_synonyms():
    """
    Load synonyms for 'yes' from Firestore symptom_mappings.
    
    Returns:
        set: Set of synonyms for 'yes'.
    """
    try:
        doc_ref = db.collection('symptom_mappings').document('yes')
        doc = doc_ref.get()
        if doc.exists:
            synonyms = set(doc.to_dict().get('synonyms', []))
            synonyms.add('yes')  # Include the standard term
            return synonyms
        else:
            return {'yes', 'sure', 'okay', 'please'}
    except Exception as e:
        logger.error(f"Error loading yes synonyms from Firestore: {str(e)}")
        return {'yes', 'sure', 'okay', 'please'}

def load_no_synonyms():
    """
    Load synonyms for 'no' from Firestore symptom_mappings (or use defaults).
    
    Returns:
        set: Set of synonyms for 'no'.
    """
    try:
        doc_ref = db.collection('symptom_mappings').document('no')
        doc = doc_ref.get()
        if doc.exists:
            synonyms = set(doc.to_dict().get('synonyms', []))
            synonyms.add('no')  # Include the standard term
            return synonyms
        else:
            return {'no', 'not really', 'nah', 'nope'}
    except Exception as e:
        logger.error(f"Error loading no synonyms from Firestore: {str(e)}")
        return {'no', 'not really', 'nah', 'nope'}

def load_maybe_synonyms():
    """
    Load synonyms for 'maybe' from Firestore symptom_mappings (or use defaults).
    
    Returns:
        set: Set of synonyms for 'maybe'.
    """
    try:
        doc_ref = db.collection('symptom_mappings').document('maybe')
        doc = doc_ref.get()
        if doc.exists:
            synonyms = set(doc.to_dict().get('synonyms', []))
            synonyms.add('maybe')  # Include the standard term
            return synonyms
        else:
            return {'maybe', 'possibly', 'perhaps', 'i think so'}
    except Exception as e:
        logger.error(f"Error loading maybe synonyms from Firestore: {str(e)}")
        return {'maybe', 'possibly', 'perhaps', 'i think so'}

def load_unknown_synonyms():
    """
    Load synonyms for 'unknown' responses from Firestore symptom_mappings (or use defaults).
    
    Returns:
        set: Set of synonyms for 'unknown'.
    """
    try:
        doc_ref = db.collection('symptom_mappings').document('unknown_response')
        doc = doc_ref.get()
        if doc.exists:
            synonyms = set(doc.to_dict().get('synonyms', []))
            synonyms.add('i don\'t know')  # Include the standard term
            return synonyms
        else:
            return {'i don\'t know', 'not sure', 'unsure', 'i\'m not sure'}
    except Exception as e:
        logger.error(f"Error loading unknown synonyms from Firestore: {str(e)}")
        return {'i don\'t know', 'not sure', 'unsure', 'i\'m not sure'}

def normalize_transcription(text, state, yes_synonyms, no_synonyms, maybe_synonyms, unknown_synonyms):
    """
    Normalize transcriptions based on the conversation state.
    
    Args:
        text (str): The raw transcription.
        state (str): The conversation state (e.g., 'INITIAL', 'CONFIRM').
        yes_synonyms (set): Set of synonyms for 'yes'.
        no_synonyms (set): Set of synonyms for 'no'.
        maybe_synonyms (set): Set of synonyms for 'maybe'.
        unknown_synonyms (set): Set of synonyms for 'unknown' responses.
    
    Returns:
        str: Normalized transcription (symptoms for symptom states, preserved responses for non-symptom states, or 'unknown').
    """
    # First pass: Regex normalization
    text = normalize_transcription_regex(text)
    
    # Define states where symptoms are expected
    symptom_states = {'INITIAL', 'SEVERITY', 'TRIGGERS'}
    non_symptom_states = {'CONFIRM_SEVERITY', 'CONFIRM'}
    
    # For non-symptom states, check if the transcription is a known response
    if state in non_symptom_states:
        cleaned_text = text.strip().lower()
        # Check for yes synonyms
        for synonym in yes_synonyms:
            if cleaned_text.startswith(synonym):
                return synonym  # Preserve the affirmation
        # Check for no synonyms
        for synonym in no_synonyms:
            if cleaned_text.startswith(synonym):
                return synonym  # Preserve the negation
        # Check for maybe synonyms
        for synonym in maybe_synonyms:
            if cleaned_text.startswith(synonym):
                return synonym  # Preserve the uncertainty
        # Check for unknown synonyms
        for synonym in unknown_synonyms:
            if cleaned_text.startswith(synonym):
                return synonym  # Preserve the unknown response
        return "unknown"  # Unrecognized response in non-symptom state
    
    # For DURATION state, extract duration information
    if state == 'DURATION':
        duration = extract_duration(text)
        return duration
    
    # For symptom states, use LLM to extract symptoms
    return normalize_transcription_llm(text)

def analyze_conversations():
    """
    Analyze conversations in Firestore, update symptom mappings, and log fine-tuning data to GCS.
    Run locally to process low-reward interactions and STT errors.
    """
    try:
        # Load synonyms for affirmations, negations, and other responses from Firestore
        yes_synonyms = load_yes_synonyms()
        no_synonyms = load_no_synonyms()
        maybe_synonyms = load_maybe_synonyms()
        unknown_synonyms = load_unknown_synonyms()
        logger.info(f"Loaded yes synonyms: {yes_synonyms}")
        logger.info(f"Loaded no synonyms: {no_synonyms}")
        logger.info(f"Loaded maybe synonyms: {maybe_synonyms}")
        logger.info(f"Loaded unknown synonyms: {unknown_synonyms}")

        # Query all conversations from the last 24 hours
        yesterday = datetime.now() - timedelta(days=1)
        conversations_ref = db.collection('conversations')\
            .where(filter=firestore.FieldFilter('timestamp', '>=', yesterday))
        conversations = conversations_ref.stream()

        # Step 1: Normalize transcriptions in Firestore
        batch = db.batch()
        update_count = 0
        for conv in conversations:
            data = conv.to_dict()
            raw_transcription = data.get('transcription', '')
            state = data.get('state', 'UNKNOWN')
            if raw_transcription:
                # Normalize based on state
                normalized = normalize_transcription(raw_transcription, state, yes_synonyms, no_synonyms, maybe_synonyms, unknown_synonyms)
                # Update the normalized_transcription field if different
                if normalized != data.get('normalized_transcription'):
                    doc_ref = db.collection('conversations').document(conv.id)
                    batch.update(doc_ref, {'normalized_transcription': normalized})
                    update_count += 1
                    logger.info(f"Updated normalized_transcription for conversation {conv.id}: {raw_transcription} → {normalized}")
        batch.commit()
        logger.info(f"Updated {update_count} conversations with normalized transcriptions")

        # Step 2: Re-query conversations for analysis
        conversations = conversations_ref.stream()
        errors = []
        for conv in conversations:
            data = conv.to_dict()
            # Check for STT errors (transcription != normalized_transcription) or repeated questions
            if data.get('is_repeat') or (data.get('transcription') != data.get('normalized_transcription') and data.get('transcription')):
                errors.append({
                    'transcription': data['transcription'],
                    'normalized_transcription': data['normalized_transcription'],
                    'state': data['state'],
                    'response': data['response'],
                    'conversation_id': data['conversation_id']
                })

        # Cluster errors by state
        state_errors = {}
        for error in errors:
            state = error['state']
            transcription = error['transcription'].lower()
            state_errors.setdefault(state, []).append({
                'transcription': transcription,
                'normalized': error['normalized_transcription'],
                'conversation_id': error['conversation_id']
            })

        # Identify misrecognitions (log all errors, no frequency threshold)
        updates = []
        fine_tuning_data = []
        for state, transcriptions in state_errors.items():
            # Process each transcription error
            for t in transcriptions:
                transcription = t['transcription']
                normalized = t['normalized']
                logger.info(f"State: {state}, STT error: {transcription} → {normalized}")
                
                # Handle CONFIRM state errors (e.g., "Divya" → "yes")
                if state == "CONFIRM" and transcription not in yes_synonyms:
                    doc_ref = db.collection('symptom_mappings').document('yes')
                    doc = doc_ref.get()
                    if doc.exists:
                        synonyms = doc.to_dict().get('synonyms', [])
                        if transcription not in synonyms:
                            synonyms.append(transcription)
                            updates.append({
                                'doc_ref': doc_ref,
                                'data': {'synonyms': synonyms, 'timestamp': firestore.SERVER_TIMESTAMP}
                            })
                            logger.info(f"Queued update: Added '{transcription}' as synonym for 'yes' in CONFIRM state")
                
                # Log fine-tuning data for all STT errors
                fine_tuning_data.append({
                    'conversation_id': t['conversation_id'],
                    'incorrect_transcription': transcription,
                    'correct_transcription': normalized,
                    'state': state
                })

        # Apply updates to symptom_mappings
        batch = db.batch()
        for update in updates:
            batch.update(update['doc_ref'], update['data'])
        batch.commit()
        logger.info(f"Applied {len(updates)} updates to symptom_mappings")

        # Log fine-tuning data to GCS (append to the existing file)
        if fine_tuning_data:
            gcs_path = "fine_tuning/whisper_fine_tuning_20250427_044746.jsonl"
            blob = gcs_bucket.blob(gcs_path)
            
            # Download existing data if it exists
            existing_data = []
            try:
                if blob.exists():
                    existing_content = blob.download_as_string().decode('utf-8')
                    existing_data = [json.loads(line) for line in existing_content.splitlines() if line.strip()]
                    logger.info(f"Loaded {len(existing_data)} existing fine-tuning records from GCS")
            except Exception as e:
                logger.error(f"Error loading existing fine-tuning data from GCS: {str(e)}")
                existing_data = []

            # Append new data (deduplicate by conversation_id)
            existing_ids = {item['conversation_id'] for item in existing_data}
            new_data = [item for item in fine_tuning_data if item['conversation_id'] not in existing_ids]
            existing_data.extend(new_data)
            
            # Upload updated data
            jsonl_content = '\n'.join(json.dumps(item) for item in existing_data)
            blob.upload_from_string(jsonl_content, content_type='application/jsonl')
            logger.info(f"Appended {len(new_data)} new records to GCS: {gcs_path}")

        # Update RL policies (contextual bandit)
        states = ['INITIAL', 'SEVERITY', 'CONFIRM_SEVERITY', 'DURATION', 'TRIGGERS', 'CONFIRM']
        for state in states:
            state_convs = db.collection('conversations')\
                .where(filter=firestore.FieldFilter('state', '==', state))\
                .where(filter=firestore.FieldFilter('timestamp', '>=', yesterday))\
                .stream()
            rewards = []
            actions = []
            for conv in state_convs:
                data = conv.to_dict()
                rewards.append(data['reward'])
                actions.append(data['next_state'])

            # Calculate action preferences (average reward per next_state)
            action_rewards = {}
            for action, reward in zip(actions, rewards):
                action_rewards.setdefault(action, []).append(reward)
            preferences = {action: sum(rs) / len(rs) if rs else 0 for action, rs in action_rewards.items()}

            # Store in rl_policies
            doc_ref = db.collection('rl_policies').document(state)
            doc_ref.set({
                'state': state,
                'action_preferences': preferences,
                'timestamp': firestore.SERVER_TIMESTAMP
            })
            logger.info(f"Updated RL policy for state: {state}")

        return {"status": "Analysis complete", "updates": len(updates), "fine_tuning_records": len(new_data)}
    except Exception as e:
        logger.error(f"Error in analyze_conversations: {str(e)}")
        return {"status": "Error", "error": str(e)}

if __name__ == "__main__":
    result = analyze_conversations()
    print(json.dumps(result, indent=2))