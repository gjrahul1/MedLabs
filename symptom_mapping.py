import re
import logging
from gemini_processor import process_text_with_gemini

# Create logger instance
logger = logging.getLogger(__name__)

valid_specialties = [
    'dermatology', 'ent', 'nephrology', 'urology', 'endocrinology', 'general medicine',
    'gastroenterology', 'general surgery', 'psychiatry', 'gynecology', 'oncology',
    'pediatrics', 'cardiology', 'neurology', 'orthopedics'
]

symptom_specialty_map = {
    'anxiety': 'psychiatry',
    'depression': 'psychiatry',
    'fever': 'general medicine',
    'cough': 'general medicine',
    'rash': 'dermatology',
    'sore throat': 'ent',
    'kidney pain': 'nephrology',
    'urination issues': 'nephrology',
    'weight changes': 'endocrinology',
    'thirst': 'endocrinology',
    'stomach pain': 'gastroenterology',
    'nausea': 'gastroenterology',
    'diarrhea': 'gastroenterology',
    'urinary tract infection': 'urology',
    'lower back pain': 'urology',
    'difficulty urinating': 'urology',
    'sleep issues': 'psychiatry',
    'inability to sleep': 'psychiatry',
    'abdominal pain': 'general surgery',
    'injury': 'general surgery',
    'menstrual issues': 'gynecology',
    'pelvic pain': 'gynecology',
    'unexplained weight loss': 'oncology',
    'lump': 'oncology',
    'child fever': 'pediatrics',
    'child cough': 'pediatrics',
    'chest pain': 'cardiology',
    'shortness of breath': 'cardiology',
    'headache': 'neurology',
    'dizziness': 'neurology',
    'tingling numbness': 'neurology',
    'tremors': 'neurology',
    'joint pain': 'orthopedics',
    'muscle pain': 'orthopedics',
    'fracture': 'orthopedics'
}

def clean_symptom(symptom):
    """
    Clean and normalize the symptom string by removing punctuation and extra spaces.
    
    Args:
        symptom (str): The raw symptom string.
    
    Returns:
        str: The cleaned and normalized symptom in lowercase.
    """
    try:
        symptom = re.sub(r'[^\w\s]', '', symptom)
        symptom = re.sub(r'\s+', ' ', symptom).strip().lower()
        return symptom
    except Exception as e:
        logger.error(f"Error cleaning symptom '{symptom}': {e}")
        return symptom.lower().strip()

def determine_specialty(symptoms):
    """
    Determine the most appropriate medical specialty based on a list of symptoms.
    Handles multi-word symptoms and prioritizes 'neurology' for symptoms like 'tingling numbness'.
    
    Args:
        symptoms (list): A list of symptom strings.
    
    Returns:
        str: The determined specialty in lowercase, defaulting to 'general medicine' if unclear.
    """
    specialty_counts = {}
    for symptom in symptoms:
        try:
            cleaned_symptom = clean_symptom(symptom)
            specialty = symptom_specialty_map.get(cleaned_symptom, None)
            if not specialty:
                specialty = update_mapping_with_gemini(cleaned_symptom)
            if specialty:
                specialty_counts[specialty] = specialty_counts.get(specialty, 0) + 1
        except Exception as e:
            logger.error(f"Error processing symptom '{symptom}': {e}")
            continue
    
    if not specialty_counts:
        return 'general medicine'
    
    if 'neurology' in specialty_counts:
        return 'neurology'
    
    try:
        max_count = max(specialty_counts.values())
        top_specialties = [s for s, count in specialty_counts.items() if count == max_count]
        return top_specialties[0] if len(top_specialties) == 1 else 'general medicine'
    except Exception as e:
        logger.error(f"Error determining specialty: {e}")
        return 'general medicine'

def update_mapping_with_gemini(symptom):
    """
    Use Gemini to map an unknown symptom to a specialty and update the symptom_specialty_map.
    
    Args:
        symptom (str): The unknown symptom to map.
    
    Returns:
        str: The determined specialty in lowercase, defaulting to 'general medicine' on error.
    """
    prompt = (
        f"You are a medical expert. Given the symptom '{symptom}', determine the most appropriate medical specialty from the following list:\n"
        "- dermatology\n- ent\n- nephrology\n- urology\n- endocrinology\n- general medicine\n- gastroenterology\n"
        "- general surgery\n- psychiatry\n- gynecology\n- oncology\n- pediatrics\n- cardiology\n- neurology\n- orthopedics\n\n"
        "Return the specialty name only in lowercase."
    )
    try:
        result = process_text_with_gemini(
            extracted_text=prompt,
            category="symptom_mapping",
            language="en",
            patient_name="",
            existing_text="",
            uid=""
        )
        specialty = result.get('regional_summary', 'general medicine').strip().lower()
        if specialty not in valid_specialties:
            specialty = 'general medicine'
        symptom_specialty_map[symptom] = specialty
        return specialty
    except Exception as e:
        logger.error(f"Error processing with Gemini for symptom '{symptom}': {e}")
        return 'general medicine'