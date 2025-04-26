import re
import logging
from firebase_admin import firestore
from gemini_processor import process_text_with_gemini

# Create logger instance
logger = logging.getLogger(__name__)

valid_specialties = [
    'dermatology', 'ent', 'nephrology', 'urology', 'endocrinology', 'general medicine',
    'gastroenterology', 'general surgery', 'psychiatry', 'gynecology', 'oncology',
    'pediatrics', 'cardiology', 'neurology', 'orthopedics', 'none'
]

# Default symptom mappings (fallback)
default_symptom_specialty_map = {
    "rash": {"standard_term": "rash", "synonyms": ["red spots", "skin breakout"], "specialty": "dermatology"},
    "vomiting": {"standard_term": "vomiting", "synonyms": ["throwing up", "puking", "emesis"], "specialty": "gastroenterology"},
    "headache": {"standard_term": "headache", "synonyms": ["head pain", "tight band sensation"], "specialty": "neurology"},
    "fever": {"standard_term": "fever", "synonyms": ["high temperature", "pyrexia"], "specialty": "general medicine"},
    "unknown": {"standard_term": "unknown", "synonyms": [], "specialty": "general medicine"},
    "yes": {"standard_term": "yes", "synonyms": ["sure", "okay", "please"], "specialty": "none"}
}

# Initialize Firestore client
db = firestore.client()

def load_symptom_mappings():
    """
    Load symptom mappings from Firestore or use defaults.
    Returns: Dict mapping standard terms to {standard_term, synonyms, specialty}.
    """
    try:
        mappings = {}
        docs = db.collection('symptom_mappings').stream()
        for doc in docs:
            mappings[doc.id] = doc.to_dict()
        if not mappings:
            logger.warning("No symptom mappings in Firestore, using defaults")
            mappings = default_symptom_specialty_map.copy()
            for symptom, data in mappings.items():
                db.collection('symptom_mappings').document(symptom).set({
                    'standard_term': data['standard_term'],
                    'synonyms': data['synonyms'],
                    'specialty': data['specialty'],
                    'timestamp': firestore.SERVER_TIMESTAMP
                })
    except Exception as e:
        logger.error(f"Failed to load symptom mappings from Firestore: {e}")
        mappings = default_symptom_specialty_map.copy()
    return mappings

symptom_specialty_map = load_symptom_mappings()

def clean_symptom(symptom):
    """
    Clean and normalize the symptom string.
    Args:
        symptom (str): Raw symptom string.
    Returns:
        str: Cleaned symptom in lowercase.
    """
    try:
        symptom = re.sub(r'[^\w\s]', '', symptom)
        symptom = re.sub(r'\s+', ' ', symptom).strip().lower()
        return symptom
    except Exception as e:
        logger.error(f"Error cleaning symptom '{symptom}': {e}")
        return symptom.lower().strip()

def normalize_symptom(text):
    """
    Normalize input text to a standard symptom using synonym mappings.
    Args:
        text (str): Patient input (e.g., "throwing up").
    Returns:
        str: Standard symptom term or "unknown".
    """
    text = clean_symptom(text)
    for standard_term, data in symptom_specialty_map.items():
        if text == standard_term or text in data["synonyms"]:
            return standard_term
    return "unknown"

def parse_duration(duration_str):
    """
    Parse duration string to months.
    Args:
        duration_str (str): Duration (e.g., "three months").
    Returns:
        float: Duration in months, or None if parsing fails.
    """
    try:
        duration_str = duration_str.lower().strip()
        number_words = {
            "one": 1, "two": 2, "three": 3, "four": 4, "five": 5,
            "six": 6, "seven": 7, "eight": 8, "nine": 9, "ten": 10
        }
        for word, num in number_words.items():
            duration_str = duration_str.replace(word, str(num))

        if "to" in duration_str:
            parts = duration_str.split("to")
            if len(parts) == 2:
                end = parts[1].strip()
                end_num = re.search(r'(\d+(\.\d+)?)', end)
                if end_num:
                    end_value = float(end_num.group(1))
                    if "month" in end:
                        return end_value
                    elif "week" in end:
                        return end_value / 4
                    elif "day" in end:
                        return end_value / 30
            return None

        num_match = re.search(r'(\d+(\.\d+)?)', duration_str)
        if num_match:
            num = float(num_match.group(1))
            if "month" in duration_str:
                return num
            elif "week" in duration_str:
                return num / 4
            elif "day" in duration_str:
                return num / 30
        return None
    except Exception as e:
        logger.error(f"Error parsing duration '{duration_str}': {e}")
        return None

def determine_specialty(symptoms, severity=None, duration=None, additional_symptoms=None):
    """
    Determine medical specialty based on symptoms, severity, duration, and additional symptoms.
    Args:
        symptoms (list): List of symptom strings.
        severity (list): List of severity levels.
        duration (list): List of duration strings.
        additional_symptoms (list): List of additional symptoms.
    Returns:
        str: Specialty in lowercase, defaulting to 'general medicine'.
    """
    specialty_counts = {}
    additional_symptoms = additional_symptoms or []
    severity = severity or []
    duration = duration or []

    # Track unmapped symptoms to handle collectively with Gemini
    unmapped_symptoms = []
    mapped_specialties = []

    for idx, symptom in enumerate(symptoms):
        try:
            cleaned_symptom = clean_symptom(symptom)
            mapping = symptom_specialty_map.get(cleaned_symptom, None)

            if not mapping:
                unmapped_symptoms.append({
                    "symptom": cleaned_symptom,
                    "severity": severity[idx] if idx < len(severity) else "unknown",
                    "duration": duration[idx] if idx < len(duration) else "unknown"
                })
                continue

            base_specialty = mapping["specialty"]

            # Apply specific rules for certain symptoms
            if cleaned_symptom == "headache":
                current_severity = severity[idx] if idx < len(severity) else None
                current_duration = duration[idx] if idx < len(duration) else None
                has_neurological_symptoms = any(
                    clean_symptom(s) in ["tremors", "vision changes", "seizures", "numbness and tingling in hands", "pins and needles", "balance issues", "off balance"]
                    for s in additional_symptoms
                )
                duration_in_months = parse_duration(current_duration) if current_duration else None
                if (current_severity in ["mild", "moderate"]) and duration_in_months is not None and duration_in_months <= 3 and not has_neurological_symptoms:
                    specialty = "general medicine"
                else:
                    specialty = "neurology"
            elif cleaned_symptom in ["fever", "child fever"]:
                current_severity = severity[idx] if idx < len(severity) else None
                has_cardiology_symptoms = any(
                    clean_symptom(s) in ["chest pain", "shortness of breath", "palpitations"]
                    for s in additional_symptoms
                )
                if current_severity in ["moderate", "severe"] and has_cardiology_symptoms:
                    specialty = "cardiology"
                elif cleaned_symptom == "child fever":
                    specialty = "pediatrics"
                else:
                    specialty = "general medicine"
            elif cleaned_symptom in ["fatigue", "exhaustion"]:
                current_severity = severity[idx] if idx < len(severity) else None
                current_duration = duration[idx] if idx < len(duration) else None
                has_oncology_symptoms = any(
                    clean_symptom(s) in ["unexplained weight loss", "lump", "night sweats"]
                    for s in additional_symptoms
                )
                if current_severity in ["moderate", "severe"] and current_duration and "month" in current_duration.lower():
                    specialty = "oncology"
                elif any(clean_symptom(s) in ["chest pain", "shortness of breath", "palpitations"] for s in additional_symptoms):
                    specialty = "cardiology"
                else:
                    specialty = "general medicine"
            elif cleaned_symptom == "chest pain":
                current_severity = severity[idx] if idx < len(severity) else None
                has_cardiology_symptoms = any(
                    clean_symptom(s) in ["shortness of breath", "palpitations"]
                    for s in additional_symptoms
                )
                if current_severity in ["moderate", "severe"] or has_cardiology_symptoms:
                    specialty = "cardiology"
                else:
                    specialty = "general medicine"
            elif cleaned_symptom == "hematuria":
                has_nephrology_symptoms = any(
                    clean_symptom(s) in ["urination issues", "kidney pain"]
                    for s in additional_symptoms
                )
                has_urology_symptoms = any(
                    clean_symptom(s) in ["urinary tract infection", "kidney stones", "difficulty urinating", "lower back pain"]
                    for s in additional_symptoms
                )
                if has_nephrology_symptoms:
                    specialty = "nephrology"
                elif has_urology_symptoms:
                    specialty = "urology"
                else:
                    specialty = "urology"
            else:
                specialty = base_specialty

            if specialty:
                specialty_counts[specialty] = specialty_counts.get(specialty, 0) + 1
                mapped_specialties.append(specialty)

        except Exception as e:
            logger.error(f"Error processing symptom '{symptom}': {str(e)}")
            continue

    # Handle unmapped symptoms collectively with Gemini
    if unmapped_symptoms:
        # Construct a single prompt for all unmapped symptoms with improved context
        symptom_details = []
        for unmapped in unmapped_symptoms:
            symptom_details.append(
                f"Symptom: {unmapped['symptom']}, Severity: {unmapped['severity']}, Duration: {unmapped['duration']}"
            )
        symptom_text = "; ".join(symptom_details)
        additional_context = f"Additional symptoms: {', '.join(additional_symptoms)}" if additional_symptoms else "No additional symptoms."
        prompt = (
            f"You are a medical expert specializing in symptom-to-specialty mapping. Given the following symptoms, determine the most appropriate medical specialty from: "
            f"{', '.join(valid_specialties)}. Consider the severity and duration to ensure the specialty aligns with medical practice. For example, kidney stones typically map to urology, "
            f"while kidney pain might map to nephrology if accompanied by urination issues. Provide the most precise specialty based on the symptom context.\n\n"
            f"{symptom_text}\n\n{additional_context}\n\nReturn the specialty name only in lowercase."
        )
        logger.debug(f"Using Gemini to determine specialty for unmapped symptoms: {symptom_text}")
        try:
            result = process_text_with_gemini(
                extracted_text=prompt,
                category="symptom_mapping",
                language="en",
                patient_name="",
                existing_text="",
                uid=""
            )
            gemini_specialty = result.get('regional_summary', 'general medicine').strip().lower()
            if gemini_specialty not in valid_specialties:
                gemini_specialty = 'general medicine'
            # Update mappings for each unmapped symptom
            for unmapped in unmapped_symptoms:
                cleaned_symptom = unmapped['symptom']
                symptom_specialty_map[cleaned_symptom] = {
                    'standard_term': cleaned_symptom,
                    'synonyms': [],
                    'specialty': gemini_specialty
                }
                db.collection('symptom_mappings').document(cleaned_symptom).set({
                    'standard_term': cleaned_symptom,
                    'synonyms': [],
                    'specialty': gemini_specialty,
                    'timestamp': firestore.SERVER_TIMESTAMP
                })
                logger.info(f"Updated Firestore with new mapping: {cleaned_symptom} -> {gemini_specialty}")
            specialty_counts[gemini_specialty] = specialty_counts.get(gemini_specialty, 0) + len(unmapped_symptoms)
            mapped_specialties.extend([gemini_specialty] * len(unmapped_symptoms))
        except Exception as e:
            logger.error(f"Error processing with Gemini for unmapped symptoms: {str(e)}")
            gemini_specialty = 'general medicine'
            # Update mappings with default specialty
            for unmapped in unmapped_symptoms:
                cleaned_symptom = unmapped['symptom']
                symptom_specialty_map[cleaned_symptom] = {
                    'standard_term': cleaned_symptom,
                    'synonyms': [],
                    'specialty': gemini_specialty
                }
                db.collection('symptom_mappings').document(cleaned_symptom).set({
                    'standard_term': cleaned_symptom,
                    'synonyms': [],
                    'specialty': gemini_specialty,
                    'timestamp': firestore.SERVER_TIMESTAMP
                })
                logger.info(f"Defaulted to general medicine for unmapped symptom: {cleaned_symptom}")
            specialty_counts[gemini_specialty] = specialty_counts.get(gemini_specialty, 0) + len(unmapped_symptoms)
            mapped_specialties.extend([gemini_specialty] * len(unmapped_symptoms))

    if not specialty_counts:
        return 'general medicine'

    try:
        # Prioritize specialties based on frequency and context
        max_count = max(specialty_counts.values())
        top_specialties = [s for s, count in specialty_counts.items() if count == max_count]
        if len(top_specialties) == 1:
            return top_specialties[0]

        # If there's a tie, use Gemini to decide based on all symptoms
        all_symptoms = symptoms + additional_symptoms
        symptom_details = []
        for idx, symptom in enumerate(symptoms):
            detail = f"Symptom: {symptom}"
            if idx < len(severity):
                detail += f", Severity: {severity[idx]}"
            if idx < len(duration):
                detail += f", Duration: {duration[idx]}"
            symptom_details.append(detail)
        for symptom in additional_symptoms:
            symptom_details.append(f"Symptom: {symptom}")
        symptom_text = "; ".join(symptom_details)
        prompt = (
            f"You are a medical expert specializing in symptom-to-specialty mapping. Given the following symptoms, determine the most appropriate medical specialty from: "
            f"{', '.join(valid_specialties)}. Consider the severity and duration to ensure the specialty aligns with medical practice. For example, kidney stones typically map to urology, "
            f"while kidney pain might map to nephrology if accompanied by urination issues. Provide the most precise specialty based on the symptom context.\n\n"
            f"{symptom_text}\n\nReturn the specialty name only in lowercase."
        )
        logger.debug(f"Using Gemini to resolve specialty tie: {symptom_text}")
        result = process_text_with_gemini(
            extracted_text=prompt,
            category="symptom_mapping",
            language="en",
            patient_name="",
            existing_text="",
            uid=""
        )
        final_specialty = result.get('regional_summary', 'general medicine').strip().lower()
        if final_specialty not in valid_specialties:
            final_specialty = 'general medicine'
        return final_specialty
    except Exception as e:
        logger.error(f"Error determining specialty: {str(e)}")
        return 'general medicine'