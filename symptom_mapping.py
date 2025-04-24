import re
import logging
import json
import os
from gemini_processor import process_text_with_gemini

# Create logger instance
logger = logging.getLogger(__name__)

valid_specialties = [
    'dermatology', 'ent', 'nephrology', 'urology', 'endocrinology', 'general medicine',
    'gastroenterology', 'general surgery', 'psychiatry', 'gynecology', 'oncology',
    'pediatrics', 'cardiology', 'neurology', 'orthopedics'
]

# Define the default symptom_specialty_map
default_symptom_specialty_map = {
    "rash": "dermatology",
    "itchiness": "dermatology",
    "skin irritation": "dermatology",
    "acne": "dermatology",
    "hives": "dermatology",
    "sore throat": "ent",
    "ear pain": "ent",
    "nasal congestion": "ent",
    "dry cough": "ent",
    "swelling": "cardiology",
    "swollen ankles": "cardiology",
    "edema": "cardiology",
    "urination issues": "nephrology",
    "kidney pain": "nephrology",
    "hematuria": "urology",  # Default to Urology, will refine in determine_specialty
    "kidney stones": "urology",
    "fatigue": "general medicine",
    "weight changes": "endocrinology",
    "thirst": "endocrinology",
    "fever": "general medicine",
    "general weakness": "general medicine",
    "body aches": "general medicine",
    "stomach pain": "gastroenterology",
    "nausea": "gastroenterology",
    "diarrhea": "gastroenterology",
    "abdominal pain": "general surgery",
    "injury": "general surgery",
    "swelling (surgical)": "general surgery",
    "urinary tract infection": "urology",
    "lower back pain": "urology",
    "difficulty urinating": "urology",
    "anxiety": "psychiatry",
    "depression": "psychiatry",
    "sleep issues": "psychiatry",
    "inability to sleep": "psychiatry",
    "menstrual issues": "gynecology",
    "pelvic pain": "gynecology",
    "irregular periods": "gynecology",
    "spotting": "gynecology",
    "heavy periods": "gynecology",
    "unexplained weight loss": "oncology",
    "lump": "oncology",
    "chronic fatigue": "oncology",
    "child fever": "pediatrics",
    "child cough": "pediatrics",
    "child rash": "pediatrics",
    "chest pain": "cardiology",
    "shortness of breath": "cardiology",
    "palpitations": "cardiology",
    "headache": "neurology",
    "dizziness": "neurology",
    "seizures": "neurology",
    "numbness and tingling in hands": "neurology",
    "pins and needles": "neurology",
    "balance issues": "neurology",
    "off balance": "neurology",
    "joint pain": "orthopedics",
    "muscle pain": "orthopedics",
    "fracture": "orthopedics",
    "unknown": "general medicine",
    "exhaustion": "oncology",
    "lumps in neck": "general surgery",
    "cough": "general medicine",
    "tingling numbness": "neurology",
    "night sweats": "oncology"
}

# Load symptom_specialty_map from symptom_mapping.json at startup and merge with default mappings
symptom_mapping_file = 'symptom_mapping.json'
symptom_specialty_map = {}

if os.path.exists(symptom_mapping_file):
    try:
        with open(symptom_mapping_file, 'r') as f:
            symptom_specialty_map = json.load(f)
        logger.info("Loaded symptom mappings from symptom_mapping.json")
        # Merge default mappings into loaded map to ensure all default symptoms are included
        updated = False
        for symptom, specialty in default_symptom_specialty_map.items():
            if symptom not in symptom_specialty_map:
                symptom_specialty_map[symptom] = specialty
                updated = True
        if updated:
            # Save the updated map back to the file
            try:
                with open(symptom_mapping_file, 'w') as f:
                    json.dump(symptom_specialty_map, f, indent=4)
                logger.info("Updated symptom_mapping.json with missing default mappings")
            except Exception as e:
                logger.error(f"Failed to update symptom_mapping.json with default mappings: {e}")
    except Exception as e:
        logger.error(f"Failed to load symptom_mapping.json: {e}")
        # Fallback to default mappings if the file cannot be loaded
        symptom_specialty_map = default_symptom_specialty_map.copy()
else:
    logger.warning("symptom_mapping.json not found, creating with default mappings")
    symptom_specialty_map = default_symptom_specialty_map.copy()
    # Create the file with default mappings
    try:
        with open(symptom_mapping_file, 'w') as f:
            json.dump(symptom_specialty_map, f, indent=4)
        logger.info("Created symptom_mapping.json with default mappings")
    except Exception as e:
        logger.error(f"Failed to create symptom_mapping.json: {e}")

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

def parse_duration(duration_str):
    """
    Parse a duration string and return the maximum duration in months.
    
    Args:
        duration_str (str): The duration string (e.g., "three months", "2 to three months", "90 days").
    
    Returns:
        float: The duration in months, or None if parsing fails.
    """
    try:
        duration_str = duration_str.lower().strip()
        # Replace textual numbers with digits
        number_words = {
            "one": 1, "two": 2, "three": 3, "four": 4, "five": 5,
            "six": 6, "seven": 7, "eight": 8, "nine": 9, "ten": 10
        }
        for word, num in number_words.items():
            duration_str = duration_str.replace(word, str(num))

        # Handle ranges like "2 to 3 months"
        if "to" in duration_str:
            parts = duration_str.split("to")
            if len(parts) == 2:
                start = parts[0].strip()
                end = parts[1].strip()
                # Extract the end duration (use the higher value for conservative assignment)
                end_num = re.search(r'(\d+(\.\d+)?)', end)
                if end_num:
                    end_value = float(end_num.group(1))
                    if "month" in end:
                        return end_value
                    elif "week" in end:
                        return end_value / 4  # Approximate 4 weeks per month
                    elif "day" in end:
                        return end_value / 30  # Approximate 30 days per month
            return None

        # Handle single durations like "3 months" or "90 days"
        num_match = re.search(r'(\d+(\.\d+)?)', duration_str)
        if num_match:
            num = float(num_match.group(1))
            if "month" in duration_str:
                return num
            elif "week" in duration_str:
                return num / 4  # Approximate 4 weeks per month
            elif "day" in duration_str:
                return num / 30  # Approximate 30 days per month
        return None
    except Exception as e:
        logger.error(f"Error parsing duration '{duration_str}': {e}")
        return None

def determine_specialty(symptoms, severity=None, duration=None, additional_symptoms=None):
    """
    Determine the most appropriate medical specialty based on a list of symptoms, severity, duration, and additional symptoms.
    
    Args:
        symptoms (list): A list of symptom strings.
        severity (list): A list of severity levels (e.g., 'mild', 'moderate', 'severe').
        duration (list): A list of duration strings (e.g., 'two months').
        additional_symptoms (list): A list of additional symptoms to consider.
    
    Returns:
        str: The determined specialty in lowercase, defaulting to 'general medicine' if unclear.
    """
    specialty_counts = {}
    additional_symptoms = additional_symptoms or []

    for idx, symptom in enumerate(symptoms):
        try:
            cleaned_symptom = clean_symptom(symptom)
            base_specialty = symptom_specialty_map.get(cleaned_symptom, None)
            if not base_specialty:
                base_specialty = update_mapping_with_gemini(cleaned_symptom, additional_symptoms)
            
            # Apply nuanced rules for specific symptoms
            if cleaned_symptom == "headache":
                current_severity = severity[idx] if idx < len(severity) else None
                current_duration = duration[idx] if idx < len(duration) else None
                has_neurological_symptoms = any(
                    clean_symptom(s) in ["tremors", "vision changes", "seizures", "numbness and tingling in hands", "pins and needles", "balance issues", "off balance"]
                    for s in additional_symptoms
                )
                duration_in_months = parse_duration(current_duration) if current_duration else None
                logger.debug(f"Evaluating headache: severity={current_severity}, duration={current_duration}, duration_in_months={duration_in_months}, has_neurological_symptoms={has_neurological_symptoms}")
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
                # Hematuria can be associated with Urology (e.g., UTI, kidney stones) or Nephrology (e.g., kidney issues)
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
                    specialty = "urology"  # Default to Urology if no clear context
            else:
                specialty = base_specialty

            if specialty:
                specialty_counts[specialty] = specialty_counts.get(specialty, 0) + 1

        except Exception as e:
            logger.error(f"Error processing symptom '{symptom}': {e}")
            continue
    
    if not specialty_counts:
        return 'general medicine'
    
    try:
        max_count = max(specialty_counts.values())
        top_specialties = [s for s, count in specialty_counts.items() if count == max_count]
        return top_specialties[0] if len(top_specialties) == 1 else 'general medicine'
    except Exception as e:
        logger.error(f"Error determining specialty: {e}")
        return 'general medicine'

def update_mapping_with_gemini(symptom, additional_symptoms=None):
    """
    Use Gemini to map an unknown symptom to a specialty and update the symptom_specialty_map.
    
    Args:
        symptom (str): The unknown symptom to map.
        additional_symptoms (list): List of additional symptoms for context.
    
    Returns:
        str: The determined specialty in lowercase, defaulting to 'general medicine' on error.
    """
    additional_context = f"Additional symptoms for context: {', '.join(additional_symptoms)}" if additional_symptoms else "No additional symptoms provided."
    prompt = (
        f"You are a medical expert. Given the symptom '{symptom}', determine the most appropriate medical specialty from the following list:\n"
        "- dermatology\n- ent\n- nephrology\n- urology\n- endocrinology\n- general medicine\n- gastroenterology\n"
        "- general surgery\n- psychiatry\n- gynecology\n- oncology\n- pediatrics\n- cardiology\n- neurology\n- orthopedics\n\n"
        f"{additional_context}\n\n"
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
        # Save the updated symptom_specialty_map to symptom_mapping.json
        try:
            with open(symptom_mapping_file, 'w') as f:
                json.dump(symptom_specialty_map, f, indent=4)
            logger.info(f"Updated symptom_mapping.json with new mapping: {symptom} -> {specialty}")
        except Exception as e:
            logger.error(f"Failed to update symptom_mapping.json: {e}")
        return specialty
    except Exception as e:
        logger.error(f"Error processing with Gemini for symptom '{symptom}': {e}")
        return 'general medicine'