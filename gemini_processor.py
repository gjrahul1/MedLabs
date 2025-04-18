from langchain_core.prompts import PromptTemplate
from langchain_core.runnables import RunnableLambda
from langchain_core.output_parsers import StrOutputParser
from langchain_google_genai import ChatGoogleGenerativeAI
import pdfplumber
import logging
import re
import firebase_admin
from firebase_admin import firestore
import os
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv(dotenv_path=os.path.join(os.getcwd(), '.env'))

# Configure logging
logging.basicConfig(level=logging.DEBUG, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def extract_text_from_pdf(pdf_path):
    with pdfplumber.open(pdf_path) as pdf:
        text = ""
        for page in pdf.pages:
            text += page.extract_text() or ""
    return text

def process_text_with_gemini(extracted_text: str, category: str = None, language: str = "kannada", patient_name: str = "ರೋಗಿ", existing_text: str = None, uid: str = None) -> dict:
    # Initialize Firestore client inside the function
    db = firestore.client()

    # Load Gemini API key from environment variable
    api_key = os.getenv('GEMINI_API_KEY')
    logger.debug(f"Loaded Gemini API key: {api_key}")
    if not api_key:
        logger.error("GEMINI_API_KEY not found in .env file")
        raise ValueError("GEMINI_API_KEY not found in .env file")

    # Clean and sanitize the extracted text
    cleaned_text = re.sub(r'[^\x00-\x7F]+', ' ', extracted_text)
    
    # Remove personal identifiers
    # Remove names (e.g., Mr. CH. SAMUEL, Dr. Naveen Polavarapu)
    cleaned_text = re.sub(r'(?:Mr\.|Mrs\.|Ms\.|Dr\.)\s+[A-Z][a-zA-Z]*(?:\s+[A-Z][a-zA-Z]*)*', '[Name]', cleaned_text, flags=re.IGNORECASE)
    # Remove other identifiers (address, phone, contact, id, date, time, location)
    cleaned_text = re.sub(r'\b(?:address|phone|contact|id|date|time|location)\b.*?(?=\n|$)', '', cleaned_text, flags=re.IGNORECASE)
    # Remove standalone dates (e.g., 2025-04-20, 20/04/2025)
    cleaned_text = re.sub(r'\b\d{4}-\d{2}-\d{2}\b|\b\d{2}/\d{2}/\d{4}\b', '[Date]', cleaned_text)
    # Remove phone numbers (e.g., +91 1234567890, 123-456-7890)
    cleaned_text = re.sub(r'\b\+?\d{1,3}[-.\s]?\d{3}[-.\s]?\d{3}[-.\s]?\d{4}\b', '[Phone]', cleaned_text)
    # Remove curly braces content not part of a larger structure
    cleaned_text = re.sub(r'\{(?![^{}]*\})', '', cleaned_text)
    cleaned_text = re.sub(r'(?<!\{)[^{}]*\}(?![^{}]*\})', '', cleaned_text)

    if existing_text:
        existing_text = re.sub(r'(?:Mr\.|Mrs\.|Ms\.|Dr\.)\s+[A-Z][a-zA-Z]*(?:\s+[A-Z][a-zA-Z]*)*', '[Name]', existing_text, flags=re.IGNORECASE)
        existing_text = re.sub(r'\b(?:address|phone|contact|id|date|time|location)\b.*?(?=\n|$)', '', existing_text, flags=re.IGNORECASE)
        existing_text = re.sub(r'\b\d{4}-\d{2}-\d{2}\b|\b\d{2}/\d{2}/\d{4}\b', '[Date]', existing_text)
        existing_text = re.sub(r'\b\+?\d{1,3}[-.\s]?\d{3}[-.\s]?\d{3}[-.\s]?\d{4}\b', '[Phone]', existing_text)
        existing_text = re.sub(r'\{(?![^{}]*\})', '', existing_text)
        existing_text = re.sub(r'(?<!\{)[^{}]*\}(?![^{}]*\})', '', existing_text)

    # Anonymize for professional summary
    professional_cleaned_text = cleaned_text
    if patient_name:
        professional_cleaned_text = re.sub(re.escape(patient_name), '[Patient]', professional_cleaned_text, flags=re.IGNORECASE)
    professional_cleaned_text = re.sub(r'\b(age|aged)?\s*\d+\s*(years|yrs|year|old)?\b', '[Age]', professional_cleaned_text, flags=re.IGNORECASE)

    # Define language prompts for patient summaries
    language_prompts = {
        "kannada": "in simple Kannada (ಕನ್ನಡ), using warm and clear language for the patient, avoiding medical jargon and English text",
        "tamil": "in simple Tamil (தமிழ்), using warm and clear language for the patient, avoiding medical jargon and English text",
        "english": "in simple English, using warm and clear language for the patient, avoiding complex medical jargon"
    }
    selected_language = language.lower()
    if selected_language not in language_prompts:
        selected_language = "kannada"
    logger.debug(f"Selected language for regional summary: {selected_language}")

    # Regional/English Summary for Patient
    if category == 'prescriptions' and existing_text:
        regional_template_text = (
            f"You are Gemini, a helpful language model. Analyze the following English prescription text and create a brief but detailed, "
            f"point-wise summary {language_prompts[selected_language]}. Personalize it with the patient's name '{{patient_name}}'. Focus only on medical "
            f"information (e.g., medicines, dosages). Include the patient's condition, recommended actions or tests, and medicines with dosages. "
            f"Existing text:\n{{existing_text}}\n\nNew text:\n{{new_text}}\n\nOutput only the summary points."
        )
        regional_prompt = PromptTemplate.from_template(regional_template_text)
        regional_inputs = {
            "patient_name": patient_name,
            "existing_text": existing_text,
            "new_text": cleaned_text
        }
    else:
        regional_template_text = (
            f"You are Gemini, a helpful language model. Analyze the following English medical text and create a brief but detailed, "
            f"point-wise summary {language_prompts[selected_language]}. Personalize it with the patient's name '{{patient_name}}'. Focus only on medical "
            f"information. Include the patient's condition and recommended actions or follow-ups. "
            f"Text:\n{{text}}\n\nOutput only the summary points."
        )
        regional_prompt = PromptTemplate.from_template(regional_template_text)
        regional_inputs = {
            "patient_name": patient_name,
            "text": cleaned_text
        }

    # Professional Summary for Doctor (always in English)
    professional_template = (
        f"You are Gemini, a helpful language model. Analyze the following English medical text and create a concise, professional medical summary in English "
        f"for a doctor. Start with the patient's condition as the first point, followed by key findings, prescribed medications with dosages (if applicable), "
        f"and recommended follow-ups or tests. Do not include the patient's name or age. Text:\n{{text}}\n\nOutput only the summary points."
    )
    professional_prompt = PromptTemplate.from_template(professional_template)
    professional_inputs = {"text": professional_cleaned_text}

    # Gemini model
    llm = ChatGoogleGenerativeAI(
        model="gemini-1.5-pro",
        temperature=0,
        max_tokens=None,
        timeout=None,
        max_retries=2,
        google_api_key=api_key
    )

    parser = StrOutputParser()

    # Chains using | operator
    regional_chain = regional_prompt | llm | parser
    professional_chain = professional_prompt | llm | parser

    try:
        regional_summary = regional_chain.invoke(regional_inputs)
        professional_summary = professional_chain.invoke(professional_inputs)

        logger.debug(f"Regional Summary ({selected_language}): {regional_summary}")
        logger.debug(f"Professional Summary (English): {professional_summary}")

        # Generate Medical History if UID is provided
        medical_history = None
        if uid:
            try:
                medical_history = generate_medical_history(uid, db)
                logger.info(f"Generated medical history for UID: {uid}")
            except Exception as e:
                logger.error(f"Failed to generate medical history for UID {uid}: {str(e)}")
                medical_history = "Failed to generate medical history."

        return {
            "regional_summary": regional_summary,
            "professional_summary": professional_summary,
            "medical_history": medical_history,
            "language": selected_language  # Include the language used for the regional summary
        }
    except Exception as e:
        logger.error(f"Error processing text with Gemini: {e}")
        raise

def generate_medical_history(uid: str, db) -> str:
    """
    Generate a concise medical history summary for a patient using initial screening, prescriptions, and lab records.
    Args:
        uid (str): The unique identifier of the patient.
        db: Firestore client instance.
    Returns:
        str: A concise medical history summary.
    """
    # Fetch all relevant data for the patient
    initial_screening = db.collection('initial_screenings').document(f'initial_screening_{uid}').get()
    prescriptions = db.collection('prescriptions').where('uid', '==', uid).order_by('timestamp', direction=firestore.Query.DESCENDING).get()
    lab_records = db.collection('lab_records').where('uid', '==', uid).order_by('timestamp', direction=firestore.Query.DESCENDING).get()

    # Combine data into a single text input
    history_text = "Patient Medical History:\n\nInitial Screening:\n"
    if initial_screening.exists:
        screening_data = initial_screening.to_dict()
        history_text += f"- Symptoms: {screening_data.get('symptoms', 'N/A')}\n"
        history_text += f"  Severity: {screening_data.get('severity', 'N/A')}\n"
        history_text += f"  Duration: {screening_data.get('duration', 'N/A')}\n"
        history_text += f"  Triggers: {screening_data.get('triggers', 'N/A')}\n"
        history_text += f"  Date: {screening_data.get('timestamp', 'N/A')}\n"
    else:
        history_text += "No initial screening data available.\n"

    history_text += "\nPrescriptions:\n"
    if not prescriptions:
        history_text += "No prescriptions available.\n"
    else:
        for doc in prescriptions:
            data = doc.to_dict()
            history_text += f"- Date: {data['timestamp'].strftime('%Y-%m-%d')}\n"
            history_text += f"  Professional Summary: {data['professional_summary']}\n"

    history_text += "\nLab Records:\n"
    if not lab_records:
        history_text += "No lab records available.\n"
    else:
        for doc in lab_records:
            data = doc.to_dict()
            history_text += f"- Date: {data['timestamp'].strftime('%Y-%m-%d')}\n"
            history_text += f"  Professional Summary: {data['professional_summary']}\n"

    # Define the prompt for generating a concise medical history
    history_template = (
        "You are Gemini, a helpful language model. Analyze the following patient medical data and create a concise medical history summary in English. "
        "The summary should be brief, point-wise, and include key conditions, treatments, and metrics (e.g., test results, dosages) from initial screening, prescriptions, and lab records. "
        "Focus on the most relevant information, combining data where appropriate. Do not include the patient's name or age. "
        "Text:\n{text}\n\nOutput only the summary points."
    )
    history_prompt = PromptTemplate.from_template(history_template)
    history_inputs = {"text": history_text}

    # Load Gemini API key from environment variable
    api_key = os.getenv('GEMINI_API_KEY')
    logger.debug(f"Loaded Gemini API key in generate_medical_history: {api_key}")
    if not api_key:
        logger.error("GEMINI_API_KEY not found in .env file")
        raise ValueError("GEMINI_API_KEY not found in .env file")

    # Gemini model
    llm = ChatGoogleGenerativeAI(
        model="gemini-1.5-pro",
        temperature=0,
        max_tokens=None,
        timeout=None,
        max_retries=2,
        google_api_key=api_key
    )

    parser = StrOutputParser()
    history_chain = history_prompt | llm | parser

    try:
        summary = history_chain.invoke(history_inputs)
        logger.debug(f"Generated Medical History Summary: {summary}")
        return summary
    except Exception as e:
        logger.error(f"Error generating medical history: {str(e)}")
        raise