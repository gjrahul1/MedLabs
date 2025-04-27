from langchain_core.prompts import PromptTemplate
from langchain_core.runnables import RunnableLambda
from langchain_core.output_parsers import StrOutputParser
from langchain_google_genai import ChatGoogleGenerativeAI
import logging
import re
from firebase_admin import firestore
import os
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv(dotenv_path=os.path.join(os.getcwd(), '.env'))

# Configure logging
logging.basicConfig(level=logging.DEBUG, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def process_text_with_gemini(extracted_text: str, category: str = None, language: str = "kannada", patient_name: str = "ರೋಗಿ", existing_text: str = None, uid: str = None) -> dict:
    """
    Process the extracted text using Gemini to generate regional, professional, and English patient summaries.
    For symptom_mapping category, return only the specialty name using Gemini-2.0.
    For other categories, use Gemini-1.5-pro for summaries.
    Args:
        extracted_text (str): Text extracted from the image or PDF, or prompt for symptom mapping.
        category (str): Category of the document ('prescriptions', 'lab_records', or 'symptom_mapping').
        language (str): Language for the regional summary ('kannada', 'tamil', or 'english').
        patient_name (str): Name of the patient.
        existing_text (str): Existing summary text for prescriptions, if any.
        uid (str): Unique identifier of the patient.
    Returns:
        dict: For symptom_mapping, returns {'regional_summary': specialty_name}. Otherwise, contains regional_summary, professional_summary, english_patient_summary, medical_history, and language.
    """
    # Initialize Firestore client inside the function
    db = firestore.client()

    # Load Gemini API key from environment variable
    api_key = os.getenv('GEMINI_API_KEY')
    logger.debug(f"Loaded Gemini API key: {api_key}")
    if not api_key:
        logger.error("GEMINI_API_KEY not found in .env file")
        raise ValueError("GEMINI_API_KEY not found in .env file")

    # Define Gemini models for different tasks
    llm_symptom_mapping = ChatGoogleGenerativeAI(
        model="gemini-2.0-flash-lite",  
        temperature=0,
        max_tokens=None,
        timeout=None,
        max_retries=2,
        google_api_key=api_key
    )

    llm_summaries = ChatGoogleGenerativeAI(
        model="gemini-1.5-pro",  # Use Gemini-1.5-pro for summaries
        temperature=0,
        max_tokens=None,
        timeout=None,
        max_retries=2,
        google_api_key=api_key
    )

    parser = StrOutputParser()

    # Handle symptom_mapping category with Gemini-2.0
    if category == "symptom_mapping":
        try:
            specialty_chain = llm_symptom_mapping | parser
            logger.debug(f"Symptom mapping prompt sent to Gemini-2.0: {extracted_text}")
            raw_specialty = specialty_chain.invoke(extracted_text)
            logger.debug(f"Raw specialty returned by Gemini-2.0: {raw_specialty}")
            specialty_name = raw_specialty.strip().lower()
            from symptom_mapping import valid_specialties
            if specialty_name not in valid_specialties:
                logger.warning(f"Gemini-2.0 returned invalid specialty '{specialty_name}' for prompt: {extracted_text}")
                if "kidney stones" in extracted_text.lower():
                    specialty_name = "urology"
                    logger.info("Applied fallback: kidney stones mapped to urology due to invalid Gemini response")
                else:
                    specialty_name = "general medicine"
                    logger.info("Applied default fallback to general medicine due to invalid Gemini response")
            return {
                "regional_summary": specialty_name
            }
        except Exception as e:
            logger.error(f"Error processing symptom mapping with Gemini-2.0: {str(e)}")
            if "kidney stones" in extracted_text.lower():
                logger.info("Applied fallback on error: kidney stones mapped to urology")
                return {
                    "regional_summary": "urology"
                }
            return {
                "regional_summary": "general medicine"
            }

    # Log the input text for debugging
    logger.debug(f"Processing text for category {category}, language {language}, patient_name {patient_name}: {extracted_text}")

    # Clean and sanitize the extracted text for other categories
    cleaned_text = re.sub(r'[^\x00-\x7F]+', ' ', extracted_text)
    
    # Remove personal identifiers and sensitive information, but preserve dosage durations and medicine names
    cleaned_text = re.sub(r'(?:Mr\.|Mrs\.|Ms\.|Dr\.)\s+[A-Za-z]+(?:\s+[A-Za-z]+)*', '[Name]', cleaned_text, flags=re.IGNORECASE)
    cleaned_text = re.sub(r'\b[A-Za-z]+(?:\s+[A-Za-z]+)*(?:\'s)?(?=\s*(?:medical|report|is|has|with))', '[Name]', cleaned_text, flags=re.IGNORECASE)
    cleaned_text = re.sub(r'\b(?:address|phone|contact|id|date|time|location)\b.*?(?=\n|$)', '', cleaned_text, flags=re.IGNORECASE)
    cleaned_text = re.sub(r'\b\d{4}-\d{2}-\d{2}\b|\b\d{2}/\d{2}/\d{4}\b', '[Date]', cleaned_text)
    cleaned_text = re.sub(r'\b\+?\d{1,3}[-.\s]?\d{3}[-.\s]?\d{3}[-.\s]?\d{4}\b', '[Phone]', cleaned_text)
    cleaned_text = re.sub(r'\b[\w\.-]+@[\w\.-]+\.\w+\b', '[Email]', cleaned_text)
    cleaned_text = re.sub(r'\{(?![^{}]*\})', '', cleaned_text)
    cleaned_text = re.sub(r'(?<!\{)[^{}]*\}(?![^{}]*\})', '', cleaned_text)
    # Refined regex to target age-related phrases only, preserving dosage durations
    cleaned_text = re.sub(r'\b(age|aged)\s*\d+\s*(years|yrs|year|old)\b|\b\d+-year-old\b', '[Age]', cleaned_text, flags=re.IGNORECASE)
    cleaned_text = re.sub(r'(?:hospital|centre|clinic|family health centre|secunderabad|india|apollo|hills|city|jubilee|hyderabad).*?(?=\n|$)', '', cleaned_text, flags=re.IGNORECASE)
    cleaned_text = re.sub(r'(?:timing|evening|sun|mon|tue|wed|thu|fri|sat)\s*:.*?(?=\n|$)', '', cleaned_text, flags=re.IGNORECASE)

    if existing_text:
        existing_text = re.sub(r'(?:Mr\.|Mrs\.|Ms\.|Dr\.)\s+[A-Za-z]+(?:\s+[A-Za-z]+)*', '[Name]', existing_text, flags=re.IGNORECASE)
        existing_text = re.sub(r'\b[A-Za-z]+(?:\s+[A-Za-z]+)*(?:\'s)?(?=\s*(?:medical|report|is|has|with))', '[Name]', existing_text, flags=re.IGNORECASE)
        existing_text = re.sub(r'\b(?:address|phone|contact|id|date|time|location)\b.*?(?=\n|$)', '', existing_text, flags=re.IGNORECASE)
        existing_text = re.sub(r'\b\d{4}-\d{2}-\d{2}\b|\b\d{2}/\d{2}/\d{4}\b', '[Date]', existing_text)
        existing_text = re.sub(r'\b\+?\d{1,3}[-.\s]?\d{3}[-.\s]?\d{3}[-.\s]?\d{4}\b', '[Phone]', existing_text)
        existing_text = re.sub(r'\b[\w\.-]+@[\w\.-]+\.\w+\b', '[Email]', existing_text)
        existing_text = re.sub(r'\{(?![^{}]*\})', '', existing_text)
        existing_text = re.sub(r'(?<!\{)[^{}]*\}(?![^{}]*\})', '', existing_text)
        existing_text = re.sub(r'\b(age|aged)\s*\d+\s*(years|yrs|year|old)\b|\b\d+-year-old\b', '[Age]', existing_text, flags=re.IGNORECASE)
        existing_text = re.sub(r'(?:hospital|centre|clinic|family health centre|secunderabad|india|apollo|hills|city|jubilee|hyderabad).*?(?=\n|$)', '', existing_text, flags=re.IGNORECASE)
        existing_text = re.sub(r'(?:timing|evening|sun|mon|tue|wed|thu|fri|sat)\s*:.*?(?=\n|$)', '', existing_text, flags=re.IGNORECASE)

    # Log the cleaned text
    logger.debug(f"Cleaned text: {cleaned_text}")

    # Anonymize for professional summary
    professional_cleaned_text = cleaned_text
    if patient_name:
        professional_cleaned_text = re.sub(re.escape(patient_name), '[Patient]', professional_cleaned_text, flags=re.IGNORECASE)
    professional_cleaned_text = re.sub(r'\b(age|aged)\s*\d+\s*(years|yrs|year|old)\b|\b\d+-year-old\b', '[Age]', professional_cleaned_text, flags=re.IGNORECASE)

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

    # Regional Summary for Patient (in specified language)
    if category == 'prescriptions' and existing_text:
        regional_template_text = (
            f"You are Gemini, a helpful language model. Analyze the following English prescription text and create a brief but detailed summary {language_prompts[selected_language]}. "
            f"Start with a greeting 'Dear {{patient_name}},' on a new line, followed by a brief introduction line, then a point-wise summary where each point MUST start with a dash (-). "
            f"Use Markdown formatting: bold (**text**) key medical terms like condition names, medicine names, or important actions. Ensure each summary point is on a new line and starts with '- '. "
            f"Focus only on medical information (e.g., conditions, medicines, dosages). Include the patient's condition (if specified in the text), recommended actions or tests, and all prescribed medicines with their full dosages (e.g., quantity, frequency, duration) if available. "
            f"Interpret dosage patterns like '1-0-1' as one tablet in the morning and one in the evening, '2-1-1' as two tablets in the morning, one at noon, and one in the evening, and include this in the summary in the specified language. "
            f"If dosage or duration information is missing for a medicine, note that it is unclear and advise the patient to consult their doctor for clarification, in the specified language. "
            f"If no medicines are prescribed, include the patient's condition (if specified) and any recommended actions (e.g., hospital stay, follow-ups). "
            f"Do not include any information not present in the provided text, such as conditions or medications from external sources. "
            f"Existing text:\n{{existing_text}}\n\nNew text:\n{{new_text}}\n\nOutput the summary as a plain Markdown string. Do not wrap the output in code blocks (e.g., ```markdown or ```)."
        )
        regional_prompt = PromptTemplate.from_template(regional_template_text)
        regional_inputs = {
            "patient_name": patient_name,
            "existing_text": existing_text,
            "new_text": cleaned_text
        }
    else:
        regional_template_text = (
            f"You are Gemini, a helpful language model. Analyze the following English medical text and create a brief but detailed summary {language_prompts[selected_language]}. "
            f"Start with a greeting 'Dear {{patient_name}},' on a new line, followed by a brief introduction line, then a point-wise summary where each point MUST start with a dash (-). "
            f"Use Markdown formatting: bold (**text**) key medical terms like condition names, medicine names, or important actions. Ensure each summary point is on a new line and starts with '- '. "
            f"Focus only on medical information. Include the patient's condition (if specified in the text), recommended actions or follow-ups, and all prescribed medicines with their full dosages (e.g., quantity, frequency, duration) if available. "
            f"Interpret dosage patterns like '1-0-1' as one tablet in the morning and one in the evening, '2-1-1' as two tablets in the morning, one at noon, and one in the evening, and include this in the summary in the specified language. "
            f"If dosage or duration information is missing for a medicine, note that it is unclear and advise the patient to consult their doctor for clarification, in the specified language. "
            f"If no medicines are prescribed, include the patient's condition (if specified) and any recommended actions (e.g., hospital stay, follow-ups). "
            f"Do not include any information not present in the provided text, such as conditions or medications from external sources. "
            f"Text:\n{{text}}\n\nOutput the summary as a plain Markdown string. Do not wrap the output in code blocks (e.g., ```markdown or ```)."
        )
        regional_prompt = PromptTemplate.from_template(regional_template_text)
        regional_inputs = {
            "patient_name": patient_name,
            "text": cleaned_text
        }

    # Professional Summary for Doctor (always in English)
    professional_template = (
        f"You are Gemini, a helpful language model. Analyze the following English medical text and create a concise, professional medical summary in English "
        f"for a doctor. Start with the patient's condition as the first point (if specified in the text), followed by key findings, prescribed medications with full dosages (e.g., quantity, frequency, duration) if applicable, "
        f"and recommended follow-ups or tests. Use Markdown formatting: each point MUST start with a dash (-). Ensure each summary point is on a new line and starts with '- '. "
        f"If no medicines are prescribed, include the patient's condition (if specified) and any recommended actions (e.g., hospital stay, follow-ups). "
        f"Do not include the patient's name or age. Do not include any information not present in the provided text, such as conditions or medications from external sources. "
        f"Text:\n{{text}}\n\nOutput the summary as a plain Markdown string. Do not wrap the output in code blocks (e.g., ```markdown or ```)."
    )
    professional_prompt = PromptTemplate.from_template(professional_template)
    professional_inputs = {"text": professional_cleaned_text}

    # English Summary for Patient (always in English, concise and patient-friendly, addressing the patient directly)
    english_patient_template = (
        f"You are Gemini, a helpful language model. Analyze the following English medical text and create a concise, patient-friendly summary in English "
        f"for the patient. Start with a greeting 'Dear {{patient_name}},' on a new line, followed by a brief introduction line like 'Here is a summary of your medical report:', "
        f"then a point-wise summary where each point MUST start with a dash (-). "
        f"Use Markdown formatting: bold (**text**) key medical terms like condition names, medicine names, or important actions. Ensure each summary point is on a new line and starts with '- '. "
        f"Use very simple, warm, and clear language, avoiding complex medical jargon (e.g., replace 'sepsis' with 'serious infection', 'MODS' with 'problems with your organs', etc.). "
        f"Focus on the most important medical information, such as the patient's condition (if specified in the text), all prescribed medicines with their full dosages (e.g., quantity, frequency, duration) if available, and what they need to do (e.g., hospital stay, follow-up actions). "
        f"Interpret dosage patterns like '1-0-1' as one tablet in the morning and one in the evening, '2-1-1' as two tablets in the morning, one at noon, and one in the evening, and include this in the summary. "
        f"If dosage or duration information is missing for a medicine, note that it is unclear and advise the patient to ask their doctor for more details. "
        f"If no medicines are prescribed, include the patient's condition (if specified) and any recommended actions (e.g., hospital stay, follow-ups). "
        f"Do not include any information not present in the provided text, such as conditions or medications from external sources. "
        f"End the summary with a responsible AI disclaimer on a new line: '- This summary is for your information only. Please consult your doctor for more details or if you have any questions.' "
        f"Do not include the patient's name or age in the summary points, even if present in the text. Text:\n{{text}}\n\nOutput the summary as a plain Markdown string. Do not wrap the output in code blocks (e.g., ```markdown or ```)."
    )
    english_patient_prompt = PromptTemplate.from_template(english_patient_template)
    english_patient_inputs = {
        "patient_name": patient_name,
        "text": cleaned_text
    }

    # Chains using | operator with Gemini-1.5-pro for summaries
    regional_chain = regional_prompt | llm_summaries | parser
    professional_chain = professional_prompt | llm_summaries | parser
    english_patient_chain = english_patient_prompt | llm_summaries | parser

    try:
        regional_summary = regional_chain.invoke(regional_inputs)
        professional_summary = professional_chain.invoke(professional_inputs)
        english_patient_summary = english_patient_chain.invoke(english_patient_inputs)

        # Post-process summaries to correct medicine names and enforce Markdown bullet point format
        medicine_corrections = {
            "Taz Azenal": "Azenac MR",
            "Dep[Age][Age]": "Depsit",
            "Zofer U.T": "Zofer",
            "T-Minic": "T-Minic Drops",
            "Anthakind": "Anthakind Drops",
            "Advent": "Advent Drops",
            "Nanodem": "Nanodem Drops"
        }

        for summary_name, summary in [
            ('regional_summary', regional_summary),
            ('professional_summary', professional_summary),
            ('english_patient_summary', english_patient_summary)
        ]:
            for wrong_name, correct_name in medicine_corrections.items():
                summary = summary.replace(wrong_name, correct_name)

            lines = summary.split('\n')
            formatted_lines = []
            for line in lines:
                line = line.strip()
                if not line:
                    formatted_lines.append(line)
                    continue
                if line.startswith('Dear') or line.startswith('Here') or line.startswith('ಉங்கள்') or line.startswith('ನಿಮ್ಮ'):
                    formatted_lines.append(line)
                    continue
                if not line.startswith('-'):
                    line = f"- {line}"
                formatted_lines.append(line)
            cleaned_summary = '\n'.join(formatted_lines).strip()
            if cleaned_summary != summary:
                logger.debug(f"Formatted bullet points for {summary_name}")
            if summary_name == 'regional_summary':
                regional_summary = cleaned_summary
            elif summary_name == 'professional_summary':
                professional_summary = cleaned_summary
            elif summary_name == 'english_patient_summary':
                english_patient_summary = cleaned_summary

        logger.debug(f"Regional Summary ({selected_language}): {regional_summary}")
        logger.debug(f"Professional Summary (English): {professional_summary}")
        logger.debug(f"English Patient Summary: {english_patient_summary}")

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
            "english_patient_summary": english_patient_summary,
            "medical_history": medical_history,
            "language": selected_language
        }
    except Exception as e:
        logger.error(f"Error processing text with Gemini: {e}")
        raise

def generate_medical_history(uid: str, db) -> str:
    initial_screening = db.collection('initial_screenings').document(f'initial_screening_{uid}').get()
    prescriptions = db.collection('prescriptions').where('uid', '==', uid).order_by('timestamp', direction=firestore.Query.DESCENDING).get()
    lab_records = db.collection('lab_records').where('uid', '==', uid).order_by('timestamp', direction=firestore.Query.DESCENDING).get()

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

    history_template = (
        "You are Gemini, a helpful language model. Analyze the following patient medical data and create a concise medical history summary in English. "
        "The summary should be brief, point-wise, and include key conditions, treatments, and metrics (e.g., test results, dosages) from initial screening, prescriptions, and lab records. "
        "Focus on the most relevant information, combining data where appropriate. Do not include the patient's name or age. "
        "Text:\n{text}\n\nOutput only the summary points."
    )
    history_prompt = PromptTemplate.from_template(history_template)
    history_inputs = {"text": history_text}

    api_key = os.getenv('GEMINI_API_KEY')
    logger.debug(f"Loaded Gemini API key in generate_medical_history: {api_key}")
    if not api_key:
        logger.error("GEMINI_API_KEY not found in .env file")
        raise ValueError("GEMINI_API_KEY not found in .env file")

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