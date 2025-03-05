from langchain_core.prompts import PromptTemplate
from langchain.chains import LLMChain
from langchain.schema import BaseOutputParser
from langchain_google_genai import ChatGoogleGenerativeAI
import pdfplumber
import logging
import re

# Configure logging
logging.basicConfig(level=logging.DEBUG, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class SimpleOutputParser(BaseOutputParser):
    """A simple output parser that strips surrounding whitespace."""
    def parse(self, text: str) -> str:
        return text.strip()
    
    @property
    def _type(self) -> str:
        return "simple"

def extract_text_from_pdf(pdf_path):
    with pdfplumber.open(pdf_path) as pdf:
        text = ""
        for page in pdf.pages:
            text += page.extract_text() or ""
    return text

def process_text_with_gemini(extracted_text: str, category: str = None, language: str = "kannada", patient_name: str = "ರೋಗಿ", existing_text: str = None) -> dict:
    """
    Process English medical text using Gemini to generate both a regional summary (Kannada/Tamil) and a professional English summary.
    Returns a dict with both summaries.
    """
    api_key_path = './Cred/Gemini_Key/key.txt'
    with open(api_key_path, 'r') as file:
        api_key = file.read().strip()

    # Filter non-English characters and non-medical data
    cleaned_text = re.sub(r'[^\x00-\x7F]+', ' ', extracted_text)
    cleaned_text = re.sub(r'\b(?:address|phone|contact|id|date|time|location)\b.*?(?=\n|$)', '', cleaned_text, flags=re.IGNORECASE)

    # Sanitize cleaned_text and existing_text to remove unmatched curly braces
    cleaned_text = re.sub(r'\{(?![^{}]*\})', '', cleaned_text)  # Remove { without matching }
    cleaned_text = re.sub(r'(?<!\{)[^{}]*\}(?![^{}]*\})', '', cleaned_text)  # Remove } without matching {
    if existing_text:
        existing_text = re.sub(r'\{(?![^{}]*\})', '', existing_text)
        existing_text = re.sub(r'(?<!\{)[^{}]*\}(?![^{}]*\})', '', existing_text)

    # Remove patient's name and age from cleaned_text for professional summary
    professional_cleaned_text = cleaned_text
    if patient_name:
        professional_cleaned_text = re.sub(re.escape(patient_name), '[Patient]', professional_cleaned_text, flags=re.IGNORECASE)
    # Remove age patterns (e.g., "25 years", "30 yrs", "age 45")
    professional_cleaned_text = re.sub(r'\b(age|aged)?\s*\d+\s*(years|yrs|year|old)?\b', '[Age]', professional_cleaned_text, flags=re.IGNORECASE)

    # Define language-specific prompts
    language_prompts = {
        "kannada": "in simple Kannada (ಕನ್ನಡ), using warm and clear language for the patient, avoiding medical jargon and English text",
        "tamil": "in simple Tamil (தமிழ்), using warm and clear language for the patient, avoiding medical jargon and English text"
    }
    regional_language = language.lower()
    if regional_language not in language_prompts:
        regional_language = "kannada"  # Default to Kannada if unsupported

    # Regional summary prompt
    if category == 'prescriptions' and existing_text:
        regional_template = (
            f"You are Gemini, a helpful language model. Analyze the following English prescription text and create a brief but detailed, "
            f"point-wise summary {language_prompts[regional_language]}. Personalize it with the patient's name '{{patient_name}}'. Focus only on medical "
            f"information (e.g., medicines, dosages). Include the patient's condition, recommended actions or tests, and medicines with dosages. "
            f"Existing text:\n{existing_text}\n\nNew text:\n{cleaned_text}\n\nOutput only the summary points."
        )
    else:
        regional_template = (
            f"You are Gemini, a helpful language model. Analyze the following English medical text and create a brief but detailed, "
            f"point-wise summary {language_prompts[regional_language]}. Personalize it with the patient's name '{{patient_name}}'. Focus only on medical "
            f"information. Include the patient's condition and recommended actions or follow-ups. "
            f"Text:\n{cleaned_text}\n\nOutput only the summary points."
        )

    # Professional English summary prompt (using anonymized text, ensuring condition is included)
    professional_template = (
        f"You are Gemini, a helpful language model. Analyze the following English medical text and create a concise, professional medical summary in English "
        f"for a doctor. Start with the patient's condition as the first point, followed by key findings, prescribed medications with dosages (if applicable), "
        f"and recommended follow-ups or tests. Do not include the patient's name or age. Text:\n{professional_cleaned_text}\n\nOutput only the summary points."
    )

    llm = ChatGoogleGenerativeAI(
        model="gemini-1.5-pro",
        temperature=0,
        max_tokens=None,
        timeout=None,
        max_retries=2,
        google_api_key=api_key
    )

    # Regional chain
    regional_prompt = PromptTemplate(input_variables=["patient_name", "existing_text", "new_text"] if (category == 'prescriptions' and existing_text) else ["patient_name", "text"], template=regional_template)
    regional_chain = LLMChain(prompt=regional_prompt, llm=llm)
    
    # Professional chain
    professional_prompt = PromptTemplate(input_variables=["text"], template=professional_template)
    professional_chain = LLMChain(prompt=professional_prompt, llm=llm)

    parser = SimpleOutputParser()
    try:
        # Generate regional summary
        if category == 'prescriptions' and existing_text:
            regional_output = regional_chain.run(patient_name=patient_name, existing_text=existing_text, new_text=cleaned_text)
        else:
            regional_output = regional_chain.run(patient_name=patient_name, text=cleaned_text)
        
        # Generate professional summary
        professional_output = professional_chain.run(text=professional_cleaned_text)

        regional_summary = parser.parse(regional_output)
        professional_summary = parser.parse(professional_output)
        
        logger.debug(f"Regional Summary ({regional_language}): {regional_summary}")
        logger.debug(f"Professional Summary (English): {professional_summary}")
        
        return {
            "regional_summary": regional_summary,
            "professional_summary": professional_summary
        }
    except Exception as e:
        logger.error(f"Error processing text with Gemini: {e}")
        raise