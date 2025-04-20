from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_google_genai import ChatGoogleGenerativeAI
import logging
import os
from dotenv import load_dotenv
import time
import random

# Load environment variables from .env file
load_dotenv(dotenv_path=os.path.join(os.getcwd(), '.env'))

# Configure logging
logging.basicConfig(level=logging.DEBUG, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Fallback poems in case Gemini API fails or repeats
FALLBACK_POEMS = [
    "Blank space resonates\n404 hums creation\nNew worlds now beckon",
    "Zero result found\nImagination triggers\nInspiration dawns",
    "Page drifts into mist\nA hush within the network\nLet ideas soar",
    "Missing code pathway\nInvitation in blank time\nBold mind forges on",
    "This link has vanished\nLet your mind wander freely\nSeek new terrain now"
]

# In-memory cache to track recently generated poems
recent_poems = []
MAX_RECENT_POEMS = 5  # Maximum number of recent poems to track

def generate_error_poem(error_type: str) -> str:
    """
    Generate a poetic error message using Gemini and LangChain.
    Args:
        error_type (str): The type of error (e.g., "token_missing", "authentication", "404").
    Returns:
        str: A haiku-style poem (3 lines, 5-7-5 syllables).
    """
    global recent_poems

    # Load Gemini API key from environment variable
    api_key = os.getenv('GEMINI_API_KEY')
    logger.debug(f"Loaded Gemini API key: {api_key}")
    if not api_key:
        logger.error("GEMINI_API_KEY not found in .env file, using fallback poem")
        return select_fallback_poem()

    # Define the prompt for generating a haiku-style error poem
    # Add a random seed (timestamp) to encourage variation
    seed = str(time.time()) + str(random.randint(1, 1000))
    error_prompt_template = PromptTemplate.from_template(
        "You are Gemini, a poetic AI. Create a unique haiku-style error message (3 lines, 5-7-5 syllables) in English for the following error type: {error_type}. "
        "The poem should be reflective, imaginative, and inspire creativity, avoiding technical jargon. Use this seed to ensure uniqueness: {seed}. "
        "Example for 404 error: 'Blank space resonates\n404 hums creation\nNew worlds now beckon'\n\n"
        "Output only the poem, with each line separated by a newline character (\n)."
    )

    # Gemini model with higher temperature for more randomness
    llm = ChatGoogleGenerativeAI(
        model="gemini-1.5-pro",
        temperature=1.0,  # Increased for more creative variation
        max_tokens=None,
        timeout=None,
        max_retries=2,
        google_api_key=api_key
    )

    parser = StrOutputParser()
    error_chain = error_prompt_template | llm | parser

    try:
        poem = error_chain.invoke({"error_type": error_type, "seed": seed})
        logger.debug(f"Generated error poem for {error_type}: {poem}")

        # Check if the poem has already been generated recently
        if poem in recent_poems:
            logger.debug(f"Poem already generated recently, selecting fallback: {poem}")
            return select_fallback_poem()
        
        # Add the poem to recent poems
        recent_poems.append(poem)
        if len(recent_poems) > MAX_RECENT_POEMS:
            recent_poems.pop(0)  # Remove the oldest poem
        
        return poem
    except Exception as e:
        logger.error(f"Error generating poem for {error_type}: {str(e)}, using fallback poem")
        return select_fallback_poem()

def select_fallback_poem() -> str:
    """
    Select a fallback poem that hasn't been used recently.
    Returns:
        str: A haiku-style poem from the fallback list.
    """
    global recent_poems
    available_poems = [poem for poem in FALLBACK_POEMS if poem not in recent_poems]
    if not available_poems:
        # If all fallback poems have been used recently, reset and use the first one
        recent_poems.clear()
        selected_poem = FALLBACK_POEMS[0]
    else:
        selected_poem = random.choice(available_poems)
    
    recent_poems.append(selected_poem)
    if len(recent_poems) > MAX_RECENT_POEMS:
        recent_poems.pop(0)
    
    return selected_poem