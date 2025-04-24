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
from flask import Flask, render_template, request, jsonify, session, redirect,url_for, send_from_directory
from func_timeout import func_timeout, FunctionTimedOut
import firebase_admin
from firebase_admin import credentials, auth, firestore, storage
from openai import OpenAI
from gtts import gTTS
import httpx
from langchain_openai import ChatOpenAI
from langchain.prompts import PromptTemplate

# Import the determine_specialty function from symptom_mapping.py
from symptom_mapping import determine_specialty, clean_symptom, symptom_specialty_map

# Setup logging
logging.basicConfig(level=logging.DEBUG, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Load environment variables
load_dotenv(dotenv_path=os.path.join(os.getcwd(), '.env'))
openai_api_key = os.getenv("OPENAI_API_KEY")
if not openai_api_key:
    logger.error("OPENAI_API_KEY not found in .env")
    raise ValueError("OPENAI_API_KEY not found in .env file")

# Validate OpenAI API key
try:
    test_client = OpenAI(api_key=openai_api_key)
    test_client.models.list()
    logger.info("✅ OpenAI API key validated successfully")
except Exception as e:
    logger.error(f"Invalid OpenAI API key: {str(e)}")
    raise ValueError(f"Invalid OpenAI API key: {str(e)}")

client = OpenAI(api_key=openai_api_key)

# Initialize Flask app
app = Flask(__name__)
app.secret_key = os.environ.get("FLASK_SECRET_KEY", secrets.token_hex(32))

# Initialize Firebase
cred_path = './Cred/Firebase/Med App/med-labs-42f13-firebase-adminsdk-fbsvc-43b78dfef7.json'
if not firebase_admin._apps:
    cred = credentials.Certificate(cred_path)
    firebase_admin.initialize_app(cred, {
        'storageBucket': 'med-labs-42f13.appspot.com'
    })
    logger.info("✅ Firebase initialized successfully with bucket: med-labs-42f13. SDK Version: %s", firebase_admin.__version__)

db = firestore.client()
bucket = storage.bucket()

# Validate bucket access
try:
    bucket.get_blob('test-check')
    logger.info("Bucket exists and is accessible")
except Exception as e:
    logger.error(f"Bucket validation failed: {str(e)}. Please ensure the bucket 'med-labs-42f13' exists.")
    raise

# LLM Setup for Conversation, JSON Parsing, Symptom Extraction, and Doctor Mapping Validation
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
        "- Interpret symptoms accurately and map them to the appropriate specialty based on severity, duration, and additional symptoms:\n"
        "  - 'sadness' or 'heavy sadness' as 'depression' (Psychiatry).\n"
        "  - 'anxiety' or 'heart racing' as 'anxiety' (Psychiatry).\n"
        "  - 'focus issues' or 'feeling stuck' as 'difficulty concentrating' (Psychiatry).\n"
        "  - 'irritability' as 'irritability' (Psychiatry).\n"
        "  - 'sleep problems' or 'trouble sleeping' as 'sleep issues' (Psychiatry).\n"
        "  - 'weight loss' as 'unexplained weight loss' if unintentional (Oncology).\n"
        "  - 'lump' or 'lumps' as 'lump' (Oncology).\n"
        "  - 'night sweats' as 'night sweats' (Oncology).\n"
        "  - 'tight band sensation around head' as 'headache'.\n"
        "  - 'hand shaking' or 'tremors' as 'tremors' (Neurology).\n"
        "  - 'numbness' or 'tingling' as 'numbness and tingling in hands' (Neurology).\n"
        "  - 'pins and needles' as 'pins and needles' (Neurology).\n"
        "  - 'balance problems' or 'off balance' as 'balance issues' (Neurology).\n"
        "  - 'child fever' or 'child cough' as is (Pediatrics).\n"
        "  - 'irregular periods', 'spotting', or 'heavy periods' as 'menstrual issues' (Gynecology).\n"
        "  - For 'headache':\n"
        "    - If severity is 'mild' or 'moderate', duration is less than or equal to 3 months, and there are no additional neurological symptoms (e.g., 'tremors', 'vision changes', 'seizures', 'numbness and tingling in hands', 'pins and needles', 'balance issues'), assign to General Medicine.\n"
        "    - Otherwise, assign to Neurology.\n"
        "  - For 'fever' or 'child fever':\n"
        "    - If severity is 'moderate' or 'severe' and additional symptoms include 'chest pain', 'shortness of breath', or 'palpitations', assign to Cardiology.\n"
        "    - If the symptom is 'child fever', assign to Pediatrics.\n"
        "    - Otherwise, assign to General Medicine.\n"
        "  - For 'fatigue' or 'exhaustion':\n"
        "    - If severity is 'moderate' or 'severe' and duration includes 'month' (indicating chronic) with additional symptoms like 'unexplained weight loss', 'lump', or 'night sweats', assign to Oncology.\n"
        "    - If additional symptoms include 'chest pain', 'shortness of breath', or 'palpitations', assign to Cardiology.\n"
        "    - Otherwise, assign to General Medicine.\n"
        "  - For 'chest pain':\n"
        "    - If severity is 'moderate' or 'severe' or additional symptoms include 'shortness of breath' or 'palpitations', assign to Cardiology.\n"
        "    - Otherwise, assign to General Medicine.\n"
        "  - For 'rash', 'itchiness', 'skin irritation', 'acne', or 'hives', assign to Dermatology.\n"
        "  - For 'sore throat', 'ear pain', 'nasal congestion', or 'dry cough', assign to ENT.\n"
        "  - For 'swelling', 'swollen ankles', or 'edema', assign to Cardiology.\n"
        "  - For 'urination issues' or 'kidney pain', assign to Nephrology.\n"
        "  - For 'kidney stones', 'urinary tract infection', 'lower back pain', or 'difficulty urinating', assign to Urology.\n"
        "  - For 'weight changes' or 'thirst', assign to Endocrinology.\n"
        "  - For 'stomach pain', 'nausea', or 'diarrhea', assign to Gastroenterology.\n"
        "  - For 'abdominal pain', 'injury', 'swelling (surgical)', or 'lumps in neck', assign to General Surgery.\n"
        "  - For 'menstrual issues' or 'pelvic pain', assign to Gynecology.\n"
        "  - For 'child cough' or 'child rash', assign to Pediatrics.\n"
        "  - For 'shortness of breath' or 'palpitations', assign to Cardiology.\n"
        "  - For 'dizziness', 'seizures', 'numbness and tingling in hands', 'pins and needles', or 'balance issues', assign to Neurology.\n"
        "  - For 'joint pain', 'muscle pain', or 'fracture', assign to Orthopedics.\n"
        "  - For 'general weakness' or 'body aches', assign to General Medicine.\n"
        "  - If the symptom is 'unknown' or cannot be mapped, assign to General Medicine.\n"
        "- For state INITIAL: Extract symptoms from input, update medical_data['symptoms'], ask for severity ('How severe is your {{symptoms}}? Is it mild, moderate, or severe?'), set next_state to SEVERITY.\n"
        "- For state SEVERITY: Extract severity (mild/moderate/severe). If the input contains a numerical value (e.g., '7'), map it as follows: 1-3 as 'mild', 4-6 as 'moderate', 7-10 as 'severe', then ask for confirmation ('You said your {{symptoms}} severity is {{input}}, which I interpret as {{severity}}. Is that correct?'), set next_state to CONFIRM_SEVERITY. Otherwise, update medical_data['severity'], ask for duration ('How long have you been experiencing {{symptoms}}?'), set next_state to DURATION.\n"
        "- For state CONFIRM_SEVERITY: If input confirms (e.g., 'yes', 'correct'), update medical_data['severity'] with the interpreted value, ask for duration ('How long have you been experiencing {{symptoms}}?'), set next_state to DURATION. If input denies (e.g., 'no', 'incorrect'), ask for severity again ('Please specify the severity of your {{symptoms}}. Is it mild, moderate, or severe?'), set next_state to SEVERITY. If unclear, ask for clarification ('Could you please confirm if this is correct?'), keep next_state as CONFIRM_SEVERITY.\n"
        "- For state DURATION: Extract duration, update medical_data['duration'], ask for triggers ('What triggers your {{symptoms}}, if anything? Say \"unknown\" if unsure.'), set next_state to TRIGGERS.\n"
        "- For state TRIGGERS: Extract triggers, update medical_data['triggers'], ask to assign doctor ('May I recommend a doctor for you?'), set next_state to CONFIRM. Treat 'random situations', 'I'm not sure', 'I don't know', 'It's unknown', or similar phrases as 'unknown' for triggers.\n"
        "- For state CONFIRM: If input confirms (e.g., 'yes', 'please'), set assign_doctor to true, set next_state to COMPLETE; otherwise, ask again.\n"
        "- If input is unclear at any state, ask for clarification (e.g., 'Could you please repeat or clarify your symptoms?') and do not advance state.\n"
        "- Severity should be 'mild', 'moderate', or 'severe'. If ambiguous (e.g., 'mild to moderate'), use the higher value ('moderate').\n"
        "- Ensure 'medical_data' is a JSON object with lists: 'symptoms', 'severity', 'duration', 'triggers'.\n"
        "- Example outputs:\n"
        "  - If state is INITIAL and input is 'I have a headache', update medical_data['symptoms'] to ['headache'], return: {{\"response\": \"How severe is your headache? Is it mild, moderate, or severe?\", \"medical_data\": {{\"symptoms\": [\"headache\"], \"severity\": [\"\"], \"duration\": [\"\"], \"triggers\": [\"unknown\"]}}, \"next_state\": \"SEVERITY\", \"assign_doctor\": false}}\n"
        "  - If state is SEVERITY and input is '7', update medical_data['severity'] to ['severe'], return: {{\"response\": \"You said your headache severity is 7, which I interpret as severe. Is that correct?\", \"medical_data\": {{\"symptoms\": [\"headache\"], \"severity\": [\"severe\"], \"duration\": [\"\"], \"triggers\": [\"unknown\"]}}, \"next_state\": \"CONFIRM_SEVERITY\", \"assign_doctor\": false}}\n"
        "  - If state is CONFIRM_SEVERITY and input is 'yes', return: {{\"response\": \"How long have you been experiencing headache?\", \"medical_data\": {{\"symptoms\": [\"headache\"], \"severity\": [\"severe\"], \"duration\": [\"\"], \"triggers\": [\"unknown\"]}}, \"next_state\": \"DURATION\", \"assign_doctor\": false}}\n"
        "- Return a valid JSON object (ensure proper formatting) with the following keys:\n"
        "  - \"response\": [Text response with one question or confirmation]\n"
        "  - \"medical_data\": [JSON object with 'symptoms', 'severity', 'duration', 'triggers' as lists]\n"
        "  - \"next_state\": [INITIAL, SEVERITY, CONFIRM_SEVERITY, DURATION, TRIGGERS, CONFIRM, or COMPLETE]\n"
        "  - \"assign_doctor\": [true/false]\n"
        "Ensure the output is a properly formatted JSON string."
    )
)


json_prompt_template = PromptTemplate(
    input_variables=["conversation_output"],
    template=(
        "You are a JSON formatting assistant. Parse the conversation output into a JSON object.\n\n"
        "**Conversation Output:**\n{conversation_output}\n\n"
        "**Instructions:**\n"
        "- Parse the output into a JSON object with the following keys:\n"
        "  - \"response\": [Text]\n"
        "  - \"medical_data\": [JSON object with 'symptoms', 'severity', 'duration', 'triggers' as lists]\n"
        "  - \"next_state\": [INITIAL, SEVERITY, DURATION, TRIGGERS, CONFIRM, or COMPLETE]\n"
        "  - \"assign_doctor\": [true/false]\n"
        "- Ensure the output is a properly formatted JSON string.\n"
        "Return only the JSON string."
    )
)

symptom_extraction_prompt_template = PromptTemplate(
    input_variables=["input", "history", "current_state"],
    template=(
        "You are a medical assistant tasked with extracting symptoms from a patient's input. Your goal is to identify and list all potential symptoms mentioned, even if the input is vague, descriptive, or uses medical terminology. Use medical knowledge to interpret ambiguous or descriptive terms and map them to known medical symptoms. If a symptom is a medical term or not directly listed, map it to the closest known symptom or flag it for dynamic mapping.\n\n"
        "**Current State:** {current_state}\n"
        "**Conversation History:**\n{history}\n\n"
        "**Patient's Latest Input:**\n{input}\n\n"
        "**Instructions:**\n"
        "- Identify all symptoms mentioned in the input and history, combining information if necessary.\n"
        "- Interpret ambiguous, descriptive, or medical terms using medical knowledge and map them to the most appropriate known medical symptom. If the symptom is a medical term (e.g., 'hematuria') or not directly known, map it to the closest symptom in the following list, or flag it as unknown for dynamic mapping:\n"
        "  - Known symptoms: rash, itchiness, skin irritation, acne, hives, sore throat, ear pain, nasal congestion, dry cough, swelling, swollen ankles, edema, urination issues, kidney pain, hematuria, kidney stones, fatigue, weight changes, thirst, fever, general weakness, body aches, stomach pain, nausea, diarrhea, abdominal pain, injury, swelling (surgical), urinary tract infection, lower back pain, difficulty urinating, anxiety, depression, sleep issues, inability to sleep, menstrual issues, pelvic pain, irregular periods, spotting, heavy periods, unexplained weight loss, lump, chronic fatigue, child fever, child cough, child rash, chest pain, shortness of breath, palpitations, headache, dizziness, seizures, numbness and tingling in hands, pins and needles, balance issues, off balance, joint pain, muscle pain, fracture, unknown, exhaustion, lumps in neck, cough, tingling numbness, night sweats\n"
        "- Use context from additional symptoms to refine mappings. For example:\n"
        "  - If 'hematuria' is mentioned with 'urination issues' or 'kidney pain', it aligns with Nephrology.\n"
        "  - If 'hematuria' is mentioned with 'burning sensation when I pee' or 'difficulty urinating', it aligns with Urology.\n"
        "- For ambiguous or medical terms not in the known list, flag them as 'unknown' and let the system dynamically map them.\n"
        "- For each specialty, map descriptive phrases to known symptoms as follows:\n"
        "  - For Dermatology:\n"
        "    - 'red spots on my skin', 'bumpy rash', 'skin breakout', or 'itchy patches' should map to 'rash'.\n"
        "    - 'skin feels itchy', 'constant scratching', or 'itching all over' should map to 'itchiness'.\n"
        "    - 'irritated skin', 'skin feels raw', or 'burning sensation on skin' should map to 'skin irritation'.\n"
        "    - 'pimples on my face', 'zits everywhere', or 'breakouts on my back' should map to 'acne'.\n"
        "    - 'raised red welts', 'itchy bumps after a hike', or 'allergic skin reaction' should map to 'hives'.\n"
        "  - For ENT:\n"
        "    - 'throat feels scratchy', 'pain when swallowing', or 'sore throat for days' should map to 'sore throat'.\n"
        "    - 'ear hurts a lot', 'sharp pain in my ear', or 'earache after swimming' should map to 'ear pain'.\n"
        "    - 'stuffy nose', 'can’t breathe through my nose', 'blocked nasal passages', or 'nasal stuffiness' should map to 'nasal congestion'.\n"
        "    - 'persistent cough without phlegm', 'scratchy throat cough', 'coughing without mucus', or 'dry hacking cough' should map to 'dry cough'.\n"
        "  - For Nephrology:\n"
        "    - 'peeing less than usual', 'not urinating as much', 'decreased urination', 'going less often with less urine', or 'reduced urine output' should map to 'urination issues'.\n"
        "    - 'pain in my kidney area', 'dull ache in my lower back near kidneys', 'sharp pain around my kidneys', or 'back hurting near kidneys' should map to 'kidney pain'.\n"
        "    - 'blood in my urine', 'urine looks pinkish', 'reddish urine', 'hematuria', or 'pinkish pee' should map to 'hematuria'.\n"
        "  - For Endocrinology:\n"
        "    - 'gained weight suddenly', 'losing weight for no reason', 'weight keeps fluctuating', or 'unexpected weight gain/loss' should map to 'weight changes'.\n"
        "    - 'always thirsty', 'can’t stop drinking water', 'dry mouth all the time', or 'excessive thirst' should map to 'thirst'.\n"
        "  - For Gastroenterology:\n"
        "    - 'stomach hurts after eating', 'cramping in my belly', 'sharp pain in my gut', 'upset stomach', or 'belly ache' should map to 'stomach pain'.\n"
        "    - 'feeling nauseous', 'about to throw up', 'queasy stomach', 'sick to my stomach', or 'nausea after eating' should map to 'nausea'.\n"
        "    - 'loose stools', 'frequent watery bowel movements', 'can’t stop going to the bathroom', 'runny stools', or 'persistent loose motions' should map to 'diarrhea'.\n"
        "  - For Urology:\n"
        "    - 'sharp pain in my lower back', 'feels like a stone in my urinary tract', 'burning pain when peeing with stones', or 'severe pain with urination' should map to 'kidney stones'.\n"
        "    - 'burning sensation when I pee', 'frequent UTIs', 'painful urination with urgency', or 'stinging when urinating' should map to 'urinary tract infection'.\n"
        "    - 'lower back ache near my pelvis', 'dull pain in my lower spine', 'persistent back pain near my bladder', or 'back pain around pelvic area' should map to 'lower back pain'.\n"
        "    - 'can’t pee easily', 'struggling to urinate', 'weak urine stream', 'trouble starting to pee', or 'feeling like I can’t fully empty my bladder' should map to 'difficulty urinating'.\n"
        "    - 'blood in my urine', 'urine looks pinkish', 'reddish urine', 'hematuria', or 'pinkish pee' should map to 'hematuria'.\n"
        "  - For Psychiatry:\n"
        "    - 'feeling nervous all the time', 'constant worry', 'panic attacks', 'always anxious', or 'overwhelmed with worry' should map to 'anxiety'.\n"
        "    - 'feeling down', 'sad all the time', 'can’t stop crying', 'hopeless feeling', or 'always feeling blue' should map to 'depression'.\n"
        "    - 'trouble falling asleep', 'can’t sleep at night', 'restless nights', 'sleep problems', or 'insomnia issues' should map to 'sleep issues'.\n"
        "    - 'haven’t slept in days', 'always awake at night', 'can’t get to sleep', 'no sleep at all', or 'staying up all night' should map to 'inability to sleep'.\n"
        "  - For General Medicine:\n"
        "    - 'super tired', 'exhausted all the time', 'no energy to do anything', 'always drained', or 'feeling worn out' should map to 'fatigue'.\n"
        "    - 'running a fever', 'feeling hot and sweaty', 'high temperature', 'feverish feeling', or 'temperature spike' should map to 'fever'.\n"
        "    - 'feeling weak all over', 'no strength in my body', 'general body weakness', 'lacking energy', or 'overall weakness' should map to 'general weakness'.\n"
        "    - 'whole body aches', 'muscle soreness everywhere', 'body feels sore', 'all-over body pain', or 'general achiness' should map to 'body aches'.\n"
        "    - 'persistent cough', 'coughing a lot', 'can’t stop coughing', 'constant cough', or 'ongoing cough' should map to 'cough'.\n"
        "    - 'I don’t know what’s wrong', 'feeling off', or 'something feels wrong' should map to 'unknown'.\n"
        "  - For General Surgery:\n"
        "    - 'sharp pain in my abdomen', 'stabbing pain in my belly', 'severe stomach ache', 'intense belly pain', or 'gut pain' should map to 'abdominal pain'.\n"
        "    - 'hurt myself recently', 'got injured playing sports', 'accidentally cut myself', 'fell and got hurt', or 'recent trauma' should map to 'injury'.\n"
        "    - 'swelling after surgery', 'post-op swelling', 'swollen area after procedure', or 'surgery site swelling' should map to 'swelling (surgical)'.\n"
        "    - 'lump in my neck area', 'swollen gland in my neck', 'bump on my neck', 'neck mass', or 'growth in my neck' should map to 'lumps in neck'.\n"
        "  - For Gynecology:\n"
        "    - 'irregular periods', 'periods all over the place', 'missed periods', 'unpredictable menstrual cycle', or 'period timing issues' should map to 'menstrual issues'.\n"
        "    - 'pain in my pelvic area', 'cramping in my lower abdomen', 'sharp pain in my pelvis', 'lower belly pain', or 'pelvic discomfort' should map to 'pelvic pain'.\n"
        "    - 'spotting between periods', 'light bleeding mid-cycle', 'random spotting', 'unexpected light bleeding', or 'bleeding between periods' should map to 'spotting'.\n"
        "    - 'very heavy periods', 'bleeding too much during periods', 'soaking through pads quickly', 'excessive menstrual bleeding', or 'heavy flow' should map to 'heavy periods'.\n"
        "  - For Oncology:\n"
        "    - 'losing weight without trying', 'unintentionally lost weight', 'unexpectedly dropped weight', 'weight dropping for no reason', or 'sudden weight loss' should map to 'unexplained weight loss'.\n"
        "    - 'lump in my neck', 'noticeable lump', 'bump that won’t go away', 'hard mass under skin', or 'growth that persists' should map to 'lump'.\n"
        "    - 'always tired even after rest', 'persistent exhaustion', 'chronic tiredness', 'ongoing fatigue', or 'tiredness that won’t go away' should map to 'chronic fatigue'.\n"
        "    - 'extremely exhausted', 'no energy at all', 'can’t get out of bed', 'completely drained', or 'utter exhaustion' should map to 'exhaustion'.\n"
        "    - 'sweating a lot at night', 'nighttime sweats soaking my sheets', 'waking up drenched in sweat', 'night sweats every night', or 'excessive sweating at night' should map to 'night sweats'.\n"
        "  - For Pediatrics:\n"
        "    - 'my child has a fever', 'kid running a temperature', 'baby feels hot', 'child feels warm', or 'fever in my kid' should map to 'child fever'.\n"
        "    - 'my child keeps coughing', 'kid has a bad cough', 'persistent cough in my child', 'child coughing a lot', or 'cough in my baby' should map to 'child cough'.\n"
        "    - 'rash on my child’s skin', 'red spots on my kid', 'child has a skin breakout', 'skin rash on my baby', or 'itchy spots on my child' should map to 'child rash'.\n"
        "  - For Cardiology:\n"
        "    - 'swollen legs', 'puffy ankles', 'legs feel heavy and swollen', 'swelling in my limbs', or 'general body swelling' should map to 'swelling'.\n"
        "    - 'ankles are swollen', 'swelling in my feet', 'puffy feet and ankles', 'swollen ankles and feet', or 'ankle puffiness' should map to 'swollen ankles'.\n"
        "    - 'general swelling in my body', 'feeling bloated all over', 'swollen limbs', 'body feels puffy', or 'overall swelling' should map to 'edema'.\n"
        "    - 'weird tightness in my chest', 'heavy pressure in the chest', 'squeezing in my chest', 'chest discomfort', or 'pain in my chest' should map to 'chest pain'.\n"
        "    - 'lack of air', 'can’t catch my breath', 'difficulty breathing', 'short of breath', or 'breathing trouble' should map to 'shortness of breath'.\n"
        "    - 'heart racing', 'heart pounding', 'feeling my heart beat fast', 'rapid heartbeat', or 'heart fluttering' should map to 'palpitations'.\n"
        "  - For Neurology:\n"
        "    - 'slight headache', 'head pain', 'throbbing in my head', 'headache for days', or 'pain in my head' should map to 'headache'.\n"
        "    - 'feeling dizzy', 'room spinning', 'lightheaded all the time', 'dizzy when standing', or 'spinning sensation' should map to 'dizziness'.\n"
        "    - 'had a seizure', 'sudden shaking episode', 'uncontrolled jerking', 'convulsions', or 'seizure episode' should map to 'seizures'.\n"
        "    - 'numbness in my hands', 'tingling in my fingers', 'hands feel numb', 'numb fingers', or 'hand numbness' should map to 'numbness and tingling in hands'.\n"
        "    - 'pins and needles in my arms', 'prickling sensation', 'tingling like pins', 'pins and needles feeling', or 'prickly sensation' should map to 'pins and needles'.\n"
        "    - 'losing my balance', 'trouble walking straight', 'feeling unsteady', 'balance problems', or 'unsteady gait' should map to 'balance issues'.\n"
        "    - 'feeling off balance', 'can’t walk without stumbling', 'unsteady on my feet', 'always off balance', or 'balance feels off' should map to 'off balance'.\n"
        "    - 'numbness with tingling', 'tingling in my limbs', 'weird numbness sensation', 'tingling and numbness', or 'numb and tingly' should map to 'tingling numbness'.\n"
        "  - For Orthopedics:\n"
        "    - 'pain in my knees', 'joint stiffness', 'aching joints', 'sore joints', or 'knee pain' should map to 'joint pain'.\n"
        "    - 'muscle soreness', 'pain in my muscles', 'sore arms after lifting', 'muscle aches', or 'aching muscles' should map to 'muscle pain'.\n"
        "    - 'broke my arm', 'fractured my leg', 'bone injury from a fall', 'broken bone', or 'fracture after accident' should map to 'fracture'.\n"
        "- Do not include severity (e.g., 'mild', 'moderate', 'severe') in the symptom names; severity will be handled separately.\n"
        "- Only ask for clarification if the symptom description is truly ambiguous (e.g., 'I feel weird' with no further context) or cannot be reasonably mapped to a known medical term.\n"
        "- Return a valid JSON object (ensure proper formatting) with the following keys:\n"
        "  - \"symptoms\": [List of mapped symptom names as strings, e.g., [\"chest pain\", \"shortness of breath\", \"fatigue\"]]\n"
        "  - \"needs_clarification\": [true/false]\n"
        "  - \"clarification_message\": [Message to ask for clarification, if needed]\n"
        "  - \"mapped_symptoms\": [List of objects showing the original user input and the mapped symptom, e.g., [{{\"original\": \"weird tightness in my chest\", \"mapped\": \"chest pain\"}}]]\n"
        "- Example outputs:\n"
        "  - If the input is 'I have a weird tightness in my chest and can’t catch my breath', return: {{\"symptoms\": [\"chest pain\", \"shortness of breath\"], \"needs_clarification\": false, \"clarification_message\": \"\", \"mapped_symptoms\": [{{\"original\": \"weird tightness in my chest\", \"mapped\": \"chest pain\"}}, {{\"original\": \"can’t catch my breath\", \"mapped\": \"shortness of breath\"}}]}}\n"
        "  - If the input is 'I’m peeing less than usual and my back hurts near my kidneys', return: {{\"symptoms\": [\"urination issues\", \"kidney pain\"], \"needs_clarification\": false, \"clarification_message\": \"\", \"mapped_symptoms\": [{{\"original\": \"peeing less than usual\", \"mapped\": \"urination issues\"}}, {{\"original\": \"back hurts near my kidneys\", \"mapped\": \"kidney pain\"}}]}}\n"
        "  - If the input is 'I have blood in my urine and swelling in my legs', return: {{\"symptoms\": [\"hematuria\", \"swelling\"], \"needs_clarification\": false, \"clarification_message\": \"\", \"mapped_symptoms\": [{{\"original\": \"blood in my urine\", \"mapped\": \"hematuria\"}}, {{\"original\": \"swelling in my legs\", \"mapped\": \"swelling\"}}]}}\n"
        "  - If the input is 'I feel off', return: {{\"symptoms\": [], \"needs_clarification\": true, \"clarification_message\": \"Could you please clarify or provide more details about how you're feeling?\", \"mapped_symptoms\": []}}\n"
        "Return only the JSON string."
    )
)

doctor_mapping_validation_prompt_template = PromptTemplate(
    input_variables=["symptoms", "assigned_specialty"],
    template=(
        "You are a medical assistant tasked with validating the mapping of a patient's symptoms to a medical specialty. The available specialties are: Dermatology, ENT, Nephrology, Endocrinology, Gastroenterology, Urology, Psychiatry, General Medicine, General Surgery, Gynecology, Oncology, Pediatrics, Cardiology, Neurology, Orthopedics.\n\n"
        "**Patient's Symptoms:**\n{symptoms}\n\n"
        "**Assigned Specialty:**\n{assigned_specialty}\n\n"
        "**Instructions:**\n"
        "- Evaluate if the assigned specialty is appropriate for the given symptoms based on medical knowledge.\n"
        "- If the assigned specialty is incorrect, suggest the correct specialty and provide a brief reasoning.\n"
        "- Return a valid JSON object (ensure proper formatting) with the following keys:\n"
        "  - \"is_correct\": [true/false]\n"
        "  - \"correct_specialty\": [The correct specialty if incorrect, otherwise same as assigned]\n"
        "  - \"reasoning\": [Brief explanation of the validation]\n"
        "- Examples of mappings:\n"
        "  - 'palpitations', 'shortness of breath' → Cardiology\n"
        "  - 'depression', 'anxiety' → Psychiatry\n"
        "  - 'lump', 'unexplained weight loss' → Oncology\n"
        "  - 'headache', 'tremors' → Neurology\n"
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

def synthesize_audio(text, language="en", session_id=None):
    try:
        logger.debug(f"Attempting to synthesize audio with input language: {language}, text: {text[:50]}...")
        if not text or not text.strip():
            logger.warning("TTS input is empty or whitespace only")
            return None
        tts = gTTS(text=text, lang="en", slow=False)
        filename = f"audio_{session_id}.mp3" if session_id else f"audio_{int(time.time())}.mp3"
        audio_path = os.path.join("static", filename)
        tts.save(audio_path)
        audio_size = os.path.getsize(audio_path)
        if audio_size < 1024:
            logger.error(f"Generated audio file {audio_path} is too small: {audio_size} bytes")
            os.remove(audio_path)
            return None
        logger.info(f"Saved audio to {audio_path} with size {audio_size} bytes")
        return audio_path
    except Exception as e:
        logger.error(f"Audio synthesis error: {str(e)}")
        if 'audio_path' in locals() and os.path.exists(audio_path):
            os.remove(audio_path)
        return None

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

def calculate_age(dob_str):
    try:
        dob = datetime.strptime(dob_str, "%d-%m-%Y")
        today = datetime(2025, 4, 23)
        age = today.year - dob.year - ((today.month, today.day) < (dob.month, dob.day))
        return age
    except ValueError as e:
        logger.error(f"Failed to parse DOB '{dob_str}': {str(e)}")
        return None

def validate_doctor_mapping(symptoms, assigned_specialty):
    try:
        validation_input = {
            "symptoms": ", ".join(symptoms),
            "assigned_specialty": assigned_specialty
        }
        validation_response = doctor_mapping_validation_chain.invoke(validation_input)
        raw_response = validation_response.content
        logger.debug(f"Raw doctor mapping validation response: {raw_response}")

        # Strip Markdown code block markers if present
        cleaned_response = raw_response.strip()
        if cleaned_response.startswith("```json"):
            cleaned_response = cleaned_response[len("```json"):].strip()
        if cleaned_response.endswith("```"):
            cleaned_response = cleaned_response[:-len("```")].strip()

        # Parse the response as JSON
        validation_data = json.loads(cleaned_response)
        logger.debug(f"Doctor mapping validation result: {validation_data}")
        return validation_data
    except json.JSONDecodeError as e:
        logger.error(f"Failed to parse doctor mapping validation response as JSON: {str(e)}. Cleaned response: {cleaned_response}")
        return {
            "is_correct": True,
            "correct_specialty": assigned_specialty,
            "reasoning": "Validation failed due to JSON parsing error, defaulting to assigned specialty."
        }
    except Exception as e:
        logger.error(f"Error validating doctor mapping: {str(e)}")
        return {
            "is_correct": True,
            "correct_specialty": assigned_specialty,
            "reasoning": "Validation failed, defaulting to assigned specialty."
        }

def fetch_available_doctors(specialty):
    try:
        specialty = specialty.lower()
        doctors_list = [
            {"consultant_id": doc["consultant_id"], "full_name": doc["full_name"], "specialty": doc["specialty"]}
            for doc in DOCTORS if doc["specialty"] == specialty and doc["availability"]
        ]
        if not doctors_list:
            logger.warning(f"No doctors found for specialty: {specialty} with availability: True")
        return doctors_list
    except Exception as e:
        logger.error(f"Error fetching available doctors: {str(e)}")
        return []

def assign_doctor(medical_data):
    symptoms = medical_data.get('symptoms', [])
    severity = medical_data.get('severity', [''])  # List of severities
    duration = medical_data.get('duration', [''])  # List of durations
    additional_symptoms = symptoms[1:] if len(symptoms) > 1 else []  # Other symptoms

    if not symptoms:
        logger.warning("No symptoms provided for doctor assignment")
        return [], "general medicine"

    specialty = determine_specialty(symptoms, severity, duration, additional_symptoms)
    logger.debug(f"Determined specialty: {specialty} for symptoms: {symptoms}")

    # Validate the specialty mapping using the AI agent
    validation_result = validate_doctor_mapping(symptoms, specialty)
    if not validation_result["is_correct"]:
        logger.info(f"Specialty validation corrected from {specialty} to {validation_result['correct_specialty']}: {validation_result['reasoning']}")
        specialty = validation_result["correct_specialty"]

    doctors_list = fetch_available_doctors(specialty)
    if not doctors_list:
        logger.warning(f"No doctors available for specialty: {specialty}. Falling back to general medicine.")
        specialty = "general medicine"
        doctors_list = fetch_available_doctors(specialty)

    return doctors_list, specialty

def process_conversation(audio_path=None, transcript=None, history="", session_id=None):
    try:
        language = "en"
        medical_data = session.get('medical_data', {"symptoms": [], "severity": [], "duration": [], "triggers": []})
        if not medical_data.get("symptoms"):
            medical_data = {"symptoms": [], "severity": [], "duration": [], "triggers": []}

        current_state = session.get('current_state', "INITIAL")
        asked_questions = session.get('asked_questions', {})
        if not asked_questions:
            asked_questions = {symptom: {"severity": False, "duration": False, "triggers": False} 
                               for symptom in medical_data.get("symptoms", [])}
        session['asked_questions'] = asked_questions

        current_query = session.get('current_query', {"symptom": None, "field": None, "interpreted_severity": None, "raw_severity": None})
        has_initial_prompt_been_sent = session.get('has_initial_prompt_been_sent', False)

        conversation_id = session.get('conversation_id', None)
        if not conversation_id:
            conversation_id = secrets.token_hex(16)
            session['conversation_id'] = conversation_id
            logger.debug(f"New conversation started with ID: {conversation_id}")
        else:
            logger.debug(f"Continuing conversation with ID: {conversation_id}")

        if audio_path:
            transcription = transcribe_audio(audio_path, language)
            if "failed" in transcription.lower():
                return {"success": False, "response": "Audio issue detected. Please try again.", "medical_data": medical_data, "audio_url": None}, 400
        elif transcript:
            transcription = transcript
        else:
            if has_initial_prompt_been_sent:
                logger.debug("Initial prompt already sent, skipping duplicate request")
                return {"success": True, "response": "Please provide your symptoms to continue.", "medical_data": medical_data, "audio_url": None, "already_initiated": True}, 200
            intro = "What symptoms are you experiencing?"
            audio_path = synthesize_audio(intro, language, session_id)
            session['current_state'] = "INITIAL"
            session['has_initial_prompt_been_sent'] = True
            logger.info(f"Initial prompt sent for conversation {conversation_id}: {intro}")
            return {"success": True, "response": intro, "medical_data": medical_data, "audio_url": f"/static/{os.path.basename(audio_path)}"}, 200

        # Use gpt-4o-mini to extract symptoms from the transcript with a timeout
        symptom_extraction_input = {
            "input": transcription,
            "history": history,
            "current_state": current_state
        }
        logger.debug(f"Symptom extraction input: {symptom_extraction_input}")
        try:
            symptom_extraction_response = func_timeout(20, symptom_extraction_chain.invoke, args=(symptom_extraction_input,))
        except FunctionTimedOut:
            logger.error("Symptom extraction timed out after 20 seconds")
            return {
                "success": False,
                "response": "Symptom extraction took too long. Please try again.",
                "medical_data": medical_data,
                "audio_url": None
            }, 500
        raw_response = symptom_extraction_response.content
        logger.debug(f"Raw symptom extraction response: {raw_response}")

        # Strip Markdown code block markers if present
        cleaned_response = raw_response.strip()
        if cleaned_response.startswith("```json"):
            cleaned_response = cleaned_response[len("```json"):].strip()
        if cleaned_response.endswith("```"):
            cleaned_response = cleaned_response[:-len("```")].strip()

        # Parse the symptom extraction response
        try:
            symptom_extraction_data = json.loads(cleaned_response)
        except json.JSONDecodeError as e:
            logger.error(f"Failed to parse symptom extraction response as JSON: {str(e)}. Cleaned response: {cleaned_response}")
            return {
                "success": False,
                "response": "I couldn't understand your symptoms due to an internal error. Could you please repeat or clarify your symptoms?",
                "medical_data": medical_data,
                "audio_url": None
            }, 500

        extracted_symptoms = symptom_extraction_data.get("symptoms", [])
        needs_clarification = symptom_extraction_data.get("needs_clarification", False)
        clarification_message = symptom_extraction_data.get("clarification_message", "Could you please clarify or rephrase your symptoms?")

        if needs_clarification:
            logger.debug(f"Symptoms unclear, asking for clarification: {clarification_message}")
            audio_path = synthesize_audio(clarification_message, language, session_id)
            return {"success": True, "response": clarification_message, "medical_data": medical_data, "audio_url": f"/static/{os.path.basename(audio_path)}"}, 200

        # Update medical_data with extracted symptoms if in INITIAL state
        if current_state == "INITIAL":
            if not extracted_symptoms:
                response_text = "I couldn't identify any symptoms. Could you please tell me more about what you're experiencing?"
                audio_path = synthesize_audio(response_text, language, session_id)
                return {"success": True, "response": response_text, "medical_data": medical_data, "audio_url": f"/static/{os.path.basename(audio_path)}"}, 200
            medical_data["symptoms"] = extracted_symptoms
            # Initialize severity, duration, and triggers lists to match symptoms length
            medical_data["severity"] = [''] * len(extracted_symptoms)
            medical_data["duration"] = [''] * len(extracted_symptoms)
            medical_data["triggers"] = ['unknown'] * len(extracted_symptoms)
            session['medical_data'] = medical_data
            logger.debug(f"Extracted symptoms in INITIAL state: {extracted_symptoms}")

        input_vars = {"input": transcription, "history": history, "medical_data": json.dumps(medical_data), "current_state": current_state}
        try:
            conversation_response = func_timeout(20, conversation_chain.invoke, args=(input_vars,))
        except FunctionTimedOut:
            logger.error("Conversation chain invocation timed out after 20 seconds")
            return {
                "success": False,
                "response": "Processing your response took too long. Please try again.",
                "medical_data": medical_data,
                "audio_url": None
            }, 500
        raw_conversation_response = conversation_response.content
        logger.debug(f"Raw conversation response: {raw_conversation_response}")

        # Strip Markdown code block markers if present
        cleaned_conversation_response = raw_conversation_response.strip()
        if cleaned_conversation_response.startswith("```json"):
            cleaned_conversation_response = cleaned_conversation_response[len("```json"):].strip()
        if cleaned_conversation_response.endswith("```"):
            cleaned_conversation_response = cleaned_conversation_response[:-len("```")].strip()

        # Parse the conversation response through json_chain
        try:
            json_response = func_timeout(20, json_chain.invoke, args=({"conversation_output": cleaned_conversation_response},))
        except FunctionTimedOut:
            logger.error("JSON chain invocation timed out after 20 seconds")
            return {
                "success": False,
                "response": "Processing your response took too long. Please try again.",
                "medical_data": medical_data,
                "audio_url": None
            }, 500
        raw_json_response = json_response.content
        logger.debug(f"Raw JSON response: {raw_json_response}")

        # Strip Markdown code block markers if present
        cleaned_json_response = raw_json_response.strip()
        if cleaned_json_response.startswith("```json"):
            cleaned_json_response = cleaned_json_response[len("```json"):].strip()
        if cleaned_json_response.endswith("```"):
            cleaned_json_response = cleaned_json_response[:-len("```")].strip()

        # Parse the JSON response
        try:
            data = json.loads(cleaned_json_response)
        except json.JSONDecodeError as e:
            logger.error(f"Failed to parse JSON response as JSON: {str(e)}. Cleaned response: {cleaned_json_response}")
            return {
                "success": False,
                "response": "I encountered an internal error while processing your response. Could you please repeat or clarify your symptoms?",
                "medical_data": medical_data,
                "audio_url": None
            }, 500

        response_text = data["response"]
        raw_medical_data = data["medical_data"]
        next_state = data["next_state"]
        assign_doctor_flag = data["assign_doctor"]

        if not isinstance(raw_medical_data, dict):
            try:
                raw_medical_data = json.loads(raw_medical_data) if isinstance(raw_medical_data, str) else {"symptoms": [], "severity": [], "duration": [], "triggers": []}
            except json.JSONDecodeError as e:
                logger.error(f"Failed to parse raw_medical_data as JSON: {str(e)}. Raw data: {raw_medical_data}")
                raw_medical_data = {"symptoms": [], "severity": [], "duration": [], "triggers": []}

        updated_symptoms = raw_medical_data.get("symptoms", [])
        updated_severities = raw_medical_data.get("severity", [])
        updated_durations = raw_medical_data.get("duration", [])
        updated_triggers = raw_medical_data.get("triggers", [])

        # Ensure lists are the same length as updated_symptoms
        num_symptoms = len(updated_symptoms)
        if len(updated_severities) < num_symptoms:
            updated_severities.extend([''] * (num_symptoms - len(updated_severities)))
        if len(updated_durations) < num_symptoms:
            updated_durations.extend([''] * (num_symptoms - len(updated_durations)))
        if len(updated_triggers) < num_symptoms:
            updated_triggers.extend(['unknown'] * (num_symptoms - len(updated_triggers)))

        # Fallback: Extract severity from transcription if not updated by LLM, and map numerical values
        if current_state == "SEVERITY":
            for idx in range(len(updated_symptoms)):
                if not updated_severities[idx]:
                    transcription_lower = transcription.lower()
                    if "mild" in transcription_lower:
                        updated_severities[idx] = "mild"
                    elif "moderate" in transcription_lower:
                        updated_severities[idx] = "moderate"
                    elif "severe" in transcription_lower:
                        updated_severities[idx] = "severe"
                    else:
                        # Check for numerical values (e.g., "7", "6")
                        num_match = re.search(r'\b([1-9]|10)\b', transcription_lower)
                        if num_match:
                            num = int(num_match.group(1))
                            interpreted_severity = ""
                            if num <= 3:
                                interpreted_severity = "mild"
                            elif num <= 6:
                                interpreted_severity = "moderate"
                            else:
                                interpreted_severity = "severe"
                            updated_severities[idx] = interpreted_severity
                            # Store the raw and interpreted severity in current_query for CONFIRM_SEVERITY
                            current_query["raw_severity"] = num_match.group(1)
                            current_query["interpreted_severity"] = interpreted_severity
                        else:
                            updated_severities[idx] = "moderate"  # Default to moderate if unclear
                    logger.debug(f"Fallback: Extracted severity '{updated_severities[idx]}' for symptom {updated_symptoms[idx]} from transcription")

        # Fallback: Extract duration from transcription if not updated by LLM
        if current_state == "DURATION":
            for idx in range(len(updated_symptoms)):
                if not updated_durations[idx]:
                    transcription_lower = transcription.lower()
                    duration_match = re.search(r'(\d+\s*(to\s*\d+\s*)?(month|week|day))', transcription_lower)
                    if duration_match:
                        updated_durations[idx] = duration_match.group(0)
                    else:
                        updated_durations[idx] = "unknown"
                    logger.debug(f"Fallback: Extracted duration '{updated_durations[idx]}' for symptom {updated_symptoms[idx]} from transcription")

        # Fallback: Extract triggers from transcription if not updated by LLM
        if current_state == "TRIGGERS":
            for idx in range(len(updated_symptoms)):
                if not updated_triggers[idx]:
                    transcription_lower = transcription.lower()
                    if any(phrase in transcription_lower for phrase in ["i'm not sure", "it's unknown", "i don't know", "unknown"]):
                        updated_triggers[idx] = "unknown"
                    else:
                        updated_triggers[idx] = transcription_lower
                    logger.debug(f"Fallback: Extracted triggers '{updated_triggers[idx]}' for symptom {updated_symptoms[idx]} from transcription")

        # Handle CONFIRM_SEVERITY state
        if current_state == "CONFIRM_SEVERITY":
            transcription_lower = transcription.lower()
            symptom = current_query.get("symptom")
            interpreted_severity = current_query.get("interpreted_severity")
            raw_severity = current_query.get("raw_severity")
            if any(x in transcription_lower for x in ["yes", "correct", "right", "okay"]):
                # User confirmed the interpreted severity
                for idx, s in enumerate(medical_data["symptoms"]):
                    if s == symptom:
                        updated_severities[idx] = interpreted_severity
                        break
                response_text = f"How long have you been experiencing {symptom}?"
                session['current_query'] = {"symptom": symptom, "field": "duration"}
                next_state = "DURATION"
            elif any(x in transcription_lower for x in ["no", "incorrect", "wrong"]):
                # User denied the interpreted severity, re-ask
                response_text = f"Please specify the severity of your {symptom}. Is it mild, moderate, or severe?"
                session['current_query'] = {"symptom": symptom, "field": "severity"}
                next_state = "SEVERITY"
            else:
                # Unclear response, ask for clarification
                response_text = "Could you please confirm if this is correct? Please say 'yes' or 'no'."
                session['current_query'] = current_query  # Keep current_query unchanged
                next_state = "CONFIRM_SEVERITY"
            audio_path = synthesize_audio(response_text, language, session_id)
            medical_data["severity"] = updated_severities
            session['medical_data'] = medical_data
            session['current_state'] = next_state
            return {"success": True, "response": response_text, "medical_data": medical_data, "audio_url": f"/static/{os.path.basename(audio_path)}"}, 200

        # Validate that symptoms are present before advancing from INITIAL state
        if current_state == "INITIAL" and not updated_symptoms:
            response_text = "I couldn't identify any symptoms. Could you please tell me more about what you're experiencing?"
            audio_path = synthesize_audio(response_text, language, session_id)
            return {"success": True, "response": response_text, "medical_data": medical_data, "audio_url": f"/static/{os.path.basename(audio_path)}"}, 200

        # Symptom clarification
        if current_state == "INITIAL":
            unclear_symptoms = [s for s in updated_symptoms if not symptom_specialty_map.get(clean_symptom(s))]
            if unclear_symptoms:
                response_text = f"I didn't understand the symptom '{unclear_symptoms[0]}'. Could you please clarify or rephrase your symptom?"
                session['current_state'] = "INITIAL"
                session['medical_data'] = medical_data
                audio_path = synthesize_audio(response_text, language, session_id)
                return {"success": True, "response": response_text, "medical_data": medical_data, "audio_url": f"/static/{os.path.basename(audio_path)}"}, 200

        medical_data = {
            "symptoms": updated_symptoms,
            "severity": updated_severities,
            "duration": updated_durations,
            "triggers": updated_triggers
        }

        # Update asked_questions based on medical_data and handle multiple symptoms
        for idx, symptom in enumerate(medical_data["symptoms"]):
            if not symptom: continue
            if symptom not in asked_questions:
                asked_questions[symptom] = {"severity": False, "duration": False, "triggers": False}
            # Check if severity is a valid value (mild, moderate, severe) or a numerical value
            if idx < len(medical_data["severity"]):
                severity = medical_data["severity"][idx].lower()
                # Map numerical values to mild/moderate/severe
                if severity.isdigit():
                    num = int(severity)
                    if num <= 3:
                        medical_data["severity"][idx] = "mild"
                    elif num <= 6:
                        medical_data["severity"][idx] = "moderate"
                    else:
                        medical_data["severity"][idx] = "severe"
                    severity = medical_data["severity"][idx]
                if severity in ["mild", "moderate", "severe"]:
                    asked_questions[symptom]["severity"] = True
                else:
                    asked_questions[symptom]["severity"] = False  # Reset if not a valid severity
            if idx < len(medical_data["duration"]) and medical_data["duration"][idx].strip():
                asked_questions[symptom]["duration"] = True
            else:
                asked_questions[symptom]["duration"] = False  # Reset if not answered
            if idx < len(medical_data["triggers"]) and medical_data["triggers"][idx]:
                asked_questions[symptom]["triggers"] = True
            else:
                asked_questions[symptom]["triggers"] = False  # Reset if not answered

        session['asked_questions'] = asked_questions
        session['medical_data'] = medical_data
        session['current_state'] = next_state

        # Log the state of asked_questions for debugging
        logger.debug(f"Current state: {current_state}, Asked questions: {asked_questions}")

        # Check if all questions are answered
        all_questions_answered = all(
            asked_questions[symptom]["severity"] and 
            asked_questions[symptom]["duration"] and 
            asked_questions[symptom]["triggers"]
            for symptom in medical_data["symptoms"] if symptom
        )
        logger.debug(f"All questions answered: {all_questions_answered}")

        # Handle CONFIRM state explicitly to ensure state transition
        if current_state == "CONFIRM" and any(x in transcription.lower() for x in ["yes", "please", "okay", "sure"]):
            logger.debug(f"Medical data before assigning doctor: {medical_data}")
            doctors_list, specialty = assign_doctor(medical_data)
            if not doctors_list:
                response_text = f"No doctors available for {specialty or 'unknown specialty'}. Please try again later or contact support."
                audio_path = synthesize_audio(response_text, language, session_id)
                return {
                    "success": False,
                    "response": response_text,
                    "medical_data": medical_data,
                    "audio_url": f"/static/{os.path.basename(audio_path)}" if audio_path else None,
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
                    logger.info(f"Successfully saved initial screening data for UID: {uid}")
                except Exception as e:
                    logger.error(f"Failed to update Firestore initial_screenings: {str(e)}")
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
                    logger.info(f"Successfully updated patient registration for UID: {uid}")
                except Exception as e:
                    logger.error(f"Failed to update Firestore patient_registrations: {str(e)}")
                    save_success = False

                response_text = f"You have been assigned to {doctor['full_name']} (specializing in {specialty.capitalize()}). You will be redirected to your dashboard."
                if not save_success:
                    response_text += " Note: There was an issue saving your data, but your doctor assignment has been recorded."

                audio_path = synthesize_audio(response_text, language, session_id)
                return {
                    "success": True,
                    "response": response_text,
                    "medical_data": medical_data,
                    "audio_url": f"/static/{os.path.basename(audio_path)}" if audio_path else None,
                    "redirect": "/dashboard",
                    "conversationComplete": True
                }, 200

        if not all_questions_answered:
            for idx, symptom in enumerate(medical_data["symptoms"]):
                if not symptom: continue
                if not asked_questions[symptom]["severity"]:
                    response_text = response_text.replace("{{symptoms}}", symptom)
                    session['current_query'] = {"symptom": symptom, "field": "severity"}
                    audio_path = synthesize_audio(response_text, language, session_id)
                    return {"success": True, "response": response_text, "medical_data": medical_data, "audio_url": f"/static/{os.path.basename(audio_path)}"}, 200
                elif not asked_questions[symptom]["duration"]:
                    response_text = response_text.replace("{{symptoms}}", symptom)
                    session['current_query'] = {"symptom": symptom, "field": "duration"}
                    audio_path = synthesize_audio(response_text, language, session_id)
                    return {"success": True, "response": response_text, "medical_data": medical_data, "audio_url": f"/static/{os.path.basename(audio_path)}"}, 200
                elif not asked_questions[symptom]["triggers"]:
                    response_text = response_text.replace("{{symptoms}}", symptom)
                    session['current_query'] = {"symptom": symptom, "field": "triggers"}
                    audio_path = synthesize_audio(response_text, language, session_id)
                    return {"success": True, "response": response_text, "medical_data": medical_data, "audio_url": f"/static/{os.path.basename(audio_path)}"}, 200

        if all_questions_answered and current_state != "CONFIRM" and current_state != "COMPLETE":
            response_text = "May I recommend a doctor for you?"
            session['current_state'] = "CONFIRM"
            audio_path = synthesize_audio(response_text, language, session_id)
            return {"success": True, "response": response_text, "medical_data": medical_data, "audio_url": f"/static/{os.path.basename(audio_path)}"}, 200

        audio_path = synthesize_audio(response_text, language, session_id)
        return {"success": True, "response": response_text, "medical_data": medical_data, "audio_url": f"/static/{os.path.basename(audio_path)}"}, 200

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
        logger.error("index.html not found: " + str(e))
        return "Homepage not found. Please create templates/index.html", 404

@app.route('/register', methods=['GET', 'POST'])
def register():
    try:
        if request.method == 'POST':
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
                user_ref = db.collection('assistant_registrations').document(uid)
            else:
                user_ref = db.collection('patient_registrations').document(uid)

            user_data['uid'] = uid
            user_ref.set(user_data)
            logger.info(f"✅ Registered new {role} with UID: {uid}")

            redirect_url = '/login' if role == 'patient' else '/login'
            return jsonify({
                'success': True,
                'message': 'Registration successful.',
                'redirect': redirect_url
            })
        return render_template('registration.html')
    except Exception as e:
        logger.exception(f"Registration error: {e}")
        return jsonify({"error": str(e)}), 500

@app.route('/login', methods=['GET', 'POST'])
def login():
    if request.method == 'GET':
        try:
            if 'user_info' in session:
                id_token = session.get('idToken')
                if id_token:
                    try:
                        decoded_token = auth.verify_id_token(id_token, clock_skew_seconds=60)
                        email = decoded_token.get('email')
                        logger.debug(f"Session token valid for email: {email}")
                        role = session['user_info'].get('role')
                        return redirect('/dashboard')
                    except auth.InvalidIdTokenError as e:
                        logger.error(f"Invalid session token: {str(e)}. Clearing session.")
                        session.pop('user_info', None)
                        session.pop('idToken', None)
                    except auth.ExpiredIdTokenError as e:
                        logger.error(f"Expired session token: {str(e)}. Clearing session.")
                        session.pop('user_info', None)
                        session.pop('idToken', None)
            return render_template('login.html')
        except Exception as e:
            logger.error("login.html not found: " + str(e))
            return "Login page not found. Please create templates/login.html", 404
    try:
        id_token = request.headers.get('Authorization', '').replace('Bearer ', '') or request.form.get('idToken')
        if not id_token:
            logger.error("No ID token provided")
            return jsonify({"error": "No ID token provided"}), 400

        try:
            decoded_token = auth.verify_id_token(id_token, clock_skew_seconds=60)
            current_time = int(time.time())
            token_iat = decoded_token.get('iat')
            logger.debug(f"Server time: {current_time}, Token iat: {token_iat}, Skew: {token_iat - current_time} seconds")
        except auth.InvalidIdTokenError as e:
            if "Token used too early" in str(e):
                logger.error(f"Clock skew error during token verification: {str(e)}")
                return jsonify({
                    "error": "Login failed",
                    "details": "Token used too early. Please ensure your device's clock is synchronized with the correct time and try again."
                }), 400
            raise e

        email = decoded_token.get('email')
        uid = decoded_token.get('uid')

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

        full_name = user_data.get('full_name', uid)
        session['user_info'] = {
            'email': email,
            'role': role,
            'uid': doc_id if role == 'doctor' else uid,
            'full_name': full_name
        }
        session["idToken"] = id_token
        logger.info(f"🔑 User logged in: {email} with role: {role}, UID: {doc_id if role == 'doctor' else uid}")

        redirect_url = '/dashboard'
        return jsonify({"success": True, "redirect": redirect_url})
    except Exception as e:
        logger.exception(f"Unexpected error during login: {str(e)}")
        return jsonify({"error": "Login failed", "details": str(e)}), 500
    
@app.route('/check-auth', methods=['GET'])
def check_auth():
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
        return jsonify({"authenticated": False}), 401

    try:
        decoded_token = auth.verify_id_token(token, clock_skew_seconds=60)
        current_time = int(time.time())
        token_iat = decoded_token.get('iat')
        logger.debug(f"Server time: {current_time}, Token iat: {token_iat}, Skew: {token_iat - current_time} seconds")
        logger.info(f"✅ Token verified for: {decoded_token.get('email')}, UID: {decoded_token.get('uid')}")
        return jsonify({"authenticated": True})
    except auth.InvalidIdTokenError as e:
        logger.error(f"Invalid ID token: {str(e)}")
        return jsonify({"authenticated": False}), 401
    except auth.ExpiredIdTokenError as e:
        logger.error(f"Expired ID token: {str(e)}")
        return jsonify({"authenticated": False}), 401
    except Exception as e:
        logger.error(f"Token verification error: {str(e)}")
        return jsonify({"authenticated": False}), 401
    
@app.route('/further_patient_registration', methods=['GET'])
@token_required
def further_patient_registration():
    try:
        user = request.user
        uid = user.get('uid')
        session_info = session.get('user_info', {})
        patient_name = session_info.get('full_name', uid)

        doc_id = f'initial_screening_{uid}'
        doc_ref = db.collection('initial_screenings').document(doc_id)

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

        return render_template('further_patient_registration.html', user_info={'uid': uid, 'patient_name': patient_name})
    except Exception as e:
        logger.exception(f"Further patient registration error: {e}")
        return jsonify({"error": "Error loading further patient registration", "details": str(e)}), 500

@app.route('/synthesize_audio', methods=['POST'])
@token_required
def synthesize_audio_endpoint():
    try:
        data = request.get_json()
        if not data or 'text' not in data or 'language' not in data:
            logger.error("Missing text or language in synthesize request")
            return jsonify({"error": "Missing text or language parameter"}), 400

        text = data['text']
        language = data['language']

        audio_path = synthesize_audio(text, language)
        if not audio_path:
            logger.error("Failed to synthesize audio")
            return jsonify({"error": "Failed to synthesize audio"}), 500

        audio_url = f"/static/{os.path.basename(audio_path)}"
        logger.info(f"Text synthesized successfully: {text}, audio URL: {audio_url}")
        return jsonify({"audio_url": audio_url})

    except Exception as e:
        logger.error(f"Error in /synthesize_audio endpoint: {str(e)}")
        return jsonify({"error": "Failed to synthesize text", "details": str(e)}), 500
    
@app.route('/transcribe', methods=['POST'])
@token_required
def transcribe():
    try:
        language = "en"
        if 'audio' not in request.files:
            logger.error("No audio file provided in the request")
            return jsonify({"error": "No audio file provided"}), 400

        audio_file = request.files['audio']
        if audio_file.filename == '':
            logger.error("No audio file selected")
            return jsonify({"error": "No audio file selected"}), 400

        temp_dir = tempfile.mkdtemp()
        audio_path = os.path.join(temp_dir, f"temp_audio_{request.user.get('uid')}.webm")
        audio_file.save(audio_path)
        logger.debug(f"Audio saved to: {audio_path}")

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
            data = request.get_json()
            transcript = data.get('transcript', '')
            logger.debug(f"Received transcript for UID {uid}: {transcript}")
            processed_data, status_code = process_conversation(transcript=transcript, history=history)

            if status_code != 200:
                return jsonify(processed_data), status_code

            medical_data = processed_data.get('medical_data')
            expected_keys = ["symptoms", "severity", "duration", "triggers"]
            for key in expected_keys:
                if key not in medical_data or not isinstance(medical_data[key], list):
                    logger.warning(f"medical_data[{key}] is not a list, resetting: {medical_data.get(key)}")
                    medical_data[key] = []

            session['conversation_history'] = f"{history}\nPatient: {transcript}\nAgent: {processed_data['response']}" if transcript else f"{history}\nAgent: {processed_data['response']}"

            return jsonify({
                'success': True,
                'response': processed_data['response'],
                'medical_data': medical_data,
                'audio_url': processed_data.get('audio_url'),
                'redirect': processed_data.get('redirect', '/dashboard')
            })
        elif 'audio' in request.files:
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

            medical_data = processed_data.get('medical_data')
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

        blob = bucket.blob(file_path)
        temp_file = os.path.join(tempfile.gettempdir(), f'temp_{uid}.jpg')
        os.makedirs(os.path.dirname(temp_file), exist_ok=True)
        blob.download_to_filename(temp_file)

        # Placeholder for extract_text_from_image (not provided)
        extracted_text = "Extracted text placeholder"

        existing_text = None
        if category == 'prescriptions':
            existing_summaries = db.collection('prescriptions').where('uid', '==', uid).order_by('timestamp', direction=firestore.Query.DESCENDING).limit(1).get()
            if existing_summaries:
                existing_text = existing_summaries[0].to_dict().get('summary', '')

        patient_ref = db.collection('patient_registrations').document(uid).get()
        patient_name = patient_ref.to_dict().get('full_name', 'Patient') if patient_ref.exists else 'Patient'

        # Placeholder for process_text_with_gemini (not provided)
        result = {
            'regional_summary': "Summary in English",
            'professional_summary': "Professional summary",
            'language': "en"
        }

        doc_ref = db.collection(category).document()
        doc_ref.set({
            'uid': uid,
            'consultant_id': consultant_id,
            'patient_name': patient_name,
            'summary': result['regional_summary'],
            'professional_summary': result['professional_summary'],
            'language': result['language'],
            'timestamp': firestore.SERVER_TIMESTAMP
        })
        return jsonify({'success': True, 'language': result['language']})
    except Exception as e:
        logger.error(f"Processing failed: {str(e)}")
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

        # Placeholder for generate_medical_history (not provided)
        summary = "Medical history summary placeholder"
        return jsonify({'success': True, 'summary': summary})
    except Exception as e:
        logger.error(f"Error processing medical history: {str(e)}")
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
            logger.error(f"Invalid role {role} for dashboard")
            raise ValueError(f"No template found for role {role}")

        return render_template(template, user_info={'uid': uid, 'email': email, 'role': role}, login_success=True)
    except Exception as e:
        logger.exception(f"Dashboard error: {e}")
        return jsonify({"error": "Error rendering dashboard", "details": str(e)}), 500

@app.route('/logout')
def logout():
    logger.info("User logging out")
    if request.args.get('confirm') != 'yes':
        return "Are you sure you want to log out? <a href='/logout?confirm=yes'>Confirm</a> | <a href='/dashboard'>Cancel</a>", 200
    session.clear()
    logger.info("Session cleared, redirecting to /login")
    return redirect('/login')

@app.errorhandler(404)
def page_not_found(e):
    static_poem = "Blank space resonates\n404 hums creation\nNew worlds now beckon"
    return render_template('errors/404.html', error_poem=static_poem), 404

if __name__ == '__main__':
    port = int(os.environ.get('PORT', 5000))
    logger.info(f"Starting Flask app on port {port}")
    app.run(debug=False, port=port)