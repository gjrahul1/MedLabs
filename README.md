<div align="center">
  <img src="https://osf.io/static/img/osf-sharing.png" alt="OSF Preprints" height="90" style="margin-right: 20px;">
  <img src="https://th.bing.com/th/id/R.47e8daaa41ec963d5cbfd458f37045d1?rik=kkJARg5sDo9zew&riu=http%3a%2f%2fairo.co.in%2fimages%2flogo.png&ehk=bR%2fdGNcUnfVIZUKhfrdJyv2b9ribA4RSeDSehAtGE%2fM%3d&risl=&pid=ImgRaw&r=0" alt="Airo Journal" height="90">
</div>

<h2 align="center">🧬 MedLabs: An AI-Powered System for Smart Medical Triage and Record Summarization</h2>

## About the Project

**MedLabs** is an advanced AI-powered medical diagnostics assistant that integrates speech-to-text, medical reasoning, and Firebase-backed health data pipelines. It helps patients input symptoms via voice, dynamically maps them to medical specialties, and routes them to the appropriate consultant. It is voice-enabled, cross-lingual (supports Tamil, Kannada), and integrates OpenAI's LLMs and Google's Gemini for intelligent summarization of prescriptions and lab records.

🔬 This work is also:
- [Published as a preprint on OSF](https://osf.io/preprints/osf/t7sbp_v1)
- [Accepted in Airo International Journal](https://www.airo.co.in/view-publication/2440)
- [Published Paper in Airo International Journal](https://drive.google.com/file/d/1Lftt_PuVVfSphpoOstpipbw3xMfNC2rq/view?usp=sharing)
- [Conference Proceeding Book](https://drive.google.com/file/d/1okVGr3w6MSqQ46iYY6DFEY8hmfdNzbIT/view?usp=sharing)

---

## Prerequisites

Before running the project, ensure the following setup:

```bash
### 1. Install Dependencies

pip install -r requirements.txt

### 2. Set Up Firebase

Create a Firebase project.
Enable Authentication, Firestore, and Storage.

Place your Firebase Admin SDK JSON inside:
Cred/Firebase/Med App/

### 3. Set Up Google Cloud
Enable the Cloud Vision API and Google Cloud Storage.

Create a bucket (e.g., med-labs-new-bucket-2025).

Place your Google Cloud service account JSON inside:
Cred/Med_Labs_GC/

### 4. OpenAI Account
Create an account and generate an API key.

Save your key inside a .env file as:
OPENAI_API_KEY=your_key_here
GEMINI_API_KEY=your_key_here

### 5. Prepare Folder Structure
Ensure this directory structure is in place:
Cred/
├── Firebase/
│   └── Med App/
│       └── med-labs-42f13-firebase-adminsdk-*.json
└── Med_Labs_GC/
    └── med-labs-gcs-credentials.json


## Firebase Utility Scripts
These are JavaScript-based setup scripts to populate Firestore collections:

firebase-setup.js
Upload PDFs to GCS and Firestore

firebase-setup-2.js
Register assistants and consultants

firebase-setup-3.js
Add prescriptions and lab records

firebase-setup-4.js
Create consultant Firestore entries

firebase-setup-5.js
Create initial screenings

Run any of them using:
node firebase-setup.js path/to/sample.pdf

## 🚀 Run the App
python app.py

The app will be live at: http://127.0.0.1:5000
```
## 📜 Citation
If you use MedLabs for academic or research purposes, please cite the Airo Journal version.

## 📬 Contact
Developed by: G J Rahul, Rithika Ramesh Chettiar, S A Vinit

For queries: gjrahul21@gmail.com
