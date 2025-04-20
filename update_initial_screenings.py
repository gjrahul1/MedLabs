import os
import glob
import json
import firebase_admin
from firebase_admin import credentials, firestore
from datetime import datetime

# Initialize Firebase
cred_path = './Cred/Firebase/Med App/med-labs-42f13-firebase-adminsdk-fbsvc-43b78dfef7.json'
cred = credentials.Certificate(cred_path)
firebase_admin.initialize_app(cred)
db = firestore.client()

def get_latest_json_file(directory):
    """Find the latest medical_data_*.json file in the specified directory."""
    json_files = glob.glob(os.path.join(directory, "medical_data_*.json"))
    if not json_files:
        raise FileNotFoundError("No medical_data_*.json files found in the directory.")
    
    # Sort files by creation time (most recent first)
    json_files.sort(key=os.path.getctime, reverse=True)
    return json_files[0]

def update_initial_screenings(json_file_path):
    """Read the JSON file and update the corresponding initial_screenings document in Firestore."""
    try:
        # Read the JSON file
        with open(json_file_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        print(f"Loaded JSON data from {json_file_path}: {data}")

        # Extract medical_data and UID
        medical_data = data.get('medical_data', {})
        if not medical_data:
            raise ValueError("No medical_data found in the JSON file.")
        
        uid = medical_data.get('uid')
        if not uid:
            raise ValueError("No UID found in medical_data.")
        
        # Prepare the updated data
        updated_data = {
            'consultant_id': medical_data.get('consultant_id'),
            'symptoms': medical_data.get('symptoms', ''),
            'severity': medical_data.get('severity', ''),
            'duration': medical_data.get('duration', ''),
            'triggers': medical_data.get('triggers', ''),
            'patient_name': medical_data.get('patient_name', 'Unknown'),
            'uid': uid,
            'timestamp': firestore.SERVER_TIMESTAMP
        }

        # Find the most recent initial_screenings document for the UID
        doc_ref = db.collection('initial_screenings').where(
            filter=firestore.FieldFilter('uid', '==', uid)
        ).order_by('timestamp', direction=firestore.Query.DESCENDING).limit(1).get()

        if not doc_ref:
            raise ValueError(f"No initial_screenings document found for UID: {uid}")

        # Update the document
        for doc in doc_ref:
            doc.reference.set(updated_data, merge=True)
            print(f"Updated initial_screenings document for UID: {uid} with data: {updated_data}")
            break

    except Exception as e:
        print(f"Error updating initial_screenings: {str(e)}")
        raise

if __name__ == "__main__":
    try:
        # Directory where the JSON files are stored
        json_directory = os.path.join("C:", "Users", "gjrah", "Documents", "Major Project", "App Development","Users")
        
        # Get the latest JSON file
        latest_json_file = get_latest_json_file(json_directory)
        print(f"Found latest JSON file: {latest_json_file}")

        # Update Firestore
        update_initial_screenings(latest_json_file)
        print("Successfully updated initial_screenings in Firestore.")

    except Exception as e:
        print(f"Failed to update initial_screenings: {str(e)}")