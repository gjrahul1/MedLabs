import re
import secrets
import logging
import os
import tempfile
from functools import wraps
from flask import Flask, render_template, request, jsonify, session, redirect, url_for, send_from_directory
import firebase_admin
from firebase_admin import credentials, auth, firestore, storage
from google_vision import extract_text_from_image
from gemini_processor import process_text_with_gemini

app = Flask(__name__)
app.secret_key = os.environ.get("FLASK_SECRET_KEY", secrets.token_hex(32))

logging.basicConfig(level=logging.DEBUG, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

cred_path = './Cred/Firebase/Med App/med-labs-42f13-firebase-adminsdk-fbsvc-43b78dfef7.json'
if not firebase_admin._apps:
    cred = credentials.Certificate(cred_path)
    firebase_admin.initialize_app(cred, {
        'storageBucket': 'med-labs-42f13'
    })
    logger.info("✅ Firebase initialized successfully with bucket: med-labs-42f13. SDK Version: %s", firebase_admin.__version__)

db = firestore.client()
bucket = storage.bucket()

# Verify bucket existence
try:
    bucket.get_blob('test-check')
    logger.info("Bucket exists and is accessible")
except Exception as e:
    logger.error(f"Bucket validation failed: {str(e)}. Please ensure the bucket 'med-labs-42f13' exists.")
    raise

def firebase_token_required(f):
    @wraps(f)
    def decorated_function(*args, **kwargs):
        id_token = request.headers.get('Authorization', '').replace('Bearer ', '')
        if not id_token and 'idToken' in session:
            id_token = session['idToken']
        if not id_token:
            logger.error("No Authorization header or session token found")
            return jsonify({"error": "Authorization header with Bearer token is required"}), 400
        try:
            logger.debug(f"Verifying ID token: {id_token[:20]}... (Full length: {len(id_token)})")
            decoded_token = auth.verify_id_token(id_token)
            request.user = decoded_token
            logger.info(f"✅ Token verified for: {decoded_token.get('email')}, UID: {decoded_token.get('uid')}")
        except auth.InvalidIdTokenError as e:
            logger.error(f"❌ Invalid ID token error: {str(e)} with token: {id_token[:50]}...")
            return jsonify({"error": "Invalid ID token", "details": str(e)}), 401
        except Exception as e:
            logger.error(f"❌ Token verification error: {str(e)} with token: {id_token[:50]}...")
            return jsonify({"error": "Token verification failed", "details": str(e)}), 401
        return f(*args, **kwargs)
    return decorated_function

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

# --------------------------------------------------------------------
# INDEX ROUTE (GET)
# --------------------------------------------------------------------
@app.route('/', methods=['GET'])
def index():
    try:
        return render_template('index.html')
    except Exception as e:
        logger.error("index.html not found: " + str(e))
        return "Homepage not found. Please create templates/index.html", 404

# --------------------------------------------------------------------
# LOGIN ROUTE (GET/POST) - MODIFIED TO SUPPORT GET
# --------------------------------------------------------------------
@app.route('/login', methods=['GET', 'POST'])
def login():
    if request.method == 'GET':
        try:
            if 'user_info' in session:
                logger.debug("User already logged in, redirecting to dashboard")
                return redirect(url_for('dashboard'))
            return render_template('login.html')
        except Exception as e:
            logger.error("login.html not found: " + str(e))
            return "Login page not found. Please create templates/login.html", 404

    try:
        id_token = request.headers.get('Authorization', '').replace('Bearer ', '') or request.form.get('idToken')
        logger.debug(f"Received ID token: {id_token[:20]}... (Full length: {len(id_token)})")
        if not id_token:
            logger.error("No ID token provided")
            return jsonify({"error": "No ID token provided"}), 400

        decoded_token = auth.verify_id_token(id_token)
        email = decoded_token.get('email')
        uid = decoded_token.get('uid')
        logger.debug(f"Decoded token for email: {email}, uid: {uid}")

        request_data = request.get_json() or {}
        request_uid = request_data.get('uid')
        if request_uid and request_uid != uid:
            logger.error(f"UID mismatch: Request UID {request_uid} does not match token UID {uid}")
            return jsonify({"error": "UID mismatch", "details": f"Expected: {uid}, Got: {request_uid}"}), 401

        role_mapping = {
            'patient': 'patient_registrations',
            'doctor': 'consultant_registrations',
            'assistant': 'assistant_registrations'
        }
        user_data = None
        role = None
        doc_id = None

        for role_key, collection_name in role_mapping.items():
            query = db.collection(collection_name).where('email', '==', email).limit(1).get()
            for doc in query:
                user_data = doc.to_dict()
                role = role_key
                doc_id = doc.id
                break
            if role:
                break

        if not user_data:
            logger.error(f"User not found in any role collection for email: {email} or uid: {uid}")
            return jsonify({"error": "User not found in any role collection", "details": f"Checked UID: {uid}, Email: {email}"}), 401
        elif not user_data.get('email') or user_data.get('email') != email:
            logger.error(f"Email mismatch for uid: {uid}, expected {email}, got {user_data.get('email')}")
            return jsonify({"error": "Email mismatch", "details": f"Expected: {email}, Got: {user_data.get('email')}"}), 401

        full_name = user_data.get('full_name', uid)
        logger.info(f"User found with role: {role}, email: {email}, uid: {uid}, full_name: {full_name}")

        session['user_info'] = {
            'email': email,
            'role': role,
            'uid': doc_id if role == 'doctor' else uid,
            'full_name': full_name
        }
        session["idToken"] = id_token
        logger.info(f"🔑 User logged in: {email} with role: {role}, UID: {doc_id if role == 'doctor' else uid}, Full Name: {full_name}")

        session_user_ref = db.collection(role_mapping[role]).document(doc_id if role == 'doctor' else uid)
        session_user_snap = session_user_ref.get()
        if session_user_snap.exists:
            session_user_data = session_user_snap.to_dict()
            if session_user_data.get('email') != email or session_user_data.get('role') != role:
                logger.error("Session data mismatch with Firestore")
                session.pop('user_info', None)
                session.pop('idToken', None)
                return jsonify({"error": "Session data mismatch", "details": "Please log in again"}), 401

        return jsonify({"success": True, "redirect": "/dashboard"})
    except auth.InvalidIdTokenError as e:
        logger.error(f"Invalid ID token error: {str(e)} with token: {id_token[:50]}...")
        return jsonify({"error": "Invalid ID token", "details": str(e)}), 401
    except Exception as e:
        logger.exception(f"Unexpected error during login: {str(e)} with token: {id_token[:50]}...")
        return jsonify({"error": "Login failed", "details": str(e)}), 500

# --------------------------------------------------------------------
# REGISTER ROUTE (GET/POST)
# --------------------------------------------------------------------
@app.route('/register', methods=['GET', 'POST'])
def register():
    try:
        if request.method == 'POST':
            id_token = request.headers.get('Authorization', '').replace('Bearer ', '') or request.form.get('idToken')
            if not id_token:
                email = request.form.get('email')
                password = request.form.get('password')
                role = request.form.get('role')
                if not email or not password or not role:
                    return jsonify({"error": "Email, password, and role are required"}), 400

                user = auth.create_user(email=email, password=password, uid=None)
                uid = user.uid
                logger.info(f"✅ Created new user in Firebase Auth: {email}, UID: {uid}")
            else:
                decoded_token = auth.verify_id_token(id_token)
                uid = decoded_token['uid']
                email = decoded_token.get('email')
                logger.info(f"✅ Verified existing user token: {email}, UID: {uid}")

            user_data = request.get_json() if request.is_json else request.form.to_dict()

            required_fields = ["full_name", "email", "phone", "dob", "location", "role"]
            missing_fields = [field for field in required_fields if field not in user_data or not user_data[field].strip()]
            if missing_fields:
                logger.error(f"Missing fields in registration: {', '.join(missing_fields)}")
                return jsonify({"error": f"Missing fields: {', '.join(missing_fields)}"}), 400

            role = user_data.get('role')
            if role == 'doctor':
                additional = ["doctor_id", "department"]
                missing_add = [field for field in additional if field not in user_data or not user_data[field].strip()]
                if missing_add:
                    return jsonify({"error": "Missing fields for doctor: " + ", ".join(missing_add)}), 400
                consultant_id = generate_consultant_id()
                user_data["consultant_id"] = consultant_id
                user_ref = db.collection('consultant_registrations').document(consultant_id)
            elif role == 'assistant':
                additional = ["lab_id", "assistant_id"]
                missing_add = [field for field in additional if field not in user_data or not user_data[field].strip()]
                if missing_add:
                    return jsonify({"error": "Missing fields for assistant: " + ", ".join(missing_add)}), 400
                user_ref = db.collection('assistant_registrations').document(uid)

            user_data['uid'] = uid
            user_data['firebase_id'] = uid

            user_ref.set(user_data)
            session['user_info'] = {
                'email': user_data['email'],
                'role': role,
                'uid': consultant_id if role == 'doctor' else uid,
                'full_name': user_data.get('full_name', '')
            }
            session["idToken"] = id_token if id_token else auth.create_custom_token(uid).decode('utf-8')
            logger.info(f"✅ Registered new {role} with UID: {uid}, Document ID: {consultant_id if role == 'doctor' else uid}")
            return jsonify({"success": True, "redirect": "/dashboard"})
        logger.debug("Rendering registration page (registration.html)")
        return render_template('registration.html')
    except Exception as e:
        logger.exception(f"Registration error: {e}")
        return jsonify({"error": str(e)}), 500

# --------------------------------------------------------------------
# Static File Route
# --------------------------------------------------------------------
@app.route('/static/<path:path>')
def send_static(path):
    return send_from_directory('static', path)

# --------------------------------------------------------------------
# UPLOAD IMAGE ROUTE (POST)
# --------------------------------------------------------------------
@app.route('/upload-image', methods=['POST'])
@firebase_token_required
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
        logger.debug(f"Attempting to upload file to GCS path: {storage_path}")
        
        blob = bucket.blob(storage_path)
        try:
            blob.upload_from_file(file.stream, content_type=file.content_type)
            logger.info(f"File successfully uploaded to GCS: {storage_path}")
        except Exception as e:
            logger.error(f"Failed to upload to GCS: {str(e)}", exc_info=True)
            return jsonify({"error": f"Failed to upload to GCS: {str(e)}"}), 500
        
        file_url = blob.public_url
        logger.info(f"File available at: {file_url}")
        
        return jsonify({"success": True, "filePath": storage_path})
    except Exception as e:
        logger.exception(f"Upload error: {e}")
        return jsonify({"error": str(e)}), 500

# --------------------------------------------------------------------
# PROCESS UPLOAD ROUTE (POST)
# --------------------------------------------------------------------
@app.route('/process-upload', methods=['POST'])
@firebase_token_required
def process_upload():
    temp_dir = None
    local_image_path = None
    try:
        uid = request.user.get('uid')
        form_data = request.form
        
        # Log incoming request data
        logger.debug(f"Received form data: {dict(form_data)}")
        
        language_text = form_data.get('languageText', 'kannada').lower()
        image_path = form_data.get('filePath')
        category = form_data.get('category')
        consultant_id = form_data.get('consultantId')  # Get consultantId from form data
        is_public = form_data.get('isPublic', 'true').lower() == 'true'
        patient_name = session.get('user_info', {}).get('full_name', uid)
        role = session.get('user_info', {}).get('role')

        if not image_path or not category:
            logger.error(f"Missing required fields: image_path={image_path}, category={category}")
            return jsonify({"error": "Image path and category are required"}), 400
        
        if category not in ['prescriptions', 'lab_records']:
            logger.error(f"Invalid category provided: {category}")
            return jsonify({"error": "Invalid category"}), 400

        logger.info(f"Processing upload for UID: {uid}, Category: {category}, Image Path: {image_path}, Language: {language_text}")

        # Create temp directory and download image
        temp_dir = tempfile.mkdtemp()
        local_image_path = os.path.join(temp_dir, f"temp_{uid}.jpg")
        blob = bucket.blob(image_path)
        try:
            blob.download_to_filename(local_image_path)
            logger.debug(f"Image downloaded to: {local_image_path}")
        except Exception as e:
            logger.error(f"Failed to download image from GCS: {str(e)}")
            return jsonify({"error": f"Failed to download image from GCS: {str(e)}"}), 500

        # Extract text from image
        extracted_text = extract_text_from_image(local_image_path)
        if not extracted_text:
            logger.error(f"No text extracted from image {local_image_path} with Cloud Vision.")
            return jsonify({"error": "No text could be extracted from the image with Cloud Vision"}), 400
        logger.debug(f"Extracted English text: {extracted_text[:100]}... (length: {len(extracted_text)})")

        # Get existing text for prescriptions
        existing_text = ""
        if category == 'prescriptions':
            prescriptions_query = db.collection('prescriptions') \
                .where('uid', '==', uid) \
                .order_by('timestamp', direction=firestore.Query.DESCENDING) \
                .limit(1).get()
            for doc in prescriptions_query:
                existing_text = doc.to_dict().get('professional_summary', '')
                logger.debug(f"Found existing prescription summary: {existing_text[:50]}...")
                break

        # Process with Gemini, with validation error handling
        try:
            summaries = process_text_with_gemini(
                extracted_text,
                category=category,
                language=language_text,
                patient_name=patient_name,
                existing_text=existing_text
            )
            professional_summary = summaries["professional_summary"]
            summary = summaries["regional_summary"]
            logger.debug(f"Professional Summary (English): {professional_summary[:50]}...")
            logger.debug(f"Summary ({language_text}): {summary[:50]}...")
        except ValueError as ve:
            logger.error(f"Gemini processing failed due to validation error: {str(ve)}")
            return jsonify({"error": f"Text processing failed: Validation error - {str(ve)}"}), 500
        except Exception as e:
            logger.error(f"Gemini processing failed: {str(e)}")
            return jsonify({"error": f"Text processing failed: {str(e)}"}), 500

        # Get consultant name
        consultant_name = "Unknown Consultant"
        if consultant_id:
            consultant_ref = db.collection('consultant_registrations').document(consultant_id)
            consultant_snap = consultant_ref.get()
            if consultant_snap.exists:
                consultant_name = consultant_snap.to_dict().get('full_name', "Unknown Consultant")
                logger.debug(f"Consultant name: {consultant_name}")
            else:
                logger.warning(f"No consultant found for ID: {consultant_id}")

        # Store metadata in Firestore
        doc_ref = db.collection(category).document()
        doc_id = doc_ref.id
        metadata = {
            'doc_id': doc_id,
            'uid': uid,
            'category': category,
            'original_image_path': image_path,
            'language': language_text,
            'timestamp': firestore.SERVER_TIMESTAMP,
            'extracted_text': extracted_text,
            'professional_summary': professional_summary,
            'summary': summary,
            'patient_name': patient_name
        }
        # Add consultant_id to metadata for both prescriptions and lab_records
        if consultant_id:
            metadata['consultant_id'] = consultant_id
            logger.debug(f"Added consultant_id: {consultant_id} to metadata for {category}")

        try:
            doc_ref.set(metadata)
            logger.info(f"Metadata stored in Firestore with doc_id={doc_id}")
        except Exception as e:
            logger.error(f"Failed to store metadata in Firestore: {str(e)}")
            return jsonify({"error": f"Failed to store metadata: {str(e)}"}), 500

        return jsonify({
            'success': True,
            'message': f"{category.capitalize()} processed and stored successfully. Summaries are available in the dashboard."
        })
    except Exception as e:
        logger.exception(f"Process upload error: {e}")
        return jsonify({'error': str(e)}), 500
    finally:
        # Clean up temporary files
        if local_image_path and os.path.exists(local_image_path):
            try:
                os.remove(local_image_path)
                logger.debug(f"Cleaned up temporary file: {local_image_path}")
            except Exception as e:
                logger.warning(f"Failed to remove temp file {local_image_path}: {str(e)}")
        if temp_dir and os.path.exists(temp_dir):
            try:
                os.rmdir(temp_dir)
                logger.debug(f"Cleaned up temporary directory: {temp_dir}")
            except Exception as e:
                logger.warning(f"Failed to remove temp directory {temp_dir}: {str(e)}")

# --------------------------------------------------------------------
# DASHBOARD ROUTE (GET)
# --------------------------------------------------------------------
@app.route('/dashboard', methods=['GET'])
@firebase_token_required
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
                logger.warning(f"Unexpected document ID for doctor: {consultant_id}, falling back to uid")
                consultant_id = user_data.get('consultant_id', uid)
            session['user_info']['uid'] = consultant_id
            uid = consultant_id
            logger.info(f"Updated session uid to consultant_id: {uid} for doctor (document ID: {user_snap.id})")

        logger.info(f"Loading dashboard for {email} with role: {role}, uid: {uid}")
        dashboards = {
            'patient': 'patient_dashboard.html',
            'doctor': 'consultantDashboard.html',
            'assistant': 'assistant_dashboard.html'
        }
        template = dashboards.get(role)
        if not template:
            logger.error(f"Invalid role {role} for dashboard")
            raise ValueError(f"No template found for role {role}")

        logger.debug(f"Rendering dashboard template: {template} with uid: {uid}")
        try:
            rendered = render_template(template, user_info={'uid': uid, 'email': email, 'role': role}, login_success=True)
            logger.debug("Template rendered successfully with consultantId: " + uid)
            return rendered
        except Exception as render_error:
            logger.error(f"Failed to render template {template}: {render_error}")
            raise
    except Exception as e:
        logger.exception(f"Dashboard error: {e}")
        return jsonify({"error": "Error rendering dashboard", "details": str(e)}), 500

# --------------------------------------------------------------------
# LOGOUT ROUTE
# --------------------------------------------------------------------
@app.route('/logout')
def logout():
    logger.info("User logging out")
    # Check for confirmation parameter
    if request.args.get('confirm') != 'yes':
        return render_template('confirm_logout.html', message="Are you sure you want to log out?")
    # Clear session and redirect to /login
    session.pop('user_info', None)
    session.pop('idToken', None)
    logger.info("Session cleared, redirecting to /login")
    return redirect('/login')

if __name__ == '__main__':
    port = int(os.environ.get('PORT', 5000))
    logger.info(f"Starting Flask app on port {port}")
    app.run(debug=True, port=port)