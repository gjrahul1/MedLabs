from google.cloud import vision
from google.oauth2 import service_account

def extract_text_from_image(local_image_path):
    """
    Uses Google Cloud Vision to perform text detection on the image file located at local_image_path.
    Returns the extracted text as a string.
    """
    # Path to your Google Cloud service account JSON key file
    creds_path = "./Cred/Med_Labs_GC/med-labs-42f13-b311d40cb690.json"  
    credentials = service_account.Credentials.from_service_account_file(creds_path)

    # Initialize the Vision client with explicit credentials
    vision_client = vision.ImageAnnotatorClient(credentials=credentials)

    # Read the image file
    with open(local_image_path, "rb") as image_file:
        content = image_file.read()
    
    image = vision.Image(content=content)
    response = vision_client.text_detection(image=image)
    
    if response.error.message:
        raise Exception(f"Vision API error: {response.error.message}")
    
    extracted_text = response.text_annotations[0].description if response.text_annotations else ""
    return extracted_text