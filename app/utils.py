import os
import base64
import json
from google.cloud import firestore
from google.oauth2 import service_account
from app.config import PROJECT_ID

def get_firestore_client():
    """
    Create and return a Firestore client using credentials from env.
    """
    # Read and decode the base64-encoded credentials
    encoded_credentials = os.getenv("GOOGLE_CREDENTIALS")
    if not encoded_credentials:
        raise ValueError("GOOGLE_CREDENTIALS environment variable not set.")

    credentials_info = json.loads(base64.b64decode(encoded_credentials))
    credentials = service_account.Credentials.from_service_account_info(credentials_info)

    return firestore.Client(project=PROJECT_ID, credentials=credentials)
