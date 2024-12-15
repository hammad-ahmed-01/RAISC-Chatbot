from google.cloud import firestore
from app.config import PROJECT_ID

def get_firestore_client():
    """
    Create and return a Firestore client.
    """
    return firestore.Client(project=PROJECT_ID)
