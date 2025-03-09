import os

# Firestore Configuration
PROJECT_ID = "raisc-20b3d"
COLLECTION_NAME = "chat_history"

# Directory for the RAG's Chroma DB
current_dir = os.path.dirname(os.path.abspath(__file__))
PERSISTENT_DIRECTORY = os.path.join(current_dir, "../db/chroma_db_with_metadata")

# LLM and Embedding Model Configuration
MODEL_NAME = "all-MiniLM-L6-v2"
#worth testing
# MODEL_NAME = "nomic-embed-text-v1.5"
GROQ_API_KEY = "REMOVED_GROQ_API_KEY"
