import numpy as np
import os
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
from langchain_openai import ChatOpenAI
from dotenv import load_dotenv

load_dotenv()
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

# ---------------------------------------------------------
# 1. Your LLM (for fallback classification)
# ---------------------------------------------------------
llm = ChatOpenAI(model="gpt-4o-mini", api_key=OPENAI_API_KEY, temperature=1)

# ---------------------------------------------------------
# 2. Load OPEN SOURCE embedding model
# ---------------------------------------------------------
# Recommended: all-mpnet-base-v2 (state-of-the-art semantic similarity)
print("Going for embedding model")
embedding_model = SentenceTransformer("sentence-transformers/all-mpnet-base-v2")
print("Got embedding model")

# ---------------------------------------------------------
# 3. Embedding utilities
# ---------------------------------------------------------
def embed(text: str):
    """Compute embedding using open-source model."""
    return embedding_model.encode(text, convert_to_numpy=True)

def cosine_sim(vec1, vec2):
    """Cosine similarity from sklearn."""
    return float(cosine_similarity([vec1], [vec2])[0][0])

# ---------------------------------------------------------
# 5. Main Relevance Score Function
# ---------------------------------------------------------
def relevance_score(
    question: str,
    user_msg: str,
    borderline_low=0.30,
    borderline_high=0.60
):
    """
    Returns:
    {
        'embedding_similarity': float,
        'keyword_similarity': float or None,
        'final_label': 'ON_TOPIC' | 'OFF_TOPIC'
    }
    """

    # STEP 1: Embedding similarity
    q_emb = embed(question)
    u_emb = embed(user_msg)
    emb_sim = cosine_sim(q_emb, u_emb)

    # STEP 2: Expected-answer semantic similarity
    keyword_sim = None
    expected_keywords = ["alright", "unwell", "theek", "sad", "tabiyat", "kharab"]
    if expected_keywords:
        vectors = [embed(k) for k in expected_keywords]
        centroid = np.mean(vectors, axis=0)
        keyword_sim = cosine_sim(centroid, u_emb)

    # STEP 3: Decision logic

    # Strong match → ON_TOPIC
    if emb_sim > borderline_low:
        return {
            "embedding_similarity": emb_sim,
            "keyword_similarity": keyword_sim,
            "final_label": "ON_TOPIC"
        }

    # Very weak → OFF_TOPIC
    if emb_sim <= borderline_low:
        return {
            "embedding_similarity": emb_sim,
            "keyword_similarity": keyword_sim,
            "final_label": "OFF_TOPIC"
        }
