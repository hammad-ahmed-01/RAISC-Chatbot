# app/services/questionnaire_service.py

import requests
from datetime import datetime, timedelta
from app.config import DJANGO_BACKEND_URL, QUESTIONNAIRE_CACHE_TTL
import logging

logger = logging.getLogger(__name__)

# In-memory cache
_questionnaire_cache = {
    "data": None,
    "loaded_at": None,
}

# Hardcoded fallback questionnaire (used if Django API fails)
DEFAULT_REQUIRED_INFORMATION = {
    "current_condition": {
        "collected": False,
        "value": None,
        "required": True,
        "description": "how the user is feeling currently"
    },
    "duration": {
        "collected": False,
        "value": None,
        "required": True,
        "description": "how long has the user been feeling like this"
    },
    "mental_health_history": {
        "collected": False,
        "value": None,
        "required": True,
        "description": "user's mental health history or family history"
    },
    "physical_activity": {
        "collected": False,
        "value": None,
        "required": True,
        "description": "Does the user perform any sort of physical exercise"
    },
    "suicidal_thoughts": {
        "collected": False,
        "value": None,
        "required": True,
        "description": "Whether the user has experienced suicidal thoughts"
    }
}


def is_cache_valid() -> bool:
    """Check if the cached questionnaire is still valid"""
    if not _questionnaire_cache["data"] or not _questionnaire_cache["loaded_at"]:
        return False
    
    time_elapsed = (datetime.now() - _questionnaire_cache["loaded_at"]).total_seconds()
    return time_elapsed < QUESTIONNAIRE_CACHE_TTL


def fetch_questionnaire_from_django(session_key: str) -> dict:
    """
    Fetch questionnaire configuration from Django API
    Returns the fields dict or None if fetch fails
    """
    try:
        url = f"{DJANGO_BACKEND_URL}/api/questionnaire/config"
        headers = {
            "Authorization": f"Token {session_key}",
            "Content-Type": "application/json"
        }
        
        logger.info(f"Fetching questionnaire from Django: {url}")
        
        response = requests.get(url, headers=headers, timeout=5)
        response.raise_for_status()
        
        data = response.json()
        
        if data.get("success") and data.get("data") and data["data"].get("fields"):
            logger.info("Successfully fetched questionnaire from Django")
            return data["data"]["fields"]
        else:
            logger.warning(f"Django API returned success=false or missing fields: {data}")
            return None
            
    except requests.Timeout:
        logger.error("Django API request timed out")
        return None
    except requests.RequestException as e:
        logger.error(f"Error fetching questionnaire from Django: {e}")
        return None
    except (KeyError, ValueError) as e:
        logger.error(f"Error parsing Django API response: {e}")
        return None


def normalize_questionnaire_format(django_fields: dict) -> dict:
    """
    Convert Django API format to FastAPI format
    Django returns: {"field_key": {"required": true, "description": "..."}}
    FastAPI needs: {"field_key": {"collected": false, "value": null, "required": true, "description": "..."}}
    """
    normalized = {}
    
    for field_key, field_data in django_fields.items():
        normalized[field_key] = {
            "collected": False,  # Always false in template
            "value": None,       # Always null in template
            "required": field_data.get("required", True),
            "description": field_data.get("description", "")
        }
    
    return normalized


def load_required_information(session_key: str) -> dict:
    """
    Load questionnaire configuration with caching and fallback
    
    Flow:
    1. Check cache - if valid, return cached data
    2. Try to fetch from Django API
    3. If Django fails, use default fallback
    4. Cache successful fetches
    
    Returns: REQUIRED_INFORMATION dict
    """
    
    # Check cache first
    if is_cache_valid():
        logger.info("Using cached questionnaire")
        return _questionnaire_cache["data"]
    
    # Cache expired or empty, try to fetch from Django
    logger.info("Cache expired, fetching fresh questionnaire from Django")
    django_fields = fetch_questionnaire_from_django(session_key)
    
    if django_fields:
        # Normalize to FastAPI format
        questionnaire = normalize_questionnaire_format(django_fields)
        
        # Update cache
        _questionnaire_cache["data"] = questionnaire
        _questionnaire_cache["loaded_at"] = datetime.now()
        
        logger.info(f"Cached questionnaire with {len(questionnaire)} fields")
        return questionnaire
    else:
        # Django API failed, use default fallback
        logger.warning("Django API failed, using default questionnaire fallback")
        
        # Don't cache the fallback - try Django again next time
        return DEFAULT_REQUIRED_INFORMATION


def clear_questionnaire_cache():
    """
    Clear the questionnaire cache
    Useful if you need to force a refresh
    """
    _questionnaire_cache["data"] = None
    _questionnaire_cache["loaded_at"] = None
    logger.info("Questionnaire cache cleared")


def get_cache_status() -> dict:
    """
    Get current cache status for debugging
    """
    if not _questionnaire_cache["loaded_at"]:
        return {
            "cached": False,
            "loaded_at": None,
            "age_seconds": None,
            "ttl_seconds": QUESTIONNAIRE_CACHE_TTL,
            "fields_count": 0
        }
    
    age = (datetime.now() - _questionnaire_cache["loaded_at"]).total_seconds()
    
    return {
        "cached": True,
        "loaded_at": _questionnaire_cache["loaded_at"].isoformat(),
        "age_seconds": age,
        "ttl_seconds": QUESTIONNAIRE_CACHE_TTL,
        "is_valid": age < QUESTIONNAIRE_CACHE_TTL,
        "fields_count": len(_questionnaire_cache["data"]) if _questionnaire_cache["data"] else 0
    }