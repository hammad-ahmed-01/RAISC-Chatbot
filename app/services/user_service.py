import requests

# DJANGO_BACKEND_URL = "https://web-production-deb22.up.railway.app"
DJANGO_BACKEND_URL = "http://localhost:8000"
def get_user_data(session_key: str) -> dict:
    """
    Fetch user data from Django backend.
    """
    try:
        response = requests.get(f"{DJANGO_BACKEND_URL}/users/patient/data/{session_key}/")
        if response.status_code == 200:
            data = response.json()
            if data.get("profile_data"):  # Ensure 'profile_data' exists in the response
                return data["profile_data"]
            return None  # Return None if 'profile_data' is not available
        return None  # Return None for non-200 status codes
    except Exception as e:
        print(f"Error fetching user data: {e}")
        return None


def get_doctor_summary(session_key: str) -> list:
    """
    Fetch doctor summary from Django backend.
    Returns a list containing the summary if available, otherwise an empty list.
    """
    try:
        response = requests.get(f"{DJANGO_BACKEND_URL}/users/patient/doctor-summary/{session_key}/")
        if response.status_code == 200:
            data = response.json()
            return [data["doctor_summary"]] if "doctor_summary" in data and data["doctor_summary"] else []
        elif response.status_code == 404:  # No doctor summary found case
            print("No doctor summary found.")
            return []
        else:
            print(f"Error fetching doctor summary: Received status code {response.status_code}")
            return []
    except Exception as e:
        print(f"Error fetching doctor summary: {e}")
        return []


def store_user_data(session_key: str, user_data: dict) -> bool:
    """
    Store user data in Django backend.
    """
    try:
        response = requests.post(
            f"{DJANGO_BACKEND_URL}/users/patient/data/{session_key}/",
            json=user_data,
        )

        return response.status_code == 200
    except Exception as e:
        print(f"Error storing user data: {e}")
        return False
