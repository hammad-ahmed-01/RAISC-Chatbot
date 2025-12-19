# app/services/language_service.py
from langchain_groq import ChatGroq
from langchain_core.messages import SystemMessage, HumanMessage
from langchain_openai import ChatOpenAI
import re
import os
from langchain_openai import ChatOpenAI
from dotenv import load_dotenv

load_dotenv()
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")


llm = ChatOpenAI(model="gpt-4o-mini", api_key=OPENAI_API_KEY, temperature=1)

ROMAN_URDU_MARKERS = {
    "salam","assalam","walaikum","allah","khuda","inshaallah","mashaallah",
    "ap","aap","tum","mein","main","mai","hum","ham","wo","woh","ye","yahan",
    "wahan","idhar","udhar","kuch","kisi","kis","kon","kaun","kya","kia","kyun",
    "q","ky","tha","thi","the","thy","hoga","hogaya","hogayi","hogaye","hora",
    "horaha","horahi","horahay","hoon","hun","hain","hai","tha","tha",
    "acha","achha","bhai","behan","bhaiya","jan","jaan","yar","yaar",
    "bohat","bahut","zyada","kam","thoda","thora","aise","waise","kaisa",
    "kaise","kesi","kesy","sahi","galat","theek","thik","masla","masail",
    "zindagi","dunya","insan","log","sab","har","kisi","koi","shukriya",
    "meh","meri","mera","mere","tera","teri","tere","hamari","hamara","hamare",
    "paisa","paise","ghareeb","ameer","khushi","gham","dard","takleef",
    "mushkil","asan","asani","mushkilat","dar","khauf","tanhai","sakoon",
    "shanti","fikr","tension","parishan","parishani",
    "madad","dua","ibadat","namaz","roza","deen","islam","muslim",
    "quran","masjid","allahumma","alhamdulillah","subhanallah","astaghfirullah",
    "jazakallah","rizq","sabr","shukar","shukr","khair",
    "ghar","kamra","kamre","gharwalay","makan","chhat","darwaza","khidki",
    "dost","dosti","rishta","rishtay","mohabbat","pyar","ishq",
    "roti","khana","pani","chai","chaiye","khareedna","lena","dena",
    "school","class","padhai","parhai","kitab","ustad","madrasa",
    "kaam","job","nokri","mehnat","salary","tankhwa",
    "beemar","tabiyat","dawa","ilaaj","sehat",
    "mujhe","tujhe","use","usse","unko","inko","humko","aapko","mujsay",
    "tumse","uske","iske","unke","merey","tumhare","aapke",
    "aaj","kal","abhi","baad","pehle","der","raat","subah","din","sham",
    "jaldi","dhair","barish","garmi","sardi","hawaa","dhoop","andhera",
    "gaari","bus","rickshaw","safar","raasta","manzil",
    "phone","message","msg","baat","batana","sun","suno","suni",
    "samjha","samjhi","samjhay","soch","socha","sochna",
    "ha","haan","na","nahi","nhi","ji","jee","chalo","theekhai", "kabhi", "shayad", "pata"
}

def detect_language(message: str):
    """
    NO LLM language detection.
    Simply:
    - If >=2 Roman Urdu markers → classify as Roman Urdu
    - Otherwise → classify as English
    - If Roman Urdu → translate to English (LLM)
    
    Returns:
        {
            "language": "roman_urdu" | "english",
            "normalized_text": "<english translation or original>"
        }
    """

    msg = message.lower()

    # ---------------------------------------
    # 1. Heuristic detection of Roman Urdu
    # ---------------------------------------
    # Count matches using whole words only
    hits = 0
    for w in ROMAN_URDU_MARKERS:
        if re.search(rf"\b{re.escape(w)}\b", msg):
            hits += 1

    if hits >= 1:
        lang = "roman_urdu"
    else:
        lang = "english"

    # ---------------------------------------
    # 2. Translation ONLY if Roman Urdu
    # ---------------------------------------
    if lang == "roman_urdu":
        translate_prompt = [
            SystemMessage(content="Translate Roman Urdu into English."),
            HumanMessage(content=f"Translate this Roman Urdu sentence into English:\n\n{message}")
        ]
        try:
            translation = llm.invoke(translate_prompt).content.strip()
            normalized_text = translation
        except:
            normalized_text = message  # fallback if translation fails
    else:
        normalized_text = message

    return {
        "language": lang,
        "normalized_text": normalized_text
    }
        
def get_greeting_message(name: str = None) -> str:
    """
    Get appropriate greeting based on language
    """
    if name:
        return f"Hello {name}! I'm your mental health assistant. I'm here to support you through whatever you're going through! Let's start with a few questions okay?"
    else:
        return "Hi! I'm your mental health assistant. I'm here to support you through whatever you're going through! Let's start with a few questions okay?"
    
def get_information_prompts(language: str) -> dict:
    """
    Get information gathering prompts in appropriate language
    """
    return {
            'current_condition': "How are you feeling right now? What's your current emotional state?",
            'duration': "How long have you been feeling this way?",
            'mental_health_history': "Do you have any history of mental health issues?",
            # 'current_condition': "How are you feeling right now? What's your current emotional state?",
            'physical_activity': "Are you physically active? Do you do exercise of any kind?",
        }
        
def get_completion_message(name: str = None) -> str:
    """
    Get questionnaire completion message
    """
    if name:
        return f"Thank you for sharing that with me, {name}! I feel like I have a good understanding of your situation now. "
    else:
        return "Thank you for sharing that with me! I feel like I have a good understanding of your situation now. "

def get_risk_intervention_message(language: str) -> str:
    """
    Get risk intervention message
    """
    return (
        "I've noticed that our conversation seems to reflect some distress. "
        "It might be helpful to consider speaking with a mental health professional. "
        "Please remember, I'm not a substitute for professional advice. "
        "Would you like some resources or help finding support?"
    )

def get_session_end_message(language: str) -> str:
    """
    Get session ending message
    """
    return "Here is your session summary:"

def get_error_message(language: str) -> str:
    """
    Get error message
    """
    return "I'm sorry, I'm experiencing some technical difficulties. Please try again in a moment."


if __name__ == "__main__":
    text = detect_language("I do not work out")
    print (text)