# app/services/language_service.py
from langchain_groq import ChatGroq
from langchain.schema import SystemMessage, HumanMessage
from app.config import GROQ_API_KEY
import re

llm = ChatGroq(model="openai/gpt-oss-20b", groq_api_key=GROQ_API_KEY, temperature=0)

def detect_language(message: str) -> str:
    """
    Detect if the message is in Roman Urdu (Pakistani) or English.
    Returns 'roman_urdu' or 'english'
    Uses LLM-first approach for better accuracy.
    """
    # Quick checks for obvious cases first (for performance)
    message_lower = message.lower().strip()
    
    # Very obvious English patterns
    obvious_english = [
        'hello', 'hi there', 'good morning', 'good evening', 'how are you',
        'i am', 'i need', 'can you', 'please help', 'thank you', 'thanks'
    ]
    
    # Very obvious Pakistani Roman Urdu patterns
    obvious_urdu = [
        'salam', 'assalam', 'walaikum', 'assalamu alaikum', 'wa alaikum assalam',
        'allah hafiz', 'khuda hafiz', 'inshaallah', 'insha allah', 'mashallah', 'mash allah',
        'alhamdulillah', 'alhamdulillahi', 'subhanallah', 'subhan allah'
    ]
    
    # Check for obvious cases
    for phrase in obvious_english:
        if phrase in message_lower:
            print(f"Quick detection: Found obvious English phrase '{phrase}'")
            return 'english'
    
    for phrase in obvious_urdu:
        if phrase in message_lower:
            print(f"Quick detection: Found obvious Urdu phrase '{phrase}'")
            return 'roman_urdu'
    
    # For all other cases, use LLM detection (which is more accurate)
    print(f"Using LLM detection for: '{message}'")
    return detect_language_with_llm(message)

def detect_language_with_llm(message: str) -> str:
    """
    Use LLM to detect language - this is now the primary detection method
    """
    detection_prompt = f"""
    You are an expert in Pakistani Roman Urdu and English language detection.
    
    Analyze this message and determine if it's written in:
    1. "roman_urdu" - Pakistani Urdu written in Latin/Roman script
    2. "english" - Standard English
    
    PAKISTANI ROMAN URDU indicators:
    - Islamic greetings: salam, assalam, assalamu alaikum, walaikum
    - Common words: aap, main, mein, hai, hoon, hain, kya, kyun, kaise, kahan, kab
    - Urdu vocabulary: baat, pareshani, takleef, masla, samajh, chahiye, theek, sahi
    - Pakistani terms: tabiyat, mizaj, halat, ilaaj, dawai
    - Islamic phrases: inshaAllah, mashAllah, alhamdulillah, subhanAllah
    - Urdu grammar patterns and sentence structure
    
    ENGLISH indicators:
    - Standard English vocabulary and grammar
    - English sentence patterns
    - Common English phrases and idioms
    
    HINDI WORDS (classify as english since we don't support Hindi):
    - anubhav, vyakti, samasya, upchar, vyavahar, samvidhan, prabandhan
    
    SPECIAL CASES:
    - Single words like "salam" = roman_urdu
    - Islamic phrases in any form = roman_urdu  
    - Mixed language: classify by the dominant language
    - If unsure, lean towards the language with more specific indicators
    
    Message to analyze: "{message}"
    
    Think step by step:
    1. What specific words indicate the language?
    2. What is the sentence structure/grammar pattern?
    3. Are there Islamic/Pakistani cultural references?
    4. What is the overall linguistic pattern?
    
    Respond with ONLY: roman_urdu OR english
    """
    
    try:
        prompt = [
            SystemMessage(content="You are an expert Pakistani Roman Urdu and English language detection specialist. You excel at distinguishing Pakistani Roman Urdu from English and Hindi."),
            HumanMessage(content=detection_prompt)
        ]
        
        response = llm(prompt)
        detected = response.content.strip().lower()
        
        # Extract the decision
        if 'roman_urdu' in detected:
            result = 'roman_urdu'
        elif 'english' in detected:
            result = 'english'
        else:
            # Fallback - look for key indicators in the LLM response
            if any(word in message.lower() for word in ['salam', 'aap', 'main', 'hai', 'kya', 'inshaallah', 'mashallah']):
                result = 'roman_urdu'
            else:
                result = 'english'
        
        print(f"LLM detected: '{message}' -> {result}")
        return result
        
    except Exception as e:
        print(f"Error in LLM language detection: {e}")
        # Intelligent fallback based on common patterns
        message_lower = message.lower()
        
        # Strong Pakistani Urdu indicators for fallback
        strong_urdu_words = ['salam', 'aap', 'main', 'mein', 'hai', 'hoon', 'kya', 'inshaallah', 'mashallah', 'pareshani']
        urdu_count = sum(1 for word in strong_urdu_words if word in message_lower)
        
        if urdu_count > 0:
            print(f"Fallback: Found {urdu_count} Urdu indicators, classifying as roman_urdu")
            return 'roman_urdu'
        else:
            print(f"Fallback: No clear Urdu indicators, classifying as english")
            return 'english'

def get_language_context_for_prompts(language: str) -> str:
    """
    Get language-specific context for LLM prompts with strict Pakistani Urdu enforcement
    """
    if language == 'roman_urdu':
        return """
        CRITICAL LANGUAGE INSTRUCTION: Respond STRICTLY in Pakistani Roman Urdu (Urdu written in Latin script).
        
        MANDATORY PAKISTANI URDU VOCABULARY - USE THESE EXACT WORDS:
        - Problem = masla (NOT samasya)
        - Solution = hal (NOT samadhan) 
        - Person = shakhs/insaan (NOT vyakti)
        - Experience = tajurba (NOT anubhav)
        - Behavior = rawayya (NOT vyavahar)
        - Treatment = ilaaj (NOT upchar)
        - Worry = pareshani/fikar (NOT chinta)
        - Help = madad (NOT sahayata)
        - Way/Method = tarika (NOT upay)
        - Feeling = ehsas/jazbat (NOT vyatha)
        - Mind = dimagh/zehn (NOT man)
        - Health = sehat/tandrusti (NOT swasthya)
        - Reason = waja (NOT karan)
        - Beautiful = khoobsurat/haseen (NOT sundar)
        - Prosperous = khushal (NOT samriddh)
        - Best wishes = naik khuwahishat (NOT shubhkamnayein)
        - Time = waqt (NOT samay)
        
        PAKISTANI SENTENCE PATTERNS:
        - Use "aap" (you), "main" (I), "hai" (is), "hoon" (am)
        - Use "kya" (what), "kyun" (why), "kaise" (how), "kab" (when)
        - Use Pakistani expressions: "theek hai", "bilkul sahi", "bohat acha"
        
        ABSOLUTELY FORBIDDEN HINDI WORDS:
        - samasya, samadhan, vyakti, anubhav, vyavahar, upchar, chinta, sahayata, upay, vyatha
        - swasthya, prabandhan, samvidhan, vyavastha, adhyayan, pariksha, karan
        
        CULTURAL CONTEXT:
        - Use Pakistani social norms and Islamic references
        - Use respectful titles: sahib, sahiba, bhai, behen
        - Keep tone warm and respectful as per Pakistani culture
        
        VERIFICATION: Before responding, check your answer for ANY Hindi words and replace them with Pakistani Urdu equivalents.
        """
    else:
        return """
        LANGUAGE INSTRUCTION: Respond in clear, simple English.
        Use a warm, supportive tone appropriate for mental health conversations.
        Avoid overly complex vocabulary and keep sentences conversational.
        """



def get_greeting_message(language: str, user_message: str, name: str = None) -> str:
    """
    Get appropriate greeting based on language
    """
                #Prompt llm to generate a more natural greeting based on user message
    if language == 'roman_urdu':
        
            
           response=llm.invoke([
                SystemMessage(content="You are an expert Pakistani Roman Urdu mental health assistant. You excel at generating warm, culturally appropriate greetings in Pakistani Roman Urdu."),
                HumanMessage(content=f"The user said: '{user_message}'. Generate a warm, friendly greeting in Pakistani Roman Urdu that acknowledges their message and makes them feel comfortable. Use respectful language and a supportive tone. If the user's message contains any Urdu words, incorporate them naturally into the greeting. If the user's message is neutral or doesn't provide much context, use a general warm greeting. If the user's message is very short or just a greeting, respond with a friendly greeting back. If the user's message contains any emotional cues (like feeling sad, anxious, happy), acknowledge that in your greeting. If the user's message is in English, still respond in Pakistani Roman Urdu. If the user's message is very formal, keep your greeting slightly more formal but still warm and friendly. If the user's message is informal or casual, use a more casual tone in your greeting. If the user's message contains slang or colloquial terms, you can use similar casual language in your greeting. Make sure to include an introduction of yourself as a mental health assistant and express your willingness to help.\n\nIf the user's name is '{name}', include it in the greeting to make it more personal.")
            ])
           return response.content.strip()
       
            
    else:
        
        response=llm.invoke([
                SystemMessage(content="You are an expert English-speaking mental health assistant. You excel at generating warm, culturally appropriate greetings in English."),
                HumanMessage(content=f"The user said: '{user_message}'. Generate a warm, friendly greeting in English that acknowledges their message and makes them feel comfortable. Use respectful language and a supportive tone. If the user's message contains any Urdu words, do not include them in your greeting. If the user's message is neutral or doesn't provide much context, use a general warm greeting. If the user's message is very short or just a greeting, respond with a friendly greeting back. If the user's message contains any emotional cues (like feeling sad, anxious, happy), acknowledge that in your greeting. If the user's message is very formal, keep your greeting slightly more formal but still warm and friendly. If the user's message is informal or casual, use a more casual tone in your greeting. If the user's message contains slang or colloquial terms, you can use similar casual language in your greeting. Make sure to include an introduction of yourself as a mental health assistant and express your willingness to help.\n\nIf the user's name is '{name}', include it in the greeting to make it more personal.")
            ])
        return response.content.strip()
        
def get_completion_message(language: str, name: str = None) -> str:
    """
    Get questionnaire completion message
    """
    if language == 'roman_urdu':
        if name:
            return f"Bohat shukriya {name}! Ab mujhe aapki situation samajh aa gayi hai. "
        else:
            return "Bohat shukriya! Ab mujhe aapki situation samajh aa gayi hai. "
    else:
        if name:
            return f"Thank you for sharing that with me, {name}! I feel like I have a good understanding of your situation now. "
        else:
            return "Thank you for sharing that with me! I feel like I have a good understanding of your situation now. "

def get_risk_intervention_message(language: str) -> str:
    """
    Get risk intervention message
    """
    if language == 'roman_urdu':
        return (
            "Mujhe laga hai ke aap thoda pareshan hain. Shayad aapko kisi professional se baat karni chahiye. "
            "Main yaad dilana chahta hoon ke main kisi professional doctor ka replacement nahi hoon. "
            "Kya aap chahenge ke main aapko koi resources ya professional help dhoondhne mein madad karun?"
        )
    else:
        return (
            "I've noticed that our conversation seems to reflect some distress. "
            "It might be helpful to consider speaking with a mental health professional. "
            "Please remember, I'm not a substitute for professional advice. "
            "Would you like some resources or help finding support?"
        )
