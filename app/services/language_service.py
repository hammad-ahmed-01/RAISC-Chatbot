# app/services/language_service.py
from langchain_groq import ChatGroq
from langchain.schema import SystemMessage, HumanMessage
from app.config import GROQ_API_KEY
import re

llm = ChatGroq(model="openai/gpt-oss-20b", groq_api_key=GROQ_API_KEY, temperature=0)

def get_language_context_for_prompts(language: str) -> str:
    """
    Get language-specific context for LLM prompts with strict ENGLISH Only enforcement
    """
    return """
    LANGUAGE INSTRUCTION: Respond in clear, simple ENGLISH ONLY.
    Use a warm, supportive tone appropriate for mental health conversations.
    Avoid overly complex vocabulary and keep sentences conversational.
    """

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