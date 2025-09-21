# app/services/rag_service.py - FIXED to use both RAG chains properly

# Import both RAG chains from updated rag_config
from app.rag_config import (
    english_rag_chain,      # English RAG chain
    pakistani_rag_chain,    # Pakistani RAG chain (NEW!)
    retriever               # Shared retriever (if needed for manual retrieval)
)
from app.services.language_service import get_language_context_for_prompts
from langchain_core.messages import AIMessage

def _to_text(result):
    # plain prompt chains usually return AIMessage
    if isinstance(result, AIMessage):
        return result.content
    # old retrieval/stuff chains return dicts
    if isinstance(result, dict):
        # try common keys in older LangChain chains
        for k in ("answer", "output_text", "result"):
            if k in result:
                return result[k]
        return str(result)
    # fallback
    return str(result)

def process_user_message(
    user_message: str, 
    chat_history: list, 
    user_data: dict = None, 
    max_summaries: int = 3, 
    max_doctor_summaries: int = 2,
    additional_context: str = "",
    language: str = "english"
) -> str:
    """
    Process user message through appropriate RAG chain based on language
    BOTH English and Roman Urdu now use full RAG processing!
    """
    
    print(f"Processing message in {language} using full RAG chain")
    
    if language == "english":
        return process_via_english_rag_chain(
            user_message, chat_history, user_data, 
            max_summaries, max_doctor_summaries, additional_context
        )
    
    elif language == "roman_urdu":  
        return process_via_pakistani_rag_chain(
            user_message, chat_history, user_data,
            max_summaries, max_doctor_summaries, additional_context
        )
    
    else:
        # Fallback to English
        return process_via_english_rag_chain(
            user_message, chat_history, user_data,
            max_summaries, max_doctor_summaries, additional_context
        )

def process_via_english_rag_chain(
    user_message: str, 
    chat_history: list, 
    user_data: dict = None, 
    max_summaries: int = 3, 
    max_doctor_summaries: int = 2,
    additional_context: str = ""
) -> str:
    """
    Process English messages through English RAG chain
    """
    
    # Build user context
    user_context = ""
    if user_data:
        collected_info = []
        info_needed = user_data.get("information_needed", {})
        for key, info in info_needed.items():
            if info.get("collected", False) and info.get("value"):
                collected_info.append(f"{key}: {info['value']}")
        
        if collected_info:
            user_context = f"Known about this person: {', '.join(collected_info)}. Use this to personalize your support."
        else:
            user_context = "Getting to know this person to provide better support."

    # Add past summaries
    past_summaries_context = ""
    if user_data and "past_summaries" in user_data:
        past_summaries = user_data["past_summaries"]
        if past_summaries:
            limited_summaries = past_summaries[-max_summaries:]
            past_summaries_context = "Past conversation summaries:\n"
            for summary in limited_summaries:
                past_summaries_context += f"[{summary['timestamp']}] {summary['summary']}\n"

    # Add doctor summaries
    doctor_summaries_context = ""
    if user_data and "doctor_summary" in user_data:
        doctor_summaries = user_data["doctor_summary"]
        if doctor_summaries:
            limited_doctor_summaries = doctor_summaries[-max_doctor_summaries:]
            doctor_summaries_context = "Professional notes:\n"
            for summary in limited_doctor_summaries:
                if isinstance(summary, dict) and "summary" in summary and "timestamp" in summary:
                    doctor_summaries_context += f"[{summary['timestamp']}] {summary['summary']}\n"
                else:
                    doctor_summaries_context += f"- {summary}\n"

    # Get English language context
    language_context = get_language_context_for_prompts("english")
    
    # Combine all context
    full_context = f"{user_context}\n{past_summaries_context}\n{doctor_summaries_context}\n{additional_context}".strip()

    # Use English RAG chain (full RAG processing: retrieval → context → generation)
    result = english_rag_chain.invoke({
        "input": user_message,
        "chat_history": chat_history,
        "context": full_context,
        "language_context": language_context,
    })

    return _to_text(result)

def process_via_pakistani_rag_chain(
    user_message: str,
    chat_history: list, 
    user_data: dict = None,
    max_summaries: int = 3,
    max_doctor_summaries: int = 2, 
    additional_context: str = ""
) -> str:
    """
    Process Roman Urdu messages through Pakistani RAG chain
    NOW USES FULL RAG PROCESSING like English!
    """
    
    # Build user context in Roman Urdu style
    user_context = ""
    if user_data:
        collected_info = []
        info_needed = user_data.get("information_needed", {})
        for key, info in info_needed.items():
            if info.get("collected", False) and info.get("value"):
                collected_info.append(f"{key}: {info['value']}")
        
        if collected_info:
            user_context = f"Is shakhs ke baare mein maloom hai: {', '.join(collected_info)}. Is information se personalized support dijiye."
        else:
            user_context = "Is shakhs ko jaanne ki koshish kar rahe hain taake behtar support de sakein."

    # Add past summaries in Roman Urdu style
    past_summaries_context = ""
    if user_data and "past_summaries" in user_data:
        past_summaries = user_data["past_summaries"]
        if past_summaries:
            limited_summaries = past_summaries[-max_summaries:]
            past_summaries_context = "Pichli conversations ka summary:\n"
            for summary in limited_summaries:
                past_summaries_context += f"[{summary['timestamp']}] {summary['summary']}\n"

    # Add doctor summaries in Roman Urdu style
    doctor_summaries_context = ""
    if user_data and "doctor_summary" in user_data:
        doctor_summaries = user_data["doctor_summary"]
        if doctor_summaries:
            limited_doctor_summaries = doctor_summaries[-max_doctor_summaries:]
            doctor_summaries_context = "Doctor ke professional notes:\n"
            for summary in limited_doctor_summaries:
                if isinstance(summary, dict) and "summary" in summary and "timestamp" in summary:
                    doctor_summaries_context += f"[{summary['timestamp']}] {summary['summary']}\n"
                else:
                    doctor_summaries_context += f"- {summary}\n"

    # Get Pakistani language context
    language_context = get_language_context_for_prompts("roman_urdu")
    
    # Combine all contexts
    full_context = f"{user_context}\n{past_summaries_context}\n{doctor_summaries_context}\n{additional_context}".strip()

    # 🎯 KEY FIX: Use Pakistani RAG chain for FULL RAG processing
    # This now does: retrieval → context → Pakistani LLM generation
    result = pakistani_rag_chain.invoke({
        "input": user_message,
        "chat_history": chat_history,
        "context": full_context,
        "language_context": language_context,
    })

    return _to_text(result)

# Backwards compatibility function (if needed)
def process_user_message_legacy(user_message, chat_history, user_data=None, max_summaries=3, max_doctor_summaries=2, additional_context=""):
    """
    Legacy function for backwards compatibility - defaults to English
    """
    return process_user_message(
        user_message=user_message,
        chat_history=chat_history,
        user_data=user_data,
        max_summaries=max_summaries,
        max_doctor_summaries=max_doctor_summaries,
        additional_context=additional_context,
        language="english"
    )
