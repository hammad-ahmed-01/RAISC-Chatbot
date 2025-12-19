# app/services/rag_service.py - FIXED to use both RAG chains properly

# Import both RAG chains from updated rag_config
from app.rag_config import (
    english_rag_chain,      # English RAG chain
    retriever               # Shared retriever (if needed for manual retrieval)
)

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
    detected_language: str = "english",
    max_summaries: int = 3, 
    max_doctor_summaries: int = 2,
    additional_context: str = ""
) -> str:
    """
    Process user message through appropriate RAG chain based on language
    BOTH English and Roman Urdu now use full RAG processing!
    """
    
    print(f"Processing message using full RAG chain")
    
    return process_via_english_rag_chain(
        user_message, chat_history, user_data, detected_language,
        max_summaries, max_doctor_summaries, additional_context
    )

def process_via_english_rag_chain(
    user_message: str, 
    chat_history: list, 
    user_data: dict = None,
    detected_language: str = "english",
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

    # Get English/roman urdr language context
    #language_context = get_language_context_for_prompts(detected_language)

    
    # Combine all context
    full_context = f"{user_context}\n{past_summaries_context}\n{doctor_summaries_context}\n{additional_context}".strip()

    # Use English RAG chain (full RAG processing: retrieval → context → generation)
    result = english_rag_chain.invoke({
        "input": user_message,
        "chat_history": chat_history,
        "context": full_context,
        # "language_context": language_context
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
        additional_context=additional_context
    )
