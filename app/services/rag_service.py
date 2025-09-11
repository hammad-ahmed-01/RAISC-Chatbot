from app.rag_config import rag_chain

def process_user_message(
    user_message: str, 
    chat_history: list, 
    user_data: dict = None, 
    max_summaries: int = 3, 
    max_doctor_summaries: int = 2,
    additional_context: str = ""
) -> str:
    """
    Process a user's message through the unified RAG model.
    
    Args:
        user_message (str): The current user message.
        chat_history (list): The current session's chat history.
        user_data (dict, optional): User data including past summaries and doctor summaries.
        max_summaries (int, optional): Maximum number of past user-chatbot summaries. Defaults to 3.
        max_doctor_summaries (int, optional): Maximum number of doctor summaries. Defaults to 2.
        additional_context (str, optional): Additional context to include in the prompt.
    """
    
    # Basic user context - include collected information if available
    user_context = ""
    if user_data:
        # Include collected information for personalization
        collected_info = []
        info_needed = user_data.get("information_needed", {})
        for key, info in info_needed.items():
            if info.get("collected", False) and info.get("value"):
                collected_info.append(f"{key}: {info['value']}")
        
        if collected_info:
            user_context = f"Known about this person: {', '.join(collected_info)}. Use this to personalize your support."
        else:
            user_context = "Getting to know this person to provide better support."

    # Add past user-chatbot summaries
    past_summaries_context = ""
    if user_data and "past_summaries" in user_data:
        past_summaries = user_data["past_summaries"]
        if past_summaries:
            limited_summaries = past_summaries[-max_summaries:]
            past_summaries_context = "Past conversation summaries:\n"
            for summary in limited_summaries:
                past_summaries_context += (
                    f"[{summary['timestamp']}] {summary['summary']}\n"
                )

    # Add doctor summaries from user_data
    doctor_summaries_context = ""
    if user_data and "doctor_summary" in user_data:
        doctor_summaries = user_data["doctor_summary"]
        if doctor_summaries:
            limited_doctor_summaries = doctor_summaries[-max_doctor_summaries:]
            doctor_summaries_context = "Professional notes:\n"
            for summary in limited_doctor_summaries:
                if isinstance(summary, dict) and "summary" in summary and "timestamp" in summary:
                    doctor_summaries_context += (
                        f"[{summary['timestamp']}] {summary['summary']}\n"
                    )
                else:
                    doctor_summaries_context += f"- {summary}\n"

    # Combine all context including additional context
    full_context = f"{user_context}\n{past_summaries_context}\n{doctor_summaries_context}\n{additional_context}".strip()

    # Invoke the unified RAG chain
    result = rag_chain.invoke({
        "input": user_message,
        "chat_history": chat_history,
        "context": full_context,
    })

    return result["answer"]