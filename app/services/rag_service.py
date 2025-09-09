from app.rag_config import get_rag_chain

def process_user_message(
    user_message: str, 
    chat_history: list, 
    user_data: dict = None, 
    max_summaries: int = 3, 
    max_doctor_summaries: int = 2,
    additional_context: str = ""
) -> str:
    """
    Process a user's message through the RAG model and return the response, using past user and doctor summaries as context.
    
    Args:
        user_message (str): The current user message.
        chat_history (list): The current session's chat history.
        user_data (dict, optional): User data including past summaries and doctor summaries.
        max_summaries (int, optional): Maximum number of past user-chatbot summaries. Defaults to 3.
        max_doctor_summaries (int, optional): Maximum number of doctor summaries. Defaults to 2.
    """
    # Determine which phase we're in
    is_onboarding_complete = user_data.get("questionnaire_completed", False) if user_data else False
    
    # Get the appropriate RAG chain for the current phase
    rag_chain = get_rag_chain(is_onboarding_complete)
    
    # Basic user context
    user_context = ""
    if user_data:
        if is_onboarding_complete:
            # Phase 2: Include full personalization context
            collected_info = []
            info_needed = user_data.get("information_needed", {})
            for key, info in info_needed.items():
                if info.get("collected", False) and info.get("value"):
                    collected_info.append(f"{key}: {info['value']}")
            
            if collected_info:
                user_context = f"Patient Profile: {', '.join(collected_info)}. Use this information to personalize your therapeutic responses."
            else:
                user_context = f"User information: {user_data}. This includes their mental state and history. Personalize your responses using this data."
        else:
            # Phase 1: Minimal context, focus on information gathering
            user_context = "Patient is currently completing onboarding process."

    # Add past user-chatbot summaries (only for therapeutic phase)
    past_summaries_context = ""
    if is_onboarding_complete and user_data and "past_summaries" in user_data:
        past_summaries = user_data["past_summaries"]
        if past_summaries:
            limited_summaries = past_summaries[-max_summaries:]
            past_summaries_context = "Past user-chatbot conversation summaries:\n"
            for summary in limited_summaries:
                past_summaries_context += (
                    f"[{summary['timestamp']}] {summary['summary']}\n"
                )

    # Add doctor summaries from user_data (only for therapeutic phase)
    doctor_summaries_context = ""
    if is_onboarding_complete and user_data and "doctor_summary" in user_data:
        doctor_summaries = user_data["doctor_summary"]
        if doctor_summaries:
            limited_doctor_summaries = doctor_summaries[-max_doctor_summaries:]
            doctor_summaries_context = "Doctor summaries:\n"
            for summary in limited_doctor_summaries:
                if isinstance(summary, dict) and "summary" in summary and "timestamp" in summary:
                    doctor_summaries_context += (
                        f"[{summary['timestamp']}] {summary['summary']}\n"
                    )
                else:
                    doctor_summaries_context += f"- {summary}\n"

    # Combine all context
    if is_onboarding_complete:
        # Phase 2: Full context for therapeutic conversations
        full_context = f"{user_context}\n{past_summaries_context}\n{doctor_summaries_context}".strip()
    else:
        # Phase 1: Onboarding context + information gathering goals
        full_context = f"{user_context}\n{additional_context}".strip()

    # Invoke the appropriate RAG chain
    result = rag_chain.invoke({
        "input": user_message,
        "chat_history": chat_history,
        "context": full_context,
    })

    return result["answer"]

# def process_user_message(user_message: str, chat_history: list, user_data: dict = None, max_summaries: int = 3, doctor_summary: list) -> str:
#     """
#     Process a user's message through the RAG model and return the response, using a limited number of past summaries as context.
    
#     Args:
#         user_message (str): The current user message.
#         chat_history (list): The current session's chat history.
#         user_data (dict, optional): User data including past summaries.
#         max_summaries (int, optional): Maximum number of past summaries to include in context. Defaults to 3.
#     """
#     # Basic user context
#     user_context = (
#         f"User information: {user_data}. "
#         "This includes their mental state and history. "
#         "Personalize your responses using this data."
#     ) if user_data else ""

#     # Add past summaries as additional context, limited by max_summaries
#     past_summaries_context = ""
#     if user_data and "past_summaries" in user_data:
#         past_summaries = user_data["past_summaries"]
#         if past_summaries:
#             # Take the most recent summaries up to max_summaries
#             limited_summaries = past_summaries[-max_summaries:]
#             past_summaries_context = "Past conversation summaries:\n"
#             for summary in limited_summaries:
#                 past_summaries_context += (
#                     f"[{summary['timestamp']}] {summary['summary']}\n"
#                 )

#     # Combine all context
#     full_context = f"{user_context}\n{past_summaries_context}".strip()

#     # Invoke the RAG chain with the combined context
#     result = rag_chain.invoke({
#         "input": user_message,
#         "chat_history": chat_history,
#         "context": full_context,
#     })

#     return result["answer"]