from app.rag_config import rag_chain

# def process_user_message(user_message: str, chat_history: list, user_data: dict = None) -> str:
#     """
#     Process a user's message through the RAG model and return the response.
#     """
#     user_context = (
#         f"User information: {user_data}. "
#         "This includes their mental state and history. "
#         "Personalize your responses using this data."
#     ) if user_data else ""

#     result = rag_chain.invoke({
#         "input": user_message,
#         "chat_history": chat_history,
#         "context": user_context,
#     })

#     return result["answer"]

from app.rag_config import rag_chain

def process_user_message(
    user_message: str, 
    chat_history: list, 
    user_data: dict = None, 
    max_summaries: int = 3, 
    max_doctor_summaries: int = 2
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
    # Basic user context
    user_context = (
        f"User information: {user_data}. "
        "This includes their mental state and history. "
        "Personalize your responses using this data."
    ) if user_data else ""

    # Add past user-chatbot summaries
    past_summaries_context = ""
    if user_data and "past_summaries" in user_data:
        past_summaries = user_data["past_summaries"]
        if past_summaries:
            limited_summaries = past_summaries[-max_summaries:]
            past_summaries_context = "Past user-chatbot conversation summaries:\n"
            for summary in limited_summaries:
                past_summaries_context += (
                    f"[{summary['timestamp']}] {summary['summary']}\n"
                )

    # Add doctor summaries from user_data
    doctor_summaries_context = ""
    if user_data and "doctor_summary" in user_data:  # Match the key you defined
        doctor_summaries = user_data["doctor_summary"]
        if doctor_summaries:
            limited_doctor_summaries = doctor_summaries[-max_doctor_summaries:]
            doctor_summaries_context = "Doctor summaries:\n"
            for summary in limited_doctor_summaries:
                # Handle case where doctor_summary might be a list of strings or objects
                if isinstance(summary, dict) and "summary" in summary and "timestamp" in summary:
                    doctor_summaries_context += (
                        f"[{summary['timestamp']}] {summary['summary']}\n"
                    )
                else:
                    doctor_summaries_context += f"- {summary}\n"  # Fallback for plain strings

    # Combine all context
    full_context = f"{user_context}\n{past_summaries_context}\n{doctor_summaries_context}".strip()

    # Invoke the RAG chain with the combined context
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