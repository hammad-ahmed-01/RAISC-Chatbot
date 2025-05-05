from app.rag_config import rag_chain

def process_user_message(
    user_message: str,
    chat_history: list,
    user_data: dict = None,
    max_summaries: int = 3,
    max_doctor_summaries: int = 2
) -> str:
    user_context = (
        f"User information: {user_data}. "
        "This includes their mental state and history. "
        "Personalize your responses using this data."
    ) if user_data else ""

    relevant_summaries_context = ""
    if user_data and "relevant_summaries" in user_data:
        relevant_summaries = user_data["relevant_summaries"]
        if relevant_summaries:
            relevant_summaries_context = "Most relevant past conversation summaries:\n"
            for summary in relevant_summaries:
                relevant_summaries_context += f"- {summary}\n"

    past_summaries_context = ""
    if user_data and "past_summaries" in user_data:
        past_summaries = user_data["past_summaries"]
        if past_summaries:
            limited_summaries = past_summaries[-max_summaries:]
            past_summaries_context = "Past user-chatbot conversation summaries:\n"
            for summary in limited_summaries:
                past_summaries_context += f"[{summary['timestamp']}] {summary['summary']}\n"

    doctor_summaries_context = ""
    if user_data and "doctor_summary" in user_data:
        doctor_summaries = user_data["doctor_summary"]
        if doctor_summaries:
            limited_doctor_summaries = doctor_summaries[-max_doctor_summaries:]
            doctor_summaries_context = "Doctor summaries:\n"
            for summary in limited_doctor_summaries:
                if isinstance(summary, dict) and "summary" in summary and "timestamp" in summary:
                    doctor_summaries_context += f"[{summary['timestamp']}] {summary['summary']}\n"
                else:
                    doctor_summaries_context += f"- {summary}\n"

    full_context = f"{user_context}\n{relevant_summaries_context}\n{past_summaries_context}\n{doctor_summaries_context}".strip()

    result = rag_chain.invoke({
        "input": user_message,
        "chat_history": chat_history,
        "context": full_context,
    })

    return result["answer"]
