from fastapi import APIRouter, HTTPException
from app.models import ChatRequest
from app.services.firestore_service import get_chat_history, save_chat_history
from app.services.user_service import get_user_data, store_user_data
from app.services.rag_service import process_user_message

router = APIRouter()

@router.post("/chat")
async def chat(request: ChatRequest):
    """
    Handle chat interactions with the user.
    """
    session_key = request.session_key
    user_message = request.message

    # Fetch user data and chat history
    user_data = get_user_data(session_key) or {}
    chat_history = get_chat_history(session_key) or []

    # Always include the user's message in the chat history
    chat_history.append({"role": "user", "content": user_message})

    # Greet new users if the chat history is empty
    if not chat_history or len(chat_history) == 1:  # Check if only the current user message exists
        greeting = "Hi! Welcome to our mental health assistant. Let me ask you a few questions to get started."
        chat_history.append({"role": "assistant", "content": greeting})
        save_chat_history(session_key, chat_history)
        return {"response": greeting}

    # Questions for the questionnaire
    questionnaire = {
        "name": "What's your name?",
        "age": "How old are you?",
        "gender": "What is your gender?",
        "current_state": "How have you been feeling recently?",
        "history": "Do you have any history of mental health issues, treatments, or family history of such conditions?",
    }

    # Check for missing data in the user's profile
    missing_keys = [key for key in questionnaire if key not in user_data]

    # Determine the last question asked by the bot
    last_question = next(
        (entry["content"] for entry in reversed(chat_history) if entry["role"] == "assistant"), None
    )

    if missing_keys:
        # Handle the questionnaire
        current_key = missing_keys[0]
        if last_question == questionnaire[current_key]:
            # Save the user's response to the last question
            user_data[current_key] = user_message.strip()
            store_user_data(session_key, user_data)  # Save the updated data to Django backend
    
            # Check if there are more questions to ask
            remaining_keys = missing_keys[1:]  # Exclude the current question
            if remaining_keys:
                # Ask the next question
                next_question = questionnaire[remaining_keys[0]]
                chat_history.append({"role": "assistant", "content": next_question})
                save_chat_history(session_key, chat_history)
                return {"response": next_question}
            else:
                # All questions answered
                follow_up = "Thank you for answering the questions. How can I help you today?"
                chat_history.append({"role": "assistant", "content": follow_up})
                save_chat_history(session_key, chat_history)
                return {"response": follow_up}
        else:
            # Ask the current question again
            next_question = questionnaire[current_key]
            chat_history.append({"role": "assistant", "content": next_question})
            save_chat_history(session_key, chat_history)
            return {"response": next_question}


    # If all questions are answered, transition to normal chatbot flow
    if not missing_keys:
        if last_question == "Hi! Welcome to our mental health assistant. Let me ask you a few questions to get started.":
            follow_up = "Thank you for answering the questions. How can I help you today?"
            chat_history.append({"role": "assistant", "content": follow_up})
            save_chat_history(session_key, chat_history)
            return {"response": follow_up}

        # Normal chatbot flow
        ai_response = process_user_message(user_message, chat_history, user_data)
        chat_history.append({"role": "assistant", "content": ai_response})
        save_chat_history(session_key, chat_history)
        return {"response": ai_response}
