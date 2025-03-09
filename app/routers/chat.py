from fastapi import APIRouter, HTTPException
from app.models import ChatRequest
from app.services.firestore_service import get_chat_history, save_chat_history
from app.services.user_service import get_user_data, store_user_data
from app.services.rag_service import process_user_message

import nltk
from nltk.sentiment.vader import SentimentIntensityAnalyzer
import json
from transformers import pipeline


nltk.download('vader_lexicon')

router = APIRouter()
# Initializing the VADER analyzer once 
analyzer = SentimentIntensityAnalyzer()
# Initializing the summarization pipeline (ensure transformers is installed and configured)
summarizer = pipeline("summarization")

# Steps for emotion analysis & then summarization of the convo
# Step 1: Collect all message contents into one text string.
# Step 2: Passing that text to the summarization model to produce a concise summary.
# Step 3: Running emotional analysis on the chat history and gettt key metrics (average, min, and max comp score).
# Step 4: Combine both pieces of information into a dict output that provides summary of the conversation and
# the emotion throughout the session.

# Function to get SUMMARY of the whole convo texts
def summarize_conversation(chat_history):
    # Combine only the text of the messages (ignoring role or other metadata)
    conversation_text = " ".join([msg["content"] for msg in chat_history])
    # Adjust parameters such as max_length/min_length as needed for your conversations
    summary_output = summarizer(conversation_text, max_length=500, min_length=50, do_sample=False)
    return summary_output[0]['summary_text']

# Function to get sentiment for EACH message individually
def analyze_sentiment(message: str) -> dict:
    return analyzer.polarity_scores(message)

#    function to get the overall metrics of the WHOLE convo
def analyze_emotions(chat_history):
    # Filter out user messages that include a sentiment score
    user_sentiments = [
        msg["sentiment"] for msg in chat_history
        if msg.get("role") == "user" and "sentiment" in msg
    ]
    
    if user_sentiments:
        compound_scores = [score.get("compound", 0) for score in user_sentiments]
        avg_compound = sum(compound_scores) / len(compound_scores)
        # max_compound = max(compound_scores)
        # min_compound = min(compound_scores)
    else:
        avg_compound = max_compound = min_compound = 0

    return {
        "average_compound": avg_compound,
        # "max_compound": max_compound,
        # "min_compound": min_compound,
        # "individual_scores": user_sentiments,
        # "num_messages": len(user_sentiments)
    }

def generate_conversation_summary(chat_history):
    
    # Generate summary using the summarization model
    summary_text = summarize_conversation(chat_history)
    
    # Analyze emotional data from the conversation
    emotional_info = analyze_emotions(chat_history)
    
    # Derive an overall tone based on the average compound score
    if emotional_info["num_messages"] >= 0:
        if emotional_info["average_compound"] > 0.1:
            overall_emotion = "positive"
        elif emotional_info["average_compound"] < -0.1:
            overall_emotion = "negative"
        else:
            overall_emotion = "neutral"
    else:
        overall_emotion = "neutral"
    
    emotional_summary = (
        f"The conversation included {emotional_info['num_messages']} user messages. "
        f"Average compound sentiment score was {emotional_info['average_compound']:.2f} "
        f"(min: {emotional_info['min_compound']:.2f}, max: {emotional_info['max_compound']:.2f}) "
        f"indicating an overall {overall_emotion} tone. "
    )
    
    # Combine the generated summary and emotional analysis into a JSON-compatible dict
    # return {
    #     "conversation_summary": summary_text,
    #     "emotional_info": emotional_info,
    #     "emotional_summary": emotional_summary
    # }
    # Return the JSON string (or the dict if your framework automatically serializes it)
    # print "Summary"
    # return json.dumps(summary_data, indent=4)

    # return f"conversation_summary: {summary_text}, emotional_info: {emotional_info}, emotional_summary: {emotional_summary}"
    return "Summary Saved"
    # return "Summary saved"
    
    

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

    # Analyze sentiment of the user's message in real-time
    # sentiment_scores = analyze_sentiment(user_message)
    # Always include the user's message in the chat history
    chat_history.append({"role": "user", "content": user_message, "sentiment": analyze_sentiment(user_message)})

    
    # Check if the user wants to end the session
    if user_message.lower() in ["end session", "goodbye", "exit", "bye", "end"]:
        # Generate the conversation summary including emotional analysis
        summary = generate_conversation_summary(chat_history)
        # Optionally, append the summary as an assistant message in the chat history
        chat_history.append({
            "role": "assistant",
            "content": "Here is your session summary:\n" + summary,
        })
        save_chat_history(session_key, chat_history)
        return {"response": summary}

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
