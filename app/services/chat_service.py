# app/services/chat_service.py
import nltk
from nltk.sentiment.vader import SentimentIntensityAnalyzer
from datetime import datetime, timedelta
from langchain_groq import ChatGroq
from langchain.schema import SystemMessage, HumanMessage
from app.services.firestore_service import get_chat_history, save_chat_history
from app.services.user_service import get_user_data, store_user_data, get_doctor_summary
from app.services.rag_service import process_user_message
from dotenv import load_dotenv
load_dotenv()
import os

nltk.download('vader_lexicon')

# Initialize the VADER analyzer and LLM once
analyzer = SentimentIntensityAnalyzer()
llm = ChatGroq(model="llama-3.3-70b-versatile", groq_api_key=os.environ.get("GROQ_API_KEY"), temperature=0)

# Inactivity threshold
INACTIVITY_THRESHOLD = timedelta(minutes=1)

# Function to get sentiment for each message individually
def analyze_sentiment(message: str) -> dict:
    return analyzer.polarity_scores(message)

# Function to get the overall metrics of the current session
def analyze_emotions(chat_history, session_start_index):
    user_sentiments = [
        msg["sentiment"] for msg in chat_history[session_start_index:]
        if msg.get("role") == "user" and "sentiment" in msg
    ]
    
    if user_sentiments:
        compound_scores = [score.get("compound", 0) for score in user_sentiments]
        avg_compound = sum(compound_scores) / len(compound_scores) if compound_scores else 0
        min_compound = min(compound_scores) if compound_scores else 0
        max_compound = max(compound_scores) if compound_scores else 0
        num_messages = len(user_sentiments)
    else:
        avg_compound = min_compound = max_compound = 0
        num_messages = len([msg for msg in chat_history[session_start_index:] if msg.get("role") == "user"])
        
    return {
        "average_compound": avg_compound,
        "min_compound": min_compound,
        "max_compound": max_compound,
        "num_messages": num_messages
    }

# Incrementally update the sentiment aggregate for the current session
def update_sentiment_aggregate(agg: dict, new_score: float) -> dict:
    agg["sum"] += new_score
    agg["count"] += 1
    agg["average"] = agg["sum"] / agg["count"] if agg["count"] > 0 else 0
    if agg["count"] == 1:
        agg["min"] = new_score
        agg["max"] = new_score
    else:
        agg["min"] = min(agg.get("min", new_score), new_score)
        agg["max"] = max(agg.get("max", new_score), new_score)
    return agg

# Reset sentiment aggregate for a new session
def reset_sentiment_aggregate() -> dict:
    return {"sum": 0.0, "count": 0, "average": 0.0, "min": 0.0, "max": 0.0}

# Check risk using the session-specific aggregate
def check_risk(agg: dict, threshold: float = -0.1) -> bool:
    return agg["average"] < threshold

# Generate a summary and emotional analysis for the current session
def generate_conversation_summary(chat_history, session_start_index, previous_summary=None):
    user_messages = [msg["content"] for msg in chat_history[session_start_index:] if msg.get("role") == "user"]
    if not user_messages:
        summary_text = "No messages to summarize in this session."
    else:
        conversation_text = " ".join(user_messages)
        prompt_text = (
            "You are a helpful assistant that summarizes conversations. "
            "Summarize the following conversation concisely, focusing on the key points and overall tone. "
            "If a previous summary is provided, relate the new summary to it, noting any changes, continuations, or new topics. "
            "Otherwise just summarize the available conversation."
        )
        if previous_summary:
            prompt_text += f"\nPrevious summary: {previous_summary}\n"
        prompt_text += f"Current conversation: {conversation_text}"

        prompt = (
            SystemMessage(content=prompt_text),
            HumanMessage(content="")
        )
        summary_output = llm(prompt)
        summary_text = summary_output.content.strip()

    emotions = analyze_emotions(chat_history, session_start_index)
    num_messages = emotions["num_messages"]
    avg_compound = emotions["average_compound"]
    min_compound = emotions["min_compound"]
    max_compound = emotions["max_compound"]

    if avg_compound > 0.1:
        overall_emotion = "positive"
    elif avg_compound < -0.1:
        overall_emotion = "negative"
    else:
        overall_emotion = "neutral"

    emotional_summary = (
        f"This session included {num_messages} user messages. "
        f"Average compound sentiment score was {avg_compound:.2f} "
        f"(min: {min_compound:.2f}, max: {max_compound:.2f}) "
        f"indicating an overall {overall_emotion} tone."
    )
    
    timestamp = datetime.now().isoformat()
    return {
        "summary": summary_text,
        "emotional_summary": emotional_summary,
        "timestamp": timestamp
    }

# Main chat processing logic
async def process_chat(session_key: str, user_message: str):
    # Fetch user data and chat history
    user_data = get_user_data(session_key) or {}
    chat_history = get_chat_history(session_key) or []
    doctor_summary = get_doctor_summary(session_key)

    # Initialize fields in user_data if not present
    user_data["doctor_summary"] = doctor_summary
    if "past_summaries" not in user_data:
        user_data["past_summaries"] = []
    if "last_summarized_index" not in user_data:
        user_data["last_summarized_index"] = 0
    if "questionnaire_completed" not in user_data:
        user_data["questionnaire_completed"] = False
    if "session_start_index" not in user_data:
        user_data["session_start_index"] = 0
    if "session_agg_sentiment" not in user_data:
        user_data["session_agg_sentiment"] = reset_sentiment_aggregate()

    # Get current time
    current_time = datetime.now()

    # Check for inactivity and auto-generate summary if detected
    last_interaction = user_data.get("last_interaction")
    if last_interaction:
        last_interaction_time = datetime.fromisoformat(last_interaction)
        time_since_last = current_time - last_interaction_time
        if time_since_last >= INACTIVITY_THRESHOLD and chat_history:
            session_start_index = user_data["session_start_index"]
            previous_summary = user_data["past_summaries"][-1]["summary"] if user_data.get("past_summaries") else None
            summary_data = generate_conversation_summary(chat_history, session_start_index, previous_summary)
            summary_text = summary_data["summary"]
            emotional_summary = summary_data["emotional_summary"]

            user_data["past_summaries"].append({
                "summary": summary_text,
                "emotional_summary": emotional_summary,
                "timestamp": summary_data["timestamp"]
            })
            user_data["last_summarized_index"] = len(chat_history)
            user_data["session_start_index"] = len(chat_history)  # Start a new session
            user_data["session_agg_sentiment"] = reset_sentiment_aggregate()  # Reset sentiment metrics
            chat_history.append({
                "role": "assistant",
                "content": f"Auto-generated summary due to inactivity:\n{summary_text}"
            })
            save_chat_history(session_key, chat_history)
            store_user_data(session_key, user_data)
            print(f"Here's a summary of our last session:\n{summary_text}")

    # Analyze sentiment of the user's message
    sentiment = analyze_sentiment(user_message)
    chat_history.append({"role": "user", "content": user_message, "sentiment": sentiment})

    # Update session-specific sentiment aggregate
    session_agg_sentiment = user_data["session_agg_sentiment"]
    new_score = sentiment.get("compound", 0.0)
    session_agg_sentiment = update_sentiment_aggregate(session_agg_sentiment, new_score)
    user_data["session_agg_sentiment"] = session_agg_sentiment
    
    # Update last interaction time
    user_data["last_interaction"] = current_time.isoformat()
    store_user_data(session_key, user_data)

    # Check if user explicitly ends the session
    if user_message.lower() in ["end session", "goodbye", "exit", "bye", "end"]:
        session_start_index = user_data["session_start_index"]
        previous_summary = user_data["past_summaries"][-1]["summary"] if user_data.get("past_summaries") else None
        summary_data = generate_conversation_summary(chat_history, session_start_index, previous_summary)
        summary_text = summary_data["summary"]
        emotional_summary = summary_data["emotional_summary"]
        
        user_data["past_summaries"].append({
            "summary": summary_text,
            "emotional_summary": emotional_summary,
            "timestamp": summary_data["timestamp"]
        })
        user_data["last_summarized_index"] = len(chat_history)
        user_data["session_start_index"] = len(chat_history)  # Start a new session
        user_data["session_agg_sentiment"] = reset_sentiment_aggregate()  # Reset sentiment metrics
        chat_history.append({
            "role": "assistant",
            "content": "Here is your session summary:\n" + summary_text,
        })
        save_chat_history(session_key, chat_history)
        store_user_data(session_key, user_data)
        return {"response": summary_text}

    # Risk intervention logic
    if check_risk(session_agg_sentiment) and not user_data.get("risk_intervention_shown", False):
        intervention_message = (
            "I've noticed that our conversation seems to reflect some distress. "
            "It might be helpful to consider speaking with a mental health professional. "
            "Please remember, I'm not a substitute for professional advice. "
            "Would you like some resources or help finding support?"
        )
        user_data["risk_intervention_shown"] = True
        store_user_data(session_key, user_data)
        chat_history.append({"role": "assistant", "content": intervention_message})
        save_chat_history(session_key, chat_history)
        return {"response": intervention_message}

    # Greet new users or returning users
    if len(chat_history) == 1:  # Only the current user message
        if user_data.get("questionnaire_completed", False):
            greeting = "Welcome back! How can I assist you today?"
        else:
            greeting = "Hi! Welcome to our mental health assistant. Let me ask you a few questions to get started."
            user_data["session_start_index"] = 1  # Start a new session for new users
        chat_history.append({"role": "assistant", "content": greeting})
        save_chat_history(session_key, chat_history)
        store_user_data(session_key, user_data)
        return {"response": greeting}

    # Questionnaire logic with validation
    questionnaire = {
        "name": "Hey there! What is your good name?",
        "Age": "How old are you?",
        "Gender": "What is your gender?",
        "Condition": "How are you feeling in general at this time?",
        "History": "Do you have any history of mental health issues? or maybe in the family?"
    }
    missing_keys = [key for key in questionnaire if key not in user_data]
    last_question = next(
        (entry["content"] for entry in reversed(chat_history) if entry["role"] == "assistant"), None
    )

    # If questionnaire is incomplete
    if missing_keys and not user_data.get("questionnaire_completed", False):
        current_key = missing_keys[0]
        if last_question == questionnaire[current_key]:
            user_data[current_key] = user_message.strip()
            store_user_data(session_key, user_data)
            remaining_keys = missing_keys[1:]
            if not remaining_keys:
                user_data["questionnaire_completed"] = True
                follow_up = f"Thank you, {user_data.get('name', 'friend')}. I appreciate your responses. How can I help you today?"
                chat_history.append({"role": "assistant", "content": follow_up})
                save_chat_history(session_key, chat_history)
                return {"response": follow_up}
            next_question = questionnaire[remaining_keys[0]]
            chat_history.append({"role": "assistant", "content": next_question})
            save_chat_history(session_key, chat_history)
            return {"response": next_question}
        else:
            next_question = questionnaire[current_key]
            chat_history.append({"role": "assistant", "content": next_question})
            save_chat_history(session_key, chat_history)
            return {"response": next_question}

    # Normal chatbot flow for users who have completed the questionnaire
    last_question = next(
        (entry["content"] for entry in reversed(chat_history) if entry["role"] == "assistant"), None
    )

    if last_question == "Hi! Welcome to our mental health assistant. Let me ask you a few questions to get started.":
        if not user_data.get("questionnaire_completed", False):
            next_question = questionnaire["name"]
            chat_history.append({"role": "assistant", "content": next_question})
            save_chat_history(session_key, chat_history)
            return {"response": next_question}
        follow_up = "Thank you for answering the questions. How can I help you today?"
        chat_history.append({"role": "assistant", "content": follow_up})
        save_chat_history(session_key, chat_history)
        return {"response": follow_up}
    elif last_question == "Welcome back! How can I assist you today?":
        session_start_index = user_data["session_start_index"]
        ai_response = process_user_message(
            user_message, 
            chat_history, 
            user_data, 
            max_summaries=3, 
            max_doctor_summaries=2
        )
        chat_history.append({"role": "assistant", "content": ai_response})
        save_chat_history(session_key, chat_history)
        return {"response": ai_response}

    session_start_index = user_data["session_start_index"]
    ai_response = process_user_message(
        user_message, 
        chat_history, 
        user_data, 
        max_summaries=3, 
        max_doctor_summaries=2
    )
    chat_history.append({"role": "assistant", "content": ai_response})
    save_chat_history(session_key, chat_history)
    return {"response": ai_response}