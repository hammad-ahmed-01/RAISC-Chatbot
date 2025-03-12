from fastapi import APIRouter, HTTPException
from app.models import ChatRequest
from app.services.firestore_service import get_chat_history, save_chat_history
from app.services.user_service import get_user_data, store_user_data, get_doctor_summary
from app.services.rag_service import process_user_message
from langchain_groq import ChatGroq
from datetime import datetime, timedelta
import nltk
from nltk.sentiment.vader import SentimentIntensityAnalyzer
import json
from transformers import pipeline
from app.config import GROQ_API_KEY
from langchain.schema import SystemMessage, HumanMessage

nltk.download('vader_lexicon')

router = APIRouter()
# Initializing the VADER analyzer once 
analyzer = SentimentIntensityAnalyzer()
llm = ChatGroq(model="llama-3.3-70b-versatile", groq_api_key=GROQ_API_KEY, temperature=0)

# Steps for emotion analysis & then summarization of the convo
# Step 1: Collect all message contents into one text string.
# Step 2: Passing that text to the summarization model to produce a concise summary.
# Step 3: Running emotional analysis on the chat history and gettt key metrics (average, min, and max comp score).
# Step 4: Combine both pieces of information into a dict output that provides summary of the conversation and
# the emotion throughout the session.

# Function to get SUMMARY of the whole convo texts of users
def summarize_conversation(chat_history, last_n=10):
    # Combine only the text from user messages
    user_messages = [msg["content"] for msg in chat_history if msg.get("role") == "user"]
    # Keep only the last N user messages
    user_messages = user_messages[-last_n:]
    # Combine them into a single string
    conversation_text = " ".join(user_messages)  # keep only the last N
    # summary extraction prompt
    prompt = (
        SystemMessage(content="You are a helpful assistant that summarizes conversations. "
                              "Summarize the following conversation concisely, focusing on the key points and overall tone:"),
        HumanMessage(content=conversation_text)
    )
    summary_output = llm(prompt)  # Adjust this call based on your LLM integration
    return summary_output.content.strip()


# Function to get sentiment for EACH message individually
def analyze_sentiment(message: str) -> dict:
    return analyzer.polarity_scores(message)

#    function to get the overall metrics of the WHOLE convo
# Fallback emotion analysis by scanning the entire chat history
def analyze_emotions(chat_history):
    user_sentiments = [
        msg["sentiment"] for msg in chat_history
        if msg.get("role") == "user" and "sentiment" in msg
    ]
    
    if user_sentiments:
        compound_scores = [score.get("compound", 0) for score in user_sentiments]
        avg_compound = sum(compound_scores) / len(compound_scores)
        min_compound = min(compound_scores)
        max_compound = max(compound_scores)
        num_messages = len(user_sentiments)
    else:
        avg_compound = min_compound = max_compound = 0
        num_messages = 0

    return {
        "average_compound": avg_compound,
        "min_compound": min_compound,
        "max_compound": max_compound,
        "num_messages": num_messages
    }

# Incrementally update the sentiment aggregate stored in user_data
def update_sentiment_aggregate(agg: dict, new_score: float) -> dict:
    agg["sum"] += new_score
    agg["count"] += 1
    agg["average"] = agg["sum"] / agg["count"]
    if agg["count"] == 1:
        agg["min"] = new_score
        agg["max"] = new_score
    else:
        agg["min"] = min(agg.get("min", new_score), new_score)
        agg["max"] = max(agg.get("max", new_score), new_score)
    return agg

# Check risk using the incremental aggregate (threshold can be tuned)
def check_risk(agg: dict, threshold: float = -0.1) -> bool:
    return agg["average"] < threshold


def generate_conversation_summary(chat_history, agg_sentiment=None, last_summarized_index=0, previous_summary=None):
    """
    Generate a summary for the conversation, relating it to the previous summary if provided.
    """
    # Filter chat history to include only new user messages
    new_messages = [msg for msg in chat_history[last_summarized_index:] if msg.get("role") == "user"]
    if not new_messages:
        return {"summary": "No new messages to summarize.", "timestamp": datetime.now().isoformat()}

    # Combine text from new user messages
    conversation_text = " ".join(msg["content"] for msg in new_messages)
    
    # Build prompt with previous summary context
    prompt_text = (
        "You are a helpful assistant that summarizes conversations. "
        "Summarize the following conversation concisely, focusing on the key points and overall tone. "
        "If a previous summary is provided, relate the new summary to it, noting any changes, continuations, or new topics."
    )
    if previous_summary:
        prompt_text += f"\nPrevious summary: {previous_summary}\n"
    prompt_text += f"Current conversation: {conversation_text}"

    prompt = (
        SystemMessage(content=prompt_text),
        HumanMessage(content="")
    )
    summary_output = llm(prompt)  # Replace with your actual LLM call
    summary_text = summary_output.content.strip()

    # Emotional analysis for new messages
    if agg_sentiment is None:
        user_sentiments = [msg["sentiment"] for msg in new_messages if "sentiment" in msg]
        if user_sentiments:
            compound_scores = [score.get("compound", 0) for score in user_sentiments]
            avg_compound = sum(compound_scores) / len(compound_scores)
            min_compound = min(compound_scores)
            max_compound = max(compound_scores)
            num_messages = len(user_sentiments)
        else:
            avg_compound = min_compound = max_compound = 0
            num_messages = 0
    else:
        num_messages = agg_sentiment["count"]
        avg_compound = agg_sentiment["average"]
        min_compound = agg_sentiment.get("min", 0)
        max_compound = agg_sentiment.get("max", 0)

    if avg_compound > 0.1:
        overall_emotion = "positive"
    elif avg_compound < -0.1:
        overall_emotion = "negative"
    else:
        overall_emotion = "neutral"
        
    emotional_summary = (
        f"The conversation included {num_messages} new user messages. "
        f"Average compound sentiment score was {avg_compound:.2f} "
        f"(min: {min_compound:.2f}, max: {max_compound:.2f}) "
        f"indicating an overall {overall_emotion} tone."
    )
    
    timestamp = datetime.now().isoformat()
    full_summary = f"{summary_text}\n{emotional_summary}"

    return {
        "summary": summary_text,
        "emotional_summary": emotional_summary,
        "timestamp": timestamp
    }

# added feature where only new msgs are sumamrized
# Inactivity threshold (e.g., 5 minutes)
INACTIVITY_THRESHOLD = timedelta(minutes=1)

@router.post("/chat")
async def chat(request: ChatRequest):
    session_key = request.session_key
    user_message = request.message

    # Fetch user data and chat history
    user_data = get_user_data(session_key) or {}
    chat_history = get_chat_history(session_key) or []
    
    # Fetch doctor summary
    doctor_summary = get_doctor_summary(session_key)

    # Initialize fields in user_data if not present
    user_data["doctor_summary"] = doctor_summary
    
    if "past_summaries" not in user_data:
        user_data["past_summaries"] = []
    if "last_summarized_index" not in user_data:
        user_data["last_summarized_index"] = 0

    # Get current time
    current_time = datetime.now()

    # Check for inactivity and auto-generate summary if needed
    last_interaction = user_data.get("last_interaction")
    if last_interaction:
        last_interaction_time = datetime.fromisoformat(last_interaction)
        time_since_last = current_time - last_interaction_time
        if time_since_last >= INACTIVITY_THRESHOLD and chat_history:
            agg_sentiment = user_data.get("agg_sentiment", {"sum": 0.0, "count": 0, "average": 0.0})
            last_index = user_data["last_summarized_index"]
            # Get the most recent previous summary
            previous_summary = user_data["past_summaries"][-1]["summary"] if user_data.get("past_summaries") else None
            summary_data = generate_conversation_summary(chat_history, agg_sentiment, last_index, previous_summary)
            summary_text = summary_data["summary"]
            emotional_summary = summary_data["emotional_summary"]

            user_data["past_summaries"].append({
                "summary": summary_text,
                "emotional_summary": emotional_summary,
                "timestamp": summary_data["timestamp"]
            })
            # Update last_summarized_index to the last index before the new message
            user_data["last_summarized_index"] = len(chat_history)
            # Append summary to chat_history for visibility
            chat_history.append({
                "role": "assistant",
                "content": f"Auto-generated summary due to inactivity:\n{summary_text}"
            })
            save_chat_history(session_key, chat_history)
            store_user_data(session_key, user_data)
            print(f"Here's a summary of our last conversation:\n{summary_text}")
            # Return the summary but continue processing the new message
            # (No immediate return here to allow flow continuation)

    # Analyze sentiment of the user's message
    sentiment = analyze_sentiment(user_message)
    chat_history.append({"role": "user", "content": user_message, "sentiment": sentiment})

    # Update sentiment aggregate
    agg_sentiment = user_data.get("agg_sentiment", {"sum": 0.0, "count": 0, "average": 0.0})
    new_score = sentiment.get("compound", 0.0)
    agg_sentiment = update_sentiment_aggregate(agg_sentiment, new_score)
    user_data["agg_sentiment"] = agg_sentiment

    # Update last interaction time
    user_data["last_interaction"] = current_time.isoformat()
    store_user_data(session_key, user_data)

    # Check if user explicitly ends the session
    if user_message.lower() in ["end session", "goodbye", "exit", "bye", "end"]:
        last_index = user_data["last_summarized_index"]
        # Get the most recent previous summary
        previous_summary = user_data["past_summaries"][-1]["summary"] if user_data.get("past_summaries") else None
        summary_data = generate_conversation_summary(chat_history, agg_sentiment, last_index, previous_summary)
        summary_text = summary_data["summary"]
        emotional_summary = summary_data["emotional_summary"]
        
        user_data["past_summaries"].append({
            "summary": summary_text,
            "emotional_summary": emotional_summary,
            "timestamp": summary_data["timestamp"]
        })
        user_data["last_summarized_index"] = len(chat_history)
        chat_history.append({
            "role": "assistant",
            "content": "Here is your session summary:\n" + summary_text,
        })
        
        save_chat_history(session_key, chat_history)
        store_user_data(session_key, user_data)
        return {"response": summary_text}
    
    # Risk intervention logic
    if check_risk(agg_sentiment) and not user_data.get("risk_intervention_shown", False):
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

    # Greet new users
    if len(chat_history) == 1:  # Only the current user message
        greeting = "Hi! Welcome to our mental health assistant. Let me ask you a few questions to get started."
        chat_history.append({"role": "assistant", "content": greeting})
        save_chat_history(session_key, chat_history)
        return {"response": greeting}

    # Questionnaire logic
    questionnaire = {
        "name": " Hey there! What is your good name?",
        "Age" : " How old are you?",
        "Gender": "What is your gender?",
        "Condition": "How are you feeling in general at this time?",
        "History": "Do you have any history of mental health issues? or maybe in the family?"

    }
    missing_keys = [key for key in questionnaire if key not in user_data]
    last_question = next(
        (entry["content"] for entry in reversed(chat_history) if entry["role"] == "assistant"), None
    )

    if missing_keys:
        current_key = missing_keys[0]
        if last_question == questionnaire[current_key]:
            user_data[current_key] = user_message.strip()
            store_user_data(session_key, user_data)
            remaining_keys = missing_keys[1:]
            if remaining_keys:
                next_question = questionnaire[remaining_keys[0]]
                chat_history.append({"role": "assistant", "content": next_question})
                save_chat_history(session_key, chat_history)
                return {"response": next_question}
            else:
                follow_up = "Thank you for answering the questions. How can I help you today?"
                chat_history.append({"role": "assistant", "content": follow_up})
                save_chat_history(session_key, chat_history)
                return {"response": follow_up}
        else:
            next_question = questionnaire[current_key]
            chat_history.append({"role": "assistant", "content": next_question})
            save_chat_history(session_key, chat_history)
            return {"response": next_question}

    # Normal chatbot flow
    if not missing_keys:
        if last_question == "Hi! Welcome to our mental health assistant. Let me ask you a few questions to get started.":
            follow_up = "Thank you for answering the questions. How can I help you today?"
            chat_history.append({"role": "assistant", "content": follow_up})
            save_chat_history(session_key, chat_history)
            return {"response": follow_up}

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
