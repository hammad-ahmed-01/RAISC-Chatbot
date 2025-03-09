from fastapi import APIRouter, HTTPException
from app.models import ChatRequest
from app.services.firestore_service import get_chat_history, save_chat_history
from app.services.user_service import get_user_data, store_user_data
from app.services.rag_service import process_user_message
from langchain_groq import ChatGroq
import nltk
from nltk.sentiment.vader import SentimentIntensityAnalyzer
import json
from transformers import pipeline
from app.config import GROQ_API_KEY
from langchain.schema import SystemMessage, HumanMessage
from datetime import datetime

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
    
# Generate a combined conversation summary and emotional analysis using the aggregate
def generate_conversation_summary(chat_history, agg_sentiment=None):
    summary_text = summarize_conversation(chat_history)
    if agg_sentiment is None:
        emotional_info = analyze_emotions(chat_history)
        num_messages = emotional_info["num_messages"]
        avg_compound = emotional_info["average_compound"]
        min_compound = emotional_info["min_compound"]
        max_compound = emotional_info["max_compound"]
    else:
        num_messages = agg_sentiment["count"]
        avg_compound = agg_sentiment["average"]
        min_compound = agg_sentiment["min"]
        max_compound = agg_sentiment["max"]

    if avg_compound > 0.1:
        overall_emotion = "positive"
    elif avg_compound < -0.1:
        overall_emotion = "negative"
    else:
        overall_emotion = "neutral"
        
    emotional_summary = (
        f"The conversation included {num_messages} user messages. "
        f"Average compound sentiment score was {avg_compound:.2f} "
        f"(min: {min_compound:.2f}, max: {max_compound:.2f}) "
        f"indicating an overall {overall_emotion} tone. "
    )
    # ################ ENDPOINT TYPE SHI? ######################## missing key information atm
    # print( f"conversation_summary: {summary_text}, emotional_summary: {emotional_summary}")
    # return summary_text

    # Add timestamp
    timestamp = datetime.now().isoformat()
    full_summary = f"{summary_text}\n{emotional_summary}"

    # Return a structured object with timestamp
    return {
        "summary": full_summary,
        "timestamp": timestamp
    }

# @router.post("/chat")
# async def chat(request: ChatRequest):
#     """
#     Handle chat interactions with the user.
#     """
#     session_key = request.session_key
#     user_message = request.message

#     # Fetch user data and chat history
#     user_data = get_user_data(session_key) or {}
#     chat_history = get_chat_history(session_key) or []
#     doctor_summary = []

#     # Analyze sentiment of the user's message in real-time
#     sentiment = analyze_sentiment(user_message)
#     chat_history.append({"role": "user", "content": user_message, "sentiment": sentiment})

#     # Update incremental sentiment aggregate stored in user_data
#     agg_sentiment = user_data.get("agg_sentiment", {"sum": 0.0, "count": 0, "average": 0.0})
#     new_score = sentiment.get("compound", 0.0)
#     agg_sentiment = update_sentiment_aggregate(agg_sentiment, new_score)
#     user_data["agg_sentiment"] = agg_sentiment
#     store_user_data(session_key, user_data)
    
#     # Check if the user wants to end the session
#     if user_message.lower() in ["end session", "goodbye", "exit", "bye", "end"]:
#         summary_data = generate_conversation_summary(chat_history, agg_sentiment)
#         summary_text = summary_data["summary"]
        
#         # Store the summary with timestamp in user_data
#         if "past_summaries" not in user_data:
#             user_data["past_summaries"] = []
#         user_data["past_summaries"].append({
#             "summary": summary_text,
#             "timestamp": summary_data["timestamp"]
#         })
#         store_user_data(session_key, user_data)

#         chat_history.append({
#             "role": "assistant",
#             "content": "Here is your session summary:\n" + summary_text,
#         })
#         save_chat_history(session_key, chat_history)
#         return {"response": summary_text}
    
#     # If we haven't shown an intervention yet AND the risk check is triggered
#     if check_risk(agg_sentiment) and not user_data.get("risk_intervention_shown", False):
#         intervention_message = (
#             "I've noticed that our conversation seems to reflect some distress. "
#             "It might be helpful to consider speaking with a mental health professional. "
#             "Please remember, I'm not a substitute for professional advice. "
#             "Would you like some resources or help finding support?"
#         )
#         user_data["risk_intervention_shown"] = True
#         store_user_data(session_key, user_data)

#         chat_history.append({"role": "assistant", "content": intervention_message})
#         save_chat_history(session_key, chat_history)
#         return {"response": intervention_message}

#     # Greet new users if the chat history is empty
#     if not chat_history or len(chat_history) == 1:
#         greeting = "Hi! Welcome to our mental health assistant. Let me ask you a few questions to get started."
#         chat_history.append({"role": "assistant", "content": greeting})
#         save_chat_history(session_key, chat_history)
#         return {"response": greeting}

#     # Questionnaire logic (unchanged)
#     questionnaire = {
#         "name": "What's your name?",
#         "age": "How old are you?",
#         "gender": "What is your gender?",
#         "current_state": "How have you been feeling recently?",
#         "history": "Do you have any history of mental health issues, treatments, or family history of such conditions?",
#     }
#     missing_keys = [key for key in questionnaire if key not in user_data]
#     last_question = next(
#         (entry["content"] for entry in reversed(chat_history) if entry["role"] == "assistant"), None
#     )

#     if missing_keys:
#         current_key = missing_keys[0]
#         if last_question == questionnaire[current_key]:
#             user_data[current_key] = user_message.strip()
#             store_user_data(session_key, user_data)
    
#             remaining_keys = missing_keys[1:]
#             if remaining_keys:
#                 next_question = questionnaire[remaining_keys[0]]
#                 chat_history.append({"role": "assistant", "content": next_question})
#                 save_chat_history(session_key, chat_history)
#                 return {"response": next_question}
#             else:
#                 follow_up = "Thank you for answering the questions. How can I help you today?"
#                 chat_history.append({"role": "assistant", "content": follow_up})
#                 save_chat_history(session_key, chat_history)
#                 return {"response": follow_up}
#         else:
#             next_question = questionnaire[current_key]
#             chat_history.append({"role": "assistant", "content": next_question})
#             save_chat_history(session_key, chat_history)
#             return {"response": next_question}

#     # Normal chatbot flow with max_summaries
#     if not missing_keys:
#         if last_question == "Hi! Welcome to our mental health assistant. Let me ask you a few questions to get started.":
#             follow_up = "Thank you for answering the questions. How can I help you today?"
#             chat_history.append({"role": "assistant", "content": follow_up})
#             save_chat_history(session_key, chat_history)
#             return {"response": follow_up}

#         # Pass max_summaries to process_user_message
#         ai_response = process_user_message(user_message, chat_history, user_data, max_summaries=3)
#         chat_history.append({"role": "assistant", "content": ai_response})
#         save_chat_history(session_key, chat_history)
#         return {"response": ai_response}

@router.post("/chat")
async def chat(request: ChatRequest):
    session_key = request.session_key
    user_message = request.message

    # Fetch user data and chat history
    user_data = get_user_data(session_key) or {}
    chat_history = get_chat_history(session_key) or []

    # Initialize doctor_summary in user_data if not present
    if "doctor_summary" not in user_data:
        user_data["doctor_summary"] = ["I am the patient's doctor and my name is van helsing"]  # dummy list naming

    # Analyze sentiment of the user's message in real-time(each msg)
    sentiment = analyze_sentiment(user_message)
    chat_history.append({"role": "user", "content": user_message, "sentiment": sentiment})

    # Update incremental sentiment aggregate stored in user_data
    agg_sentiment = user_data.get("agg_sentiment", {"sum": 0.0, "count": 0, "average": 0.0})
    new_score = sentiment.get("compound", 0.0)
    agg_sentiment = update_sentiment_aggregate(agg_sentiment, new_score)
    user_data["agg_sentiment"] = agg_sentiment
    store_user_data(session_key, user_data)
    
    # Check if the user wants to end the session
    if user_message.lower() in ["end session", "goodbye", "exit", "bye", "end"]:
        summary_data = generate_conversation_summary(chat_history, agg_sentiment)
        summary_text = summary_data["summary"]
        
        # Store the summary with timestamp in user_data
        if "past_summaries" not in user_data:
            user_data["past_summaries"] = []
        user_data["past_summaries"].append({
            "summary": summary_text,
            "timestamp": summary_data["timestamp"]
        })
        store_user_data(session_key, user_data)

        chat_history.append({
            "role": "assistant",
            "content": "Here is your session summary:\n" + summary_text,
        })
        save_chat_history(session_key, chat_history)
        return {"response": summary_text}
    
    # If we haven't shown an intervention yet AND the risk check is triggered
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

    # Greet new users if the chat history is empty
    if not chat_history or len(chat_history) == 1:
        greeting = "Hi! Welcome to our mental health assistant. Let me ask you a few questions to get started."
        chat_history.append({"role": "assistant", "content": greeting})
        save_chat_history(session_key, chat_history)
        return {"response": greeting}

    # Questionnaire logic
    questionnaire = {
        "name": "What's your name?",
        "age": "How old are you?",
        "gender": "What is your gender?",
        "current_state": "How have you been feeling recently?",
        "history": "Do you have any history of mental health issues, treatments, or family history of such conditions?",
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

    # Normal chatbot flow with doctor_summary included
    if not missing_keys:
        if last_question == "Hi! Welcome to our mental health assistant. Let me ask you a few questions to get started.":
            follow_up = "Thank you for answering the questions. How can I help you today?"
            chat_history.append({"role": "assistant", "content": follow_up})
            save_chat_history(session_key, chat_history)
            return {"response": follow_up}

        # Pass max_summaries and max_doctor_summaries to process_user_message
        ai_response = process_user_message(
            user_message, 
            chat_history, 
            user_data, 
            max_summaries=3, 
            max_doctor_summaries=3
        )
        chat_history.append({"role": "assistant", "content": ai_response})
        save_chat_history(session_key, chat_history)
        return {"response": ai_response}