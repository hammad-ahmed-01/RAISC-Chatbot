# app/services/chat_service.py
import nltk
from nltk.sentiment.vader import SentimentIntensityAnalyzer
from datetime import datetime, timedelta
from langchain_groq import ChatGroq
from langchain.schema import SystemMessage, HumanMessage
from app.services.firestore_service import get_chat_history, save_chat_history
from app.services.user_service import get_user_data, store_user_data, get_doctor_summary
from app.services.rag_service import process_user_message
from app.config import GROQ_API_KEY
import json
import re

nltk.download('vader_lexicon')

# Initialize the VADER analyzer and LLM once
analyzer = SentimentIntensityAnalyzer()
llm = ChatGroq(model="llama-3.3-70b-versatile", groq_api_key=GROQ_API_KEY, temperature=0)

# Inactivity threshold
INACTIVITY_THRESHOLD = timedelta(minutes=1)

# Define the information we need to collect
REQUIRED_INFORMATION = {
    "name": {
        "collected": False, 
        "value": None, 
        "required": True,
        "description": "user's preferred name"
    },
    "age": {
        "collected": False, 
        "value": None, 
        "required": True,
        "description": "user's age"
    },
    "gender": {
        "collected": False, 
        "value": None, 
        "required": True,
        "description": "user's gender"
    },
    "current_condition": {
        "collected": False, 
        "value": None, 
        "required": True,
        "description": "how the user is feeling currently"
    },
    "mental_health_history": {
        "collected": False, 
        "value": None, 
        "required": True,
        "description": "user's mental health history or family history"
    }
}

def initialize_information_tracking(user_data: dict) -> dict:
    """Initialize or update the information tracking structure"""
    # Ensure user_data is a dictionary
    if not isinstance(user_data, dict):
        user_data = {}
    
    if "information_needed" not in user_data:
        # Create a deep copy to avoid modifying the original REQUIRED_INFORMATION
        user_data["information_needed"] = {}
        for key, value in REQUIRED_INFORMATION.items():
            user_data["information_needed"][key] = value.copy()
    else:
        # Update existing structure with any new fields
        for key, value in REQUIRED_INFORMATION.items():
            if key not in user_data["information_needed"]:
                user_data["information_needed"][key] = value.copy()
    
    return user_data

def get_next_missing_information(user_data: dict) -> str:
    """Get the next information that needs to be collected in the defined order"""
    info_needed = user_data.get("information_needed", {})
    
    # Check in the order defined in REQUIRED_INFORMATION
    for key in REQUIRED_INFORMATION.keys():
        if key in info_needed:
            info = info_needed[key]
            if info.get("required", True) and not info.get("collected", False):
                return key
    
    return None

def extract_information_from_message(user_message: str, missing_info_key: str) -> dict:
    """Extract information from user's message using LLM"""
    if not missing_info_key:
        return {}
    
    # Create extraction prompt for the specific piece of information
    if missing_info_key in REQUIRED_INFORMATION:
        description = REQUIRED_INFORMATION[missing_info_key]["description"]
    else:
        return {}
    
    extraction_prompt = f"""
    Analyze the following user message and extract information about: {missing_info_key} ({description})
    
    User message: "{user_message}"
    
    Return your response as a JSON object where the key is "{missing_info_key}" and the value is the extracted information.
    Only include the key if you found clear information. If no relevant information is found, return an empty JSON object.
    
    Example response format:
    {{"{missing_info_key}": "extracted_value"}}
    
    Response:
    """
    
    try:
        prompt = [
            SystemMessage(content="You are an information extraction assistant. Extract personal information from user messages and return it as valid JSON."),
            HumanMessage(content=extraction_prompt)
        ]
        
        response = llm(prompt)
        extracted_text = response.content.strip()
        
        # Try to parse JSON from the response
        try:
            # Look for JSON in the response
            json_match = re.search(r'\{.*\}', extracted_text, re.DOTALL)
            if json_match:
                extracted_info = json.loads(json_match.group())
                return extracted_info
        except json.JSONDecodeError:
            pass
        
        return {}
        
    except Exception as e:
        print(f"Error extracting information: {e}")
        return {}

def update_collected_information(user_data: dict, extracted_info: dict) -> dict:
    """Update user_data with newly extracted information"""
    if not extracted_info:
        return user_data
    
    info_needed = user_data.get("information_needed", {})
    
    for key, value in extracted_info.items():
        if key in info_needed and value and value.strip():
            info_needed[key]["collected"] = True
            info_needed[key]["value"] = value.strip()
            # Also store in the main user_data for backward compatibility
            user_data[key] = value.strip()
    
    return user_data

def is_questionnaire_complete(user_data: dict) -> bool:
    """Check if all required information has been collected"""
    info_needed = user_data.get("information_needed", {})
    
    for key, info in info_needed.items():
        if info.get("required", True) and not info.get("collected", False):
            return False
    
    return True

def create_information_gathering_context(missing_info_key: str, user_data: dict) -> str:
    """Create strict context for gathering the next specific piece of information"""
    if not missing_info_key:
        return ""
    
    # Strict, enforcement-focused context for information gathering
    context = f"""
STRICT MODE: INFORMATION GATHERING ONLY
REQUIRED ACTION: Ask for {missing_info_key} - DO NOT provide therapy yet
BLOCK: Any therapeutic advice until all information is collected
ENFORCE: Stay focused on getting {missing_info_key} only

MANDATORY APPROACH:
- Ask for {missing_info_key} warmly but directly
- Keep response to 2-3 sentences maximum  
- DO NOT provide therapeutic responses or advice
- DO NOT ask follow-up questions about their problems
- ONLY focus on collecting the {missing_info_key} information
- Be supportive but BRIEF until you have all needed information

FORBIDDEN: Therapeutic advice, problem exploration, coping strategies, or extensive emotional support until information gathering is complete.

EXAMPLE: "I'd like to help you better. Could you tell me [ask for {missing_info_key}]?"
"""
    
    return context

def create_completion_message(user_data: dict) -> str:
    """Create a message when information gathering is completed"""
    name = user_data.get("name", "")
    
    message = f"Thank you for sharing that with me{', ' + name if name else ''}! "
    message += "I feel like I have a good understanding of your situation now. "
    message += "How can I best support you today?"
    
    return message

# Sentiment analysis and summary functions
def analyze_sentiment(message: str) -> dict:
    return analyzer.polarity_scores(message)

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

def reset_sentiment_aggregate() -> dict:
    return {"sum": 0.0, "count": 0, "average": 0.0, "min": 0.0, "max": 0.0}

def check_risk(agg: dict, threshold: float = -0.1) -> bool:
    return agg["average"] < threshold

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
    # Fetch user data and chat history with proper type checking
    user_data_raw = get_user_data(session_key)
    
    # Ensure user_data is a dictionary
    if isinstance(user_data_raw, dict):
        user_data = user_data_raw
    else:
        # If get_user_data returns None, empty string, or any non-dict, create new dict
        user_data = {}
    
    chat_history = get_chat_history(session_key) or []
    doctor_summary = get_doctor_summary(session_key)

    # Initialize information tracking
    user_data = initialize_information_tracking(user_data)

    # Initialize other fields in user_data if not present
    user_data["doctor_summary"] = doctor_summary
    if "past_summaries" not in user_data:
        user_data["past_summaries"] = []
    if "last_summarized_index" not in user_data:
        user_data["last_summarized_index"] = 0
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
            user_data["session_start_index"] = len(chat_history)
            user_data["session_agg_sentiment"] = reset_sentiment_aggregate()
            chat_history.append({
                "role": "assistant",
                "content": f"Auto-generated summary due to inactivity:\n{summary_text}"
            })
            save_chat_history(session_key, chat_history)
            store_user_data(session_key, user_data)

    # Analyze sentiment of the user's message
    sentiment = analyze_sentiment(user_message)
    chat_history.append({"role": "user", "content": user_message, "sentiment": sentiment})

    # Get the next missing information in order
    next_missing_info = get_next_missing_information(user_data)
    
    # Extract information from user's message if we're looking for something specific
    extracted_info = extract_information_from_message(user_message, next_missing_info)
    user_data = update_collected_information(user_data, extracted_info)

    # Update questionnaire_completed status
    was_completed_before = user_data.get("questionnaire_completed", False)
    user_data["questionnaire_completed"] = is_questionnaire_complete(user_data)
    just_completed = not was_completed_before and user_data["questionnaire_completed"]

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
        user_data["session_start_index"] = len(chat_history)
        user_data["session_agg_sentiment"] = reset_sentiment_aggregate()
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

    # Handle first-time users
    if len(chat_history) == 1:
        greeting = "Hi! I'm here to support you with whatever you're going through. What should I call you?"
        user_data["session_start_index"] = 1
        chat_history.append({"role": "assistant", "content": greeting})
        save_chat_history(session_key, chat_history)
        store_user_data(session_key, user_data)
        return {"response": greeting}

    # Check if information gathering just completed
    if just_completed:
        completion_message = create_completion_message(user_data)
        chat_history.append({"role": "assistant", "content": completion_message})
        save_chat_history(session_key, chat_history)
        store_user_data(session_key, user_data)
        return {"response": completion_message}

    # Get the next missing information in order and enforce strict mode
    next_missing_info = get_next_missing_information(user_data)
    
    # Create context for the unified system with strict enforcement
    if next_missing_info:
        # STRICT ENFORCEMENT: Must collect information before therapy
        info_context = create_information_gathering_context(next_missing_info, user_data)
        print(f"STRICT MODE ENFORCED: Must collect {next_missing_info} before therapy")
        print(f"BLOCKING: Therapeutic responses until {next_missing_info} is collected")
    else:
        # All information gathered - allow full therapy mode
        info_context = """
THERAPY MODE ACTIVATED: All required information has been collected.
PERMISSION GRANTED: Provide full therapeutic support and guidance.
USE COLLECTED INFO: Personalize responses using the user's information.
FULL SUPPORT: Offer coping strategies, emotional support, and mental health advice.
"""
        print("THERAPY MODE: All information collected - Full therapeutic support enabled")
    print(info_context)
    # Process message through RAG with the unified system
    session_start_index = user_data["session_start_index"]
    ai_response = process_user_message(
        user_message, 
        chat_history, 
        user_data, 
        max_summaries=3, 
        max_doctor_summaries=2,
        additional_context=info_context
    )
    
    chat_history.append({"role": "assistant", "content": ai_response})
    save_chat_history(session_key, chat_history)
    return {"response": ai_response}