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

def extract_all_information_from_message(user_message: str) -> dict:
    """Extract ALL available information from user's message using LLM"""
    
    # Dynamically build field descriptions from REQUIRED_INFORMATION
    field_descriptions = []
    for field_key, field_info in REQUIRED_INFORMATION.items():
        field_descriptions.append(f"- {field_key}: {field_info['description']}")
    
    fields_text = "\n    ".join(field_descriptions)
    field_names = list(REQUIRED_INFORMATION.keys())
    
    # Create example JSON with some of the fields
    example_fields = list(field_names)[:3]  # Take first 3 fields for example
    example_json = {field: f"example_{field}_value" for field in example_fields}
    example_json_str = json.dumps(example_json, indent=8).replace("example_", "")
    
    # Create a comprehensive extraction prompt for all information at once
    extraction_prompt = f"""
    Analyze the following user message and extract ALL available personal information.
    
    User message: "{user_message}"
    
    Extract information for these fields if present in the message:
    {fields_text}
    
    Return your response as a JSON object with these exact field names: {', '.join(field_names)}
    Only include a field if you found clear, relevant information for it.
    If no information is found for a field, do not include that field in the JSON.
    
    Example response format:
    {example_json_str}
    
    Response:
    """
    
    try:
        prompt = [
            SystemMessage(content="You are an information extraction assistant. Extract all available personal information from user messages and return it as valid JSON. Only include fields where you found clear information."),
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
                # Only return fields that are in REQUIRED_INFORMATION
                filtered_info = {k: v for k, v in extracted_info.items() if k in REQUIRED_INFORMATION}
                print(f"Extracted information: {filtered_info}")
                return filtered_info
        except json.JSONDecodeError as e:
            print(f"JSON parsing error: {e}")
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
        if key in info_needed and value and str(value).strip():
            info_needed[key]["collected"] = True
            info_needed[key]["value"] = str(value).strip()
            # Also store in the main user_data for backward compatibility
            user_data[key] = str(value).strip()
            print(f"Updated {key}: {value}")
    
    return user_data

def is_questionnaire_complete(user_data: dict) -> bool:
    """Check if all required information has been collected"""
    info_needed = user_data.get("information_needed", {})
    
    for key, info in info_needed.items():
        if info.get("required", True) and not info.get("collected", False):
            print(f"Still missing: {key}")
            return False
    
    print("Questionnaire complete!")
    return True

def get_missing_information_list(user_data: dict) -> list:
    """Get list of missing information fields"""
    info_needed = user_data.get("information_needed", {})
    missing = []
    
    for key, info in info_needed.items():
        if info.get("required", True) and not info.get("collected", False):
            missing.append(key)
    
    return missing

def generate_information_gathering_response(chat_history: list, missing_info: list) -> str:
    """Generate a warm, conversational response asking for missing information (not all at once)"""
    
    # Create context about what information is still needed
    missing_info_context = ", ".join(missing_info)
    
    # Create a prompt for natural information gathering
    info_gathering_prompt = f"""
    You are a warm, empathetic mental health assistant having a conversation with someone.
    
    Based on the conversation so far, you still need to learn about: {missing_info_context}
    
    Your task:
    1. Respond naturally to what the user just said (acknowledge their message)
    2. Ask for ONLY ONE piece of missing information in a conversational, caring way
    3. Do NOT ask for all missing information at once - that feels overwhelming
    4. Keep the tone warm, supportive, and conversational
    5. Make it feel like a natural conversation, not an interview
    6. Be brief but caring (2-3 sentences maximum)
    
    IMPORTANT: Do not provide therapeutic advice yet - just gather information warmly.
    
    Conversation history:
    """
    
    # Add conversation history to the prompt
    for message in chat_history[-6:]:  # Last 6 messages for context
        role = "User" if message["role"] == "user" else "Assistant"
        info_gathering_prompt += f"\n{role}: {message['content']}"
    
    try:
        prompt = [
            SystemMessage(content="You are a warm, empathetic mental health assistant gathering information naturally through conversation."),
            HumanMessage(content=info_gathering_prompt)
        ]
        
        response = llm(prompt)
        return response.content.strip()
        
    except Exception as e:
        print(f"Error generating information gathering response: {e}")
        return "I'd like to get to know you better so I can provide the best support. Could you share a bit more about yourself?"

def create_collected_information_context(user_data: dict) -> str:
    """Create context string with collected information for RAG"""
    collected_info = []
    info_needed = user_data.get("information_needed", {})
    
    for key, info in info_needed.items():
        if info.get("collected", False) and info.get("value"):
            collected_info.append(f"{key}: {info['value']}")
    
    if collected_info:
        return f"Known information about this person: {', '.join(collected_info)}. Use this to personalize your therapeutic support."
    else:
        return ""

# Sentiment analysis and summary functions (unchanged)
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

    # NEW APPROACH: Extract ALL available information from current message
    extracted_info = extract_all_information_from_message(user_message)
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

    # NEW LOGIC: Separate information gathering vs therapeutic responses
    if not user_data["questionnaire_completed"]:
        # INFORMATION GATHERING MODE - Direct LLM call (no RAG)
        print("INFORMATION GATHERING MODE: Using direct LLM call")
        missing_info = get_missing_information_list(user_data)
        ai_response = generate_information_gathering_response(chat_history, missing_info)
        
    else:
        # THERAPEUTIC MODE - Full RAG processing
        print("THERAPEUTIC MODE: Using full RAG processing")
        
        # Create context with collected information
        collected_info_context = create_collected_information_context(user_data)
        
        # Process message through RAG with collected information context
        ai_response = process_user_message(
            user_message, 
            chat_history, 
            user_data, 
            max_summaries=3, 
            max_doctor_summaries=2,
            additional_context=collected_info_context
        )

    # Check if questionnaire was just completed to add a transition message
    if just_completed:
        name = user_data.get("name", "")
        transition_message = f"Thank you for sharing that with me{', ' + name if name else ''}! I feel like I have a good understanding of your situation now. "
        ai_response = transition_message + ai_response
    
    chat_history.append({"role": "assistant", "content": ai_response})
    save_chat_history(session_key, chat_history)
    return {"response": ai_response}