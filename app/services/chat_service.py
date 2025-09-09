# app/services/chat_service.py
import json
import re
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
    if "information_needed" not in user_data:
        user_data["information_needed"] = REQUIRED_INFORMATION.copy()
    else:
        # Update existing structure with any new fields
        for key, value in REQUIRED_INFORMATION.items():
            if key not in user_data["information_needed"]:
                user_data["information_needed"][key] = value.copy()
    
    return user_data

def get_missing_information(user_data: dict) -> list:
    """Get list of information that still needs to be collected"""
    missing = []
    info_needed = user_data.get("information_needed", {})
    
    for key, info in info_needed.items():
        if info.get("required", True) and not info.get("collected", False):
            missing.append(key)
    
    return missing

def extract_information_from_message(user_message: str, missing_info: list) -> dict:
    """Extract information from user's message using LLM"""
    if not missing_info:
        return {}
    
    # Create extraction prompt
    missing_descriptions = []
    for info_key in missing_info:
        if info_key in REQUIRED_INFORMATION:
            description = REQUIRED_INFORMATION[info_key]["description"]
            missing_descriptions.append(f"- {info_key}: {description}")
    
    extraction_prompt = f"""
    Analyze the following user message and extract any personal information that matches these categories:
    {chr(10).join(missing_descriptions)}
    
    User message: "{user_message}"
    
    Return your response as a JSON object where keys are the category names and values are the extracted information.
    Only include categories where you found clear information. If no relevant information is found, return an empty JSON object.
    
    Example response format:
    {{"name": "John", "age": "25"}}
    
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

def create_information_gathering_context(missing_info: list, user_data: dict) -> str:
    """Create context for the RAG system about what information is still needed"""
    if not missing_info:
        return ""
    
    # Create information gathering context without mentioning backend processes
    context_parts = []
    for info_key in missing_info:
        if info_key in REQUIRED_INFORMATION:
            description = REQUIRED_INFORMATION[info_key]["description"]
            context_parts.append(f"- {description}")
    
    # Check if this is a very new conversation (first few messages)
    chat_length = len(user_data.get("chat_history", []))
    is_early_conversation = chat_length <= 3
    
    if is_early_conversation:
        priority_level = "URGENT"
        guidance = "Start gathering information immediately. Be direct but warm."
    else:
        priority_level = "HIGH"
        guidance = "Continue gathering missing information. Be persistent but supportive."
    
    # Create varied example phrases for natural information gathering
    example_phrases = [
        "To better understand and support you, could you tell me [information]?",
        "I'd like to know a bit more about you - what's [information]?",
        "It would help me to know [information]. Could you share that with me?",
        "May I ask about [information]? This helps me provide better support.",
        "Could you tell me [information]? I want to understand your situation better.",
        "I'd appreciate knowing [information] so I can help you more effectively.",
        "What's [information]? This will help me tailor my responses to you.",
        "To give you the best support, could you share [information] with me?"
    ]
    
    context = f"""
INFORMATION GATHERING STATUS: INCOMPLETE - {priority_level} PRIORITY
Missing required information:
{chr(10).join(context_parts)}

GUIDANCE:
{guidance}

You need to collect this information to better understand and support the user. 
Ask for ONE missing piece of information in your response.

EXAMPLE WAYS TO ASK (vary your approach):
{chr(10).join([f"- {phrase}" for phrase in example_phrases[:4]])}

Focus on information gathering while providing brief supportive responses.
Vary your questioning style to keep the conversation natural.
"""
    
    return context
def create_completion_message(user_data: dict) -> str:
    """Create a message when onboarding is completed"""
    name = user_data.get("name", "")
    
    message = f"Thank you for sharing that with me{', ' + name if name else ''}! "
    message += "I feel like I have a good understanding of your situation now. "
    message += "How can I best support you today?"
    
    return message

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
def  generate_conversation_summary(chat_history, session_start_index, previous_summary=None):
    user_messages = [msg["content"] for msg in chat_history[session_start_index:] if msg.get("role") == "user"]
    # in case of no new messages
    if not user_messages:
        # summary_text = " "
        return None
        # summary_text = "No messages to summarize in this session."
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

    # Initialize information tracking
    user_data = initialize_information_tracking(user_data)
    print("Initializing user_data")
    print(user_data)
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
                "session_start_msg": session_start_index,
                "session_end_msg": len(chat_history)-1,
                "timestamp": summary_data["timestamp"]
            })
            user_data["last_summarized_index"] = len(chat_history)
            user_data["session_start_index"] = len(chat_history)  # Start a new session
            user_data["session_agg_sentiment"] = reset_sentiment_aggregate()  # Reset sentiment metrics
            # Prints the chat history into the UI
            # chat_history.append({
            #     "role": "assistant",
            #     "content": f"Auto-generated summary due to inactivity:\n{summary_text}"
            # })
            save_chat_history(session_key, chat_history)
            store_user_data(session_key, user_data)
            print(f"Here's a summary of our last session:\n{summary_text}")

    # Analyze sentiment of the user's message
    sentiment = analyze_sentiment(user_message)
    chat_history.append({"role": "user", "content": user_message, "sentiment": sentiment})
    
    # Extract information from user's message BEFORE checking completion status
    missing_info_before = get_missing_information(user_data)
    extracted_info = extract_information_from_message(user_message, missing_info_before)
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
            "session_start_msg": session_start_index,
            "session_end_msg": len(chat_history)-1,
            "timestamp": summary_data["timestamp"]
        })
        user_data["last_summarized_index"] = len(chat_history)
        user_data["session_start_index"] = len(chat_history)  # Start a new session
        user_data["session_agg_sentiment"] = reset_sentiment_aggregate()  # Reset sentiment metrics
        # Prints the chat history into the UI
        # chat_history.append({
        #     "role": "assistant",
        #     "content": "Here is your session summary:\n" + summary_text,
        # })
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
    if just_completed:
        completion_message = create_completion_message(user_data)
        chat_history.append({"role": "assistant", "content": completion_message})
        save_chat_history(session_key, chat_history)
        store_user_data(session_key, user_data)
        return {"response": completion_message}
    # Questionnaire logic with validation
    missing_info = get_missing_information(user_data)
    
    # Create context based on onboarding status
    if user_data.get("questionnaire_completed", False):
        # Phase 2: Therapeutic conversation
        info_context = ""
        print("Phase 2: Therapeutic conversation mode")
    else:
        # Phase 1: Information gathering
        info_context = create_information_gathering_context(missing_info, user_data)
        print(f"Phase 1: Natural conversation mode - Still learning about: {missing_info}")
        print(f"Info context: {info_context[:200]}...")  # Debug print
    # Process message through RAG with information gathering context
    session_start_index = user_data["session_start_index"]
    ai_response = process_user_message(
        user_message, 
        chat_history, 
        user_data, 
        max_summaries=3, 
        max_doctor_summaries=2,
        additional_context=info_context  # Pass the information gathering context
    )
    
    chat_history.append({"role": "assistant", "content": ai_response})
    save_chat_history(session_key, chat_history)
    return {"response": ai_response}