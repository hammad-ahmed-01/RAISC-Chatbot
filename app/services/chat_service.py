# app/services/enhanced_chat_service.py
# Enhanced version of your chat_service.py with the new conversation analysis system

from datetime import datetime, timedelta
from app.services.firestore_service import get_chat_history, save_chat_history
from app.services.user_service import get_user_data, store_user_data, get_doctor_summary
from app.services.rag_service import process_user_message
from app.services.language_service import (
    detect_language, 
    get_greeting_message, 
    get_completion_message,
    get_session_end_message,
    get_error_message,
)
from app.services.conversation_analysis import (
    analyze_conversation_with_enhanced_system,
    generate_adaptive_question,
    ConversationFlowManager,
    UserPatternLearning
)
from app.config import GROQ_API_KEY
import json
from langchain_groq import ChatGroq
from langchain.schema import SystemMessage, HumanMessage
    
# Your existing REQUIRED_INFORMATION (unchanged)
REQUIRED_INFORMATION = {
    "current_condition": {
        "collected": False, 
        "value": None, 
        "required": True,
        "description": "how the user is feeling currently"
    },
    "duration": {
        "collected": False, 
        "value": None, 
        "required": True,
        "description": "how long has the user been feeling like this"
    },
    "mental_health_history": {
        "collected": False, 
        "value": None, 
        "required": True,
        "description": "user's mental health history or family history"
    },
    "physical_activity": {
        "collected": False, 
        "value": None, 
        "required": True,
        "description": "Does the user perform any sort of physical exercise"
    },
    "suicidal_thoughts": {
        "collected": False, 
        "value": None, 
        "required": True,
        "description": "Whether the user has experienced suicidal thoughts"
    }
}

# Inactivity threshold (unchanged)
INACTIVITY_THRESHOLD = timedelta(minutes=1)

def initialize_information_tracking(user_data: dict) -> dict:
    """Initialize or update the information tracking structure (unchanged)"""
    if not isinstance(user_data, dict):
        user_data = {}
    
    if "information_needed" not in user_data:
        user_data["information_needed"] = {}
        for key, value in REQUIRED_INFORMATION.items():
            user_data["information_needed"][key] = value.copy()
    else:
        for key, value in REQUIRED_INFORMATION.items():
            if key not in user_data["information_needed"]:
                user_data["information_needed"][key] = value.copy()
    
    return user_data

def extract_information_with_llm(user_message: str, missing_field: str, language: str = "english") -> dict:
    """Enhanced information extraction using LLM"""
    
    
    llm = ChatGroq(model="llama-3.3-70b-versatile", groq_api_key=GROQ_API_KEY, temperature=0)
    
    field_info = REQUIRED_INFORMATION.get(missing_field, {})
    field_description = field_info.get("description", missing_field)
    
    extraction_prompt = f"""
You are extracting information from a user's message for mental health assessment.

User message: "{user_message}"
Information to extract: {missing_field}
Field description: {field_description}
Language: {language}

INSTRUCTIONS:
1. ONLY extract information that is clearly and explicitly mentioned
2. Accept both positive and negative responses as valid information
3. Treat "no", "never", "don't have" as VALID informative responses
4. If extracting from Roman Urdu, translate the response to English
5. DO NOT use placeholder values or make assumptions
6. Return empty JSON {{}} if no clear information is found

EXAMPLES of valid extractions:
- "No, I don't have depression" → {{"mental_health_history": "no history of depression"}}
- "I've been feeling anxious for 2 weeks" → {{"duration": "2 weeks"}}
- "Main bilkul theek hoon" → {{"current_condition": "feeling fine"}}
- "Kabhi suicide ka khayal nahi aya" → {{"suicidal_thoughts": "no suicidal thoughts"}}

Return ONLY valid JSON for the field "{missing_field}":
"""
    
    try:
        response = llm([
            SystemMessage(content="You extract information accurately, treating denial as valid information. Always return valid JSON."),
            HumanMessage(content=extraction_prompt)
        ])
        
        response_text = response.content.strip()
        
        # Extract JSON from response
        import re
        json_match = re.search(r'\{.*\}', response_text, re.DOTALL)
        if json_match:
            extracted_info = json.loads(json_match.group())
            
            # Filter to only include the requested field
            if missing_field in extracted_info and extracted_info[missing_field]:
                return {missing_field: extracted_info[missing_field]}
        
        return {}
        
    except Exception as e:
        print(f"LLM extraction failed: {e}")
        return {}

def update_collected_information(user_data: dict, extracted_info: dict) -> dict:
    """Update user_data with newly extracted information (unchanged)"""
    if not extracted_info:
        return user_data
    
    info_needed = user_data.get("information_needed", {})
    
    for key, value in extracted_info.items():
        if key in info_needed and value and str(value).strip():
            info_needed[key]["collected"] = True
            info_needed[key]["value"] = str(value).strip()
            user_data[key] = str(value).strip()
            print(f"✅ Successfully collected {key}: {value}")
    
    return user_data

def is_questionnaire_complete(user_data: dict) -> bool:
    """Check if all required information has been collected (unchanged)"""
    info_needed = user_data.get("information_needed", {})
    
    for key, info in REQUIRED_INFORMATION.items():
        if info.get("required", True):
            user_field_info = info_needed.get(key, {})
            if not user_field_info.get("collected", False):
                return False
    
    return True

def get_missing_information_list(user_data: dict) -> list:
    """Get list of missing information fields (unchanged)"""
    info_needed = user_data.get("information_needed", {})
    missing = []
    
    for key, info in REQUIRED_INFORMATION.items():
        if info.get("required", True):
            user_field_info = info_needed.get(key, {})
            if not user_field_info.get("collected", False):
                missing.append(key)
    
    return missing

def generate_enhanced_information_gathering_response(enhanced_analysis: dict, missing_info: list, 
                                                  user_data: dict, language: str = "english") -> str:
    """Generate response using enhanced conversation analysis"""
    
    if not missing_info:
        return get_completion_message(language)
    
    analysis = enhanced_analysis['analysis']
    next_action = enhanced_analysis['next_action']
    user_approach = enhanced_analysis['user_approach']
    
    # Check if we should switch to RAG instead of continuing questions
    if next_action['action'] in ['SWITCH_TO_RAG', 'END_SESSION']:
        return None  # Signal to switch to RAG or end session
    
    # Get the next field to ask about
    priority_order = ['current_condition', 'duration', 'mental_health_history', 'physical_activity', 'suicidal_thoughts']
    next_field = None
    
    for field in priority_order:
        if field in missing_info:
            next_field = field
            break
    
    if not next_field:
        next_field = missing_info[0]
    
    # Get attempt count for this field
    flow_manager = user_data.get('flow_manager')
    attempt_count = 1
    if flow_manager and hasattr(flow_manager, 'question_attempts'):
        attempt_count = flow_manager.question_attempts.get(next_field, 1)
    
    # Get field description
    field_info = REQUIRED_INFORMATION.get(next_field, {})
    field_description = field_info.get("description", next_field)
    
    # Generate adaptive question
    adaptive_question = generate_adaptive_question(
        missing_field=next_field,
        field_description=field_description,
        analysis=analysis,
        attempt_count=attempt_count,
        language=language,
        user_approach=user_approach
    )
    
    print(f"📝 Generated adaptive question for {next_field} (attempt {attempt_count}): {adaptive_question}")
    
    return adaptive_question

def create_collected_information_context(user_data: dict, language: str = "english") -> str:
    """Create context string with collected information for RAG (unchanged)"""
    collected_info = []
    info_needed = user_data.get("information_needed", {})
    
    for key, info in info_needed.items():
        if info.get("collected", False) and info.get("value"):
            collected_info.append(f"{key}: {info['value']}")
    
    if collected_info:
        if language == 'roman_urdu':
            return f"Is shakhs ke baare mein maloom hai: {', '.join(collected_info)}. Is information ko use karke personalized therapeutic support dijiye."
        else:
            return f"Known information about this person: {', '.join(collected_info)}. Use this to personalize your therapeutic support."
    else:
        return ""

# Legacy sentiment analysis functions (keep for compatibility)
def analyze_sentiment(message: str) -> dict:
    """Legacy function - now returns enhanced analysis in VADER format for compatibility"""
    from nltk.sentiment.vader import SentimentIntensityAnalyzer
    import nltk
    
    try:
        nltk.download('vader_lexicon', quiet=True)
        analyzer = SentimentIntensityAnalyzer()
        return analyzer.polarity_scores(message)
    except:
        # Fallback if VADER fails
        return {'neg': 0.1, 'neu': 0.7, 'pos': 0.2, 'compound': 0.1}

def generate_conversation_summary(chat_history, session_start_index, previous_summary=None, language="english"):
    """Generate conversation summary (unchanged from original)"""
    from langchain_groq import ChatGroq
    from langchain.schema import SystemMessage, HumanMessage
    
    llm = ChatGroq(model="llama-3.3-70b-versatile", groq_api_key=GROQ_API_KEY, temperature=0)
    
    user_messages = [msg["content"] for msg in chat_history[session_start_index:] if msg.get("role") == "user"]
    if not user_messages:
        if language == 'roman_urdu':
            summary_text = "Is session mein koi messages nahi hain jo summarize kar sakein."
        else:
            summary_text = "No messages to summarize in this session."
    else:
        conversation_text = " ".join(user_messages)
        
        if language == 'roman_urdu':
            prompt_text = (
                "Aap ek helpful assistant hain jo conversations ka summary banate hain. "
                "Is conversation ka concise summary dijiye, key points aur overall tone par focus karte hue."
            )
        else:
            prompt_text = (
                "You are a helpful assistant that summarizes conversations. "
                "Summarize the following conversation concisely, focusing on the key points and overall tone."
            )
            
        if previous_summary:
            if language == 'roman_urdu':
                prompt_text += f"\nPichla summary: {previous_summary}\n"
            else:
                prompt_text += f"\nPrevious summary: {previous_summary}\n"
                
        prompt_text += f"Current conversation: {conversation_text}"

        prompt = [
            SystemMessage(content=prompt_text),
            HumanMessage(content="")
        ]
        
        try:
            summary_output = llm(prompt)
            summary_text = summary_output.content.strip()
        except:
            summary_text = "Summary generation failed."

    timestamp = datetime.now().isoformat()
    return {
        "summary": summary_text,
        "emotional_summary": "Enhanced analysis system active",
        "timestamp": timestamp
    }

async def process_chat(session_key: str, user_message: str):
    """Enhanced chat processing using the new conversation analysis system"""
    
    # Detect language from user message
    detected_language = detect_language(user_message)
    print(f"🔍 Detected language: {detected_language}")
    
    # Fetch user data and chat history
    user_data_raw = get_user_data(session_key)
    if isinstance(user_data_raw, dict):
        user_data = user_data_raw
    else:
        user_data = {}
    
    chat_history = get_chat_history(session_key) or []
    doctor_summary = get_doctor_summary(session_key)

    # Store user's language preference
    user_data["preferred_language"] = detected_language

    # Initialize information tracking and conversation managers
    user_data = initialize_information_tracking(user_data)
    
    # Initialize conversation analysis managers
    if 'flow_manager' not in user_data:
        user_data['flow_manager'] = ConversationFlowManager()
    if 'pattern_learner' not in user_data:
        user_data['pattern_learner'] = UserPatternLearning()

    # Initialize other fields
    user_data["doctor_summary"] = doctor_summary
    if "past_summaries" not in user_data:
        user_data["past_summaries"] = []
    if "session_start_index" not in user_data:
        user_data["session_start_index"] = 0

    # Get current time and handle inactivity
    current_time = datetime.now()
    last_interaction = user_data.get("last_interaction")
    
    if last_interaction:
        last_interaction_time = datetime.fromisoformat(last_interaction)
        time_since_last = current_time - last_interaction_time
        if time_since_last >= INACTIVITY_THRESHOLD and chat_history:
            session_start_index = user_data["session_start_index"]
            previous_summary = user_data["past_summaries"][-1]["summary"] if user_data.get("past_summaries") else None
            
            summary_data = generate_conversation_summary(chat_history, session_start_index, previous_summary, "english")
            user_data["past_summaries"].append(summary_data)
            user_data["session_start_index"] = len(chat_history)
            
            store_user_data(session_key, user_data)
            print(f"[AUTO-SUMMARY] Generated due to inactivity: {summary_data['summary'][:100]}...")

    # Add user message to chat history with legacy sentiment for compatibility
    legacy_sentiment = analyze_sentiment(user_message)
    chat_history.append({"role": "user", "content": user_message, "sentiment": legacy_sentiment})

    # 🚀 ENHANCED CONVERSATION ANALYSIS
    enhanced_analysis = analyze_conversation_with_enhanced_system(
        message=user_message,
        chat_history=chat_history,
        user_data=user_data,
        session_key=session_key,
        language=detected_language
    )
    
    print(f"🧠 Enhanced Analysis: {enhanced_analysis['analysis']['engagement_level']} | {enhanced_analysis['next_action']['strategy']}")

    # Extract information with enhanced LLM extraction
    missing_info = get_missing_information_list(user_data)
    if missing_info:
        next_field = missing_info[0]  # Get next field to ask about
        extracted_info = extract_information_with_llm(user_message, next_field, detected_language)
        user_data = update_collected_information(user_data, extracted_info)

    # Update questionnaire completion status
    was_completed_before = user_data.get("questionnaire_completed", False)
    user_data["questionnaire_completed"] = is_questionnaire_complete(user_data)
    just_completed = not was_completed_before and user_data["questionnaire_completed"]

    # Update last interaction time
    user_data["last_interaction"] = current_time.isoformat()
    store_user_data(session_key, user_data)

    # Handle session ending requests
    end_session_phrases = {
        'english': ["end session", "goodbye", "exit", "bye", "end"],
        'roman_urdu': ["session khatam", "khuda hafiz", "allah hafiz", "bye", "alvida", "khatam"]
    }
    
    user_message_lower = user_message.lower()
    is_ending_session = any(phrase in user_message_lower for phrase in 
                           end_session_phrases.get(detected_language, end_session_phrases['english']))
    
    if is_ending_session or enhanced_analysis['next_action']['action'] == 'END_SESSION':
        session_start_index = user_data["session_start_index"]
        previous_summary = user_data["past_summaries"][-1]["summary"] if user_data.get("past_summaries") else None
        
        summary_data = generate_conversation_summary(chat_history, session_start_index, previous_summary, "english")
        user_data["past_summaries"].append(summary_data)
        user_data["session_start_index"] = len(chat_history)
        
        if detected_language == 'roman_urdu':
            goodbye_msg = "Aap se baat kar ke acha laga. Allah hafiz aur khyal rakhiye apna!"
        else:
            goodbye_msg = "Thank you for chatting with me today. Take care and feel free to return anytime!"
        
        chat_history.append({"role": "assistant", "content": goodbye_msg})
        save_chat_history(session_key, chat_history)
        store_user_data(session_key, user_data)
        
        print(f"[SESSION END] Summary stored: {summary_data['summary'][:100]}...")
        return {"response": goodbye_msg}

    # Handle first-time users
    if len(chat_history) == 1:
        greeting = get_greeting_message(detected_language)
        user_data["session_start_index"] = 1
        chat_history.append({"role": "assistant", "content": greeting})
        save_chat_history(session_key, chat_history)
        store_user_data(session_key, user_data)
        return {"response": greeting}

    # 🎯 DECISION LOGIC: Information Gathering vs Therapeutic Mode
    
    # Check if enhanced analysis says we should switch to RAG
    if enhanced_analysis['next_action']['action'] in ['SWITCH_TO_RAG', 'DE_ESCALATE']:
        print(f"🔄 SWITCHING TO RAG MODE: {enhanced_analysis['next_action']['reasoning']}")
        
        # Mark questionnaire as completed to prevent future forcing
        user_data["questionnaire_completed"] = True
        user_data["completion_reason"] = enhanced_analysis['next_action']['strategy']
        
        # Create supportive context
        supportive_context = enhanced_analysis['next_action']['context']
        collected_info_context = create_collected_information_context(user_data, detected_language)
        full_context = f"{supportive_context}\n{collected_info_context}".strip()
        
        # Process through RAG with supportive context
        ai_response = process_user_message(
            user_message, 
            chat_history, 
            user_data, 
            max_summaries=3, 
            max_doctor_summaries=2,
            additional_context=full_context,
            language=detected_language
        )
        
    elif not user_data["questionnaire_completed"]:
        print("📝 INFORMATION GATHERING MODE: Using enhanced adaptive questioning")
        
        missing_info = get_missing_information_list(user_data)
        ai_response = generate_enhanced_information_gathering_response(
            enhanced_analysis, missing_info, user_data, detected_language
        )
        
        # If enhanced analysis says to switch to RAG, do it
        if ai_response is None:
            print("🔄 Enhanced analysis recommended switching to RAG during info gathering")
            user_data["questionnaire_completed"] = True
            user_data["completion_reason"] = "enhanced_analysis_recommendation"
            
            collected_info_context = create_collected_information_context(user_data, detected_language)
            ai_response = process_user_message(
                user_message, 
                chat_history, 
                user_data,
                additional_context=collected_info_context,
                language=detected_language
            )
        
    else:
        print("🎭 THERAPEUTIC MODE: Using full RAG processing")
        
        collected_info_context = create_collected_information_context(user_data, detected_language)
        ai_response = process_user_message(
            user_message, 
            chat_history, 
            user_data, 
            max_summaries=3, 
            max_doctor_summaries=2,
            additional_context=collected_info_context,
            language=detected_language
        )

    # Add completion message if questionnaire was just completed
    if just_completed:
        name = ""
        info_needed = user_data.get("information_needed", {})
        if "name" in info_needed and info_needed["name"].get("collected", False):
            name = info_needed["name"].get("value", "")
        
        transition_message = get_completion_message(detected_language, name)
        ai_response = transition_message + " " + ai_response
    
    # Save final state
    chat_history.append({"role": "assistant", "content": ai_response})
    save_chat_history(session_key, chat_history)
    store_user_data(session_key, user_data)
    
    return {"response": ai_response}