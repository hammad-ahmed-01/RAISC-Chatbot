# app/services/chat_service.py
import nltk
from nltk.sentiment.vader import SentimentIntensityAnalyzer
from datetime import datetime, timedelta
from langchain_groq import ChatGroq
from langchain.schema import SystemMessage, HumanMessage
from app.services.firestore_service import get_chat_history, save_chat_history
from app.services.user_service import get_user_data, store_user_data, get_doctor_summary
from app.services.rag_service import process_user_message
from app.services.language_service import (
    get_greeting_message, 
    get_completion_message,
    get_risk_intervention_message,
    get_session_end_message,
    get_error_message,
    get_language_context_for_prompts
)
from app.config import GROQ_API_KEY
import json
import re

nltk.download('vader_lexicon')

# Initialize the VADER analyzer and LLM once
analyzer = SentimentIntensityAnalyzer()
llm = ChatGroq(model="openai/gpt-oss-20b", groq_api_key=GROQ_API_KEY, temperature=0)

# Inactivity threshold
INACTIVITY_THRESHOLD = timedelta(minutes=1)

# Define the information we need to collect
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
def initialize_information_tracking(user_data: dict) -> dict:
    """Initialize or update the information tracking structure
    DYNAMICALLY based on REQUIRED_INFORMATION fields"""
    # Ensure user_data is a dictionary
    if not isinstance(user_data, dict):
        user_data = {}
    
    if "information_needed" not in user_data:
        # Create a deep copy to avoid modifying the original REQUIRED_INFORMATION
        user_data["information_needed"] = {}
        for key, value in REQUIRED_INFORMATION.items():
            user_data["information_needed"][key] = value.copy()
    else:
        # Update existing structure with any new fields from REQUIRED_INFORMATION
        for key, value in REQUIRED_INFORMATION.items():
            if key not in user_data["information_needed"]:
                user_data["information_needed"][key] = value.copy()
    
    return user_data
def extract_all_information_from_message(user_message: str, current_missing_field: str) -> dict:
    """Extract ONLY information for the current missing field from user's message using LLM with language awareness
    Always stores extracted information in English for consistency
    ONLY extracts real information - never placeholder values"""
    
    # Get the description for the current missing field
    current_field_info = REQUIRED_INFORMATION.get(current_missing_field, {})
    current_field_description = current_field_info.get('description', current_missing_field)
    
    # Create example JSON with only the current field
    example_json = {current_missing_field: f"actual_{current_missing_field}_value"}
    example_json_str = json.dumps(example_json, indent=4)
    
    # Language-aware extraction prompt that ONLY outputs real information for the current field
    # English extraction with strict requirements
    # extraction_prompt = f"""
    # Analyze the following user message and extract ONLY information that is clearly and explicitly mentioned.
    
    # User message: "{user_message}"
    
    # Extract information for this field ONLY if clearly present in the message:
    # - {current_missing_field}: {current_field_description}
    
    # LENIENT REQUIREMENTS:
    # 1. Extract information that is reasonably clear or can be inferred from context
    # 2. Accept vague responses and interpret them appropriately
    # 3. DO NOT include placeholder values like "No information available", "Not specified", "Unknown"
    # 4. Include information even if it's not perfectly explicit - users often speak casually
    # 5. TREAT DENIAL/NEGATIVE RESPONSES AS VALID INFORMATION (e.g., "not aware of", "don't think so", "nothing", "no")
    # 6. Accept indirect answers and reasonable interpretations
    # 7. If absolutely no relevant information can be extracted, return an empty JSON object: {{}}
    # 8. You have to ONLY extract the following field: {current_missing_field}. If the user gives DENIAL/NEGATIVE RESPONSES, Treat it as valid information and extract that as well.
    
    # EXAMPLES of what to extract (including vague/indirect responses):
    # - "I'm feeling very anxious" → "current_condition": "very anxious"
    # - "I've been like this for 3 weeks" → "duration": "3 weeks" 
    # - "I have a history of depression" → "mental_health_history": "history of depression"
    # - "I exercise daily" → "physical_activity": "exercises daily"
    # - "I have never had suicidal thoughts" → "suicidal_thoughts": "no suicidal thoughts"
    # - "No, I don't have any mental health history" → "mental_health_history": "no mental health history"
    # - "I don't exercise at all" → "physical_activity": "no exercise"
    # - "No, I've never thought about suicide" → "suicidal_thoughts": "no suicidal thoughts"
    # - "I haven't been feeling this way for long" → "duration": "short duration"
    # - "Not that I'm aware of" → "mental_health_history": "no known mental health history"
    # - "I haven't noticed anything out of the ordinary" → "mental_health_history": "no mental health history"
    # - "There hasn't been as far as I'm aware" → "mental_health_history": "no mental health history"
    # - "Nothing comes to mind" → "mental_health_history": "no mental health history"
    # - "I don't think so" → "mental_health_history": "no mental health history"
    # - "Not really" → "mental_health_history": "no mental health history"
    # - "I'm okay I guess" → "current_condition": "okay"
    # - "Could be better" → "current_condition": "not great"
    # - "Same as usual" → "current_condition": "usual state"
    # - "A while now" → "duration": "some time"
    # - "Recently" → "duration": "recent"
    # - "It's been tough lately" → "current_condition": "struggling recently"
    # - "Sometimes I go for walks" → "physical_activity": "occasional walking"
    # - "Used to exercise but not anymore" → "physical_activity": "previously exercised, not currently"
    # - "I try to stay active" → "physical_activity": "tries to stay active"
    # - "Never really thought about it" → "suicidal_thoughts": "no suicidal thoughts"
    
    # EXAMPLES of what NOT to extract:
    # - Completely unrelated responses "What's the weather?" → {{}} (not relevant to any field)
    # - Pure greetings "Hi there!" → {{}} (not relevant information)
    
    # IMPORTANT: 
    # Be lenient and interpretive. 
    # Users often give casual, conversational answers that contain useful information even if not perfectly direct.
    # If the user gives resistance or does not want to answer a question, leave that question as empty and move on to the next. 

    # Current field to extract: {current_missing_field}
    
    # Example response format (only include the field if it has actual data):
    # {example_json_str}
    
    # Response (JSON with only the current field if it has clear information):
    # """

    extraction_prompt = f"""
    You are a therapeutic context extractor. Your goal is to identify **relevant psychological or behavioral information**
    from a casual, conversational message — as a human therapist would understand it.

    User message:
    "{user_message}"

    ---

    ### OBJECTIVE
    Extract *only* the information that clearly or reasonably relates to the specified field.
    Each field describes a single topic of mental or physical well-being.

    ### FIELD TO ANALYZE
    - Field: "{current_missing_field}"
    - Description: {current_field_description}

    ---

    ### EXTRACTION GUIDELINES

    1. **Lenient and Context-Aware**
    - Understand everyday language, slang, or vague phrases that still communicate a feeling, behavior, or history.
    - Use emotional tone or temporal cues to infer approximate meaning if the intent is clear.
    - Accept statements like “sort of okay”, “eh… not great”, “better than before” as valid data.

    2. **Negation and Denial Handling**
    - Treat any form of “no”, “not really”, “none”, “never”, “don’t think so”, “I guess not”, “not that I recall” as *explicit denials*.
    - Extract that as "no <field_name>" (e.g., "no suicidal thoughts").

    3. **Ambiguity & Hedges**
    - When users hedge (e.g., “I guess”, “maybe”, “kind of”, “sometimes”), include the information but label uncertainty.
    - Example: "physical_activity": "sometimes exercises (uncertain frequency)".

    4. **Multi-Aspect Answers**
    - If multiple ideas are given (e.g., “Mentally fine but physically tired”), prioritize the aspect relevant to the current field.
    - Ignore unrelated topics even if emotionally adjacent.

    5. **Temporal or Comparative Phrases**
    - Normalize relative expressions:
        - “A while now” → “for some time”
        - “Not long” → “recent”
        - “Used to…” → “previously, not now”
        - “Better than before” → “improving condition”

    6. **Emotional Tone Mapping (for general use)**
    - Recognize emotional expressions such as:
        - “meh”, “could be worse”, “alright I guess” → mild/neutral
        - “rough”, “bad”, “overwhelmed”, “not okay” → struggling
        - “fine”, “good”, “okay” → stable
    - Use short, normalized summaries (e.g., "current_condition": "feeling overwhelmed").

    7. **Unwillingness / Deflection**
    - If the user avoids the question (“I’d rather not say”, “skip this one”), return {{}}, do not fabricate.

    8. **Non-Relevant Responses**
    - If unrelated (e.g., greeting, joke, small talk), return {{}}

    9. **Output Format**
    - Return a single valid JSON object with only the relevant key if clear data exists.
    - Example:
        ```json
        {{ "{current_missing_field}": "some extracted text" }}
        ```
    - Otherwise:
        ```json
        {{}}
        ```

    ---

    ### EXAMPLES
    User: “I’ve been feeling weird lately, not exactly sad, just off.”
    → {{ "current_condition": "feeling off, not sad" }}

    User: “I used to exercise more but not these days.”
    → {{ "physical_activity": "previously exercised, not currently" }}

    User: “No, not really. I don’t think I ever have.”
    → {{ "suicidal_thoughts": "no suicidal thoughts" }}

    User: “Hmm depends, I guess sometimes I feel anxious when alone.”
    → {{ "current_condition": "sometimes anxious when alone" }}

    User: “Can we talk about something else?”
    → {{}}

    ---

    Now extract only the information for "{current_missing_field}".
    Return your final JSON below:
    """

    try:
        system_message = f"You are a strict information extraction assistant. You ONLY extract information for the field '{current_missing_field}' that is clearly and explicitly mentioned in the message. You NEVER use placeholder values or make assumptions. If no clear information is found, you return an empty JSON object."
        
        prompt = [
            SystemMessage(content=system_message),
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
                
                # Filter to only include the current missing field if it exists and has value
                filtered_info = {}
                if current_missing_field in extracted_info and extracted_info[current_missing_field] and str(extracted_info[current_missing_field]).strip():
                    filtered_info[current_missing_field] = extracted_info[current_missing_field]
                
                if filtered_info:
                    print(f"Extracted information for {current_missing_field}: {filtered_info}")
                    return filtered_info
                else:
                    print(f"No valid information extracted for field: {current_missing_field}")
                    return {}
        except json.JSONDecodeError as e:
            print(f"JSON parsing error: {e}")
            pass
        
        print(f"No information could be extracted from the message for field: {current_missing_field}")
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
    """Check if all required information has been collected
    DYNAMICALLY checks based on REQUIRED_INFORMATION fields"""
    info_needed = user_data.get("information_needed", {})
    
    # Check all fields in REQUIRED_INFORMATION
    for key, info in REQUIRED_INFORMATION.items():
        if info.get("required", True):
            # Check if this field exists in user's info_needed and is collected
            user_field_info = info_needed.get(key, {})
            if not user_field_info.get("collected", False):
                print(f"Still missing: {key}")
                return False
    
    print("Questionnaire complete!")
    return True

def get_missing_information_list(user_data: dict) -> list:
    """Get list of missing information fields
    DYNAMICALLY based on REQUIRED_INFORMATION fields"""
    info_needed = user_data.get("information_needed", {})
    missing = []
    
    # Check all fields in REQUIRED_INFORMATION
    for key, info in REQUIRED_INFORMATION.items():
        if info.get("required", True):
            # Check if this field exists in user's info_needed and is collected
            user_field_info = info_needed.get(key, {})
            if not user_field_info.get("collected", False):
                missing.append(key)
    
    return missing

def generate_dynamic_question(missing_field: str, field_description: str, chat_history: list) -> str:
    """
    Generate a contextual question for a specific missing field using LLM
    No premade prompts - fully dynamic based on field name and description
    """
    
    # Get the user's last message for context
    user_last_message = ""
    if chat_history:
        last_messages = [msg for msg in chat_history[-3:] if msg.get("role") == "user"]
        if last_messages:
            user_last_message = last_messages[-1]["content"]
    
    # Create dynamic question generation prompt
    # generation_prompt = f"""
    # You are a warm, empathetic mental health assistant that only speaks ENGLISH.
    
    # TASK: Generate a natural, conversational question to ask about: "{missing_field}" 
    
    # FIELD INFORMATION:
    # - Field name: {missing_field}
    # - What we need to know: {field_description}
    
    # USER'S LAST MESSAGE: "{user_last_message}"

    # The user's last message "{user_last_message}", in addition to containing answer of the asked question, can be of various types.
    # Case 1. It could contain a counter question or be unrelated to the questionnaire.

    # In case of case 1, Answer the user's counter question if it lies in the domain of mental health, 
    # or is part of conversation flow, and then ask the natural question.


    # Generate a natural question about "{missing_field}"
    # """
    generation_prompt = f"""
    You are **RAISC**, a warm, empathetic mental health assistant who speaks **only English**. 
    You engage users in a natural, therapist-like conversation to gently collect details about their mental and physical wellbeing.

    ---

    ### TASK
    Generate a natural, conversational **reply** that:
    1. Feels emotionally intelligent, validating, and non-judgmental.
    2. Gently guides the user toward answering the next question about **"{missing_field}"**.
    3. Adapts to the user's last message tone, even if it's vague, deflective, emotional, or conversationally off-topic.

    ---

    ### FIELD INFORMATION
    - **Field name:** {missing_field}
    - **Goal:** {field_description}

    ---

    ### USER'S LAST MESSAGE
    "{user_last_message}"

    ---

    ### BEHAVIORAL RULES

    1. **Empathetic First, Functional Second**
    - Always start by *acknowledging* or *reflecting* the user’s previous message naturally (e.g., “That sounds like it’s been tough” or “I’m glad you shared that”).
    - Then, smoothly transition into the next question about **{missing_field}**.

    2. **Context Awareness**
    - If the user's message seems emotional, respond with care before asking the next question.
    - If they sound neutral or factual, keep tone conversational and curious.
    - If they ask a *counter question* about mental health or something relevant, **briefly answer it** and then return to the main question.
    - If the response is **completely unrelated**, redirect politely and naturally.

    3. **Natural Question Generation**
    - The question should feel like a continuation of a conversation — **not** a form or survey.
    - Avoid robotic or repetitive phrasing.
    - Use phrasing styles like:
        - “Could you tell me a bit about…”  
        - “I’m curious to know…”  
        - “How has it been for you when it comes to…”  
        - “Would you say…”  
        - “Have you noticed…”  

    4. **Tone & Language**
    - Keep tone calm, friendly, and professional.
    - Avoid clinical terms unless the user uses them first.
    - Prefer emotionally soft and inclusive phrasing like “sometimes”, “a bit”, “generally”, “these days”.

    5. **Conversation Flow**
    - If the user gave *some* relevant info but not complete, **gently ask for clarification or elaboration**.
    - If they already answered this field fully, **acknowledge and move to the next topic** instead of repeating.

    6. **Safety Considerations**
    - If the user expresses signs of severe distress, sadness, or self-harm, respond compassionately (e.g., “I’m really sorry you’re feeling this way”) and ask the question softly or offer to pause.
    - Never ignore emotional cues.

    ---

    ### EXAMPLES

    User message: "I've just been feeling weird lately, not really sad though."
    → Response: "That makes sense, sometimes feelings can be hard to define. Could you tell me a bit about how you’ve been feeling overall these days?"

    User message: "I don't really exercise much anymore."
    → Response: "That’s okay, a lot of people’s routines change over time. How often do you find yourself doing any physical activity lately?"

    User message: "Hmm, do you think anxiety can cause this?"
    → Response: "It can sometimes, yes — our bodies and minds are closely connected. Speaking of that, how would you describe your current condition right now?"

    User message: "Can we skip this question?"
    → Response: "Of course, that’s totally fine. We can come back to it later if you’d like."

    User message: "I've been okay mentally but my body’s been so tired."
    → Response: "I see, sounds like your physical side has been more affected lately. How’s your energy been in general?"

    ---

    Now, generate a short, natural, empathetic message that follows these rules.
    It should *feel like part of a human conversation*, not a script.
    """
    
    try:
        prompt = [
            SystemMessage(content="You are a skilled mental health assistant who asks natural, empathetic questions. Generate warm, conversational questions that feel like talking to a caring friend or counselor."),
            HumanMessage(content=generation_prompt)
        ]
        
        response = llm(prompt)
        generated_question = response.content.strip()
        
        # Ensure the response isn't too long
        sentences = generated_question.split('. ')
        if len(sentences) > 3:
            generated_question = '. '.join(sentences[:3]) + '.'
        
        print(f"Generated dynamic question for {missing_field}: {generated_question}")
       
        return generated_question
        
    except Exception as e:
        print(f"Error generating dynamic question: {e}")
        # Simple fallback
        return f"Could you tell me about {field_description}?"

def generate_information_gathering_response(chat_history: list, missing_info: list) -> str:
    """Generate a warm, conversational response asking for missing information
    Uses LLM to dynamically generate questions based on REQUIRED_INFORMATION fields"""
    
    if not missing_info:
        return "Thank you! I now have a good understanding about you."
    
    # Prioritize which information to ask for first (based on conversation flow)
    priority_order = ['current_condition', 'duration', 'mental_health_history', 'physical_activity', 'suicidal_thoughts']
    next_info_to_ask = None
    
    # Find the highest priority missing information
    for priority_item in priority_order:
        if priority_item in missing_info:
            next_info_to_ask = priority_item
            break
    
    # Fallback to first missing item if none in priority list
    if not next_info_to_ask:
        next_info_to_ask = missing_info[0]
        missing_info[0].next
    
    print(f"Asking for: {next_info_to_ask} (missing: {missing_info})")
    
    
    # Get the field description from REQUIRED_INFORMATION
    field_info = REQUIRED_INFORMATION.get(next_info_to_ask, {})
    field_description = field_info.get("description", next_info_to_ask)
    
    # Generate dynamic question using LLM
    return generate_dynamic_question(next_info_to_ask, field_description, chat_history)

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

# Sentiment analysis and summary functions (unchanged from original)
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

def generate_conversation_summary(chat_history, session_start_index, previous_summary=None, language="english"):
    user_messages = [msg["content"] for msg in chat_history[session_start_index:] if msg.get("role") == "user"]
    if not user_messages:
        summary_text = "No messages to summarize in this session."
    else:
        conversation_text = " ".join(user_messages)
        
        # Language-aware summary generation
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
        overall_emotion = "positive" if language == 'english' else "positive"
    elif avg_compound < -0.1:
        overall_emotion = "negative" if language == 'english' else "negative"
    else:
        overall_emotion = "neutral" if language == 'english' else "neutral"

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

# Main chat processing logic with bilingual support
async def process_chat(session_key: str, user_message: str):
    # Detect language from user message
    
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

    # Store user's language preference

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
            
            # ALWAYS generate summary in English for backend storage (not user's language)
            summary_data = generate_conversation_summary(chat_history, session_start_index, previous_summary, "english")
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
                 
            # Save user data with the new summary (but don't save chat history since we didn't modify it)
            store_user_data(session_key, user_data)
            
            print(f"[BACKEND] Auto-generated summary stored silently: {summary_text[:100]}...")

    # Analyze sentiment of the user's message
    sentiment = analyze_sentiment(user_message)
    chat_history.append({"role": "user", "content": user_message, "sentiment": sentiment})

    missing_info = get_missing_information_list(user_data)
    current_missing_field = missing_info[0] if missing_info else None

    # Only extract if there's a missing field
    if current_missing_field:
        extracted_info = extract_all_information_from_message(user_message, current_missing_field)
        user_data = update_collected_information(user_data, extracted_info)
    else:
        extracted_info = {}
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

    # Check if user explicitly ends the session (language-aware)
    end_session_phrases = ["end session", "goodbye", "exit", "bye", "end"]

    user_message_lower = user_message.lower()
    is_ending_session = any(phrase in user_message_lower for phrase in end_session_phrases)
    
    if is_ending_session:
        session_start_index = user_data["session_start_index"]
        previous_summary = user_data["past_summaries"][-1]["summary"] if user_data.get("past_summaries") else None
        
        # ALWAYS generate summary in English for backend storage
        summary_data = generate_conversation_summary(chat_history, session_start_index, previous_summary, "english")
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
        
        # Provide a simple goodbye message in user's language instead of showing summary
        goodbye_msg = "Thank you for chatting with me today. Take care and feel free to return anytime!"
        
        chat_history.append({
            "role": "assistant",
            "content": goodbye_msg,
        })
        save_chat_history(session_key, chat_history)
        store_user_data(session_key, user_data)
        
        print(f"[BACKEND] Session end summary stored silently: {summary_text[:100]}...")
        return {"response": goodbye_msg}
    
    # Handle first-time users with language-appropriate greeting
    if len(chat_history) == 1:
        greeting = get_greeting_message()
        user_data["session_start_index"] = 1
        chat_history.append({"role": "assistant", "content": greeting})
        save_chat_history(session_key, chat_history)
        store_user_data(session_key, user_data)
        return {"response": greeting}

    # Separate information gathering vs therapeutic responses with language awareness
    if not user_data["questionnaire_completed"]:
        # INFORMATION GATHERING MODE - Dynamic LLM-generated questions
        print("INFORMATION GATHERING MODE: Using dynamic LLM-generated questions")
        missing_info = get_missing_information_list(user_data)
        ai_response = generate_information_gathering_response(chat_history, missing_info)

        # Check if questionnaire was just completed to add a transition message
        if just_completed:
            # Get collected name for personalization
            name = ""
            info_needed = user_data.get("information_needed", {})
            if "name" in info_needed and info_needed["name"].get("collected", False):
                name = info_needed["name"].get("value", "")
            
            transition_message = get_completion_message(name)
            ai_response = transition_message + ai_response
        
    else:
        # THERAPEUTIC MODE - Full RAG processing with language awareness
        print("THERAPEUTIC MODE: Using full RAG processing with language awareness")
        
        # Create context with collected information
        collected_info_context = create_collected_information_context(user_data)
        
        # Process message through RAG with collected information context and language
        ai_response = process_user_message(
            user_message, 
            chat_history, 
            user_data, 
            max_summaries=3, 
            max_doctor_summaries=2,
            additional_context=collected_info_context
        )

    chat_history.append({"role": "assistant", "content": ai_response})
    save_chat_history(session_key, chat_history)
    return {"response": ai_response}