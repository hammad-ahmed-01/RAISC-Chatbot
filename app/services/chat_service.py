# app/services/chat_service.py
import nltk
import re
from nltk.sentiment.vader import SentimentIntensityAnalyzer
from datetime import datetime, timedelta
from langchain_groq import ChatGroq
from langchain_core.messages import SystemMessage, HumanMessage
from app.services.firestore_service import get_chat_history, save_chat_history
from app.services.user_service import get_user_data, store_user_data, get_doctor_summary
from app.services.rag_service import process_user_message
from app.services.language_service import (
    detect_language,
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
    }
}

CONTROLLED_TAGS = [
    # Emotional
    "Low Mood", "Sadness", "Hopelessness", "Irritability", "Emotional Distress", "Emotional Numbness", "Mood Swings",

    # Anxiety / Stress
    "Anxiety", "Panic Symptoms", "Worry", "Overthinking", "Work Stress", "Academic Stress",
    "Social Anxiety", "Adjustment Stress", "Overwhelm",

    # Trauma
    "Trauma History", "Hypervigilance", "Fear", "Avoidance",

    # Behavioral
    "Fatigue", "Sleep Issues", "Insomnia", "Burnout", "Low Motivation",
    "Procrastination", "Isolation", "Withdrawal",

    # Cognitive
    "Negative Self-Talk", "Cognitive Distortions", "Low Self-Worth", "Perfectionism", "Rumination",

    # Risk
    "Self-Harm Thoughts", "Suicidal Ideation", "Substance Misuse", "Impulsivity",

    # Coping
    "Coping Skills", "Exercise", "Social Support", "Healthy Habits", "Mindfulness"
]

def extract_tags_from_taxonomy(conversation_text: str, max_tags: int = 5):
    """
    Uses LLM to map conversation themes to a fixed clinical taxonomy.
    Ensures clean, consistent tags.
    """

    taxonomy_str = ", ".join(CONTROLLED_TAGS)

    prompt = f"""
    You are a clinical psychologist creating session summary tags.

    TASK:
    - Read the conversation below.
    - Identify the main emotional, cognitive, behavioral, or situational themes.
    - Select ONLY from this taxonomy of clinical tags:

    {taxonomy_str}

    RULES:
    - Choose up to {max_tags} tags.
    - Tags must come ONLY from the list above.
    - Do NOT invent new tags.
    - Do NOT output explanations.
    - Output JSON list only.
    - Keep ordering logical (most prominent themes first).

    Conversation:
    \"\"\"{conversation_text}\"\"\"

    Example output:
    ["Anxiety", "Work Stress", "Low Mood"]
    
    """

    try:
        resp = llm.invoke([
            SystemMessage(content="You generate clinical tags from a controlled taxonomy."),
            HumanMessage(content=prompt)
        ])
        text = resp.content.strip()
        match = re.search(r"\[[\s\S]*\]", text)
        if match:
            tags = json.loads(match.group())
            return tags[:max_tags]
    except Exception as e:
        print(f"[TAGS] Controlled taxonomy tag generation failed: {e}")

    return []

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
def extract_all_information_from_message(user_message: str, current_missing_field: str, language: str = "english") -> dict:
    """Extract ONLY information for the current missing field from user's message using LLM with language awareness
    Always stores extracted information in English for consistency
    ONLY extracts real information - never placeholder values"""
    
    # Get the description for the current missing field
    current_field_info = REQUIRED_INFORMATION.get(current_missing_field, {})
    current_field_description = current_field_info.get('description', current_missing_field)
    
    # Create example JSON with only the current field
    example_json = {current_missing_field: f"actual_{current_missing_field}_value"}
    example_json_str = json.dumps(example_json, indent=4)

    extraction_prompt = f"""
    You are a therapeutic context extractor. Your goal is to identify *relevant psychological or behavioral information*
    from a casual, conversational message — as a human therapist would understand it.
    
    **LANGUAGE CONTEXT**: The user is speaking in {"Pakistani Roman Urdu" if language == "roman_urdu" else "English"}. 
    Understand their message in this language, but ALWAYS extract and output information in ENGLISH only for backend consistency.

    User message:
    "{user_message}"

    ---

    ### OBJECTIVE
    Extract only the information that clearly or reasonably relates to the specified field.
    Each field describes a single topic of mental or physical well-being.

    ### FIELD TO ANALYZE
    - Field: "{current_missing_field}"
    - Description: {current_field_description}

    ---

    ### EXTRACTION GUIDELINES

    1. *Lenient and Context-Aware*
    - Understand everyday language, slang, or vague phrases that still communicate a feeling, behavior, or history.
    - Use emotional tone or temporal cues to infer approximate meaning if the intent is clear.
    - Accept statements like “sort of okay”, “eh… not great”, “better than before” as valid data.

    2. *Negation and Denial Handling*
    - Treat any form of “no”, “not really”, “none”, “never”, “don’t think so”, “I guess not”, “not that I recall” as explicit denials.
    - Extract that as "no <field_name>" (e.g., "no suicidal thoughts").

    3. *Ambiguity & Hedges*
    - When users hedge (e.g., “I guess”, “maybe”, “kind of”, “sometimes”), include the information but label uncertainty.
    - Example: "physical_activity": "sometimes exercises (uncertain frequency)".

    4. *Multi-Aspect Answers*
    - If multiple ideas are given (e.g., “Mentally fine but physically tired”), prioritize the aspect relevant to the current field.
    - Ignore unrelated topics even if emotionally adjacent.

    5. *Temporal or Comparative Phrases*
    - Normalize relative expressions:
        - “A while now” → “for some time”
        - “Not long” → “recent”
        - “Used to…” → “previously, not now”
        - “Better than before” → “improving condition”

    6. *Emotional Tone Mapping (for general use)*
    - Recognize emotional expressions such as:
        - “meh”, “could be worse”, “alright I guess” → mild/neutral
        - “rough”, “bad”, “overwhelmed”, “not okay” → struggling
        - “fine”, “good”, “okay” → stable
    - Use short, normalized summaries (e.g., "current_condition": "feeling overwhelmed").

    7. *Unwillingness / Deflection*
    - If the user avoids the question (“I’d rather not say”, “skip this one”), return {{}}, do not fabricate.

    8. *Non-Relevant Responses*
    - If unrelated (e.g., greeting, joke, small talk), return {{}}

    9. *Output Format*
    - Return a single valid JSON object with only the relevant key if clear data exists.
    - Example:
        json
        {{ "{current_missing_field}": "some extracted text" }}
        
    - Otherwise:
        json
        {{}}
        

    ---

    ### EXAMPLES
    User: “I’ve been feeling weird lately, not exactly sad, just off.”
    → {{ "current_condition": "feeling off, not sad" }}

    User: “I used to exercise more but not these days.”
    → {{ "physical_activity": "previously exercised, not currently" }}

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
        
        response = llm.invoke(prompt)
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

def generate_dynamic_question(missing_field: str, field_description: str, chat_history: list, language: str = "english") -> str:
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
    
    generation_prompt = f"""
    You are *RAISC, a warm, empathetic mental health assistant. 
    
    **LANGUAGE INSTRUCTION**: Respond in {"Pakistani Roman Urdu (written in Roman/Latin script)" if language == "roman_urdu" else "English"}.
    {"Use authentic Pakistani vocabulary, expressions like 'aap', 'main', 'kya', 'kaise', etc. Be respectful and culturally appropriate." if language == "roman_urdu" else "Use natural, conversational English that feels warm and supportive."}
    
    You engage users in a natural, therapist-like conversation to gently collect details about their mental and physical wellbeing.

    ---

    ### TASK
    Generate a natural, conversational *reply* that:
    1. Feels emotionally intelligent, validating, and non-judgmental.
    2. Gently guides the user toward answering the next question about *"{missing_field}"*.
    3. Adapts to the user's last message tone, even if it's vague, deflective, emotional, or conversationally off-topic.

    ---

    ### FIELD INFORMATION
    - *Field name:* {missing_field}
    - *Goal:* {field_description}

    ---

    ### USER'S LAST MESSAGE
    "{user_last_message}"

    ---

    ### BEHAVIORAL RULES

    1. *Empathetic First, Functional Second*
    - Always start by acknowledging or reflecting the user’s previous message naturally (e.g., “That sounds like it’s been tough” or “I’m glad you shared that”).
    - Then, smoothly transition into the next question about *{missing_field}*.

    2. *Context Awareness*
    - If the user's message seems emotional, respond with care before asking the next question.
    - If they sound neutral or factual, keep tone conversational and curious.
    - If they ask a counter question about mental health or something relevant, *briefly answer it* and then return to the main question.
    - If the response is *completely unrelated*, redirect politely and naturally.

    3. *Natural Question Generation*
    - The question should feel like a continuation of a conversation — *not* a form or survey.
    - Avoid robotic or repetitive phrasing.
    - Use phrasing styles like:
        - “Could you tell me a bit about…”  
        - “I’m curious to know…”  
        - “How has it been for you when it comes to…”  
        - “Would you say…”  
        - “Have you noticed…”  

    4. *Tone & Language*
    - Keep tone calm, friendly, and professional.
    - Avoid clinical terms unless the user uses them first.
    - Prefer emotionally soft and inclusive phrasing like “sometimes”, “a bit”, “generally”, “these days”.

    5. *Conversation Flow*
    - If the user gave some relevant info but not complete, *gently ask for clarification or elaboration*.
    - If they already answered this field fully, *acknowledge and move to the next topic* instead of repeating.

    6. *Safety Considerations*
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
    It should feel like part of a human conversation, not a script.
    """
    
    try:
        prompt = [
            SystemMessage(content="You are a skilled mental health assistant who asks natural, empathetic questions. Generate warm, conversational questions that feel like talking to a caring friend or counselor."),
            HumanMessage(content=generation_prompt)
        ]
        
        response = llm.invoke(prompt)
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

def generate_information_gathering_response(chat_history: list, missing_info: list, language: str = "english") -> str:
    """Generate a warm, conversational response asking for missing information
    Uses LLM to dynamically generate questions based on REQUIRED_INFORMATION fields"""
    
    if not missing_info:
        return "Thank you! I now have a good understanding about you."
    
    # Prioritize which information to ask for first (based on conversation flow)
    priority_order = ['current_condition', 'duration', 'mental_health_history', 'physical_activity']
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
    return generate_dynamic_question(next_info_to_ask, field_description, chat_history, language)

# REMOVE THIS COMMENTED OUT FUNC 
def generate_questionnaire_insights(user_data: dict) -> dict:
    """Generate meaningful insights from questionnaire results in complete sentences"""
    insights = {}
    info_needed = user_data.get("information_needed", {})
    
    # Collect all raw values for context
    collected_data = {}
    for key, info in info_needed.items():
        if info.get("collected", False) and info.get("value"):
            collected_data[key] = info.get("value", "")
    
    if not collected_data:
        return insights
    
    # Create a comprehensive prompt for generating insights
    field_descriptions = {
        "current_condition": "current mental and emotional state",
        "duration": "duration of the current condition",
        "mental_health_history": "mental health history including personal and family history",
        "physical_activity": "physical activity and exercise patterns"
    }
    
    # Build context string
    context_parts = []
    for key, value in collected_data.items():
        field_desc = field_descriptions.get(key, key)
        context_parts.append(f"{field_desc}: {value}")
    
    context_str = "\n".join(context_parts)
    
    insight_prompt = f"""
    You are a clinical psychologist analyzing questionnaire responses. Your task is to transform brief, 
    raw user responses into concise, factual summaries written in complete sentences.
    
    Raw questionnaire responses:
    {context_str}
    
    For each field, generate ONE concise sentence that:
    1. Accurately summarizes what the user said (do NOT add information they didn't provide)
    2. Uses clear, professional language
    3. Stays factual and does not make assumptions
    4. Is written as a complete sentence (not fragments or 2-3 words)
    
    IMPORTANT: 
    - Only state what the user actually said
    - Do NOT add interpretations, assumptions, or details not mentioned
    - Keep each insight to ONE sentence only
    
    Field descriptions:
    - current_condition: Summary of the user's stated current mental/emotional state
    - duration: Summary of the timeline the user mentioned
    - mental_health_history: Summary of the mental health history the user shared
    - physical_activity: Summary of the physical activity patterns the user described
    
    Output format (JSON):
    {{
        "current_condition": "One sentence summarizing their stated condition",
        "duration": "One sentence summarizing the duration they mentioned",
        "mental_health_history": "One sentence summarizing the history they shared",
        "physical_activity": "One sentence summarizing their stated activity level"
    }}
    
    Only include fields that have data. Be factual and concise.
    """
    
    try:
        prompt = [
            SystemMessage(content="You are a clinical psychologist who writes clear, insightful summaries of patient information."),
            HumanMessage(content=insight_prompt)
        ]
        
        response = llm.invoke(prompt)
        raw = (response.content or "").strip()
        
        # Parse JSON response
        try:
            json_match = re.search(r'\{.*\}', raw, re.DOTALL)
            if json_match:
                parsed = json.loads(json_match.group())
                # Only keep insights for fields that actually have data
                for key in collected_data.keys():
                    if key in parsed and parsed[key]:
                        insights[key] = parsed[key]
        except json.JSONDecodeError:
            # Fallback: generate individual insights
            for key, value in collected_data.items():
                field_desc = field_descriptions.get(key, key)
                fallback_insight = f"The user's {field_desc} indicates: {value}."
                insights[key] = fallback_insight
        
        return insights
        
    except Exception as e:
        print(f"Error generating questionnaire insights: {e}")
        # Fallback to simple sentences
        for key, value in collected_data.items():
            field_desc = field_descriptions.get(key, key)
            insights[key] = f"The user's {field_desc} indicates: {value}."
        return insights

def generate_questionnaire_insights(user_data: dict, interpretive: bool = True) -> dict:
    """
    Generate meaningful, full-sentence insights from questionnaire responses.
    If interpretive=True, a second pass adds brief clinical interpretation.
    """
    insights = {}
    info_needed = user_data.get("information_needed", {})
    collected_data = {
        key: info.get("value", "")
        for key, info in info_needed.items()
        if info.get("collected", False) and info.get("value")
    }

    if not collected_data:
        return insights

    field_descriptions = {
        "current_condition": "the user's current mental and emotional state",
        "duration": "the duration for which they have been feeling this way",
        "mental_health_history": "their personal or family mental health history",
        "physical_activity": "their physical activity or exercise habits"
    }

    # Build descriptive context
    context_parts = [
        f"{field_descriptions.get(key, key)}: {value}"
        for key, value in collected_data.items()
    ]
    context_str = "\n".join(context_parts)

    # Descriptive factual sentences
    descriptive_prompt = f"""
    You are a clinical psychologist writing brief intake notes.
    Rewrite the user's responses into short, professional clinical statements.

    RULES:
    - Use the word "Client" instead of "user".
    - One sentence per field.
    - MAXIMUM 12–16 words per sentence.
    - Use neutral, factual language (e.g., "Client reported XYZ", "Client denied ABC").
    - No interpretation, no assumptions, no clinical speculation.
    - No extra details.
    - No adjectives unless stated by client.
    - Do NOT use quotes.
    - Do NOT say "the user said".
    - Do NOT use long sentences.

    Rewrite the following responses:

    {context_str}

    Output JSON like:
    {{
    "current_condition": "Client reported feeling uncertain.",
    "duration": "Symptoms present for about two weeks.",
    "mental_health_history": "Client denied any relevant mental health history.",
    "physical_activity": "Client occasionally goes for a weekend run."
    }}

    Only include fields that have data.
    """

    try:
        prompt = [
            SystemMessage(content="You are a factual clinical summarizer."),
            HumanMessage(content=descriptive_prompt)
        ]
        response = llm.invoke(prompt)
        raw = (response.content or "").strip()
        json_match = re.search(r"\{.*\}", raw, re.DOTALL)
        if json_match:
            descriptive = json.loads(json_match.group())
        else:
            descriptive = {k: f"The user's {field_descriptions.get(k, k)} indicates: {v}." for k, v in collected_data.items()}
    except Exception as e:
        print(f"(Questionnaire) Descriptive generation error: {e}")
        descriptive = {k: f"The user's {field_descriptions.get(k, k)} indicates: {v}." for k, v in collected_data.items()}

    if not interpretive:
        return descriptive

    # Interpretive insight shit, second LLM pass for deeper meaning (kharchay)
    interpretive_prompt = f"""
    You are a licensed clinical psychologist writing short case notes.
    Using the following factual summaries, create interpretive sentences that describe what these might imply
    about the user's psychological state, coping ability, or lifestyle.

    Factual summaries:
    {json.dumps(descriptive, indent=2)}

    Write one interpretive sentence per field in JSON format.
    Example:
    {{
        "current_condition": "The user appears to be experiencing emotional exhaustion, likely related to sustained stress.",
        "duration": "The condition seems persistent rather than situational.",
        "mental_health_history": "A possible genetic or environmental predisposition may exist.",
        "physical_activity": "Their limited physical activity may contribute to low mood or energy levels."
    }}
    Each sentence should:
    - Be professional and empathetic.
    - Avoid diagnostic terms (like depression, anxiety).
    - Express insights, not judgments.
    """

    try:
        prompt2 = [
            SystemMessage(content="You are a reflective clinician who writes brief insights."),
            HumanMessage(content=interpretive_prompt)
        ]
        resp2 = llm.invoke(prompt2)
        raw2 = (resp2.content or "").strip()
        json_match2 = re.search(r"\{.*\}", raw2, re.DOTALL)
        if json_match2:
            interpretive = json.loads(json_match2.group())
        else:
            interpretive = {}
    except Exception as e:
        print(f"[Questionnaire] Interpretive generation error: {e}")
        interpretive = {}

    # Merge descriptive + interpretive
    for key, desc in descriptive.items():
        if key in interpretive:
            insights[key] = f"{desc} {interpretive[key]}"
        else:
            insights[key] = desc

    return insights

def create_collected_information_context(user_data: dict) -> str:
    """Create context string with collected information for RAG using meaningful insights"""
    # Use stored insights if available, otherwise generate them
    insights = user_data.get("questionnaire_insights", {})
    
    # If no stored insights, generate them (but don't store to avoid unnecessary writes)
    if not insights:
        insights = generate_questionnaire_insights(user_data)
    
    if not insights:
        return ""
    
    # Format insights as complete sentences
    insight_sentences = []
    for key, insight in insights.items():
        insight_sentences.append(insight)
    
    if insight_sentences:
        return f"Patient background and context: {' '.join(insight_sentences)} Use this information to provide personalized therapeutic support."
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
    assistant_messages = [msg["content"] for msg in chat_history[session_start_index:] if msg.get("role") == "assistant"]

    if not user_messages:
        summary_text = "No messages to summarize in this session."
        summary_title = "No Conversation"
    else:
        conversation_text = " ".join(user_messages).strip()
        assistant_context = " ".join(assistant_messages[-3:]).strip() if assistant_messages else ""
        prev_sum_str = previous_summary if previous_summary else "None"

        # Enhanced prompt for meaningful summaries and titles
        # here we can enhance, lengthen our summary & titles type stuff
        # prompt_text = (
        #     "You are a licensed clinical psychologist writing post-session notes. "
        #     "Write in the first-person perspective of the clinician ('Client presented with DEF idea', 'Session focused on ABC topic', 'Intervention included XYZ methods')."
        #     "Keep tone objective, professional, and consistent with clinical documentation style."

        #     "The conversation may be bilingual (English AND Roman Urdu) and you must output only in English."
        #     "Your task is to create a comprehensive summary and a meaningful title that would be useful "
        #     "for a mental health professional reviewing the session.\n\n"
            
        #     "GUIDELINES:\n"
        #     "1. *Summary*: Write 2-4 complete sentences that capture:\n"
        #     "   - Key themes, emotions, and concerns discussed\n"
        #     "   - Important details about the user's mental state\n"
        #     "   - Any significant patterns, changes, or developments\n"
        #     "   - Clinical relevance and psychological insights\n"
        #     "   - If a previous summary exists, note how this session relates to it (continuation, new topic, etc.)\n\n"
            
        #     "2. *Title*: Create a descriptive, meaningful title (2-6 words) that:\n"
        #     "   - Captures the main focus or theme of the conversation\n"
        #     "   - Is specific enough to be informative but concise\n"
        #     "   - Uses professional but accessible language\n"
        #     "   - Examples: 'Anxiety Management Discussion', 'Exploring Relationship Stress', 'Coping Strategies for Depression'\n"
        #     "   - NOT single words like 'Anxiety' or 'Stress' - be descriptive\n\n"

        #     "OUTPUT STRICTLY in JSON format exactly like:\n"
        #     "{\n"
        #     "  \"Summary\": \"The user discussed experiencing persistent anxiety over the past few weeks, particularly "
        #     "related to work deadlines and social interactions. They expressed feeling overwhelmed and having difficulty "
        #     "sleeping. The conversation explored coping mechanisms and the user showed interest in learning relaxation techniques.\",\n"
        #     "  \"Title\": \"Anxiety Management and Coping Strategies\"\n"
        #     "}"
        # )

        prompt_text = """
        You are a licensed clinical psychologist writing post-session notes. 
        Write in the first-person perspective of the clinician 
        ('Client presented with DEF idea', 'Session focused on ABC topic', 
        'Intervention included XYZ methods').

        Keep tone objective, professional, and consistent with clinical documentation style.

        The conversation may be bilingual (English AND Roman Urdu). Output only in English.

        GUIDELINES:
        1. Summary: Write 2–4 complete sentences describing:
        - Key themes, emotions, concerns
        - Mental state indicators
        - Notable patterns or changes
        - Connection to previous summary if one exists

        2. Title: 2–6 words, descriptive, clinically useful.
        Examples:
        - Anxiety Management Discussion
        - Exploring Relationship Stress
        - Coping Strategies for Low Mood

        REQUIRED OUTPUT FORMAT:
        {
        "Summary": "Example summary here.",
        "Title": "Example Title"
        }
        """

        conversation_context = f"User messages: {conversation_text}"
        if assistant_context:
            conversation_context += f"\n\nRecent therapist responses (for context): {assistant_context}"

        prompt = [
            SystemMessage(content=prompt_text),
            HumanMessage(content=f"Previous session summary: {prev_sum_str}\n\nCurrent session conversation:\n{conversation_context}")
        ]

        summary_output = llm.invoke(prompt)
        raw = (summary_output.content or "").strip()

        # Parse JSON response safely
        try:
            json_match = re.search(r'\{.*\}', raw, re.DOTALL)
            if json_match:
                parsed = json.loads(json_match.group())
                summary_text = parsed.get("Summary", "Summary not available.")
                summary_title = extract_tags_from_taxonomy(conversation_text)

            else:
                # Try to extract from non-JSON response
                summary_text = raw if raw else "Summary not available."
                summary_title = "General Discussion"
        except json.JSONDecodeError:
            # Fallback: try to create a basic summary
            summary_text = raw if raw else "Summary not available."
            # Extract a simple title from the first sentence
            first_sentence = summary_text.split('.')[0] if summary_text else "General Discussion"
            summary_title = first_sentence[:50] if len(first_sentence) > 50 else first_sentence

    emotions = analyze_emotions(chat_history, session_start_index)
    num_messages = emotions.get("num_messages", 0)
    avg_compound = emotions.get("average_compound", 0.0)
    min_compound = emotions.get("min_compound", 0.0)
    max_compound = emotions.get("max_compound", 0.0)

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
    print(f"Summary title: {summary_title} \n Summary content: {summary_text}")
    timestamp = datetime.now().isoformat()
    return {
        "summary_title": summary_title,
        "summary": summary_text,
        "emotional_summary": emotional_summary,
        "timestamp": timestamp,
        "session_start_msg": session_start_index,
        "session_end_msg": len(chat_history) - 1,

    }

# Main chat processing logic with bilingual support
async def process_chat(session_key: str, user_message: str):
    # Detect language from user message
    detected_language = detect_language(user_message)
    print(f"[LANGUAGE] Detected language: {detected_language} for message: '{user_message[:50]}'")
    
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

    # Store user's detected language for this interaction
    current_language = detected_language
    
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
            summary_tag = summary_data["summary_title"]
            summary_text = summary_data["summary"]
            emotional_summary = summary_data["emotional_summary"]
            timestamp = datetime.now().isoformat()
            user_data["past_summaries"].append({
                "summary_title": summary_tag,
                "summary": summary_text,
                "emotional_summary": emotional_summary,
                "timestamp": timestamp,
                "session_start_msg": session_start_index,
                "session_end_msg": len(chat_history) - 1,
            })
            user_data["last_summarized_index"] = len(chat_history)
            user_data["session_start_index"] = len(chat_history)
            user_data["session_agg_sentiment"] = reset_sentiment_aggregate()
                 
            # Save user data with the new summary (but don't save chat history since we didn't modify it)
            store_user_data(session_key, user_data)
            
            print(f"[BACKEND] Auto-generated summary stored silently: {summary_text[:100]}")

    # Analyze sentiment of the user's message
    sentiment = analyze_sentiment(user_message)
    chat_history.append({"role": "user", "content": user_message, "sentiment": sentiment})

    missing_info = get_missing_information_list(user_data)
    current_missing_field = missing_info[0] if missing_info else None
    print("missing info list:", missing_info)
    # Only extract if there's a missing field
    # Always extract in ENGLISH regardless of user's language
    if current_missing_field:
        extracted_info = extract_all_information_from_message(user_message, current_missing_field, current_language)
        user_data = update_collected_information(user_data, extracted_info)
        print("Current user data after extraction:", user_data)
    else:
        extracted_info = {}
    user_data = update_collected_information(user_data, extracted_info)

    # Update questionnaire_completed status
    was_completed_before = user_data.get("questionnaire_completed", False)
    user_data["questionnaire_completed"] = is_questionnaire_complete(user_data)
    just_completed = not was_completed_before and user_data["questionnaire_completed"]
    
    # Generate and store insights when questionnaire is completed
    if just_completed:
        insights = generate_questionnaire_insights(user_data)
        if insights:
            user_data["questionnaire_insights"] = insights
            print(f"Generated questionnaire insights: {insights}")

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
    end_session_patterns = [
        r"\bend session\b",
        r"\bbye\b",
        r"\bgoodbye\b",
        r"\bexit\b",
        r"^end$"  # Only match "end" if it's the entire message not substring
    ]

    user_message_lower = user_message.lower()
    is_ending_session = any(re.search(pattern, user_message_lower) for pattern in end_session_patterns)
    
    if is_ending_session:
        session_start_index = user_data["session_start_index"]
        previous_summary = user_data["past_summaries"][-1]["summary"] if user_data.get("past_summaries") else None
        
        # ALWAYS generate summary in English for backend storage
        summary_data = generate_conversation_summary(chat_history, session_start_index, previous_summary, "english")
        summary_tag = summary_data["summary_title"]
        summary_text = summary_data["summary"]
        emotional_summary = summary_data["emotional_summary"]
        timestamp = datetime.now().isoformat()

        user_data["past_summaries"].append({
            "summary_title": summary_tag,
            "summary": summary_text,
            "emotional_summary": emotional_summary,
            "timestamp": timestamp,
            "session_start_msg": session_start_index,
            "session_end_msg": len(chat_history) - 1
        })
        print(f"*session_start_index: {session_start_index}, session_end_index: {len(chat_history) - 1}*")


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
        print("stored user data")
        
        print(f"[BACKEND] Session end summary stored silently: {summary_text[:100]}")
        return {"response": goodbye_msg}
    
    # Handle first-time users with language-appropriate greeting
    if len(chat_history) == 1:
        greeting = get_greeting_message(current_language, user_message)
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
        ai_response = generate_information_gathering_response(chat_history, missing_info, current_language)
        
    else:
        # THERAPEUTIC MODE - Full RAG processing with language awareness
        print("THERAPEUTIC MODE: Using full RAG processing with language awareness")
        
        # Check if questionnaire was just completed to add a transition message
        transition_prefix = ""
        if just_completed:
            # Get collected name for personalization
            name = ""
            info_needed = user_data.get("information_needed", {})
            if "name" in info_needed and info_needed["name"].get("collected", False):
                name = info_needed["name"].get("value", "")
            
            transition_prefix = get_completion_message(current_language, name)
            print("Questionnaire just completed - transitioning to therapeutic mode")
        
        # Create context with collected information
        collected_info_context = create_collected_information_context(user_data)
        
        # Process message through RAG with collected information context and language
        ai_response = process_user_message(
            user_message, 
            chat_history, 
            user_data, 
            max_summaries=3, 
            max_doctor_summaries=2,
            additional_context=collected_info_context,
            language=current_language
        )
        
        # Prepend transition message if questionnaire was just completed
        if transition_prefix:
            ai_response = transition_prefix + ai_response

    chat_history.append({"role": "assistant", "content": ai_response})
    save_chat_history(session_key, chat_history)
    return {"response": ai_response}