from langchain_openai import ChatOpenAI
from langchain_core.messages import SystemMessage, HumanMessage
import json
from dotenv import load_dotenv
import os
load_dotenv()

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

# Shared LLM for generating questions, clarification, redirects
llm = ChatOpenAI(model="gpt-4o-mini", api_key=OPENAI_API_KEY, temperature=1)

# Questionnaire flow and descriptions
QUESTION_FLOW = [
    "current_condition",
    "duration",
    "mental_health_history",
    "physical_activity",
]

FIELD_DESCRIPTIONS = {
    "current_condition": "how you are feeling right now",
    "duration": "how long you have been feeling this way",
    "mental_health_history": "your personal or family mental health background",
    "physical_activity": "your daily physical activity or exercise routine",
}

# =========================================================
# 1️⃣ Generate the primary question for a field
# =========================================================
def generate_question(field: str, language: str) -> str:
    sys = "You are RAISC, a mental-health intake assistant. You ask one question only."
    prompt = f"""
FIELD DESCRIPTION:
"{FIELD_DESCRIPTIONS[field]}"

TASK:
- Ask ONE simple, friendly question to collect information about this field.
- Do NOT reference earlier messages.
- Do NOT give advice.
LANGUAGE:
- You MUST respond ONLY in this language: "{language}"
- If language = "english" → full English.
- If language = "urdu"   → full Roman Urdu.

STRICT JSON OUTPUT:
{{ "assistant": "<question>" }}
"""
    resp = llm.invoke([SystemMessage(content=sys), HumanMessage(content=prompt)])
    try:
        return json.loads(resp.content)["assistant"]
    except:
        return f"Could you tell me {FIELD_DESCRIPTIONS[field]}?"

# =========================================================
# 2️⃣ Dynamic Off-Topic Redirect Message (empathetic)
# =========================================================
def generate_dynamic_redirect(user_message: str, field: str, language: str) -> str:
    """
    When the user says something irrelevant,
    create a custom empathetic redirect:
    - Acknowledge what they said
    - Show brief empathy
    - Bring the user back to the correct question
    """

    sys = "You are RAISC, an empathetic mental-health assistant."
    prompt = f"""
USER MESSAGE (irrelevant):
"{user_message}"

QUESTION TOPIC:
"{FIELD_DESCRIPTIONS[field]}"

TASK:
1. Briefly acknowledge what the user said.
2. Show empathy in one short sentence.
3. Redirect them softly back to the topic.
4. Ask ONLY ONE question.
LANGUAGE:
- You MUST respond ONLY in this language: "{language}"
- If language = "english" → full English.
- If language = "urdu"   → full Roman Urdu.

GOOD EXAMPLES:
- "That sounds interesting! But returning to our topic — could you tell me ___?"
- "I understand, thank you for sharing. To continue — may I ask ___?"
- "Sun kay acha laga. Wapis sawal ki taraf atay hain ___?"

STRICT JSON OUTPUT:
{{ "assistant": "<redirect_message>" }}
"""
    resp = llm.invoke([SystemMessage(content=sys), HumanMessage(content=prompt)])
    try:
        return json.loads(resp.content)["assistant"]
    except:
        return f"Let’s return to the question — could you tell me {FIELD_DESCRIPTIONS[field]}?"

# =========================================================
# 3️⃣ MAIN QUESTIONNAIRE FLOW — (NO CLASSIFICATION HERE)
# =========================================================
def run_questionnaire_turn(user_message: str, user_data: dict, language: str, relevance_label: str):
    """
    NEW DESIGN:
    relevance_label = "ON_TOPIC" | "OFF_TOPIC"

    All relevance decisions happen in relevance.py.
    This function *only reacts* to the label.
    """

    # Create or read questionnaire state
    state = user_data.setdefault("questionnaire", {
        "current_field": "current_condition",
        "completed": False,
        "first_question_pending": True
    })

    # Already completed
    if state["completed"]:
        return None, None

    current_field = state["current_field"]

    # =====================================================
    # FIRST QUESTION OF A FIELD → Ask without evaluating user message
    # =====================================================
    if state["first_question_pending"]:
        first_q = generate_question(current_field, language)
        state["first_question_pending"] = False
        return first_q, None
    # =====================================================
    # Handle 2-label relevance
    # =====================================================

    # 🟥 OFF_TOPIC → Empathetic redirect, stay on same field
    if relevance_label == "OFF_TOPIC":
        redirect_msg = generate_dynamic_redirect(user_message, current_field, language)
        return redirect_msg, None

    # 🟩 ON_TOPIC → Accept answer, extract, move to next field
    if relevance_label == "ON_TOPIC":

        extraction_field = current_field  # Tell chat_service to extract field info

        # Move to next field in flow
        idx = QUESTION_FLOW.index(current_field)

        if idx + 1 < len(QUESTION_FLOW):
            next_field = QUESTION_FLOW[idx + 1]
            state["current_field"] = next_field

            #changed below from true to false
            state["first_question_pending"] = False

            next_question = generate_question(next_field, language)
            return next_question, extraction_field

        else:
            # Questionnaire completed
            state["completed"] = True
            return "Thank you for telling me about yourself. Now, how can I help you?", extraction_field


    # Fallback (should never happen)
    return "Could you repeat that?", None
