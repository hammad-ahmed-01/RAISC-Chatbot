# app/services/conversation_analysis.py
# Unified conversation analysis system with built-in serialization

import json
import re
from datetime import datetime
from typing import Dict, List, Optional
from langchain_groq import ChatGroq
from langchain.schema import SystemMessage, HumanMessage
from app.config import GROQ_API_KEY

# Initialize LLM for analysis
analysis_llm = ChatGroq(
    model="llama-3.3-70b-versatile", 
    groq_api_key=GROQ_API_KEY, 
    temperature=0.1,
    max_tokens=500
)

class ConversationFlowManager:
    """JSON-serializable conversation flow manager"""
    
    def __init__(self, data: dict = None):
        if data:
            self.resistance_count = data.get('resistance_count', 0)
            self.hostility_count = data.get('hostility_count', 0)
            self.consecutive_low_engagement = data.get('consecutive_low_engagement', 0)
            self.question_attempts = data.get('question_attempts', {})
        else:
            self.resistance_count = 0
            self.hostility_count = 0
            self.consecutive_low_engagement = 0
            self.question_attempts = {}
    
    def to_dict(self) -> dict:
        """Convert to JSON-serializable dictionary"""
        return {
            'resistance_count': self.resistance_count,
            'hostility_count': self.hostility_count,
            'consecutive_low_engagement': self.consecutive_low_engagement,
            'question_attempts': self.question_attempts,
            '_type': 'ConversationFlowManager'
        }
    
    @classmethod
    def from_dict(cls, data: dict):
        """Create instance from dictionary"""
        return cls(data)
        
    def reset_session_counters(self):
        """Reset counters for new session"""
        self.resistance_count = 0
        self.hostility_count = 0
        self.consecutive_low_engagement = 0
        
    def update_resistance_tracking(self, analysis: dict):
        """Track resistance patterns over time"""
        engagement = analysis.get('engagement_level', 'moderate')
        
        if engagement == "hostile":
            self.hostility_count += 1
            self.consecutive_low_engagement += 1
        elif engagement == "resistant":
            self.resistance_count += 1
            self.consecutive_low_engagement += 1
        elif engagement in ["low"]:
            self.consecutive_low_engagement += 1
        else:
            self.consecutive_low_engagement = 0
            
    def determine_next_action(self, analysis: dict, user_data: dict, 
                            chat_history: list, missing_field: str = None) -> dict:
        """Decide conversation strategy based on comprehensive analysis"""
        
        self.update_resistance_tracking(analysis)
        
        engagement = analysis.get('engagement_level', 'moderate')
        flow_rec = analysis.get('conversation_flow', 'continue_questioning')
        emotional_state = analysis.get('emotional_state', 'neutral')
        
        if missing_field:
            self.question_attempts[missing_field] = self.question_attempts.get(missing_field, 0) + 1
        
        # Decision logic
        if engagement == "hostile" or self.hostility_count >= 2:
            return self._handle_hostility()
        elif engagement == "resistant" or self.resistance_count >= 3:
            return self._handle_resistance()
        elif self.consecutive_low_engagement >= 4:
            return self._handle_disengagement()
        elif flow_rec in ["end_conversation", "de_escalate"]:
            return self._handle_conversation_end()
        elif emotional_state == "distressed":
            return self._handle_distress()
        elif missing_field and self.question_attempts.get(missing_field, 0) >= 3:
            return self._handle_repeated_question_failure(missing_field)
        else:
            return self._handle_normal_flow(analysis)
    
    def _handle_hostility(self) -> dict:
        return {
            'strategy': 'DE_ESCALATE',
            'action': 'SWITCH_TO_RAG',
            'context': 'The user seems upset. I should back off from questions and provide gentle support.',
            'force_complete_questionnaire': True,
            'reasoning': f"Hostility detected (count: {self.hostility_count})"
        }
    
    def _handle_resistance(self) -> dict:
        return {
            'strategy': 'RESPECT_BOUNDARIES', 
            'action': 'SWITCH_TO_RAG',
            'context': 'The user has shown resistance. I should respect their boundaries.',
            'force_complete_questionnaire': True,
            'reasoning': f"Resistance detected (count: {self.resistance_count})"
        }
    
    def _handle_disengagement(self) -> dict:
        return {
            'strategy': 'RE_ENGAGE',
            'action': 'SWITCH_TO_RAG',
            'context': 'The user seems disengaged. I should try a supportive approach.',
            'force_complete_questionnaire': True,
            'reasoning': f"Low engagement streak: {self.consecutive_low_engagement}"
        }
    
    def _handle_conversation_end(self) -> dict:
        return {
            'strategy': 'END_GRACEFULLY',
            'action': 'END_SESSION',
            'context': 'The user wants to end the conversation.',
            'force_complete_questionnaire': True,
            'reasoning': "User requested conversation end"
        }
    
    def _handle_distress(self) -> dict:
        return {
            'strategy': 'SUPPORTIVE_PRIORITY',
            'action': 'GENTLE_QUESTIONING',
            'context': 'The user seems distressed. Prioritize emotional support but continue gently.',
            'force_complete_questionnaire': False,
            'reasoning': "Emotional distress detected"
        }
    
    def _handle_repeated_question_failure(self, field: str) -> dict:
        return {
            'strategy': 'SKIP_QUESTION',
            'action': 'CONTINUE_WITHOUT_INFO',
            'context': f'I\'ve asked about {field} multiple times. I should move on.',
            'force_complete_questionnaire': False,
            'reasoning': f"Failed to get {field} after {self.question_attempts[field]} attempts"
        }
    
    def _handle_normal_flow(self, analysis: dict) -> dict:
        engagement = analysis.get('engagement_level', 'moderate')
        readiness = analysis.get('therapeutic_readiness', 'conditional')
        
        if engagement in ['high_positive', 'moderate'] and readiness == 'ready':
            return {
                'strategy': 'CONTINUE_NORMAL',
                'action': 'CONTINUE_QUESTIONING',
                'context': 'The user is engaged and ready for conversation.',
                'force_complete_questionnaire': False,
                'reasoning': "Normal positive engagement"
            }
        else:
            return {
                'strategy': 'GENTLE_APPROACH',
                'action': 'GENTLE_QUESTIONING',
                'context': 'The user is somewhat engaged but I should be gentle, I should not keep asking "if you dont mind sharing.", I will vary my responses to have the user feel more comfortable. I wont keep saying that its optional, I will just ask the question in a gentle way.',
                'force_complete_questionnaire': False,
                'reasoning': "Moderate engagement, proceeding carefully"
            }

class UserPatternLearning:
    """JSON-serializable user pattern learning"""
    
    def __init__(self, data: dict = None):
        if data:
            self.user_patterns = data.get('user_patterns', {})
        else:
            self.user_patterns = {}
    
    def to_dict(self) -> dict:
        """Convert to JSON-serializable dictionary"""
        return {
            'user_patterns': self.user_patterns,
            '_type': 'UserPatternLearning'
        }
    
    @classmethod
    def from_dict(cls, data: dict):
        """Create instance from dictionary"""
        return cls(data)
    
    def analyze_user_communication_style(self, session_key: str, 
                                       message: str, analysis: dict):
        """Learn how this specific user communicates"""
        
        if session_key not in self.user_patterns:
            self.user_patterns[session_key] = {
                'positive_responses': [],
                'resistance_triggers': [],
                'total_interactions': 0
            }
        
        patterns = self.user_patterns[session_key]
        patterns['total_interactions'] += 1
        
        # Learn what works
        if analysis['engagement_level'] in ['high_positive', 'moderate']:
            patterns['positive_responses'].append({
                'length': len(message.split()),
                'style': analysis['therapeutic_readiness']
            })
            # Keep only last 5 to limit size
            if len(patterns['positive_responses']) > 5:
                patterns['positive_responses'] = patterns['positive_responses'][-5:]
        
        # Learn resistance triggers
        if analysis['engagement_level'] in ['resistant', 'hostile']:
            patterns['resistance_triggers'].append({
                'length': len(message.split()),
                'context': message[:30]
            })
            # Keep only last 3 to limit size
            if len(patterns['resistance_triggers']) > 3:
                patterns['resistance_triggers'] = patterns['resistance_triggers'][-3:]
    
    def get_personalized_approach(self, session_key: str) -> dict:
        """Get personalized communication approach"""
        
        if session_key not in self.user_patterns:
            return {'approach': 'standard', 'confidence': 0.0}
        
        patterns = self.user_patterns[session_key]
        
        if len(patterns['resistance_triggers']) > 2:
            return {
                'approach': 'extra_gentle',
                'recommendation': 'Use very gentle questions, avoid pushing',
                'confidence': min(0.8, patterns['total_interactions'] / 10)
            }
        elif len(patterns['positive_responses']) > 3:
            return {
                'approach': 'direct_warm',
                'recommendation': 'User responds well to direct but warm questions',
                'confidence': min(0.7, patterns['total_interactions'] / 8)
            }
        
        return {'approach': 'standard', 'confidence': 0.3}

def create_or_restore_managers(user_data: dict) -> tuple:
    """Create or restore managers from user_data"""
    
    # Handle flow_manager
    flow_data = user_data.get('flow_manager_data')
    if flow_data and isinstance(flow_data, dict):
        flow_manager = ConversationFlowManager.from_dict(flow_data)
    else:
        flow_manager = ConversationFlowManager()
    
    # Handle pattern_learner
    pattern_data = user_data.get('pattern_learner_data')
    if pattern_data and isinstance(pattern_data, dict):
        pattern_learner = UserPatternLearning.from_dict(pattern_data)
    else:
        pattern_learner = UserPatternLearning()
    
    return flow_manager, pattern_learner

def save_managers_to_user_data(user_data: dict, flow_manager: ConversationFlowManager, 
                              pattern_learner: UserPatternLearning) -> dict:
    """Save managers to user_data in serializable format"""
    
    # Remove old objects if they exist
    user_data.pop('flow_manager', None)
    user_data.pop('pattern_learner', None)
    
    # Save as serializable data
    user_data['flow_manager_data'] = flow_manager.to_dict()
    user_data['pattern_learner_data'] = pattern_learner.to_dict()
    
    return user_data

def llm_therapeutic_sentiment_analysis(message: str, chat_history: list, 
                                     language: str = "english") -> dict:
    """Use LLM to understand therapeutic context and user intent"""
    
    # Get recent context
    recent_context = ""
    if chat_history:
        recent_messages = chat_history[-3:]
        context_parts = []
        for msg in recent_messages:
            role = msg.get('role', 'unknown')
            content = msg.get('content', '')[:100]
            context_parts.append(f"{role}: {content}")
        recent_context = " | ".join(context_parts)
    
    analysis_prompt = f"""
You are an expert mental health conversation analyzer. Analyze this user message.

Language: {language}
Recent context: {recent_context}
Current message: "{message}"

Classify the user's response:

1. ENGAGEMENT_LEVEL:
- "high_positive": Actively sharing, asking for help
- "moderate": Answering questions, somewhat engaged  
- "low": Short answers, minimal engagement
- "resistant": Avoiding questions, setting boundaries
- "hostile": Aggressive, confrontational

2. INFORMATION_SHARING:
- "informative_denial": Clearly stating "no" (e.g., "I don't have depression") 
- "informative_positive": Sharing personal details
- "evasive": Avoiding answers without hostility
- "refused": Explicitly refusing to answer

3. EMOTIONAL_STATE:
- "distressed": Signs of emotional pain
- "frustrated": Annoyed but not hostile
- "neutral": Calm responses
- "positive": Good mood

4. THERAPEUTIC_READINESS:
- "ready": Open to therapeutic conversation
- "conditional": Willing under certain conditions
- "not_ready": Not ready but not hostile
- "resistant": Pushing back against therapy

5. CONVERSATION_FLOW:
- "continue_questioning": Safe to ask more
- "gentle_questioning": Ask but carefully
- "switch_to_support": Stop questions, provide support
- "de_escalate": User upset, calm situation
- "end_conversation": User wants to stop

IMPORTANT: "No, I don't have anxiety" = informative_denial (POSITIVE engagement)

Return ONLY JSON:
{{
    "engagement_level": "",
    "information_sharing": "",
    "emotional_state": "",
    "therapeutic_readiness": "", 
    "conversation_flow": "",
    "confidence": 0.85,
    "reasoning": "brief explanation"
}}
"""
    
    try:
        response = analysis_llm([
            SystemMessage(content="You are an expert therapeutic conversation analyst. Return valid JSON only."),
            HumanMessage(content=analysis_prompt)
        ])
        
        response_text = response.content.strip()
        json_match = re.search(r'\{.*\}', response_text, re.DOTALL)
        
        if json_match:
            analysis = json.loads(json_match.group())
            required_fields = ['engagement_level', 'information_sharing', 'emotional_state', 'therapeutic_readiness', 'conversation_flow']
            if all(field in analysis for field in required_fields):
                return analysis
        
        return _get_fallback_analysis(message)
        
    except Exception as e:
        print(f"LLM analysis failed: {e}")
        return _get_fallback_analysis(message)

def _get_fallback_analysis(message: str) -> dict:
    """Fallback analysis when LLM fails"""
    
    message_lower = message.lower().strip()
    
    # Simple pattern-based fallback
    hostile_patterns = ['leave me alone', 'shut up', 'stop asking', 'annoying', 'psycho']
    resistant_patterns = ['don\'t want to', 'too personal', 'none of your business']
    denial_patterns = ['no i don\'t', 'never had', 'i\'m not', 'nothing like']
    
    if any(pattern in message_lower for pattern in hostile_patterns):
        return {
            "engagement_level": "hostile",
            "information_sharing": "refused",
            "emotional_state": "frustrated",
            "therapeutic_readiness": "resistant",
            "conversation_flow": "de_escalate",
            "confidence": 0.7,
            "reasoning": "Pattern-based fallback: hostile language detected"
        }
    elif any(pattern in message_lower for pattern in resistant_patterns):
        return {
            "engagement_level": "resistant",
            "information_sharing": "refused",
            "emotional_state": "frustrated",
            "therapeutic_readiness": "not_ready",
            "conversation_flow": "switch_to_support",
            "confidence": 0.6,
            "reasoning": "Pattern-based fallback: resistance detected"
        }
    elif any(pattern in message_lower for pattern in denial_patterns):
        return {
            "engagement_level": "moderate",
            "information_sharing": "informative_denial",
            "emotional_state": "neutral",
            "therapeutic_readiness": "conditional",
            "conversation_flow": "continue_questioning",
            "confidence": 0.5,
            "reasoning": "Pattern-based fallback: informative denial"
        }
    else:
        return {
            "engagement_level": "moderate",
            "information_sharing": "evasive",
            "emotional_state": "neutral",
            "therapeutic_readiness": "conditional",
            "conversation_flow": "gentle_questioning",
            "confidence": 0.3,
            "reasoning": "Pattern-based fallback: default moderate"
        }

def generate_adaptive_question(missing_field: str, field_description: str, 
                             analysis: dict, attempt_count: int, 
                             language: str,chat_history: list,  user_approach: dict = None) -> str:
    """Generate adaptive questions based on user state"""
    
    engagement = analysis.get('engagement_level', 'moderate')
    readiness = analysis.get('therapeutic_readiness', 'conditional')
    emotional_state = analysis.get('emotional_state', 'neutral')
    
    # Determine questioning style
    if engagement in ['resistant', 'low'] or attempt_count > 2:
        style = "optional_gentle"
    elif readiness == "ready" and engagement == "high_positive":
        style = "conversational_natural"  # Changed from "direct_warm"
    elif emotional_state == "distressed":
        style = "supportive_gentle"
    else:
        style = "casual_indirect"
    
    if user_approach and user_approach.get('approach') == 'extra_gentle':
        style = "optional_gentle"
    
    # Using the last 3 messages to provide recent context without overwhelming the prompt
    generation_prompt = f"""
Generate a {style} question about {missing_field} in {language}.

User state:
- Engagement: {engagement}
- Readiness: {readiness}  
- Emotional state: {emotional_state}
- Attempt #{attempt_count}
- Field: {field_description}
- Chat history: {chat_history if chat_history else 'None'}
Questioning styles:
- optional_gentle: Make it clear they can skip if uncomfortable but keep the conversation flowing.
- conversational_natural: Natural flowing conversation, not an interview question.
- casual_indirect: Weave into conversation organically
- supportive_gentle: Acknowledge their sharing first, then gently ask

CRITICAL RULES - Make it sound HUMAN:
1. Don'T repeat the same phrases in every question as given in chat_history.
2. DON'T make it sound like a clinical interview
3. DO use natural conversation flow
4. DO acknowledge what they just said before asking
5. Keep it SHORT (one sentence when possible)
6. Vary your phrasing - never repeat the same question structure
7. Given chat_history, only give response to {chat_history[-1]['role']}. 
8. If chat_history is empty, just ask the question naturally.

EXAMPLES of NATURAL questions:
- "How long has this been going on?" (not "Could you share how long...")
- "What's your sleep been like?" (not "I'd like to understand your sleep patterns")
- "Do you exercise at all?" (not "Could you tell me about your physical activity levels")
- "Any history of mental health stuff in your family?" (not "I'm wondering if...")

EXAMPLES of ROBOTIC questions (AVOID):
- "If you'd like to share, how long have you been feeling like this?"
- "I'd like to understand more about your current condition."
- "Could you tell me about your mental health history?"

Generate ONLY the question (no explanation):
"""
    
    try:
        response = analysis_llm([
            SystemMessage(content=f"Generate natural, conversational questions in {language}. Sound like a human friend, not a robot therapist."),
            HumanMessage(content=generation_prompt)
        ])
        
        question = response.content.strip().strip('"')
        return question
        
    except Exception as e:
        print(f"Question generation failed: {e}")
        # Simple, natural fallbacks
        if language == 'roman_urdu':
            fallbacks = {
                'current_condition': 'Aap kaisa feel kar rahe hain abhi?',
                'duration': 'Yeh kab se ho raha hai?',
                'mental_health_history': 'Aapke family mein kisi ko mental health issues hain?',
                'physical_activity': 'Koi exercise waghaira karte hain?',
                'suicidal_thoughts': 'Kabhi suicide ke thoughts aye hain?'
            }
        else:
            fallbacks = {
                'current_condition': "How are you feeling right now?",
                'duration': "How long has this been going on?",
                'mental_health_history': "Any mental health stuff in your family history?",
                'physical_activity': "Do you exercise at all?",
                'suicidal_thoughts': "Have you had any thoughts of suicide?"
            }
        
        return fallbacks.get(missing_field, f"Tell me about {field_description}?")

def get_next_missing_field(user_data: dict) -> str:
    """Get the next field that needs to be collected"""
    # Define priority order for information gathering
    priority_order = ['current_condition', 'duration', 'mental_health_history', 'physical_activity', 'suicidal_thoughts']
    
    info_needed = user_data.get("information_needed", {})
    
    # Check each field in priority order
    for field in priority_order:
        field_info = info_needed.get(field, {})
        if field_info.get("required", True) and not field_info.get("collected", False):
            return field
    
    return None

def analyze_conversation_with_enhanced_system(message: str, chat_history: list, 
                                            user_data: dict, session_key: str,
                                            language: str = "english") -> dict:
    """Main conversation analysis function"""
    
    # Create/restore managers
    flow_manager, pattern_learner = create_or_restore_managers(user_data)
    
    # Get LLM analysis
    analysis = llm_therapeutic_sentiment_analysis(message, chat_history, language)
    
    # Learn from interaction
    pattern_learner.analyze_user_communication_style(session_key, message, analysis)
    
    # Get personalized approach
    user_approach = pattern_learner.get_personalized_approach(session_key)
    
    # Determine next action
    missing_field = get_next_missing_field(user_data)
    next_action = flow_manager.determine_next_action(analysis, user_data, chat_history, missing_field)
    
    # Save managers back to user_data
    user_data = save_managers_to_user_data(user_data, flow_manager, pattern_learner)
    
    return {
        'analysis': analysis,
        'next_action': next_action,
        'user_approach': user_approach,
        'flow_manager_state': {
            'resistance_count': flow_manager.resistance_count,
            'hostility_count': flow_manager.hostility_count,
            'consecutive_low_engagement': flow_manager.consecutive_low_engagement
        }
    }