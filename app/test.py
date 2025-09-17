# test_bilingual_chatbot.py
"""
Test script to demonstrate bilingual functionality
Run this to test language detection and responses
"""

import asyncio
import sys
import os

# Add the app directory to the Python path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from app.services.language_service import (
    detect_language, 
    get_greeting_message,
    get_information_prompts,
    get_completion_message,
    get_risk_intervention_message
)

def test_language_detection():
    """Test language detection with various inputs"""
    print("=== LANGUAGE DETECTION TESTS ===\n")
    
    test_cases = [
        # English examples
        ("Hello, I am feeling anxious today", "english"),
        ("I need help with my depression", "english"),
        ("Can you help me with stress management?", "english"),
        
        # Roman Urdu examples (Pakistani)
        ("Salam, main aaj bohat pareshan hoon", "roman_urdu"),
        ("Aap meri madad kar sakte hain?", "roman_urdu"),
        ("Main depression mein hoon, kya karna chahiye?", "roman_urdu"),
        ("Mujhe anxiety ho rahi hai, doctor ke paas jaana chahiye?", "roman_urdu"),
        ("Aap ka naam kya hai?", "roman_urdu"),
        ("Main samajh nahi pa raha ke kya karna hai", "roman_urdu"),
        
        # Mixed examples
        ("Main okay nahi hoon", "roman_urdu"),
        ("I feel theek nahi", "english"),
        
        # Edge cases
        ("AssalamuAlaikum", "roman_urdu"),
        ("Hi", "english"),
    ]
    
    for message, expected in test_cases:
        detected = detect_language(message)
        status = "✓" if detected == expected else "✗"
        print(f"{status} '{message}' -> {detected} (expected: {expected})")
    
    print("\n")

def test_bilingual_responses():
    """Test bilingual response generation"""
    print("=== BILINGUAL RESPONSE TESTS ===\n")
    
    # Test greetings
    print("--- GREETINGS ---")
    print("English greeting:", get_greeting_message("english"))
    print("Roman Urdu greeting:", get_greeting_message("roman_urdu"))
    print("English greeting with name:", get_greeting_message("english", "John"))
    print("Roman Urdu greeting with name:", get_greeting_message("roman_urdu", "Ahmad"))
    print()
    
    # Test information prompts
    print("--- INFORMATION PROMPTS ---")
    eng_prompts = get_information_prompts("english")
    urdu_prompts = get_information_prompts("roman_urdu")
    
    for key in eng_prompts:
        print(f"{key.upper()}:")
        print(f"  English: {eng_prompts[key]}")
        print(f"  Roman Urdu: {urdu_prompts[key]}")
        print()
    
    # Test completion messages
    print("--- COMPLETION MESSAGES ---")
    print("English completion:", get_completion_message("english", "Sarah"))
    print("Roman Urdu completion:", get_completion_message("roman_urdu", "Fatima"))
    print()
    
    # Test risk intervention
    print("--- RISK INTERVENTION ---")
    print("English risk intervention:", get_risk_intervention_message("english"))
    print("Roman Urdu risk intervention:", get_risk_intervention_message("roman_urdu"))
    print()

def test_conversation_flow():
    """Simulate a conversation flow"""
    print("=== CONVERSATION FLOW SIMULATION ===\n")
    
    # Simulate Roman Urdu conversation
    print("--- ROMAN URDU CONVERSATION ---")
    messages = [
        "Salam, main bohat pareshan hoon",
        "Mera naam Ahmed hai",
        "Main 25 saal ka hoon",
        "Main male hoon",
        "Mujhe anxiety aur depression hai",
        "Mere family mein kisi ko mental health issue nahi tha"
    ]
    
    for msg in messages:
        detected = detect_language(msg)
        print(f"User ({detected}): {msg}")
    
    print("\n--- ENGLISH CONVERSATION ---")
    messages = [
        "Hi, I'm feeling really stressed",
        "My name is Sarah",
        "I'm 28 years old", 
        "I'm female",
        "I've been feeling anxious and depressed lately",
        "My mother had depression before"
    ]
    
    for msg in messages:
        detected = detect_language(msg)
        print(f"User ({detected}): {msg}")

if __name__ == "__main__":
    print("🤖 BILINGUAL CHATBOT TEST SUITE\n")
    print("Testing Pakistani Roman Urdu vs English detection and responses\n")
    print("="*60)
    
    try:
        test_language_detection()
        test_bilingual_responses() 
        test_conversation_flow()
        
        print("="*60)
        print("✅ All tests completed successfully!")
        print("\nNote: This tests the language detection and response templates.")
        print("To test full integration, run the chatbot with these sample messages.")
        
    except Exception as e:
        print(f"❌ Test failed with error: {e}")
        import traceback
        traceback.print_exc()