# run_stt_agent.py
"""
Standalone script to run the STT Transcription Agent
This should be run separately from your main FastAPI application

Usage: python run_stt_agent.py
"""

import sys
import os
from pathlib import Path

# Add the app directory to the Python path so we can import our modules
sys.path.append(str(Path(__file__).parent))

if __name__ == "__main__":
    print("🎙️  Starting STT Transcription Agent...")
    print("This agent handles voice message transcription for text chat")
    print("Make sure your .env file has the required LiveKit variables:")
    print("- LIVEKIT_API_KEY")
    print("- LIVEKIT_API_SECRET") 
    print("- LIVEKIT_URL")
    print("- DEEPGRAM_API_KEY ")
    print("-" * 50)
    
    try:
        from stt_agent import cli, WorkerOptions, entrypoint, handle_request
        
        cli.run_app(WorkerOptions(
            entrypoint_fnc=entrypoint,
            request_fnc=handle_request
        ))
        
    except ImportError as e:
        print(f"❌ Import error: {e}")
        print("Make sure you have installed all required dependencies:")
        print("pip install livekit-agents livekit-plugins-deepgram")
        sys.exit(1)
    except Exception as e:
        print(f"❌ Error starting STT agent: {e}")
        sys.exit(1)