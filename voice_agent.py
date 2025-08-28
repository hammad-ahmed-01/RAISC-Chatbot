# voice_agent.py
import asyncio
import os
from datetime import datetime
from dotenv import load_dotenv
import logging

from livekit.agents import (
    Agent,
    AgentSession,
    JobContext,
    WorkerOptions,
    cli,
    RoomInputOptions,
    ModelSettings,
    FunctionTool,
    stt,
)

from livekit.plugins import azure, groq, silero, noise_cancellation
from livekit import rtc
from typing import AsyncIterable, Optional

# Import your existing services
from app.services.chat_service import process_chat
from app.services.user_service import get_user_data, store_user_data
from app.services.firestore_service import get_chat_history

load_dotenv()

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class MentalHealthVoiceAgent(Agent):
    """
    Mental Health Voice Agent that integrates with existing chatbot logic
    """
    
    def __init__(self) -> None:
        super().__init__(
            instructions="""
            You are a compassionate mental health assistant. Your role is to:
            
            1. Provide emotional support and guidance through voice conversation
            2. Listen actively and respond empathetically
            3. Ask appropriate follow-up questions to understand the user's needs
            4. Recognize signs of distress and provide resources when needed
            5. Maintain professional boundaries at all times
            6. Keep responses concise and conversational for voice interaction
            
            Important guidelines:
            - Always prioritize user safety and wellbeing
            - Encourage professional help when appropriate
            - Be non-judgmental and supportive in all interactions
            - Use a calm, therapeutic tone
            - Keep responses brief (1-3 sentences) for natural voice conversations
            - Ask one question at a time to avoid overwhelming the user
            - Acknowledge the user's feelings before providing guidance
            
            CRITICAL: You are NOT a replacement for professional mental health care.
            Always remind users to seek professional help for serious mental health concerns.
            
            For voice conversations, be more conversational and less formal than text chat.
            Use natural speech patterns and appropriate pauses.
            """
        )
        self.session_key = None
        self.chat_history = []
        self.latest_user_message = None  # Store the latest user message from STT
        self.rag_response = None  # Store the RAG response
        self.initial_greeting_sent = False  # Track if initial greeting was sent
        
    async def process_voice_message(self, session_key: str, message: str) -> str:
        try:
            logger.info(f"Processing message: '{message}' with session_key: '{session_key}'")
            
            # Check current user data state
            user_data = get_user_data(session_key) or {}
            logger.info(f"Current user data: {user_data}")
            
            response = await process_chat(session_key, message)
            
            # Log the response to see what your chatbot is returning
            logger.info(f"Chat service response: {response}")
            
            if response is None:
                logger.warning("process_chat returned None")
                return "I'm sorry, I'm having trouble processing that right now. Could you please try again?"
            
            response_text = response.get("response", "I'm sorry, I didn't understand that.")
            return self._make_voice_friendly(response_text)
            
        except Exception as e:
            logger.error(f"Error processing voice message: {e}")
            return "I'm experiencing some technical difficulties. Please try again."
    async def say(self, message: str, *, allow_interruptions: bool = True, add_to_chat_ctx: bool = True):
        """
        Override say method for logging and voice optimization
        """
        try:
            # Just log and pass through - the real processing happens in llm_node
            logger.info(f"Agent speaking: {message[:100]}...")
            return await super().say(message, 
                                   allow_interruptions=allow_interruptions, 
                                   add_to_chat_ctx=add_to_chat_ctx)
        except Exception as e:
            logger.error(f"Error in say method: {e}")
            try:
                fallback_msg = "I'm having trouble responding. Please try again."
                return await super().say(fallback_msg, 
                                       allow_interruptions=allow_interruptions, 
                                       add_to_chat_ctx=add_to_chat_ctx)
            except:
                pass
            
        except Exception as e:
            logger.error(f"Error processing voice message: {e}")
            return "I'm experiencing some technical difficulties. Please try again."
    
    def _make_voice_friendly(self, text: str) -> str:
        """
        Adapt text responses for voice conversation
        """
        # Split long responses into shorter, more conversational chunks
        sentences = text.split('. ')
        if len(sentences) > 3:
            # For voice, keep responses shorter - take first 2-3 key sentences
            return '. '.join(sentences[:3]) + '.'
        return text
    
    async def initialize_session(self, room_name: str):
        """
        Initialize voice session using a known test user token for backend compatibility
        """
        # For testing, use a known test user token that exists in your Django backend
        self.session_key = "10546c621bb12ee4f96d240992b1516a0ed0d131"  
        
        logger.info(f"Voice session using test user token: {self.session_key}")
        logger.info(f"LiveKit room name: {room_name}")
        
       # Fetch user data BEFORE generating greeting to give appropriate welcome message
        user_data = get_user_data(self.session_key) or {}
        logger.info(f"Retrieved user data for initial greeting: {user_data}")
        
        # Initialize chat history from Firestore (using the user token as session key)
        self.chat_history = get_chat_history(self.session_key) or []
        
        # Generate appropriate greeting based on user's profile data
        if user_data and user_data.get("name"):
            # Returning user with profile
            return f"Hi {user_data.get('name')}! Welcome back to our mental health assistant. Say 'start' when you're ready to continue."
        else:
            # New user or user without profile
            return "Hi! Welcome to our mental health voice assistant. Say 'start' when you're ready to begin."

    async def stt_node(
        self, 
        audio: AsyncIterable[rtc.AudioFrame], 
        model_settings: ModelSettings
    ) -> Optional[AsyncIterable[stt.SpeechEvent]]:
        """
        Override STT node to intercept user transcriptions and process through RAG
        This is where we get the user's speech converted to text
        """
        try:
            logger.info("STT node processing audio...")
            
            # Use the default STT processing first
           # Use the default STT processing first
            async for event in Agent.default.stt_node(self, audio, model_settings):
                logger.info(f"STT event type: {type(event)}")
                logger.info(f"STT event: {event}")
                
                # Check if this is a final transcription event
                if hasattr(event, 'type') and hasattr(event, 'alternatives'):
                    if event.type == stt.SpeechEventType.FINAL_TRANSCRIPT:
                        # Get the transcribed text
                        if event.alternatives and len(event.alternatives) > 0:
                            transcribed_text = event.alternatives[0].text
                            logger.info(f"Final transcript received: {transcribed_text}")
                            
                            # Store the user message and process through RAG
                            self.latest_user_message = transcribed_text
                            
                            # Process all messages through your existing RAG system
                            # Let process_chat() handle all logic including "start", "okay", etc.
                            try:
                                rag_result = await self.process_voice_message(
                                    self.session_key, 
                                    transcribed_text
                                )
                                
                                # Handle the case where process_voice_message returns None
                                if rag_result:
                                    self.rag_response = rag_result
                                    logger.info(f"RAG response ready: {self.rag_response[:100]}...")
                                else:
                                    logger.warning("RAG processing returned None")
                                    self.rag_response = "I'm sorry, I'm having trouble processing that. Could you please try again?"
                                    
                            except Exception as rag_error:
                                logger.error(f"RAG processing failed: {rag_error}")
                                self.rag_response = "I'm having trouble understanding. Could you please try again?"
                
                # Always yield the original event to maintain the pipeline
                yield event
                
        except Exception as e:
            logger.error(f"Error in stt_node: {e}")
            # Fallback to default STT if our processing fails
            async for event in Agent.default.stt_node(self, audio, model_settings):
                yield event

    async def llm_node(
        self,
        chat_ctx,  # Remove type hint since we're not using it anyway
        tools: list[FunctionTool],
        model_settings: ModelSettings
    ) -> AsyncIterable[str]:  # Return simple string instead of ChatChunk
        """
        Override LLM node to use our pre-processed RAG response
        The real processing happens in stt_node, this just returns the result
        """
        try:
            # Check if we have a RAG response ready from stt_node
            if self.rag_response:
                logger.info(f"Using RAG response from stt_node: {self.rag_response[:100]}...")
                
                # Return the response as a simple string
                response = self.rag_response
                
                # Clear the response so it's not reused
                self.rag_response = None
                self.latest_user_message = None
                
                yield response
                return
            
            # If this is the initial greeting (no user input yet), don't use default LLM
            if not self.initial_greeting_sent:
                logger.info("Initial greeting phase - skipping LLM processing")
                self.initial_greeting_sent = True
                return
            
            # Fallback to default LLM if no RAG response available
            logger.info("No RAG response available, using default LLM")
            async for chunk in Agent.default.llm_node(self, chat_ctx, tools, model_settings):
                yield chunk
                
        except Exception as e:
            logger.error(f"Error in llm_node: {e}")
            # Fallback to default LLM
            async for chunk in Agent.default.llm_node(self, chat_ctx, tools, model_settings):
                yield chunk

async def entrypoint(ctx: JobContext):
    """
    Main entry point for the LiveKit voice agent
    """
    try:
        logger.info("Starting mental health voice agent initialization...")
        
        await ctx.connect()
        
        # Validate environment variables
        required_env_vars = [
            "LIVEKIT_URL",
            "LIVEKIT_API_KEY", 
            "LIVEKIT_API_SECRET",
            "GROQ_API_KEY",
            "AZURE_SPEECH_KEY",
            "AZURE_SPEECH_REGION"
        ]
        
        missing_vars = [var for var in required_env_vars if not os.getenv(var)]
        if missing_vars:
            logger.error(f"Missing required environment variables: {', '.join(missing_vars)}")
            return
        
        # Initialize our mental health agent
        mental_health_agent = MentalHealthVoiceAgent()
        logger.info("Mental Health Voice Agent created successfully")
        
        # Initialize components with error handling
        try:
            stt = groq.STT(
                model="whisper-large-v3-turbo",
                language="en",
            )
            logger.info("STT initialized with Groq Whisper")
        except Exception as e:
            logger.error(f"STT initialization failed: {e}")
            return
        
        try:
            llm = groq.LLM(
                model="llama-3.3-70b-versatile",
                temperature=0.3,
            )
            logger.info("LLM initialized with Groq Llama")
        except Exception as e:
            logger.error(f"LLM initialization failed: {e}")
            return
        
        try:
            tts = azure.TTS(
                speech_key=os.getenv("AZURE_SPEECH_KEY"),
                speech_region=os.getenv("AZURE_REGION"),
                voice="en-US-AriaNeural",  # Calm, therapeutic voice
                sample_rate=24000,
            )
            logger.info("TTS initialized with Azure")
        except Exception as e:
            logger.error(f"TTS initialization failed: {e}")
            return
        
        try:
            vad = silero.VAD.load(
                min_silence_duration=0.5,  # Appropriate for therapy conversations
                min_speech_duration=0.3,
            )
            logger.info("VAD initialized")
        except Exception as e:
            logger.error(f"VAD initialization failed: {e}")
            return
        
        # Create session - no need to override LLM processing here anymore
        session = AgentSession(
            stt=stt,
            llm=llm,
            tts=tts,
            vad=vad,
        )
        
        logger.info("Session created, starting mental health voice agent...")
        
        # Start the session with our mental health agent
        await session.start(
            room=ctx.room,
            agent=mental_health_agent,
            room_input_options=RoomInputOptions(
                noise_cancellation=noise_cancellation.BVC(),
            ),
        )
        
        logger.info("Mental health voice agent started successfully")
        
        # Initialize the session and generate welcome message
        room_name = ctx.room.name or f"voice_session_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        welcome_message = await mental_health_agent.initialize_session(room_name)
        
        try:
            # Use session.say() instead of agent.say() for initial greeting
            await session.say(welcome_message, add_to_chat_ctx=False)
            logger.info(f"Initial greeting sent: {welcome_message}")
        except Exception as e:
            logger.error(f"Failed to send initial greeting via session.say(): {e}")
            # Fallback: try the generate_reply method
            try:
                await asyncio.wait_for(
                    session.generate_reply(instructions=f"Say this exact message: {welcome_message}"),
                    timeout=30.0
                )
                logger.info(f"Initial greeting sent via generate_reply: {welcome_message}")
            except asyncio.TimeoutError:
                logger.warning("Initial greeting generation timed out")
            except Exception as fallback_e:
                logger.error(f"Fallback greeting also failed: {fallback_e}")
        
    except Exception as e:
        logger.error(f"Entrypoint error: {e}")
        raise

def main():
    """
    Main function to run the voice agent
    """
    print("Starting Mental Health Voice Agent...")
    print(f"LiveKit URL: {os.getenv('LIVEKIT_URL')}")
    print("Agent is ready and waiting for voice connections...")
    
    # Run the agent
    cli.run_app(
        WorkerOptions(
            entrypoint_fnc=entrypoint,
        )
    )

if __name__ == "__main__":
    main()