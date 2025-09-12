# app/agents/stt_agent.py
import logging
import json
from dotenv import load_dotenv
from livekit import rtc
from livekit.agents import Agent, AgentSession, JobContext, JobRequest, RoomIO, WorkerOptions, cli
from livekit.agents.llm import ChatContext, ChatMessage, StopResponse
from livekit.plugins import deepgram  # Using OpenAI Whisper Turbo as requested
from app.services.chat_service import process_chat


logger = logging.getLogger("stt-transcription")
logger.setLevel(logging.INFO)
load_dotenv()

class TranscriptionAgent(Agent):
    """
    STT-only agent for voice message transcription in text chat
    No TTS or conversational responses - just transcription
    """
    def __init__(self) -> None:
        super().__init__(
            instructions="You are a transcription agent that converts speech to text for voice messages.",
            #   stt = groq.STT(
            #     model="whisper-large-v3-turbo",
            #     language="en",
            # ), # Whisper Turbo
            #    vad=silero.VAD.load()
            stt=deepgram.STT(model="nova-3")
            # No LLM or TTS - we only need transcription
        )
        self._room = None
        
    async def on_user_turn_completed(self, turn_ctx: ChatContext, new_message: ChatMessage) -> None:
        """
        Called when user finishes speaking (after end_turn RPC call)
        This is where we get the final transcription and process it
        """
        try:
            if not new_message.text_content:
                logger.info("Empty transcription received, ignoring")
                raise StopResponse()
            
            transcribed_text = new_message.text_content.strip()
            logger.info(f"Transcription received: {transcribed_text}")
            
            
           
            transcribed_text = new_message.text_content.strip()
            logger.info(f"Transcription received: {transcribed_text}")
            
            # For now, just send the transcription back
            # Later you can integrate with your chat_service.process_chat()
            result = {
                "type": "voice_message_result",
                "transcription": transcribed_text,
                "response": f"You said: {transcribed_text}",  # Placeholder response
                "success": True
            }
            
            
            
            # Send result to frontend
            await self._send_result_to_frontend(result)
            
            
            
        except Exception as e:
            logger.error(f"Error processing transcription: {e}")
            error_result = {
                "type": "voice_message_result",
                "transcription": new_message.text_content if new_message.text_content else "",
                "response": f"Error processing voice message: {str(e)}",
                "success": False
            }
            await self._send_result_to_frontend(error_result)
        
        # Always stop - we don't want the agent to generate speech responses
        raise StopResponse()
    
    def _get_session_key(self, turn_ctx: ChatContext) -> str:
        """Extract session key from room context"""
        # You can get session key from room name, participant attributes, or metadata
        # This will depend on how you structure your room creation
        try:
            # Option 1: From room name (if you embed session key in room name)
            room_name = turn_ctx.room.name if hasattr(turn_ctx, 'room') else None
            if room_name and room_name.startswith('transcription_'):
                return room_name.replace('transcription_', '')
            
            # Option 2: From participant metadata (set when creating token)
            # This would need to be passed when creating the access token
            
            # Fallback - you might need to adjust this based on your implementation
            return "default_session"  # Replace with actual session key logic
            
        except Exception as e:
            logger.error(f"Error getting session key: {e}")
            return None
    
    async def _send_result_to_frontend(self, result: dict):
        """Send transcription result to frontend via data channel"""
        try:
            # This would send data to the frontend participant
            # You'll need to store reference to the room to use this
            data_bytes = json.dumps(result).encode('utf-8')
            # Note: You'll need room reference here - see entrypoint function
            logger.info(f"Sending result to frontend: {result}")
        except Exception as e:
            logger.error(f"Error sending result to frontend: {e}")

async def entrypoint(ctx: JobContext):
    """Main entrypoint for the transcription agent"""
    try:
        logger.info("Starting transcription agent")
        
        session = AgentSession(turn_detection="manual")  # Manual turn detection for push-to-talk
        room_io = RoomIO(session, room=ctx.room)
        await room_io.start()
        
        agent = TranscriptionAgent()
        
        # Store room reference in agent for data channel communication
        agent._room = ctx.room
        
        await session.start(agent=agent)
        
        # Disable audio input by default - only enable on push-to-talk
        session.input.set_audio_enabled(False)
        
        logger.info("Transcription agent started, audio input disabled")
        
        @ctx.room.local_participant.register_rpc_method("start_turn")
        async def start_turn(data: rtc.RpcInvocationData):
            """Called when user presses voice message button"""
            try:
                logger.info(f"Starting voice message recording for: {data.caller_identity}")
                session.interrupt()  # Stop any ongoing processing
                session.clear_user_turn()  # Clear previous transcription
                
                # Listen to the specific participant who pressed the button
                room_io.set_participant(data.caller_identity)
                
                # Enable audio input to start transcription
                session.input.set_audio_enabled(True)
                
                # # Log the current audio tracks for debugging
                # caller = ctx.room.get_participant(data.caller_identity)
                # if caller:
                #     mic_track = caller.get_track_publication(rtc.TrackSource.MICROPHONE)
                #     if mic_track:
                #         logger.info(f"Microphone track found: {mic_track.track_name}")
                #     else:
                #         logger.warning("No microphone track found for caller")
                # else:
                #     logger.warning(f"Caller participant not found: {data.caller_identity}")
                
                logger.info("Audio input enabled, ready to record")
                
            except Exception as e:
                logger.error(f"Error starting turn: {e}")
                # Send error back to frontend
                error_result = {
                    "type": "voice_message_result",
                    "transcription": "",
                    "response": f"Error starting recording: {str(e)}",
                    "success": False
                }
                data_bytes = json.dumps(error_result).encode('utf-8')
                await ctx.room.local_participant.publish_data(data_bytes)
        @ctx.room.local_participant.register_rpc_method("end_turn")
        async def end_turn(data: rtc.RpcInvocationData):
            """Called when user releases voice message button"""
            try:
                logger.info(f"Ending voice message recording for: {data.caller_identity}")
                
                # Disable audio input
                # Check if we have any audio input
                # caller = ctx.room.get_participant(data.caller_identity)
                # if caller:
                #     mic_track = caller.get_track_publication(rtc.TrackSource.MICROPHONE)
                #     if not mic_track or not mic_track.is_subscribed:
                #         logger.warning("No subscribed microphone track found")
                #         # Send error to frontend
                #         error_result = {
                #             "type": "voice_message_result",
                #             "transcription": "",
                #             "response": "No audio input detected. Please check your microphone permissions.",
                #             "success": False
                #         }
                #         data_bytes = json.dumps(error_result).encode('utf-8')
                #         await ctx.room.local_participant.publish_data(data_bytes)
                #         return
                
                # Disable audio input
                session.input.set_audio_enabled(False)
                
                # Commit the user turn to finalize transcription
                # This will trigger on_user_turn_completed
                session.commit_user_turn(
                    transcript_timeout=15.0  # Wait up to 15 seconds for final transcript
                )
                
                logger.info("Audio input disabled, processing transcription")
                
            except Exception as e:
                logger.error(f"Error ending turn: {e}")
                # Send error back to frontend
                error_result = {
                    "type": "voice_message_result",
                    "transcription": "",
                    "response": f"Error processing recording: {str(e)}",
                    "success": False
                }
                data_bytes = json.dumps(error_result).encode('utf-8')
                await ctx.room.local_participant.publish_data(data_bytes)
        @ctx.room.local_participant.register_rpc_method("cancel_turn")
        async def cancel_turn(data: rtc.RpcInvocationData):
            """Called if user cancels voice message"""
            try:
                logger.info(f"Cancelling voice message for: {data.caller_identity}")
                
                # Disable audio input
                session.input.set_audio_enabled(False)
                
                # Clear the turn without processing
                session.clear_user_turn()
                
                # Send cancellation message to frontend
                result = {
                    "type": "voice_message_cancelled",
                    "message": "Voice message cancelled"
                }
                
                data_bytes = json.dumps(result).encode('utf-8')
                await ctx.room.local_participant.publish_data(data_bytes)
                
                logger.info("Voice message cancelled")
                
            except Exception as e:
                logger.error(f"Error cancelling turn: {e}")
        
        # Enhanced data sending method
        async def send_result_to_frontend(result: dict):
            """Send transcription result to frontend participants"""
            try:
                data_bytes = json.dumps(result).encode('utf-8')
                await ctx.room.local_participant.publish_data(data_bytes)
                logger.info(f"Sent result to frontend: {result['type']}")
            except Exception as e:
                logger.error(f"Error sending data to frontend: {e}")
        
        # Update agent's send method
        agent._send_result_to_frontend = send_result_to_frontend
        
        logger.info("STT Transcription agent ready and waiting for push-to-talk events")
        
    except Exception as e:
        logger.error(f"Error in transcription agent entrypoint: {e}")
        raise

async def handle_request(request: JobRequest) -> None:
    """Handle incoming room requests"""
    try:
        logger.info(f"Handling transcription request for room: {request.room}")
        
        await request.accept(
            identity="transcription-agent",
            name="STT Transcription Agent",
            attributes={
                "push-to-talk": "1",
                "transcription-only": "1",
                "agent-type": "stt"
            },
        )
        
        logger.info("Transcription agent accepted room request")
        
    except Exception as e:
        logger.error(f"Error handling request: {e}")
        raise

if __name__ == "__main__":
    logger.info("Starting STT Transcription Agent")
    cli.run_app(WorkerOptions(
        entrypoint_fnc=entrypoint, 
        request_fnc=handle_request
    ))