#!/usr/bin/env python3
"""
Virtual Robot Face with MAR Detection and FAST Speech Recognition
Ultra-fast streaming speech recognition with real-time transcription feedback.
"""
import asyncio
import time
import sys
from robot_face_system import RobotFace, RobotState
from face_detector import FaceDetector
from speech_handler import SpeechDetector
from voice_handler import RobotVoice
from llm_handler import AsyncLLMHandler, LLMResponse
import threading

class Robot:
    """Manages robot state and responses"""
    robot_face = None
    face_detection = None
    speech_detection = None
    robot_voice = None
    llm_handler = None

     # State tracking
    last_speaking = False
    last_face_visible = False
    mouth_speaking = None
    face_visible = None
    head_x = None
    head_y = None
    mar_value = None

    def __init__(self):
        """Initialize all systems."""
        print("Initializing robot")
        self.robot_face = RobotFace()
        self.robot_face.start_animation("idle")
        # Initialize facial details detector
        self.face_detection = FaceDetector(speaking_threshold=0.04)
        

        # initialize speech_detection and transcription
        self.speech_detection = SpeechDetector()
      
        # Start speech handler in background
        speech_thread = threading.Thread(target=self.start_speech_detection, daemon=True)
        speech_thread.start()
        print("Started background speech detection")

        #initialize voice_handler
        self.robot_voice = RobotVoice(
            on_speech_start=self.handleRobotVoiceStarted,
            on_speech_end=self.handleRobotVoiceEnded,
            on_speech_interrupted=self.handleRobotVoiceInterrupted,
        )
    
        #initialize LLM handler
        self.llm_handler = AsyncLLMHandler()          
        
        #delayed start
        self.face_detection.start()

        # Let systems initialize
        time.sleep(1.0) 


    def updateFaceRecognitionInfo(self):
        # Get tracking data
        status = self.face_detection.get_status()
        self.mouth_speaking = status['speaking']
        self.face_visible = status['face_visible']
        self.head_x = status['head_x']
        self.head_y = status['head_y']
        self.mar_value = status['mar']
 
        # Handle face detection changes
        if self.face_visible != self.last_face_visible:
            if self.face_visible:
                print("👤 FACE DETECTED") 
                #----here you could initiate the interaction --------------
                

                #----END: here you could initiate the interaction --------------
            else:
                print("👤 FACE LOST") #<----here you could double check this or end the interaction
            self.last_face_visible = self.face_visible


    async def handle_human_speech_pause(self, transcription: str):
            print(f"PAUSE: '{transcription}'")
            if transcription == '':
                print("no words detected")
                return
            #------------------- turn taking code here to skip the wait-----------------
            



            #------------------- end: turn taking code here to skip the wait------------
            
        
    async def handle_human_speech_end(self, transcription: str):
            print(f"END: '{transcription}'")
            if transcription != "":
                self.processWithLLM(transcription)

    def start_speech_detection(self):
        """this method starts and runs the async voice activity detection in a separate thread"""  
        async def speech_loop():
            speech_handler = self.speech_detection

            if not speech_handler.start():
                print("Failed to start speech handler")
                return
            
            try:
                await speech_handler.listen_with_voice_activity_detection(
                    voice_threshold=-40,
                    silence_timeout=3.0,
                    on_pause=self.handle_human_speech_pause,
                    on_end=self.handle_human_speech_end
                )
            finally:
                speech_handler.stop()
        
        asyncio.run(speech_loop())


    def handle_LLM_chunk(self, chunk: str):
        print(chunk, end='', flush=True)  # Display each word as it arrives
        #-------------------- buffering words code here -----------------------



        #-------------------- END: buffering words code here -----------------------
        

    def handle_LLM_complete(self, response: LLMResponse):
        print(f"\nComplete response received: {response.content}")
        self.makeTheRobotSay(response.content)

    def handleRobotVoiceEnded(self):
        print("\n unpausing speech recognition \n")
        self.speech_detection.pause_voice_detection(False)

    def handleRobotVoiceStarted(self, text):
         #We stop the transcription so that the robot does not transcribe itself
        print("\n pausing speech recognition \n")
        self.speech_detection.pause_voice_detection(True)

    def handleRobotVoiceInterrupted(self):
        pass

    def processWithLLM(self, utterance):
        asyncio.get_event_loop().create_task(self.llm_handler.send_prompt_streaming(
            utterance, 
            callback=self.handle_LLM_chunk,
            final_callback=self.handle_LLM_complete
        ))

    def makeTheRobotSay(self, utterance):
        self.robot_voice.speak(utterance)

    def reactWithRobotFace(self):
        self.robot_face.loop() #<-- main loop of the face
        
        if self.robot_voice.is_speaking():
            self.robot_face.set_state(RobotState.SPEAKING)
        #-----------outcomment below to enable listening state for robot face ----
        #elif self.mouth_speaking or self.speech_detection.is_currently_recording():
            #self.robot_face.set_state(RobotState.LISTENING)
        #-----------END outcomment -----------------------------------------------
        else:
            self.robot_face.set_state(RobotState.IDLE)
        #------------------- Update robot eyes here based on the face tracking ----------------

        
        #------------------- END: Update robot eyes here based on the face tracking ----------------

    def run_loop(self):
        """Main loop"""
        try:
            while self.robot_face.handle_events():
                #self.robot_face.loop() 
                self.updateFaceRecognitionInfo()
                self.reactWithRobotFace()
                time.sleep(0.1) #<--- we add a delay so it doesnt run wild with the cpu
                
        except KeyboardInterrupt:
            print("\n🛑 Shutdown requested...")
        except Exception as e:
            print(f"\n❌ Error in main loop: {e}")
            import traceback
            traceback.print_exc()
        
        # Cleanup
        self.face_detection.stop()
        self.speech_detection.stop()
        


def main():
    """Main entry point."""
    # Setup all systems
    robot = Robot()
    
    # Run the main loop
    robot.run_loop()
 

if __name__ == "__main__":
    sys.exit(main())