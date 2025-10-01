#!/usr/bin/env python3
"""
Fast Streaming Speech Handler using sounddevice (FFmpeg-free version)
Real-time audio processing with overlapping windows for maximum speed.
Processes audio directly with numpy arrays to avoid FFmpeg dependency.
FIXED: Prevents robot speech contamination during pause periods.
"""
import threading
import queue
import time
import asyncio
import numpy as np
import sounddevice as sd
import whisper
import wave
from collections import deque
from typing import Optional, Callable, Awaitable
import os


class SpeechDetector:
    """
    High-speed streaming speech handler with real-time processing.
    Uses sounddevice instead of PyAudio for easier installation.
    FIXED: Now properly handles pause state to prevent robot speech contamination.
    """
    
    def __init__(self, 
                 model_size: str = "tiny",
                 sample_rate: int = 16000,
                 channels: int = 1,
                 processing_window: float = 1.0,  # Process every 1 second
                 overlap: float = 0.2,  # 20% overlap between windows
                 on_partial_speech: Optional[Callable[[str], None]] = None,
                 on_final_speech: Optional[Callable[[str], None]] = None):
        """
        Initialize the fast speech handler.
        
        Args:
            model_size: Whisper model ('tiny' recommended for speed)
            sample_rate: Audio sample rate in Hz
            channels: Number of audio channels (1 for mono)
            processing_window: How often to process audio (seconds)
            overlap: Overlap between processing windows (0.0-1.0)
            on_partial_speech: Callback for partial transcriptions while speaking
            on_final_speech: Callback for final transcription when done
        """
        self.model_size = model_size
        self.sample_rate = sample_rate
        self.channels = channels
        self.processing_window = processing_window
        self.overlap = overlap
        self.on_partial_speech = on_partial_speech
        self.on_final_speech = on_final_speech
        
        # Calculate processing parameters
        self.window_samples = int(processing_window * sample_rate)
        self.overlap_samples = int(overlap * self.window_samples)
        self.step_samples = self.window_samples - self.overlap_samples
        
        # Audio stream
        self.stream = None
        
        # Whisper model (loaded lazily)
        self.whisper_model = None
        
        # Threading and queues
        self.audio_queue = queue.Queue(maxsize=100)
        self.processing_thread = None
        # Running state
        self.running = False
        
        # Rolling audio buffer for streaming processing
        self.audio_buffer = deque(maxlen=self.sample_rate * 10)  # 10 second rolling buffer
        self.buffer_lock = threading.Lock()
        
        # Recording state
        self.is_recording = False
        self.recording_start_time = None
        self.last_process_time = 0
        
        # Transcription management
        self.partial_transcriptions = deque(maxlen=10)
        self.current_transcription = ""
        self.final_transcription = ""
        self.transcription_lock = threading.Lock()
        
        # Performance optimization
        self.min_audio_length = 0.5  # Minimum audio length to process (seconds)
        self.process_counter = 0
        
        # Voice activity detection state (for async method)
        self.vad_running = False
        self.voice_detection_paused = False
        # Pause control (thread-safe)
        self._pause_evt = threading.Event()  # clear = running, set = paused
        self._pause_evt.clear()

        # NEW: external finalize control (thread-safe)
        self._skip_silence_evt = threading.Event()  # set => finalize ASAP
        
        print(f"FastSpeechHandler initialized with sounddevice: {model_size} model, {processing_window}s windows")
    
    def list_audio_devices(self):
        """List available audio input devices."""
        print("\nAvailable audio devices:")
        print(sd.query_devices())
        
    def _load_whisper_model(self):
        """Load the Whisper model (optimized for speed)."""
        if self.whisper_model is None:
            print(f"Loading Whisper {self.model_size} model...")
            try:
                self.whisper_model = whisper.load_model(self.model_size)
                print("✅ Whisper model loaded and ready!")
            except Exception as e:
                print(f"❌ Error loading Whisper model: {e}")
                return False
        return True
    
    def start(self):
        """Start the fast speech handler system."""
        if self.running:
            return True
            
        print("Starting fast speech handler with sounddevice...")
        
        # Pre-load the model to avoid first-time delay
        if not self._load_whisper_model():
            return False
        
        # Check available devices
        try:
            devices = sd.query_devices()
            default_input = sd.default.device[0]
            print(f"Using audio input device: {devices[default_input]['name']}")
        except Exception as e:
            print(f"Warning: Could not query audio devices: {e}")
        
        # Start processing thread
        self.running = True
        self.processing_thread = threading.Thread(target=self._fast_processing_loop, daemon=True)
        self.processing_thread.start()
        
        # Start audio stream
        try:
            self.stream = sd.InputStream(
                samplerate=self.sample_rate,
                channels=self.channels,
                dtype='float32',
                callback=self._audio_callback,
                blocksize=1024  # Small blocksize for low latency
            )
            self.stream.start()
            
        except Exception as e:
            print(f"❌ Error starting audio stream: {e}")
            print("Try running: pip install sounddevice")
            self.running = False
            return False
        
        print("✅ Fast speech handler ready! 🚀")
        return True
    
    def stop(self):
        """Stop the speech handler system."""
        if not self.running:
            return
            
        print("Stopping fast speech handler...")
        self.running = False
        self.vad_running = False
        
        # Stop audio stream
        if self.stream:
            self.stream.stop()
            self.stream.close()
        
        # Wait for thread to finish
        if self.processing_thread:
            self.processing_thread.join(timeout=2.0)
        
        print("✅ Fast speech handler stopped")
    
    def _audio_callback(self, indata, frames, time, status):
        """Sounddevice callback for continuous audio capture."""
        if status:
            print(f"Audio callback status: {status}")
        
        if not self.running:
            return
        
        try:
            # Convert to mono if stereo
            if indata.shape[1] > 1:
                audio_data = np.mean(indata, axis=1)
            else:
                audio_data = indata[:, 0]
            
            # Add to queue (non-blocking)
            if not self.audio_queue.full():
                self.audio_queue.put_nowait(audio_data.copy())
        except Exception as e:
            print(f"Audio callback error: {e}")
    
    def _fast_processing_loop(self):
        """FIXED: Optimized processing loop that respects pause state."""
        while self.running:
            try:
                # Get audio data with short timeout
                try:
                    audio_chunk = self.audio_queue.get(timeout=0.05)
                except queue.Empty:
                    continue
                
                # FIXED: Only add to buffer when NOT paused
                if not self._pause_evt.is_set():
                    # Add to rolling buffer only when not paused
                    with self.buffer_lock:
                        self.audio_buffer.extend(audio_chunk)
                
                # Process if recording and enough time has passed
                current_time = time.time()
                if (self.is_recording and 
                    not self._pause_evt.is_set() and  # Don't process while paused
                    current_time - self.last_process_time >= self.processing_window and
                    len(self.audio_buffer) >= int(self.min_audio_length * self.sample_rate)):
                    
                    self.last_process_time = current_time
                    self._process_current_buffer()
                    
            except Exception as e:
                print(f"Error in fast processing loop: {e}")
    
    def _process_current_buffer(self):
        """Process current audio buffer with Whisper."""
        try:
            with self.buffer_lock:
                if len(self.audio_buffer) < int(self.min_audio_length * self.sample_rate):
                    return
                
                # Get the most recent audio window WITH extra context to avoid missing first words
                audio_array = np.array(list(self.audio_buffer))
                
                # Include 1 second of extra context before the processing window
                extra_context_samples = int(1.0 * self.sample_rate)  # 1 second buffer
                extended_window_samples = self.window_samples + extra_context_samples
                
                # Take the extended window for processing to capture first words
                if len(audio_array) > extended_window_samples:
                    audio_to_process = audio_array[-extended_window_samples:]
                elif len(audio_array) > self.window_samples:
                    audio_to_process = audio_array[-self.window_samples:]
                else:
                    audio_to_process = audio_array
            
            # Quick processing with optimized Whisper settings
            self.process_counter += 1
            
            try:
                # Process directly with numpy array (no file I/O)
                # Ensure audio is float32 and normalized
                audio_normalized = np.asarray(audio_to_process, dtype=np.float32).reshape(-1)
                
                # Pad or trim to expected length for Whisper
                expected_length = self.sample_rate * 30  # Whisper expects 30-second chunks max
                if len(audio_normalized) > expected_length:
                    audio_normalized = audio_normalized[:expected_length]
                elif len(audio_normalized) < self.sample_rate:  # Less than 1 second
                    # Pad with zeros
                    audio_normalized = np.pad(audio_normalized, (0, self.sample_rate - len(audio_normalized)))
                
                # Process with Whisper directly from numpy array
                result = self.whisper_model.transcribe(
                    audio_normalized,  # Pass numpy array directly
                    fp16=False,
                    language="en",  # Specify language for speed
                    task="transcribe",
                    beam_size=1,     # greedy/beam path
                    # best_of removed to avoid beam + sampling conflict
                    temperature=0.0,  # deterministic
                    no_speech_threshold=0.6,  # Skip silence
                    logprob_threshold=-1.0,
                    condition_on_previous_text=False  # Faster processing
                )
                
                transcription = result["text"].strip()
                
                if transcription and len(transcription) > 1:
                    if self._is_hallucination(transcription):
                        print(f"🚫 Detected whisper hallucination, skipping: '{transcription[:100]}...'")
                        return
                    
                    with self.transcription_lock:
                        # Add to partial transcriptions
                        self.partial_transcriptions.append(transcription)
                        
                        # Update current transcription
                        self._update_current_transcription()
                    
                    # Call partial callback
                    if self.on_partial_speech and self.current_transcription:
                        self.on_partial_speech(self.current_transcription)
                
            except Exception as e:
                print(f"Whisper processing error: {e}")
                    
        except Exception as e:
            print(f"Buffer processing error: {e}")


    def _is_hallucination(self, text: str) -> bool:
        """Detect if text is a Whisper hallucination (repetitive patterns)."""
        if not text or len(text) < 50:
            return False
        
        # Check for repetitive phrases (common hallucination pattern)
        words = text.split()
        if len(words) < 10:
            return False
        
        # Look for repeating sequences of 3+ words
        for i in range(len(words) - 6):
            phrase = ' '.join(words[i:i+3])
            rest_of_text = ' '.join(words[i+3:])
            if rest_of_text.count(phrase) >= 3:  # Same phrase repeated 3+ times
                return True
        
        # Check for very long transcriptions (likely hallucination)
        if len(text) > 500:  # Adjust threshold as needed
            return True
        
        return False
    

    def _update_current_transcription(self):
        """Smart combination of partial transcriptions."""
        if not self.partial_transcriptions:
            self.current_transcription = ""
            print("just resetted partial transcription")
            return
        
        # Use the longest recent transcription as current
        recent_parts = list(self.partial_transcriptions)[-3:]  # Last 3 chunks
        
        if len(recent_parts) == 1:
            self.current_transcription = recent_parts[0]
        else:
            # Take the longest transcription from recent parts
            self.current_transcription = max(recent_parts, key=len)
    
    def start_recording(self):
        """FIXED: Start recording audio stream while preserving pre-voice context."""
        if not self.running:
            return
            
        if not self.is_recording:
            self.is_recording = True
            self.recording_start_time = time.time()
            self.last_process_time = time.time()
            
            # Clear previous transcriptions
            with self.transcription_lock:
                self.partial_transcriptions.clear()
                self.current_transcription = ""
                self.final_transcription = ""
            
            print("🎤 Started streaming recording (preserving pre-voice context)...")
    
    def stop_recording(self):
        """Stop recording and finalize transcription."""
        if not self.is_recording:
            return
        
        self.is_recording = False
        recording_duration = time.time() - self.recording_start_time if self.recording_start_time else 0
        
        print(f"ℹ️  Stopped recording ({recording_duration:.1f}s) - finalizing...")
        
        # Process any remaining audio one final time
        if len(self.audio_buffer) >= int(self.min_audio_length * self.sample_rate):
            self._process_final_buffer()
        else:
            # Use current transcription as final
            with self.transcription_lock:
                self.final_transcription = self.current_transcription
        
        # Call final callback
        if self.final_transcription and self.on_final_speech:
            self.on_final_speech(self.final_transcription)
        
        self.recording_start_time = None
    
    def _process_final_buffer(self):
        """Final processing of complete audio buffer (FFmpeg-free)."""
        try:
            with self.buffer_lock:
                if len(self.audio_buffer) == 0:
                    return
                
                # Get ALL available audio data for final processing (don't truncate)
                audio_array = np.array(list(self.audio_buffer))
            
            try:
                # Process directly with numpy array for final result
                audio_normalized = np.asarray(audio_array, dtype=np.float32).reshape(-1)
                
                # For final processing, use more audio but still limit to reasonable length
                max_length = self.sample_rate * 30  # 30 seconds max
                if len(audio_normalized) > max_length:
                    # Take the last 30 seconds, but include extra context at the beginning
                    extra_context_samples = int(2.0 * self.sample_rate)  # 2 seconds extra context
                    start_point = max(0, len(audio_normalized) - max_length - extra_context_samples)
                    audio_normalized = audio_normalized[start_point:]
                    
                    # Trim to max_length if still too long
                    if len(audio_normalized) > max_length:
                        audio_normalized = audio_normalized[-max_length:]
                
                # Final processing with slightly better quality settings
                result = self.whisper_model.transcribe(
                    audio_normalized,  # Pass numpy array directly
                    fp16=False,
                    language="en",
                    task="transcribe",
                    beam_size=2,  # Slightly better quality for final result
                    temperature=0.0,
                    condition_on_previous_text=False
                )
                
                final_text = result["text"].strip()
                
                if self._is_hallucination(final_text):
                    print(f"🚫 Detected whisper hallucination, skipping: '{final_text[:100]}...'")
                    return
                
                with self.transcription_lock:
                    self.final_transcription = final_text if final_text else self.current_transcription
                
            except Exception as e:
                print(f"Direct final processing failed: {e}")
                # Fallback to current transcription
                with self.transcription_lock:
                    self.final_transcription = self.current_transcription
                    
        except Exception as e:
            print(f"Final processing error: {e}")
            with self.transcription_lock:
                self.final_transcription = self.current_transcription
    
    def get_current_transcription(self) -> str:
        """Get the current streaming transcription."""
        with self.transcription_lock:
            return self.current_transcription
    
    def get_final_transcription(self) -> str:
        """Get the final transcription."""
        with self.transcription_lock:
            return self.final_transcription
    
    def is_currently_recording(self) -> bool:
        """Check if currently recording."""
        return self.is_recording
    
    def get_sound_pressure_level(self) -> float:
        """
        Get current sound pressure level in dB from the audio buffer.
        Returns None if no audio data is available.
        """
        try:
            with self.buffer_lock:
                if len(self.audio_buffer) == 0:
                    return None
                
                # Get recent audio data (last 0.1 seconds)
                recent_samples = int(0.1 * self.sample_rate)
                audio_array = np.array(list(self.audio_buffer))
                
                if len(audio_array) < recent_samples:
                    recent_audio = audio_array
                else:
                    recent_audio = audio_array[-recent_samples:]
                
                # Calculate RMS
                rms = np.sqrt(np.mean(recent_audio ** 2))
                
                # Convert to dB (avoid log of zero)
                if rms < 1e-10:
                    return -np.inf
                
                # SPL = 20 * log10(RMS / reference)
                spl_db = 20 * np.log10(rms / 1.0)  # Using 1.0 as reference for normalized audio
                
                return spl_db
                
        except Exception as e:
            print(f"SPL calculation error: {e}")
            return None

    async def listen_with_voice_activity_detection(
        self,
        voice_threshold: float = -40,  # dB threshold for voice detection
        silence_timeout: float = 3.0,  # seconds of silence before stopping
        check_interval: float = 0.1,   # check SPL every 100ms
        on_pause: Optional[Callable[[str], Awaitable[None]]] = None,  # Called when voice pauses (with current transcription)
        on_end: Optional[Callable[[str], Awaitable[None]]] = None     # Called when recording ends (with final transcription)
    ):
        """
        FIXED: Async voice activity detection method that respects pause state.
        
        This method listens continuously and automatically starts/stops recording based on voice activity.
        It will run until stopped by calling stop_voice_activity_detection() or when the handler is stopped.
        """
        if not self.running:
            raise RuntimeError("Speech handler must be started first. Call handler.start()")
        
        if self.vad_running:
            raise RuntimeError("Voice activity detection is already running")
        
        self.vad_running = True
        print("🎧 Starting async voice activity detection...")
        print(f"🔊 Voice threshold: {voice_threshold} dB | Silence timeout: {silence_timeout}s")
        
        # State tracking
        is_recording = False
        silence_start_time = None
        
        try:
            while self.vad_running and self.running:
                await self._wait_if_paused(check_interval)
                
                # FIXED: Skip SPL check if paused (since we're not buffering anyway)
                if self._pause_evt.is_set():
                    await asyncio.sleep(check_interval)
                    continue
                    
                # Get current SPL
                spl = self.get_sound_pressure_level()
                
                if spl is not None:
                    voice_detected = spl > voice_threshold
                    
                    if voice_detected:
                        # Voice detected - start recording if not already
                        if not is_recording:
                            print(f"🎤 Voice detected! SPL: {spl:.1f} dB - Starting recording...")
                            self.start_recording()
                            is_recording = True
                        # Reset silence timer
                        silence_start_time = None
                        # If there was a pending external finalize request, clear it
                        if self._skip_silence_evt.is_set():
                            self._skip_silence_evt.clear()
                    
                    else:
                        # No voice detected
                        if is_recording:
                            # If we get an external "finalize now" signal, skip waiting
                            if self._skip_silence_evt.is_set():
                                print("  External finalize requested — skipping silence wait.")
                                self._skip_silence_evt.clear()
                                print(" Stopping recording immediately due to external finalize request")
                                self.stop_recording()
                                is_recording = False
                                silence_start_time = None
                                await asyncio.sleep(0.5)  # allow final processing to complete
                                final_text = self.get_final_transcription()
                                if final_text and on_end:
                                    await on_end(final_text)
                                print("🎧 Listening for voice again...")
                                await asyncio.sleep(check_interval)
                                continue

                            if silence_start_time is None:
                                # Voice just paused - start silence timer and call onPause
                                silence_start_time = time.time()
                                print(f"🔇 Voice paused, waiting {silence_timeout}s... SPL: {spl:.1f} dB")
                                if on_pause:
                                    current_text = self.get_current_transcription()
                                    await asyncio.sleep(0.5)
                                    await on_pause(current_text)
                            else:
                                # Allow external finalize to short-circuit during countdown
                                if self._skip_silence_evt.is_set():
                                    print(" External finalize requested during silence countdown.")
                                    self._skip_silence_evt.clear()
                                    print(" Stopping recording immediately due to external finalize request")
                                    self.stop_recording()
                                    is_recording = False
                                    silence_start_time = None
                                    await asyncio.sleep(0.5)
                                    final_text = self.get_final_transcription()
                                    if final_text and on_end:
                                        await on_end(final_text)
                                    print("🎧 Listening for voice again...")
                                    await asyncio.sleep(check_interval)
                                    continue

                                # Normal countdown path
                                silence_duration = time.time() - silence_start_time
                                remaining = silence_timeout - silence_duration
                                if remaining <= 0:
                                    print(f"ℹ️  Stopping recording after {silence_timeout}s of silence")
                                    self.stop_recording()
                                    is_recording = False
                                    silence_start_time = None
                                    await asyncio.sleep(0.5)
                                    final_text = self.get_final_transcription()
                                    if not final_text:
                                        print("📝 No speech detected in recording")
                                    elif on_end:
                                        await on_end(final_text)
                                    print("🎧 Listening for voice again...")
                
                # Sleep asynchronously
                await asyncio.sleep(check_interval)
                
        except Exception as e:
            print(f"Error in voice activity detection: {e}")
        finally:
            # Clean up if recording when stopped
            if is_recording:
                print("🛑 Cleaning up recording on exit...")
                self.stop_recording()
                await asyncio.sleep(0.5)
                
                final_text = self.get_final_transcription()
                if on_end and final_text:
                    try:
                        await on_end(final_text)
                    except Exception as e:
                        print(f"Error calling onEnd callback: {e}")
            
            self.vad_running = False
            print("✅ Voice activity detection stopped")

    # helper at class scope (async)
    async def _wait_if_paused(self, check_interval: float):
        while self._pause_evt.is_set() and self.vad_running and self.running:
            await asyncio.sleep(check_interval)

    def stop_voice_activity_detection(self):
        """Stop the async voice activity detection loop."""
        self.vad_running = False
    
    def is_voice_activity_detection_running(self) -> bool:
        """Check if voice activity detection is currently running."""
        return self.vad_running
    
    def pause_voice_detection(self, should_pause: bool):
        """FIXED: Pause/unpause voice detection and buffer filling."""
        self.voice_detection_paused = should_pause
        if should_pause:
            print("🔇 Pausing voice detection and buffer filling...")
            self._pause_evt.set()
        else:
            print("🎤 Resuming voice detection and buffer filling...")
            self._pause_evt.clear()
            
            # CLEAR BUFFER ON UNPAUSE to ensure no robot speech contamination
            with self.buffer_lock:
                self.audio_buffer.clear()
                print("🗑️ Cleared audio buffer on unpause for clean start")

    def voice_detection_is_paused(self):
        return self._pause_evt.is_set()

    def get_immediate_final_transcription(self) -> None:
        """
        Trigger immediate finalization (skip silence wait).
        The final text will arrive through the normal on_final_speech/on_end callback.
        """
        if not self.is_recording:
            return

        # Set the event so the VAD loop stops recording and finalizes now
        self._skip_silence_evt.set()
        # Optional: small sleep to give the VAD loop a chance to react immediately
        # time.sleep(0.05)
        return  # no text returned


# Example usage for students
if __name__ == "__main__":
    import asyncio
    
    async def example_usage():
        """Example showing how students can use the async voice activity detection."""
        
        # Define callback functions
        async def handle_voice_pause(transcription: str):
            print(f"🟡 PAUSE: '{transcription}'")
        
        async def handle_voice_end(transcription: str):
            print(f"🟢 END: '{transcription}'")
        
        # Create and start the speech handler
        speech_handler = SpeechDetector(model_size="tiny")
        
        if not speech_handler.start():
            print("❌ Failed to start speech handler")
            return
        
        try:
            print("🚀 Starting example with async voice activity detection...")
            print("💡 Speak to test the callbacks. Press Ctrl+C to exit.")
            
            # Start voice activity detection with callbacks
            await speech_handler.listen_with_voice_activity_detection(
                voice_threshold=-40,    # Adjust based on your microphone sensitivity
                silence_timeout=3.0,    # 3 seconds of silence to end recording
                on_pause=handle_voice_pause,  # Called when voice pauses
                on_end=handle_voice_end       # Called when recording ends
            )
            
        except KeyboardInterrupt:
            print("\n🛑 Stopping example...")
        except Exception as e:
            print(f"❌ Error: {e}")
        finally:
            speech_handler.stop()
    
    # Run the example
    asyncio.run(example_usage())