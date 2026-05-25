
import os
import sys
import logging
import traceback

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

print("Current PATH:", os.environ["PATH"])

try:
    import imageio_ffmpeg
    ffmpeg_exe = imageio_ffmpeg.get_ffmpeg_exe()
    print(f"imageio-ffmpeg exe: {ffmpeg_exe}")
    ffmpeg_dir = os.path.dirname(ffmpeg_exe)
    target_ffmpeg = os.path.join(ffmpeg_dir, "ffmpeg.exe")
    if not os.path.exists(target_ffmpeg):
        import shutil
        shutil.copy(ffmpeg_exe, target_ffmpeg)
        print(f"Copied {ffmpeg_exe} to {target_ffmpeg}")
    else:
        print(f"ffmpeg.exe already exists at {target_ffmpeg}")
    if ffmpeg_dir not in os.environ["PATH"]:
        os.environ["PATH"] += os.pathsep + ffmpeg_dir
    print(f"Added {ffmpeg_dir} to PATH")
except ImportError:
    print("imageio-ffmpeg not installed")
except Exception as e:
    print(f"Error setting up ffmpeg: {e}")

try:
    import whisper
    print("Whisper imported successfully")
    
    # Create a dummy audio file for testing (empty file might fail, but let's see)
    # Better to try loading the model first
    model = whisper.load_model("base")
    print("Whisper model loaded")
    
    # Try creating a dummy wav file
    import wave
    with wave.open("test.wav", "wb") as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)
        wf.setframerate(16000)
        wf.writeframes(b"\0" * 32000) # 1 second of silence
        
    print("Created test.wav")
    
    # Try loading audio
    audio = whisper.load_audio("test.wav")
    print(f"Audio loaded successfully, shape: {audio.shape}")
    
except Exception as e:
    print("Error during Whisper test:")
    traceback.print_exc()
