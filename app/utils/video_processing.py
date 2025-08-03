# app/utils/video_processing.py

import os
import whisper
import tempfile
from moviepy.editor import VideoFileClip  # FIXED: correct import path

# Add FFmpeg to PATH explicitly
os.environ["PATH"] += os.pathsep + r"C:\ffmpeg\ffmpeg-7.1.1-essentials_build\bin"  # <- replace with your actual ffmpeg bin path


WHISPER_MODEL_DIR = "app/ml_models/whisper_models"
model = whisper.load_model("tiny", download_root=WHISPER_MODEL_DIR)



def extract_audio_from_video(video_path, output_audio_path=None):
    """Extracts audio from a video file and saves it as .wav"""
    video = VideoFileClip(video_path)

    # Ensure output_audio_path is a valid .wav file
    if not output_audio_path:
        output_audio_path = tempfile.NamedTemporaryFile(suffix=".wav", delete=False).name
    elif os.path.isdir(output_audio_path):
        output_audio_path = os.path.join(output_audio_path, "extracted_audio.wav")

    video.audio.write_audiofile(output_audio_path, codec="pcm_s16le", logger=None)
    print(f"Audio extracted to: {output_audio_path}")
    return output_audio_path


def transcribe_audio_to_text(audio_path):
    print(audio_path)
    result = model.transcribe(audio_path)
    print("text generated from audio.")
    return result["text"]

def video_to_text(video_path):
    """Main function: video → audio → text"""
    audio_path = extract_audio_from_video(video_path)
    transcript = transcribe_audio_to_text(audio_path)
    
    # Clean up temporary audio file
    if os.path.exists(audio_path):
        os.remove(audio_path)

    return transcript
