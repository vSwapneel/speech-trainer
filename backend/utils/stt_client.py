import requests
import os
from elevenlabs.client import ElevenLabs
from io import BytesIO

# FOR TESTING
from dotenv import load_dotenv
load_dotenv()
# Testing Complete

ELEVENLABS_API_KEY = os.getenv("ELEVENLABS_API_KEY")
ELEVENLABS_STT_URL = "https://api.elevenlabs.io/v1/speech-to-text"


elevenlabs = ElevenLabs(
  api_key=ELEVENLABS_API_KEY,
)

def get_transcript(audio_path):
    with open(audio_path, "rb") as f:
        audio_data = BytesIO(f.read())
    
    transcription = elevenlabs.speech_to_text.convert(
        file=audio_data,
        model_id="scribe_v1",     # Model to use
        tag_audio_events=True,    # Tag audio events like laughter, applause, etc.
        language_code="eng",      # Specify language; None for auto-detect
        diarize=True              # Annotate speakers
    )

    return transcription.text if hasattr(transcription, "text") else str(transcription)

if __name__ == "__main__":
    file = "../../datasets/sample_tests/F_0101_10y4m_1.wav"

    try:
        transcript = get_transcript(file)
        print("Transcript:\n", transcript)
    except Exception as e:
        print("Error getting transcript:", e)
    