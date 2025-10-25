import logging
import speech_recognition as sr
from pydub import AudioSegment 
from io import BytesIO

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(message)s')

def record_audio(file_path, timeout=20, phrase_time_limit=None):
    """
    Improved function to record audio with better quality settings.
    """
    recognizer = sr.Recognizer()
    
    # CRITICAL: Adjust for ambient noise and improve recognition
    recognizer.energy_threshold = 300  # Reduce background noise sensitivity
    recognizer.dynamic_energy_threshold = True
    recognizer.non_speaking_duration = 0.8
    recognizer.pause_threshold = 2   # Longer pause detection
    
    try:
        with sr.Microphone() as source:
            logging.info("Adjusting for ambient noise... Please wait.")
            recognizer.adjust_for_ambient_noise(source, duration=2)  # Increased calibration time
            logging.info("Start speaking now...")
            
            # Record with better settings
            audio_data = recognizer.listen(
                source, 
                timeout=timeout, 
                phrase_time_limit=phrase_time_limit
            )
            logging.info("Recording complete.")
            
            # Convert to MP3 with better quality
            wav_data = audio_data.get_wav_data()
            audio_segment = AudioSegment.from_wav(BytesIO(wav_data))
            
            # Increase volume if too quiet
            if audio_segment.dBFS < -25:  # If too quiet
                boost_amount = min(15, -20 - audio_segment.dBFS)
                audio_segment = audio_segment + boost_amount  # Increase volume by 10dB
                
            audio_segment = audio_segment.compress_dynamic_range(threshold=-20.0, ratio=4.0)
            
            audio_segment.export(file_path, format="mp3", bitrate="192k")  # Higher bitrate
            
            logging.info(f"Audio saved to {file_path}")
            return True

    except sr.WaitTimeoutError:
        logging.error("No speech detected within the timeout period.")
        return False
    except Exception as e:
        logging.error(f"An error occurred: {e}")
        return False

import os
from groq import Groq

GROQ_API_KEY=os.environ.get("GROQ_API_KEY")

def transcribe_with_groq(GROQ_API_KEY, audio_filepath):
    client = Groq(api_key = GROQ_API_KEY)

    
    filename = audio_filepath 

    with open(filename, "rb") as file:
        transcription = client.audio.transcriptions.create(
        file=file, 
        model="whisper-large-v3", 
        language="en",  
        temperature=0.0
        )
        
        return transcription.text