import os
from groq import Groq

from dotenv import load_dotenv

load_dotenv()

def play_audio_response(text):

  client = Groq(api_key=os.environ.get("GROQ_API_KEY"))

  speech_file_path = "output.wav" 
  model = "playai-tts"
  voice = "Fritz-PlayAI"
  response_format = "wav"

  response = client.audio.speech.create(
      model=model,
      voice=voice,
      input=text,
      response_format=response_format
  )

  response.write_to_file(speech_file_path)