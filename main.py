import os

from speech_to_text import record_audio, transcribe_with_groq
from RAG_Pipeline import rag_pipeline, delete_document_from_db, add_document_to_db
from text_to_speech import play_audio_response

from dotenv import load_dotenv

load_dotenv()

audio_file_path = "./input.mp3"
record_audio(file_path=audio_file_path)
transcription = transcribe_with_groq(os.environ.get("GROQ_API_KEY"), audio_file_path)
print(transcription)
file_path = r"data\Sagar_Resume.pdf"
response = rag_pipeline(file_path=file_path, query=transcription)
print(response)
play_audio_response(response)

