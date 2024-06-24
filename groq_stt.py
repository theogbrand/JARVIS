import os
from groq import Groq
from dotenv import load_dotenv

load_dotenv()


def groq_transcribe(relative_file_path: str):

    client = Groq(api_key=os.environ["GROQ_API_KEY"])
    # filename = os.path.dirname(__file__) + "/audio/response.wav"
    filename = os.path.dirname(__file__) + "/" + relative_file_path

    with open(filename, "rb") as file:
        transcription = client.audio.transcriptions.create(
            file=(filename, file.read()),
            model="whisper-large-v3",
            #   prompt="Specify context or spelling",  # Optional
            #   response_format="json",  # Optional
            #   language="en",  # Optional
            #   temperature=0.0  # Optional
        )
        print("Groq Whisper Large returned: ", transcription.text)
        return transcription.text
