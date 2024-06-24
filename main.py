"""Main file for the Jarvis project"""

import os
from os import PathLike
from time import time
import asyncio
from typing import Union

from dotenv import load_dotenv
from openai import AzureOpenAI
from deepgram import Deepgram
import pygame
from pygame import mixer
from tts import get_audio_response

from record import speech_to_text
from groq_stt import groq_transcribe

# Load API keys
load_dotenv()
DEEPGRAM_API_KEY = os.getenv("DEEPGRAM_API_KEY")

# Initialize APIs
deepgram = Deepgram(DEEPGRAM_API_KEY)
# mixer is a pygame module for playing audio
mixer.init()

# Change the context if you want to change Jarvis' personality
context = "You are Jarvis, Brandon's human assistant. You are witty and full of personality. Your answers should be limited to 1-2 short sentences."
conversation = {"Conversation": []}
RECORDING_PATH = "audio/recording.wav"


def request_gpt(prompt: str) -> str:
    """
    Send a prompt to the GPT-3 API and return the response.

    Args:
        - state: The current state of the app.
        - prompt: The prompt to send to the API.

    Returns:
        The response from the API.
    """
    client = AzureOpenAI(
        azure_endpoint=os.environ["AZURE_OPENAI_BASE_URL"],
        api_version=os.environ["AZURE_OPENAI_API_VERSION"],
        api_key=os.environ["AZURE_OPENAI_API_KEY"],
    )

    response = client.chat.completions.create(
        model=os.environ["AZURE_OPENAI_MODEL_ID"],
        temperature=0,
        messages=[{"role": "user", "content": prompt}],
    )
    return response.choices[0].message.content


async def transcribe(
    file_name: Union[Union[str, bytes, PathLike[str], PathLike[bytes]], int]
):
    """
    Transcribe audio using Deepgram API.

    Args:
        - file_name: The name of the file to transcribe.

    Returns:
        The response from the API.
    """
    with open(file_name, "rb") as audio:
        source = {"buffer": audio, "mimetype": "audio/wav"}
        response = await deepgram.transcription.prerecorded(source)
        return response["results"]["channels"][0]["alternatives"][0]["words"]


def log(log: str):
    """
    Print and write to status.txt
    """
    print(log)
    with open("status.txt", "w") as f:
        f.write(log)


if __name__ == "__main__":
    while True:
        # Record audio
        log("Listening...")
        speech_to_text()
        log("Done listening")

        # Transcribe audio
        current_time = time()
        string_words = groq_transcribe(RECORDING_PATH)
        # loop = asyncio.new_event_loop()
        # asyncio.set_event_loop(loop)
        # words = loop.run_until_complete(transcribe(RECORDING_PATH))
        # string_words = " ".join(
        #     word_dict.get("word") for word_dict in words if "word" in word_dict
        # )
        with open("conv.txt", "a") as f:
            f.write(f"{string_words}\n")
        transcription_time = time() - current_time
        log(f"Finished transcribing in {transcription_time:.2f} seconds.")

        # Get response from GPT-3
        current_time = time()
        context += f"\Brandon: {string_words}\nJarvis: "
        response = request_gpt(context)
        context += response
        gpt_time = time() - current_time
        log(f"Finished generating response in {gpt_time:.2f} seconds.")

        # Convert response to audio
        current_time = time()
        get_audio_response(response)
        # audio = elevenlabs.generate(
        #     text=response, voice="Adam", model="eleven_monolingual_v1"
        # )
        # elevenlabs.save(audio, "audio/response.wav")
        audio_time = time() - current_time
        log(f"Finished generating audio in {audio_time:.2f} seconds.")

        # Play response
        log("Speaking...")
        sound = mixer.Sound("audio/response.wav")
        # Add response as a new line to conv.txt
        with open("conv.txt", "a") as f:
            f.write(f"{response}\n")
        sound.play()
        pygame.time.wait(int(sound.get_length() * 1000))
        print(f"\n --- USER: {string_words}\n --- JARVIS: {response}\n")
