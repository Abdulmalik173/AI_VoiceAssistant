import io
import os
import time
import sounddevice as sd
import numpy as np
import assemblyai as aai
from elevenlabs import generate, stream
from openai import OpenAI
import ollama
import sounddevice as sd
from scipy.io.wavfile import write
import wavio as wv
from config import Config

os.makedirs(".secrets", exist_ok=True)
config_path = ".secrets/config.ini"
config = Config(config_path)
config.load()
layout = ["Keys"]
if not config.data:
    config.set_layout(layout)
    config.data= {
        "Keys":{
            "assemblyAiKey":"Key",
            "elevenLabsApiKey": "Key",
            "OpenAI_APIkey":"Key"
        }
    }

    config.create_comment("This is optional if you have ollama installed","Keys","assemblyAiKey")
    config.create_comment("OpenAI_APIkey: \"Key\"","Keys","assemblyAiKey")
    config.save()
    print("Config file Created, Please add your Keys There!")
    input()
    exit()

OpenAI_APIkey= config.data["Keys"].get("OpenAI_APIkey") # OpenAI optional
elevenLabsApiKey= config.data["Keys"].get("elevenLabsApiKey")
assemblyAiKey= config.data["Keys"].get("assemblyAiKey")
if OpenAI_APIkey == None:
    cancel= True
    
if (elevenLabsApiKey == "Key") or (assemblyAiKey== "Key")\
    (elevenLabsApiKey == None) or (assemblyAiKey== None):
    print("Please add the Keys in the file in .secret/config.ini")
    


def detect_speech(data, threshold=0.01):
    return np.max(np.abs(data)) > threshold # Checks if Stopped Talking Like

def record_until_silence(freq, silence_duration=1.0):
    print("Recording...")
    
    recording = []
    silence_counter = 0
    sample_rate = freq
    chunk_size = 1024

    with sd.InputStream(samplerate=sample_rate, channels=2, dtype='float32') as stream:
        while True:
            chunk, overflowed = stream.read(chunk_size)
            if overflowed:
                print("Audio buffer overflow")
                continue

            recording.append(chunk)

            # Check for speech
            if detect_speech(chunk):
                silence_counter = 0
            else:
                silence_counter += (chunk_size / sample_rate)

            if silence_counter > silence_duration:
                print("Silence detected, stopping recording.")
                break

    # Convert list of chunks into a single numpy array
    recording = np.concatenate(recording, axis=0)
    return recording

class AI_Assistant:
    def __init__(self):
        aai.settings.api_key = assemblyAiKey
        if not cancel:
            self.openai_client = OpenAI(api_key =OpenAI_APIkey) # OpenAI optional
        self.elevenlabs_api_key = elevenLabsApiKey
        self.freq = 44100
        self.silence_duration = 3.0  # Silence duration in seconds to stop recording
        self.transcriber = None

        # Prompt
        self.full_transcript = [
            {"role":"system", "content":"Your The user."},
        ]


# Real-Time Transcription with AssemblyAI 
    def start_transcription(self):    
        recording = record_until_silence(self.freq, self.silence_duration)
        fileRecording = io.BytesIO()
        write(fileRecording, self.freq, (recording * 32767).astype(np.int16))
        self.transcriber = aai.Transcriber()
        transcript= self.transcriber.transcribe(fileRecording)
        print(f"Transcript: {transcript.text}")
        self.on_data(transcript)

    def on_data(self, transcript: aai.transcriber.Transcript):
        if not transcript.text:
            return
        if transcript.error:
            print("An error occured:", transcript.error)
            return

        self.generate_ai_response(transcript)

    def on_close(self):
        #print("Closing Session")
        return


# Pass real-time transcript to OpenAI 


    def generate_ai_response(self, transcript):

        self.full_transcript.append({"role":"user", "content": transcript.text})
        print(f"\nPatient: {transcript.text}", end="\r\n")

        if cancel:
            response = ollama.chat(model='llama3', messages=self.full_transcript)
        else:
            response = self.openai_client.chat.completions.create(
                model = "gpt-3.5-turbo",
                messages = self.full_transcript
            )

        ai_response = response['message']['content']
        self.full_transcript.append({'role': 'assistant', 'content': ai_response})

        self.generate_audio(ai_response)

        self.start_transcription()
        print(f"\nReal-time transcription: ", end="\r\n")


# Generate audio with ElevenLabs
        
    def generate_audio(self, text):

        self.full_transcript.append({"role":"assistant", "content": text})
        print(f"\nAI Receptionist: {text}")

        audio_stream = generate(
            api_key = self.elevenlabs_api_key,
            text = text,
            voice = "Daniel",
            stream = True
        )

        stream(audio_stream)

greeting = "Hi, I'm Eonix AI"
ai_assistant = AI_Assistant()
ai_assistant.generate_audio(greeting)
ai_assistant.start_transcription()

        



# Author    : Abdulmalik Alqahtani
# Made By   : Abdulmalik Alqahtani
# Help By   : Yazeed Aloufi