import pygame
from rich.console import Console
from rich.text import Text
console = Console()
console.print(Text("Loading Libraries...", justify="center", style="bold blue"))
import io
import os
import tempfile
import sounddevice as sd
import assemblyai as aai
from elevenlabs import generate, stream
from openai import OpenAI
import ollama
import sounddevice as sd
from scipy.io.wavfile import write
import numpy as np
import soundfile as sf
import librosa
import json
import translator as tr
from data.config import Config
import warnings
import whisper

warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=UserWarning)

console.print(Text("Loading Config...", justify="center", style="bold blue"))
os.makedirs(".secrets", exist_ok=True)
config_path = ".secrets/config.ini"
config = Config(config_path)
layout = ["Keys","general","Advanced"]
try:
    config.load()
except FileNotFoundError:
    config.data = None
if not config.data:
    config.set_layout(layout)
    config.data= {
        "Keys":{
            "elevenLabsApiKey": "Key"
        },
        "general":{
            "model": "base",
            "arabic": False,
            "voice": "Will"
        },
        "Advanced":{
            "silence_duration": 3.0,
            "threshold": 0.01
        }}

    config.create_comment("This is optional if you have ollama installed","Keys","elevenLabsApiKey")
    config.create_comment("OpenAI_APIkey: \"Key\"","Keys","elevenLabsApiKey")
    config.create_comment("This is optional if you Want to use whisper instead","Keys","elevenLabsApiKey")
    config.create_comment("assemblyAiKey: \"Key\"","Keys","elevenLabsApiKey")
    config.create_comment("This is required","Keys","elevenLabsApiKey")
    config.create_comment("models: [tiny, base, small, medium, large]","general","model")
    config.create_comment("arabic: [True, False]","general","arabic")
    config.create_comment("this is the voice of the AI","general","voice")
    config.create_comment("silence_duration is in seconds","Advanced","silence_duration")
    config.create_comment("threshold is the minimum value to detect silence","Advanced","threshold")
    
    config.save()
    console.print("Config file Created, Please add your Keys There!")
    input()
    exit()

OpenAI_APIkey       = config.data["Keys"].get("OpenAI_APIkey") # OpenAI optional
assemblyAiKey       = config.data["Keys"].get("assemblyAiKey") # AssemblyAI optional
elevenLabsApiKey    = config.data["Keys"].get("elevenLabsApiKey") # ElevenLabs required
base                = config.data["general"].get("model")
arabic              = bool(config.data["general"].get("arabic"))
voice               = config.data["general"].get("voice")
silence_duration    = float(config.data["Advanced"].get("silence_duration"))
threshold           = float(config.data["Advanced"].get("threshold"))
if OpenAI_APIkey == None:
    cancel= True
elif OpenAI_APIkey == "Key":
    cancel= True
if assemblyAiKey == None:
    use_whisper= True
else:
    use_whisper= False
if elevenLabsApiKey == None:
    console.print("Please add your assemblyAiKey in the config file")
    input()
    exit()
console.print(Text("Finished All Loadings", justify="center", style="bold blue"))
    
def load_audio_from_io(io_file):
    io_file.seek(0)
    audio, sample_rate = sf.read(io_file)

    if len(audio.shape) > 1:
        audio = np.mean(audio, axis=1)

    if sample_rate != 16000:
        
        audio = librosa.resample(audio, orig_sr=sample_rate, target_sr=16000)
    return audio

def detect_speech(data, threshold=0.01):
    # Ensure data is a NumPy array of floats
    data = np.asarray(data, dtype=np.float32)
    # Ensure threshold is a float
    threshold = float(threshold)
    return np.max(np.abs(data)) > threshold  # Checks if Stopped Talking

def record_until_silence(freq):
    console.print("Recording...")
    
    recording = []
    silence_counter = 0
    sample_rate = freq
    chunk_size = 1024

    with sd.InputStream(samplerate=sample_rate, channels=2, dtype='float32') as stream:
        while True:
            chunk, overflowed = stream.read(chunk_size)
            if overflowed:
                console.print("Audio buffer overflow")
                continue

            recording.append(chunk)

            # Check for speech
            if detect_speech(chunk):
                silence_counter = 0
            else:
                silence_counter += (chunk_size / sample_rate)

            if silence_counter > float(silence_duration):
                console.print("Silence detected, stopping recording.")
                break

    # Convert list of chunks into a single numpy array
    recording = np.concatenate(recording, axis=0)
    return recording

def recognize_speech_with_whisper(audio_path):
    model = whisper.load_model(f"{base}{'' if arabic == True else '.en'}")
    audio = whisper.load_audio(audio_path)
    audio = whisper.pad_or_trim(audio)

    # Make log-Mel spectrogram and move to the same device as the model
    mel = whisper.log_mel_spectrogram(audio).to(model.device)

    # Decode the audio with language set to English
    language = "en" if arabic == False else "ar"
    options = whisper.DecodingOptions(fp16=False, language=language) 
    result = whisper.decode(model, mel, options)

    return result

class tempList(list):
    def __init__(self, *args):
        super().__init__(*args)
    def append(self, item):
        super().append(item)
        with open("data/transcript.json", "w", encoding="utf-8") as f:
            f.write(json.dumps(self))

class AI_Assistant:
    def __init__(self):
        aai.settings.api_key = assemblyAiKey
        if not cancel:
            self.openai_client = OpenAI(api_key =OpenAI_APIkey) # OpenAI optional
        self.elevenlabs_api_key = elevenLabsApiKey
        self.use_whisper = use_whisper
        self.ollamaModel = "Eonix"
        self.freq = 44100
        self.transcriber = None

        # Prompt
        self.full_transcript = tempList()
        # {"role":"system", "content":""},
        # self.full_transcript.append()


# Real-Time Transcription with AssemblyAI 
    def start_transcription(self):
        recording = record_until_silence(self.freq)
        fileRecording = io.BytesIO()
        write(fileRecording, self.freq, (recording * 32767).astype(np.int16))
        fileRecording.seek(0)

        if self.use_whisper:
            with tempfile.NamedTemporaryFile(suffix='.wav', delete=False) as temp_file:
                temp_file.write(fileRecording.read())
                temp_file_path = temp_file.name

            try:
                transcript = recognize_speech_with_whisper(temp_file_path)
            finally:
                os.remove(temp_file_path)
            console.print(f"\nPatient: {transcript.text}", end="\r\n")
            self.generate_ai_response(transcript.text)
        else:
            self.transcriber = aai.Transcriber()
            transcript = self.transcriber.transcribe(fileRecording)
            if transcript.error:
                console.print("An error occurred:", transcript.error)
                console.print("Try again")
                self.start_transcription()
            console.print(f"\nPatient: {transcript.text}", end="\r\n")
            self.generate_ai_response(transcript.text)

    def on_close(self):
        #console.print("Closing Session")
        return


# Pass real-time transcript to OpenAI 


    def generate_ai_response(self, transcript: str):

        self.full_transcript.append({"role":"user", "content": transcript})
        if cancel:
            response = ollama.chat(model=self.ollamaModel, messages=self.full_transcript)
        else:
            response = self.openai_client.chat.completions.create(
                model = "gpt-3.5-turbo",
                messages = self.full_transcript
            )

        ai_response = response['message']['content']
        translated_text = ai_response if arabic == False else tr.translateToArabic(ai_response)
        self.full_transcript.append({'role': 'assistant', 'content': ai_response})

        console.print(f"\nAI Receptionist: {translated_text}")
        self.generate_audio(translated_text)
        self.start_transcription()


# Generate audio with ElevenLabs
        
    def generate_audio(self, text):

        # self.full_transcript.append({"role":"assistant", "content": text})
        print(arabic)
        audio_stream = generate(
            api_key = self.elevenlabs_api_key,
            text = text,
            voice = voice,
            stream=True,
            model="eleven_multilingual_v2" if arabic == True else "eleven_turbo_v2"
        )

        stream(audio_stream)

# Author        : Abdulmalik Alqahtani, Yazeed Aloufi
# Structure By  : Abdulmalik Alqahtani
# Developer By  : Yazeed Aloufi
# Help by       : Meshari Alnowaishi