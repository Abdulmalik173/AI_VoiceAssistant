import pygame
import Ai as ai

# Initialize pygame mixer
pygame.mixer.init()

# Load and play the sound
pygame.mixer.music.load("data/ar.mp3" if ai.arabic else "data/en.mp3")
pygame.mixer.music.play()
while pygame.mixer.music.get_busy():
    pygame.time.Clock().tick(10)
ai_assistant = ai.AI_Assistant()
ai_assistant.start_transcription()


# Authors       : Abdulmalik Alqahtani, Yazeed Aloufi
# Structure By  : Abdulmalik Alqahtani
# Developer By  : Yazeed Aloufi
# Help by       : Meshari Alnowaishi