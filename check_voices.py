import pyttsx3
import time

# 1. Get the list of voice IDs first using a temporary engine instance
temp_engine = pyttsx3.init()
all_voices = temp_engine.getProperty('voices')
del temp_engine # Clean it up immediately

# Target common US male names on Mac
target_voices = ["ralph",] #"fred","alex", "evan", "nathan"] # daniel, junior, ralph

# Filter our list to just the ones we want to test
voices_to_test = [v for v in all_voices if any(target in v.name.lower() for target in target_voices)]

print(f"Found {len(voices_to_test)} matching voices to test.")

# 2. Loop and isolate each voice in its own clean lifecycle
for voice in voices_to_test:
    print(f"Testing Voice: {voice.name}...")
    
    # Initialize a completely fresh engine instance for this specific voice
    engine = pyttsx3.init()
    engine.setProperty('voice', voice.id)
    
    # Speak
    engine.say(f"Testing the {voice.name} voice profile for PhyzAI audio.")
    engine.runAndWait()
    
    # Explicitly stop and delete the engine instance to free up macOS resources
    engine.stop()
    del engine
    
    # Give the macOS audio subsystem a brief moment to breathe
    time.sleep(0.5)

print("Testing complete!")