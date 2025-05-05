import pyttsx3

def speak(text, rate = 210, voice_id = 0):
    engine = pyttsx3.init()
    engine.setProperty('rate', rate)
    voices = engine.getProperty('voices')
    engine.setProperty('voice', voices[voice_id].id)
    engine.say(text)
    engine.runAndWait()