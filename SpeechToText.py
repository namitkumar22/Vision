import speech_recognition as sr

def RecognizeSpeech():
    
    r = sr.Recognizer()
    with sr.Microphone() as source:
        print("Say something!")
        audio = r.listen(source)
        return r.recognize_google(audio)

    
if __name__ == "__main__":
    print(RecognizeSpeech())