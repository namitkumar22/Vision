from groq import Groq
import cv2
import base64
from PIL import Image
import io
import numpy as np
import os

def get_description(frame, query):
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    
    pil_image = Image.fromarray(rgb_frame)

    buffered = io.BytesIO()
    pil_image.save(buffered, format="PNG")
    img_str = base64.b64encode(buffered.getvalue()).decode()
    image_url = f"data:image/png;base64,{img_str}"

    api_key = os.getenv("GROQ_API_KEY")
    if api_key is None:
        raise ValueError("GROQ_API_KEY environment variable not set.")

    client = Groq(api_key=api_key)

    system_message = {
        "role": "system",
        "content": (
            "describe this image in a way a automatic system should speak as this image is seen by a POV of blind man, the system should tell whats ahead and what to do. Make sure the instructions are most accurate and helpfull to the blind person. Also try to convey to the person things fast in shorter responses. Only tell the necessary instructions to the blind person, Also you are the only hope for a blind person happy life. Make sure you also respect what user has asked and guide them on their queries what they have asked. Try to be concise, emergency responding and try to provide accurate detail so that the blind person can trust you. Make sure you are very concise and provide details in very short text. Please only reply as an personal assistance for blind person vision"
        )
    }

    user_message = {
        "role": "user",
        "content": [
            {"type": "text", "text": query},
            {"type": "image_url", "image_url": {"url": image_url}}
        ]
    }

    completion = client.chat.completions.create(
        model="meta-llama/llama-4-maverick-17b-128e-instruct",
        messages=[system_message, user_message],
        temperature=1,
        top_p=1,
        stream=True,
        stop=None,
    )

    response = ""
    for chunk in completion:
        if chunk.choices[0].delta.content:
            response += chunk.choices[0].delta.content
    
    return response