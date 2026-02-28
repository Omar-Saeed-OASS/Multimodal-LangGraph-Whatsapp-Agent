import os
import re
import uvicorn
import requests
import base64
from groq import Groq
from fastapi import FastAPI, Request
from agent import agent
from vector_store import ingestPDF
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

WHAPI_API = os.getenv("WHAPI_TOKEN")
GROQ_API_KEY = os.getenv("GROQ_API_KEY")
groq_client = Groq(api_key=GROQ_API_KEY)

app = FastAPI()

@app.post("/webhook")
async def receive_message(request: Request):
    """Receive Message From whatsapp"""

    data = await request.json()

    if "messages" in data:
        for message in data["messages"]:
            chat_id = message.get("chat_id")

            if message.get("from_me"):
                continue # skip messages sent by the agent

            if message.get("text"):
                text_obj = message.get("text", {})
                user_msg = text_obj.get("body", "")
                config = {"configurable": {"thread_id": chat_id}}
                agent_response = agent.invoke({"messages": [("user", user_msg)]}, config)
                replay_text = clean_text(agent_response["messages"][-1].content)
                send_message(replay_text, chat_id)

            elif message.get("image"):
                image_obj = message.get("image", {})
                image_media_id = image_obj.get("id", "")

                if image_media_id:
                    downloaded_image = download_image(image_media_id)

                    if downloaded_image:
                        image_description = encode_image(downloaded_image)

                        agent_prompt = f"[The user sent an image. Here is the visual description: {image_description}]. Please reply to the user based on this image."

                        config = {"configurable": {"thread_id": chat_id}}
                        agent_response = agent.invoke({"messages": [("user", agent_prompt)]}, config)
                        reply_text = clean_text(agent_response["messages"][-1].content)

                        send_message(reply_text, chat_id)

                    else:
                        print("Failed to download image.")

            elif message.get("voice"):
                voice_obj = message.get("voice", {})
                voice_media_id = voice_obj.get("id", "") # get the media id

                if voice_media_id:
                    downloaded_voice = download_audio(voice_media_id)

                    if downloaded_voice:
                        user_transcript = generate_transcript(downloaded_voice)
                        config = {"configurable": {"thread_id": chat_id}}
                        agent_response = agent.invoke({"messages": [("user", user_transcript)]}, config)
                        replay_text = clean_text(agent_response["messages"][-1].content)
                        send_message(replay_text, chat_id)

                    else:
                        continue

            elif message.get("document"):
                doc_obj = message.get("document", {})
                doc_media_id = doc_obj.get("id", "")

                mime_type = doc_obj.get("mime_type", "") # check the file type

                if doc_media_id and "pdf" in mime_type:
                    pdf_bytes = download_pdf(doc_media_id)

                    if pdf_bytes: # save the file bytes
                        file_path = "temp_doc.pdf"
                        with open(file_path, "wb") as file:
                            file.write(pdf_bytes)
                        try:
                            ingestPDF(file_path)
                            send_message("PDF was saved to memory, You can ask me anything about it.", chat_id)

                        except Exception as e:
                            send_message("Error while processing the PDF document", chat_id)
                    else:
                        send_message("Failed to download the PDF from WhatsApp.", chat_id)

    return {"status": "success"}

def download_audio(media_id: str) -> bytes:
    """
    Download the voice note from the whapi
    Args:
        media_id: media id
    """

    url = f"https://gate.whapi.cloud/media/{media_id}"

    headers = {
        "accept": "application/json",
        "authorization": f"Bearer {WHAPI_API}"
    }

    try:
        response = requests.get(url, headers=headers)

        if response.status_code == 200:
            return response.content # returns .ogg file
        else:
            print(f"Error downloading the audio")
            return None

    except Exception as e:
        print(f"Network error while downloading the audio: {e}")
        return None


def download_image(media_id: str) -> bytes:
    """Download image from whapi"""

    url = f"https://gate.whapi.cloud/media/{media_id}"
    headers = {
        "accept": "application/json",
        "authorization": f"Bearer {WHAPI_API}"
    }

    try:
        response = requests.get(url, headers=headers)

        if response.status_code == 200:
            return response.content
        else:
            print("Error downloading the image")
            return None

    except Exception as e:
        print(f"Network error while downloading the image: {e}")
        return None


def encode_image(image_bytes: bytes) -> str:
    """Encode image and send to Groq Vision model"""

    # Encode image to base64 (model requirement)
    base64_image = base64.b64encode(image_bytes).decode('utf-8')

    try:
        # Format the messages (model template)
        image_interpetation = groq_client.chat.completions.create(
            model="meta-llama/llama-4-scout-17b-16e-instruct",
            temperature=1,
            max_completion_tokens=1024,
            messages=[
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "text",
                            "text": "What is in this image? Provide a detailed description."
                        },
                        {
                            "type": "image_url",
                            "image_url": {
                                "url": f"data:image/jpeg;base64,{base64_image}",
                            },
                        },
                    ],
                }
            ]
        )

        return image_interpetation.choices[0].message.content

    except Exception as e:
        return "I was unable to process this image due to API failure"


def generate_transcript(audio_bytes: bytes) -> str:
    """
    Transcript whatsapp voice note into text
    Args:
        audio_bytes: .ogg format voice note downloaded by the download_audio function
    """

    file_tuple = ("voice_audio.ogg", audio_bytes)

    try:
        transcript = groq_client.audio.transcriptions.create(
            file = file_tuple,
            model = "whisper-large-v3-turbo",
            temperature = 0,
            response_format = "text",
            language = "en"
        )

        return transcript

    except Exception as e:
        return "I was unable to transcript this audio"


def download_pdf(media_id):
    """Download pdf from whatsapp"""

    url = f"https://gate.whapi.cloud/media/{media_id}"

    headers = {
        "accept": "application/json",
        "authorization": f"Bearer {WHAPI_API}"
    }

    try:
        response = requests.get(url, headers=headers)

        if response.status_code == 200:
            return response.content
        else:
            print(f"Error downloading the pdf")
            return None

    except Exception as e:
        print(f"Network error while downloading the PDF: {e}")
        return None


def send_message(text: str, chat_id: str):
    """
    Send Message to whatsapp
    Args:
        text: the message to send
        chat_id: to the id of the recipient
    """

    message_url = "https://gate.whapi.cloud/messages/text"
    headers = {
        "accept": "application/json",
        "content-type": "application/json",
        "authorization": f"Bearer {WHAPI_API}"
    }
    payload = {
        "to": chat_id,
        "body": text,
        "typing_time": 0
    }

    response = requests.post(message_url, json=payload, headers=headers)
    return response.json()

def clean_text(text: str) -> str:
    """Removes standard markdown and converts it to WhatsApp-friendly text."""

    # Convert **bold** to WhatsApp's *bold*
    text = re.sub(r'\*\*(.*?)\*\*', r'*\1*', text)
    # Remove # headers completely, leaving just the text
    text = re.sub(r'^#+\s*(.*?)$', r'*\1*', text, flags=re.MULTILINE)
    # Convert markdown links [text](url) to just text (url)
    text = re.sub(r'\[([^\]]+)\]\(([^)]+)\)', r'\1 (\2)', text)

    return text.strip()

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)



