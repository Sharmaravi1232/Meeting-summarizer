# app.py
import os
import requests
import tempfile
from fastapi import FastAPI, HTTPException
from fastapi.responses import HTMLResponse
from gradio_client import Client
import uvicorn

app = FastAPI(title="Meeting Summarizer API")

# Replace this with your actual deployed HF space/model if needed
HF_MODEL_SPACE = "Ravishankarsharma/voice2text-summarizer"

# Public demo audio (replace if you want your own)
DEM0_AUDIO_URL = "https://huggingface.co/datasets/Narsil/asr_dummy/resolve/main/0001.flac"

# Initialize HF client
try:
    client = Client(HF_MODEL_SPACE)
except Exception as e:
    print("⚠️ Client initialization failed:", e)
    client = None


@app.get("/", response_class=HTMLResponse)
async def home():
    return """
    <html><body>
    <h2>Meeting Summarizer API</h2>
    <p>➡️ Call <a href='/summarize'>/summarize</a> to get meeting summary</p>
    <p>➡️ Swagger docs: <a href='/docs'>/docs</a></p>
    </body></html>
    """


@app.get("/summarize")
async def summarize_meeting():
    if not client:
        raise HTTPException(status_code=500, detail="❌ Hugging Face client not initialized")

    try:
        meeting_url = DEM0_AUDIO_URL

        # Download audio
        response = requests.get(meeting_url, stream=True, timeout=30)
        if response.status_code != 200:
            raise HTTPException(
                status_code=400,
                detail=f"Failed to fetch meeting audio (status {response.status_code})"
            )

        suffix = ".flac" if meeting_url.endswith('.flac') else os.path.splitext(meeting_url)[1] or ".wav"

        with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
            for chunk in response.iter_content(chunk_size=8192):
                if chunk:
                    tmp.write(chunk)
            tmp_path = tmp.name

        # Call your HF model. The predict input depends on how your space expects input.
        # If your space expects a file, use handle_file(tmp_path) instead. Here we try both common ways.
        try:
            # First try: send file path (many gradio-based spaces accept this)
            result = client.predict(tmp_path, api_name="/predict")
        except Exception:
            # Fallback: send the raw bytes
            with open(tmp_path, "rb") as fd:
                data = fd.read()
                result = client.predict(data, api_name="/predict")

        # Clean up
        try:
            os.remove(tmp_path)
        except Exception:
            pass

        return result

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"❌ Error summarizing meeting: {e}")


if __name__ == "__main__":
    uvicorn.run("app:app", host="0.0.0.0", port=8000, reload=True)
