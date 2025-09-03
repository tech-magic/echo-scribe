# pip install gradio faster-whisper sounddevice numpy torch speechbrain scikit-learn fastapi uvicorn

import os
import threading
import queue
import uvicorn
import gradio as gr
from datetime import datetime

from fastapi import FastAPI, Request
from fastapi.responses import FileResponse, HTMLResponse

from modules.audio.audio_producer import AudioProducer
from modules.audio.audio_consumer import AudioConsumer

from modules.ai.speech_to_text import SpeechToText
from modules.ai.speaker_diarizer import SpeakerDiarizer

from modules.recording.transcript_writer import TranscriptWriter
from modules.recording.session_manager import SessionManager

from modules.coordinator.transcription_pipeline import TranscriptionPipeline

# ===================
# Global Variables
# ===================

base_session_dir = "data"

audio_queue = queue.Queue()
producer = AudioProducer(audio_queue)
consumer = AudioConsumer(audio_queue)
stt = SpeechToText()
diarizer = SpeakerDiarizer()

writer = None # Track current transcript writer
pipeline = None # Track current transcript pipeline
current_session_id = None  # Track active session

# ======================
# Application Utilities
# ======================

def start_recording():
    global pipeline, current_session_id, writer

    current_session_id = datetime.now().strftime("%d-%m-%Y-%H-%M-%S")
    writer = TranscriptWriter(base_session_dir, current_session_id)

    pipeline = TranscriptionPipeline(producer, consumer, stt, diarizer, writer)

    thread = threading.Thread(target=pipeline.run, daemon=True)
    thread.start()

    return f"✅ Recording started. Session: {current_session_id}"


def stop_recording():
    global pipeline, current_session_id
    if pipeline:
        pipeline.stop_event.set()
        return "🛑 Recording stopped. Session: {current_session_id}"
    return "⚠️ No active recording."


def get_transcript():
    global pipeline
    if pipeline:
        return "\n".join(pipeline.transcript_lines)
    return ""

def list_sessions():
    global base_session_dir
    session_manager = SessionManager(base_session_dir)

    # Collect sessions with datetime objects for sorting
    session_list = []
    for current_session in session_manager.list_sessions():
        sid = current_session[0]
        txt_file = current_session[1]
        srt_file = current_session[2]

        # Convert sid to datetime
        dt = datetime.strptime(sid, "%d-%m-%Y-%H-%M-%S")
        session_list.append((dt, sid, txt_file, srt_file))

    # Sort descending by datetime
    session_list.sort(key=lambda x: x[0], reverse=True)

    # Build HTML table
    rows = []
    for dt, sid, txt_file, srt_file in session_list:
        txt_link = f'<a href="{txt_file}" download><button>Download Script</button></a>' if txt_file else ""
        srt_link = f'<a href="{srt_file}" download><button>Download Subtitles</button></a>' if srt_file else ""
        formatted = dt.strftime("%Y %b %d at %-I:%M%p").lower()  # formatted timestamp
        rows.append(f"<tr><td>{sid}</td><td>{formatted}</td><td>{txt_link}</td><td>{srt_link}</td></tr>")

    html_header = "<h3>📂 Past Sessions</h3>"
    html_table = "<table><tr><th>Session ID</th><th>Start Time</th><th>TXT File</th><th>SRT File</th></tr>" + "".join(rows) + "</table>"

    return "<center>" + html_header + html_table + "</center>"

# ====================
# Gradio UI Layout
# ====================

svg_favicon = """
<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 100 100">
  <text y=".9em" font-size="90">🎙️</text>
</svg>
"""

# Encode as data URI
favicon_data_uri = "data:image/svg+xml," + svg_favicon.replace("\n", "").replace(" ", "%20")

with gr.Blocks() as demo:
    
    gr.Markdown("## 🎤 Real-time Transcription with Speaker Diarization")

    with gr.Row():
        start_btn = gr.Button("▶️ Start Recording")
        stop_btn = gr.Button("⏹️ Stop Recording")
        placeholder_1 = gr.Markdown("")
        placeholder_1 = gr.Markdown("")
        placeholder_1 = gr.Markdown("")
        placeholder_1 = gr.Markdown("")

    with gr.Row():
        transcript_box = gr.Textbox(label="Transcript", lines=20, interactive=False)
        session_table = gr.HTML(label="")

    # Button actions
    start_btn.click(start_recording, outputs=transcript_box)
    stop_btn.click(stop_recording, outputs=transcript_box)

    # Use gr.Timer for periodic updates
    timer1 = gr.Timer(2.0)
    timer1.tick(get_transcript, None, transcript_box)

    timer2 = gr.Timer(5.0)
    timer2.tick(list_sessions, None, session_table)

    # 👇 This ensures initial loading when UI first loads
    demo.load(list_sessions, None, session_table)
    demo.load(get_transcript, None, transcript_box)

# ==============================
# FastAPI/Uvicorn Integration
# ==============================

# ---- FastAPI app ----
app = FastAPI(title="EchoScribe 🎙️")

# Add a REST endpoint for static files
@app.get("/data/{session_id}/{filename}")
async def serve_file(session_id: str, filename: str):
    global base_session_dir
    filepath = os.path.join(base_session_dir, session_id, filename)
    if not os.path.exists(filepath):
        return {"error": "file not found"}
    return FileResponse(filepath)

# Customizing favicon icon
@app.get("/", response_class=HTMLResponse)
async def root(request: Request):
    html = f"""
    <!DOCTYPE html>
    <html>
    <head>
        <title>EchoScribe 🎤</title>
        <link rel="icon" href="data:image/svg+xml,<svg xmlns=%22http://www.w3.org/2000/svg%22 viewBox=%220 0 100 100%22><text y=%22.9em%22 font-size=%2290%22>🎤</text></svg>">
    </head>
    <body>
        <iframe src="/gradio" style="width:100%;height:100%;border:none;"></iframe>
    </body>
    </html>
    """
    return HTMLResponse(html)

# Mount Gradio at root
app = gr.mount_gradio_app(app, demo, path="/gradio")

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=7860)
