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

    return (
        gr.update(value=f"✅ Recording started for Session: {current_session_id}", visible=True),
        gr.update(interactive=False),  # disable Start button
        gr.update(interactive=True)   # enable Stop button
    )

def stop_recording():
    global pipeline, current_session_id
    status_message = "⚠️ No active recording."
    if pipeline:
        pipeline.stop_event.set()
        last_session_id = current_session_id
        current_session_id = None
        status_message = f"🛑 Recording stopped for Session: {last_session_id}"

    return (
        gr.update(value=status_message, visible=True),
        gr.update(interactive=True),   # enable Start button
        gr.update(interactive=False)  # disable Stop button
    )


def get_transcript():
    global pipeline
    if pipeline:
        return "\n".join(pipeline.get_transcript_log())
    return ""

def list_sessions():
    global base_session_dir, current_session_id
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

    # Return only the latest 5 sessions
    latest_5_sessions = session_list[:5]

    # Build HTML table
    rows = []
    for dt, sid, txt_file, srt_file in latest_5_sessions:

        start_time = dt.strftime("%Y %b %d at %-I:%M%p").lower()  # formatted timestamp

        status = "✅ Done"
        if sid == current_session_id:
            status = "🟢 Active"

        txt_link = f'<a href="{txt_file}" download><button>Download</button></a>' if txt_file else ""
        srt_link = f'<a href="{srt_file}" download><button>Download</button></a>' if srt_file else ""

        rows.append(f"<tr><td>{sid}</td><td>{start_time}</td><td>{status}</td><td>{txt_link}</td><td>{srt_link}</td></tr>")

    html_header = "<h3>📂 Latest 5 Sessions (Includes Current, if Active)</h3>"
    html_table = "<table><tr><th>Session ID</th><th>Start Time</th><th>Status</th><th>Script (.txt)</th><th>Subtitles (.srt)</th></tr>" + "".join(rows) + "</table>"

    return "<center>" + html_header + html_table + "</center>"

# ====================
# Gradio UI Layout
# ====================

# App Intro
intro_title = """
<center>
<h1>🎙️ EchoScribe </h1>
<h2>Real-Time, Speaker-Aware Meeting Transcripts — No Cloud Needed, On your Local Device, and FREE! 🕒🧑‍🤝‍🧑</h2>
</center>
"""

intro_markdown = """
> Meet 🎙️ **EchoScribe** — your **free AI-powered** meeting companion! 🚀
It captures conversations 🎤, transcribes them instantly in English 📜, separates each speaker 🧑‍🤝‍🧑, and saves your transcripts in **TXT** or **SRT**. All 100% free, open-source, and runs right on your own device. 💻✨

By feeding your speaker-labeled meeting transcripts to ChatGPT, you unlock powerful insights and productivity boosts:
- **Summarize meetings for everyone** — generate clear, concise summaries for laymen or management.
- **Automatically track action items** — identify tasks discussed and assign responsibilities without manual effort.
- **Spot issues and solutions** — quickly highlight problems raised and the solutions proposed.
- **Get AI-driven guidance** — receive actionable answers for unresolved questions or challenges discussed in the meeting.
"""

custom_gradio_css = """

.column-border {
    border: 2px solid #4CAF50;  /* Green border */
    padding: 10px;
    border-radius: 8px;
}

#transcript_box label span {
    color: #4CAF50 !important;   /* Green label */
    font-weight: bold !important;
    font-size: 18px !important;
}

"""

with gr.Blocks(css=custom_gradio_css) as demo:
    
    gr.Markdown(intro_title)
        
    with gr.Row():
        with gr.Column(scale=1, elem_classes="column-border"):
            with gr.Row():
                with gr.Column():
                    pass
                with gr.Column():
                    with gr.Row():
                        start_btn = gr.Button("▶️ Start Recording", interactive=True)
                        stop_btn = gr.Button("⏹️ Stop Recording", interactive=False)
                    with gr.Row():
                        recording_status_box = gr.Markdown("")

            with gr.Row():
                transcript_box = gr.Textbox(label="✍️📜 Transcript Log (past Hour) 🧑‍🤝‍🧑", elem_id="transcript_box", lines=25, interactive=False)

        with gr.Column(scale=1):
            with gr.Row():
                gr.Markdown(intro_markdown)

            with gr.Row():
                session_table = gr.HTML(label="")

    # Button actions
    start_btn.click(start_recording, outputs=[recording_status_box, start_btn, stop_btn])
    stop_btn.click(stop_recording, outputs=[recording_status_box, start_btn, stop_btn])

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
app = FastAPI()

# Add a REST endpoint for static files
@app.get("/gradio/data/{session_id}/{filename}")
async def serve_file(session_id: str, filename: str):
    global base_session_dir
    filepath = os.path.join(base_session_dir, session_id, filename)
    if not os.path.exists(filepath):
        return {"error": "file not found"}
    return FileResponse(filepath)

# Customizing overall Look and Feel
@app.get("/", response_class=HTMLResponse)
async def root(request: Request):
    html = f"""
    <!DOCTYPE html>
    <html>
    <head>
        <title>EchoScribe</title>
        <link rel="icon" href="data:image/svg+xml,<svg xmlns=%22http://www.w3.org/2000/svg%22 viewBox=%220 0 100 100%22><text y=%22.9em%22 font-size=%2290%22>🎙️</text></svg>">
        <style>
            html, body {{
                height: 100%;
                margin: 0;
            }}
            iframe {{
                width: 100%;
                height: 100%;
                border: none;
            }}
        </style>
    </head>
    <body>
        <iframe src="/gradio"></iframe>
    </body>
    </html>
    """
    return HTMLResponse(html)

# Mount Gradio at root
app = gr.mount_gradio_app(app, demo, path="/gradio")

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=7860)
