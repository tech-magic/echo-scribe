import numpy as np
from concurrent.futures import ThreadPoolExecutor
from threading import Event

from datetime import datetime, timedelta

# ===============================================================
# Coordinator: Transcription Pipeline
# ===============================================================
class TranscriptionPipeline:
    def __init__(self, producer, consumer, stt, diarizer, writer):
        self.producer = producer
        self.consumer = consumer
        self.stt = stt
        self.diarizer = diarizer
        self.writer = writer
        self.executor = ThreadPoolExecutor(max_workers=2)
        self.stop_event = Event()
        self.transcript_lines = []

    def process_chunk(self, chunk, chunk_start_time):
        audio_np = chunk.astype(np.float32).flatten()
        segments, _ = self.stt.transcribe(audio_np)

        for segment in segments:
            start = chunk_start_time + segment.start
            end = chunk_start_time + segment.end
            speaker = self.diarizer.get_label(audio_np, start)

            # Writer returns the console/text line
            line = self.writer.write(start, end, speaker, segment.text)

            # Store with timestamp
            self.transcript_lines.append({
                "time": datetime.utcnow(),
                "line": line
            })

            # Purge lines older than 1 hour
            cutoff = datetime.utcnow() - timedelta(hours=1)
            self.transcript_lines = [
                entry for entry in self.transcript_lines
                if entry["time"] >= cutoff
            ]

    def get_transcript_log(self):
        """Return an array of transcript lines (strings) only."""
        return [entry["line"] for entry in self.transcript_lines]

    def run(self):
        with self.producer.record_stream():
            try:
                while not self.stop_event.is_set():
                    chunk, chunk_start_time = self.consumer.get_next_chunk()
                    self.executor.submit(self.process_chunk, chunk, chunk_start_time)
            finally:
                self.writer.close()
                self.executor.shutdown(wait=True)