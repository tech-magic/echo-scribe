from concurrent.futures import ThreadPoolExecutor
from threading import Event

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
            line = self.writer.write(start, end, speaker, segment.text)
            self.transcript_lines.append(line)

    def run(self):
        with self.producer.record_stream():
            try:
                while not self.stop_event.is_set():
                    chunk, chunk_start_time = self.consumer.get_next_chunk()
                    self.executor.submit(self.process_chunk, chunk, chunk_start_time)
            finally:
                self.writer.close()
                self.executor.shutdown(wait=True)