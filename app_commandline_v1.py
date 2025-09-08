# pip install faster-whisper sounddevice numpy torch speechbrain scikit-learn

import queue
import warnings
import numpy as np
import sounddevice as sd
import torch
from faster_whisper import WhisperModel
from speechbrain.inference import SpeakerRecognition
from sklearn.cluster import AgglomerativeClustering
from concurrent.futures import ThreadPoolExecutor

warnings.filterwarnings("ignore", message=".*torchaudio.list_audio_backends.*")


# ===============================================================
# Utility: Timestamp formatter
# ===============================================================
class TimestampFormatter:
    @staticmethod
    def format(seconds: float, srt: bool = False) -> str:
        h = int(seconds // 3600)
        m = int((seconds % 3600) // 60)
        s = int(seconds % 60)
        ms = int((seconds % 1) * 1000)
        if srt:
            return f"{h:02d}:{m:02d}:{s:02d},{ms:03d}"
        return f"[{h:02d}:{m:02d}:{s:02d}]"


# ===============================================================
# Component: Audio input handler
# ===============================================================
class AudioInput:
    def __init__(self, sample_rate=16000, chunk_duration=1.5, overlap_duration=0.2):
        self.sample_rate = sample_rate
        self.chunk_duration = chunk_duration
        self.overlap_duration = overlap_duration
        self.queue = queue.Queue()

    def callback(self, indata, frames, time, status):
        if status:
            print(status)
        self.queue.put(indata.copy())

    def record_stream(self):
        return sd.InputStream(
            samplerate=self.sample_rate, channels=1, callback=self.callback
        )


# ===============================================================
# Component: Speech-to-text (Whisper)
# ===============================================================
class SpeechToText:
    def __init__(self, model_size="small.en", device="cpu", compute_type="int8"):
        self.model = WhisperModel(model_size, device=device, compute_type=compute_type)

    def transcribe(self, audio_np: np.ndarray):
        return self.model.transcribe(audio_np, language="en")


# ===============================================================
# Component: Speaker diarization
# ===============================================================
class SpeakerDiarizer:
    def __init__(self, distance_threshold=1.0, min_embeddings=5):
        self.spkrec = SpeakerRecognition.from_hparams(
            source="speechbrain/spkrec-ecapa-voxceleb"
        )
        self.speaker_embeddings = []
        self.speaker_timestamps = []
        self.cluster_to_speaker = {}
        self.next_speaker_id = 1
        self.min_embeddings = min_embeddings
        self.distance_threshold = distance_threshold

    def get_label(self, chunk: np.ndarray, start_time: float) -> str:
        waveform = torch.from_numpy(chunk).float().unsqueeze(0)
        waveform = waveform / (waveform.abs().max() + 1e-9)

        embedding = self.spkrec.encode_batch(waveform).detach().cpu().numpy().squeeze()
        self.speaker_embeddings.append(embedding)
        self.speaker_timestamps.append(start_time)

        if len(self.speaker_embeddings) >= self.min_embeddings:
            X = np.vstack(self.speaker_embeddings)
            clustering = AgglomerativeClustering(
                n_clusters=None, distance_threshold=self.distance_threshold
            )
            labels = clustering.fit_predict(X)
            cluster_idx = labels[-1]

            if cluster_idx not in self.cluster_to_speaker:
                self.cluster_to_speaker[cluster_idx] = f"Speaker_{self.next_speaker_id}"
                self.next_speaker_id += 1

            return self.cluster_to_speaker[cluster_idx]
        return "Speaker_?"


# ===============================================================
# Component: Output writer
# ===============================================================
class TranscriptWriter:
    def __init__(self, txt_path="transcript.txt", srt_path="transcript.srt"):
        self.txt_file = open(txt_path, "w", encoding="utf-8")
        self.srt_file = open(srt_path, "w", encoding="utf-8")
        self.srt_index = 1

    def write(self, start, end, speaker, text):
        # Console output
        line = f"📝 {TimestampFormatter.format(start)} -> {TimestampFormatter.format(end)} {speaker}: {text}"
        print(line)

        # TXT
        self.txt_file.write(line + "\n")
        self.txt_file.flush()

        # SRT
        self.srt_file.write(f"{self.srt_index}\n")
        self.srt_file.write(
            f"{TimestampFormatter.format(start, srt=True)} --> {TimestampFormatter.format(end, srt=True)}\n"
        )
        self.srt_file.write(f"{speaker}: {text.strip()}\n\n")
        self.srt_file.flush()

        self.srt_index += 1

    def close(self):
        self.txt_file.close()
        self.srt_file.close()


# ===============================================================
# Coordinator: Transcription pipeline
# ===============================================================
class TranscriptionPipeline:
    def __init__(self, audio: AudioInput, stt: SpeechToText,
                 diarizer: SpeakerDiarizer, writer: TranscriptWriter):
        self.audio = audio
        self.stt = stt
        self.diarizer = diarizer
        self.writer = writer
        self.executor = ThreadPoolExecutor(max_workers=2)

    def process_chunk(self, chunk, chunk_start_time):
        audio_np = chunk.astype(np.float32).flatten()
        segments, _ = self.stt.transcribe(audio_np)

        for segment in segments:
            start = chunk_start_time + segment.start
            end = chunk_start_time + segment.end
            speaker = self.diarizer.get_label(audio_np, start)
            self.writer.write(start, end, speaker, segment.text)

    def run(self):
        print("🎤 Low-latency transcription with diarization... Press Ctrl+C to stop.")
        audio_buffer = np.empty((0, 1), dtype=np.float32)
        total_audio_time = 0.0

        with self.audio.record_stream():
            try:
                while True:
                    while audio_buffer.shape[0] < int(self.audio.sample_rate * self.audio.chunk_duration):
                        audio_buffer = np.append(audio_buffer, self.audio.queue.get(), axis=0)

                    chunk_size = int(self.audio.sample_rate * self.audio.chunk_duration)
                    overlap_size = int(self.audio.sample_rate * self.audio.overlap_duration)

                    chunk = audio_buffer[:chunk_size]
                    audio_buffer = audio_buffer[chunk_size - overlap_size:]

                    chunk_start_time = total_audio_time
                    total_audio_time += (chunk_size - overlap_size) / self.audio.sample_rate

                    self.executor.submit(self.process_chunk, chunk, chunk_start_time)

            except KeyboardInterrupt:
                print("\n🛑 Stopped.")
                self.writer.close()
                self.executor.shutdown(wait=True)


# ===============================================================
# Entry point
# ===============================================================
if __name__ == "__main__":
    audio = AudioInput()
    stt = SpeechToText()
    diarizer = SpeakerDiarizer()
    writer = TranscriptWriter()

    pipeline = TranscriptionPipeline(audio, stt, diarizer, writer)
    pipeline.run()
