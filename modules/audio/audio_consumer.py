import numpy as np

# ===============================================================
# Component: Audio Consumer
# ===============================================================
class AudioConsumer:
    def __init__(self, queue_ref, sample_rate=16000, chunk_duration=1.5, overlap_duration=0.2):
        self.queue = queue_ref
        self.sample_rate = sample_rate
        self.chunk_duration = chunk_duration
        self.overlap_duration = overlap_duration
        self.audio_buffer = np.empty((0, 1), dtype=np.float32)
        self.total_audio_time = 0.0

    def get_next_chunk(self):
        while self.audio_buffer.shape[0] < int(self.sample_rate * self.chunk_duration):
            self.audio_buffer = np.append(self.audio_buffer, self.queue.get(), axis=0)

        chunk_size = int(self.sample_rate * self.chunk_duration)
        overlap_size = int(self.sample_rate * self.overlap_duration)

        chunk = self.audio_buffer[:chunk_size]
        self.audio_buffer = self.audio_buffer[chunk_size - overlap_size:]

        chunk_start_time = self.total_audio_time
        self.total_audio_time += (chunk_size - overlap_size) / self.sample_rate

        return chunk, chunk_start_time