import sounddevice as sd

# ===============================================================
# Component: Audio Producer
# ===============================================================
class AudioProducer:
    def __init__(self, queue_ref, sample_rate=16000):
        self.sample_rate = sample_rate
        self.queue = queue_ref

    def callback(self, indata, frames, time, status):
        if status:
            print(status)
        self.queue.put(indata.copy())

    def record_stream(self):
        return sd.InputStream(
            samplerate=self.sample_rate, channels=1, callback=self.callback
        )