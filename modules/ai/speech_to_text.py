import numpy as np
from faster_whisper import WhisperModel

# ===============================================================
# Component: Speech-to-text
# ===============================================================
class SpeechToText:
    def __init__(self, model_size="small.en", device="cpu", compute_type="int8"):
        self.model = WhisperModel(model_size, device=device, compute_type=compute_type)

    def transcribe(self, audio_np: np.ndarray):
        return self.model.transcribe(audio_np, language="en")