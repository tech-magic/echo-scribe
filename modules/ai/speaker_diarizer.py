import numpy as np
import torch
from speechbrain.inference import SpeakerRecognition
from sklearn.metrics.pairwise import cosine_similarity

# ===============================================================
# Component: Speaker Diarizer
# ===============================================================
class SpeakerDiarizer:
    def __init__(self, similarity_threshold=0.65):
        self.spkrec = SpeakerRecognition.from_hparams(
            source="speechbrain/spkrec-ecapa-voxceleb"
        )
        self.speakers = {}
        self.next_speaker_id = 1
        self.similarity_threshold = similarity_threshold

    def _get_embedding(self, chunk: np.ndarray):
        waveform = torch.from_numpy(chunk).float().unsqueeze(0)
        waveform = waveform / (waveform.abs().max() + 1e-9)
        return (
            self.spkrec.encode_batch(waveform)
            .detach()
            .cpu()
            .numpy()
            .squeeze()
        )

    def get_label(self, chunk: np.ndarray, start_time: float) -> str:
        embedding = self._get_embedding(chunk)

        for spk_id, embs in self.speakers.items():
            sims = cosine_similarity([embedding], embs).flatten()
            if np.max(sims) > self.similarity_threshold:
                self.speakers[spk_id].append(embedding)
                return spk_id

        spk_id = f"Speaker_{self.next_speaker_id}"
        self.speakers[spk_id] = [embedding]
        self.next_speaker_id += 1
        return spk_id