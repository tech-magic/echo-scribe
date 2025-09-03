import numpy as np
import torch
from speechbrain.inference import SpeakerRecognition
from sklearn.cluster import AgglomerativeClustering

# ===============================================================
# Component: Speaker Diarizer
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