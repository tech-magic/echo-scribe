# pip install faster-whisper torch speechbrain scikit-learn ffmpeg-python numpy tqdm

import subprocess
import os
import numpy as np
import torch
from faster_whisper import WhisperModel
from speechbrain.pretrained import EncoderClassifier
from sklearn.cluster import AgglomerativeClustering
from tqdm import tqdm

# ===============================================================
# 1️⃣ Read audio from MP4 via ffmpeg
# ===============================================================
def read_audio_from_mp4(mp4_path, sample_rate=16000):
    mp4_path = os.path.expanduser(mp4_path)
    if not os.path.exists(mp4_path):
        raise FileNotFoundError(f"MP4 file not found: {mp4_path}")

    cmd = [
        "ffmpeg",
        "-i", mp4_path,
        "-f", "s16le",
        "-acodec", "pcm_s16le",
        "-ar", str(sample_rate),
        "-ac", "1",
        "-"
    ]
    process = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.DEVNULL, bufsize=4096)
    audio_bytes = process.stdout.read()
    process.wait()

    audio = np.frombuffer(audio_bytes, np.int16).astype(np.float32) / 32768.0
    return audio, sample_rate

# ===============================================================
# 2️⃣ Extract speaker embeddings with padding + tqdm
# ===============================================================
def extract_embeddings(audio, sample_rate, window_size=3.0, step_size=1.5):
    classifier = EncoderClassifier.from_hparams(source="speechbrain/spkrec-ecapa-voxceleb")
    embeddings, segments = [], []

    win_samples = int(window_size * sample_rate)
    step_samples = int(step_size * sample_rate)

    iterator = range(0, len(audio) - win_samples + 1, step_samples) if len(audio) >= win_samples else [0]

    for start in tqdm(iterator, desc="Extracting embeddings"):
        chunk = audio if len(audio) < win_samples else audio[start:start+win_samples]
        # pad if too short
        if len(chunk) < win_samples:
            chunk = np.pad(chunk, (0, win_samples - len(chunk)))

        chunk_tensor = torch.from_numpy(chunk).float().unsqueeze(0)  # [1, time]
        emb = classifier.encode_batch(chunk_tensor)
        embeddings.append(emb.squeeze().detach().cpu().numpy())

        segments.append((start / sample_rate, min(len(audio)/sample_rate, (start + win_samples)/sample_rate)))

    return np.vstack(embeddings), segments

# ===============================================================
# 3️⃣ Cluster speakers (safe for single embedding)
# ===============================================================
def cluster_speakers(embeddings, distance_threshold=1.0):
    if len(embeddings) < 2:
        return np.array([0])  # single speaker
    clustering = AgglomerativeClustering(n_clusters=None, distance_threshold=distance_threshold)
    return clustering.fit_predict(embeddings)

# ===============================================================
# 4️⃣ Transcribe audio with Faster-Whisper + tqdm
# ===============================================================
def transcribe(mp4_path, model_size="small.en", device="cpu", compute_type="int8"):
    model = WhisperModel(model_size, device=device, compute_type=compute_type)

    segments_generator, _ = model.transcribe(
        mp4_path,
        beam_size=5,
        language="en",
        word_timestamps=False,
        vad_filter=True
    )

    transcribed_segments = []
    print("📝 Transcribing...")
    for seg in tqdm(segments_generator, desc="Transcription"):
        transcribed_segments.append((seg.start, seg.end, seg.text))
    return transcribed_segments

# ===============================================================
# 5️⃣ Main
# ===============================================================
# Use below commands to download and convert a public youtube video into mp4
#
# 1. Install yt-dlp
# brew install --formula yt-dlp
#
# 2. To Download a public youtube video (such as https://www.youtube.com/watch?v=OWC1PVQHSm4)
#
# yt-dlp OWC1PVQHSm4
# 
# The above will save youtube video in .webm format (e.g. downloaded.webm)
#
# 3. To convert from .webm to .mp4
# 
# brew install ffmpeg
#
# ffmpeg -i downloaded.webm video_to_scribe.mp4
#

if __name__ == "__main__":

    mp4_file = "~/Downloads/video_to_scribe.mp4"  # 👈 Replace with your input MP4 absolute path
    mp4_file = os.path.expanduser(mp4_file)

    print("🎬 Reading audio...")
    audio, sr = read_audio_from_mp4(mp4_file)

    print("🗣 Extracting speaker embeddings...")
    embeddings, segs = extract_embeddings(audio, sr, window_size=3.0, step_size=1.5)

    print("👥 Clustering speakers...")
    labels = cluster_speakers(embeddings)

    print("✍️ Transcribing speech...")
    transcribed = transcribe(mp4_file)

    print("\n--- Scribed Transcript ---\n")
    for start, end, text in transcribed:
        # find closest speaker embedding window
        speaker_idx = np.argmin([abs(start - seg_start) for seg_start, _ in segs])
        print(f"[Speaker {labels[speaker_idx]}] {text}")
