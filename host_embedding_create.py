'''import os
import subprocess
import numpy as np
import librosa
import torch
from transformers import Wav2Vec2ForCTC, Wav2Vec2Processor

# ---- SETTINGS ----
HOST_SOURCE = "hostsample.wav"   # <-- put your host-only file name here
OUT_WAV = "host_sample_16k.wav"
EMB_DIR = "models"
EMB_PATH = os.path.join(EMB_DIR, "host_embedding.npy")
LOCAL_MODEL = "models/wav2vec2-base-960h"
'''
'''
def ffmpeg_to_wav(input_path: str, output_path: str):
    """Convert any audio/video file to mono 16k wav using ffmpeg."""
    if os.path.exists(output_path):
        os.remove(output_path)

    cmd = [
        "ffmpeg", "-y",
        "-i", input_path,
        "-vn",
        "-acodec", "pcm_s16le",
        "-ar", "16000",
        "-ac", "1",
        output_path,
    ]
    print("▶ Running ffmpeg:", " ".join(cmd))
    subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, check=False)

    if not os.path.exists(output_path):
        raise RuntimeError("❌ ffmpeg failed – output wav not created")

'''
''' compute_wav2vec_embedding(audio, sr, processor, model) -> torch.Tensor:
    """
    Use the SAME Wav2Vec2 model as your ASR to compute a speaker embedding.
    We just take the mean of the last hidden state as a simple embedding.
    """
    inputs = processor(audio, sampling_rate=sr, return_tensors="pt")
    with torch.no_grad():
        outputs = model(
            inputs.input_values,
            output_hidden_states=True   # important!
        )
    hidden = outputs.hidden_states[-1]  # [1, T, D]
    emb = hidden.mean(dim=1).squeeze(0)  # [D]
    emb = emb / emb.norm(p=2)            # L2 normalize
    return emb


def main():
    if not os.path.exists(HOST_SOURCE):
        raise FileNotFoundError(
            f"❌ Could not find host source file: {HOST_SOURCE}\n"
            f"Place your host-only audio/video in this folder and update HOST_SOURCE if needed."
        )

    os.makedirs(EMB_DIR, exist_ok=True)

    print("🔄 Converting host sample to 16k wav...")
    #ffmpeg_to_wav(HOST_SOURCE, OUT_WAV)

    print("🎧 Loading audio...")
    audio, sr = librosa.load(HOST_SOURCE, sr=16000)

    print("📥 Loading Wav2Vec2 model (host embedding)...")
    processor = Wav2Vec2Processor.from_pretrained(LOCAL_MODEL, local_files_only=True)
    model = Wav2Vec2ForCTC.from_pretrained(LOCAL_MODEL, local_files_only=True)

    print("🧬 Computing host voice embedding...")
    emb = compute_wav2vec_embedding(audio, 16000, processor, model)

    np.save(EMB_PATH, emb.cpu().numpy())
    print(f"✅ Saved host embedding to: {EMB_PATH}")


if __name__ == "__main__":
    main()
    '''
import os
import numpy as np
import librosa
import torch
from transformers import Wav2Vec2ForCTC, Wav2Vec2Processor

# ---- SETTINGS ----
# Use your existing WAV file directly
HOST_WAV = "hostsample.wav"   # <-- put your WAV filename here

EMB_DIR = "models"
EMB_PATH = os.path.join(EMB_DIR, "host_embedding.npy")

# Folder where your wav2vec2 model is located
LOCAL_MODEL = "models/wav2vec2-base-960h"


def compute_wav2vec_embedding(audio, sr, processor, model) -> torch.Tensor:
    """Compute L2-normalized Wav2Vec2 speaker embedding."""
    inputs = processor(audio, sampling_rate=sr, return_tensors="pt")

    with torch.no_grad():
        outputs = model(
            inputs.input_values,
            output_hidden_states=True
        )

    hidden = outputs.hidden_states[-1]  # [1, T, D]
    emb = hidden.mean(dim=1).squeeze(0)  # → [D]
    emb = emb / emb.norm(p=2)            # normalize
    return emb


def main():
    # 1) Check WAV exists
    if not os.path.exists(HOST_WAV):
        raise FileNotFoundError(f"❌ Could not find WAV file: {HOST_WAV}")

    print("🎧 Loading WAV...")
    audio, sr = librosa.load(HOST_WAV, sr=16000)

    print("📥 Loading Wav2Vec2 model...")
    processor = Wav2Vec2Processor.from_pretrained(
        LOCAL_MODEL, local_files_only=True
    )
    model = Wav2Vec2ForCTC.from_pretrained(
        LOCAL_MODEL, local_files_only=True
    )

    print("🧬 Computing host embedding...")
    emb = compute_wav2vec_embedding(audio, 16000, processor, model)

    os.makedirs(EMB_DIR, exist_ok=True)
    np.save(EMB_PATH, emb.cpu().numpy())

    print(f"✅ Host embedding saved → {EMB_PATH}")


if __name__ == "__main__":
    main()


