'''import librosa
import soundfile as sf
import numpy as np
import torch
from transformers import Wav2Vec2ForCTC, Wav2Vec2Processor


CLASSIC_FILLERS = ["um", "uh", "uhh", "uhm", "erm", "hmm", "mmm"]

'''
'''def extract_wav(video_path, output="temp_audio.wav"):
    audio, sr = librosa.load(video_path, sr=None, mono=True)

    if sr != 16000:
        audio = librosa.resample(audio, sr, 16000)

    sf.write(output, audio, 16000)
    return output'''
'''import subprocess
import os


# Download Silero VAD model
silero_model, utils = torch.hub.load(repo_or_dir='snakers4/silero-vad',
                                     model='silero_vad',
                                     force_reload=False)
(get_speech_timestamps,
 save_audio,
 read_audio,
 VADIterator,
 collect_chunks) = utils


def filter_participant_speech(audio, sr, min_segment_duration=1.5):
    """
    Keep only participant's long speech segments based on Silero VAD.
    """
    # Convert audio float32 (-1..1) to int16 PCM
    int_audio = (audio * 32767).astype(np.int16)

    # Get timestamps of speech
    speech_timestamps = get_speech_timestamps(int_audio, silero_model, sampling_rate=sr)

    participant_chunks = []

    for ts in speech_timestamps:
        start = ts['start']
        end = ts['end']

        duration = (end - start) / sr

        # Only keep long segments (participant)
        if duration >= min_segment_duration:
            participant_chunks.append(int_audio[start:end])

    if len(participant_chunks) == 0:
        return audio  # fallback

    merged = np.concatenate(participant_chunks).astype("float32") / 32767.0
    return merged
def extract_wav(video_path, output="temp_audio.wav"):
    temp_raw = "temp_raw_audio.wav"

    # Remove old files if they exist
    for f in [temp_raw, output]:
        if os.path.exists(f):
            os.remove(f)

    # 1️⃣ Extract audio from MP4 using FFmpeg
    cmd_extract = [
        "ffmpeg", "-y",
        "-i", video_path,
        "-vn",
        "-acodec", "pcm_s16le",
        "-ar", "16000",
        "-ac", "1",
        temp_raw
    ]

    subprocess.run(cmd_extract, stdout=subprocess.PIPE, stderr=subprocess.PIPE)

    if not os.path.exists(temp_raw):
        raise Exception("FFmpeg failed during audio extraction")

    # 2️⃣ Convert the extracted audio into final WAV
    cmd_convert = [
        "ffmpeg", "-y",
        "-i", temp_raw,
        "-ac", "1",
        "-ar", "16000",
        output
    ]

    subprocess.run(cmd_convert, stdout=subprocess.PIPE, stderr=subprocess.PIPE)

    if not os.path.exists(output):
        raise Exception("FFmpeg failed during WAV conversion")

    return output



class SilenceDetector:
    def __init__(self, threshold=0.015, frame_ms=20):
        self.threshold = threshold
        self.frame_ms = frame_ms

    def detect(self, audio, sr):
        frame_size = int((self.frame_ms / 1000) * sr)
        voiced = 0.0
        silence = 0.0

        for i in range(0, len(audio), frame_size):
            frame = audio[i:i+frame_size]
            if len(frame) == 0:
                continue

            rms = np.sqrt(np.mean(frame**2))

            if rms > self.threshold:
                voiced += frame_size / sr
            else:
                silence += frame_size / sr

        return voiced, silence


class FillerDetector:
    def __init__(self):
        self.processor = Wav2Vec2Processor.from_pretrained(
            "facebook/wav2vec2-base-960h")
        self.model = Wav2Vec2ForCTC.from_pretrained(
            "facebook/wav2vec2-base-960h")
        self.silence = SilenceDetector()

    def analyze(self, audio_path):
        audio, sr = librosa.load(audio_path, sr=16000)
        audio = filter_participant_speech(audio, sr)
        inputs = self.processor(audio, sampling_rate=16000, return_tensors="pt")
        with torch.no_grad():
            logits = self.model(inputs.input_values).logits

        predicted_ids = torch.argmax(logits, dim=-1)
        transcript = self.processor.decode(predicted_ids[0]).lower()
        words = transcript.split()

        classic_counts = {f: transcript.count(f) for f in CLASSIC_FILLERS}
        classic_total = sum(classic_counts.values())

        repeated_count = 0
        repeated_words = []
        for i in range(len(words) - 1):
            if words[i] == words[i + 1]:
                repeated_count += 1
                repeated_words.append(words[i])

        voiced, silence = self.silence.detect(audio, sr)
        total_time = voiced + silence

        speech_rate_wps = len(words) / max(voiced, 0.001)
        speech_rate_wpm = speech_rate_wps * 60

        return {
            "transcript": transcript,
            "classic_fillers": classic_counts,
            "repeated_word_fillers": repeated_words,
            "total_fillers": classic_total + repeated_count,
            "voiced_time": round(voiced, 3),
            "silence_time": round(silence, 3),
            "total_time": round(total_time, 3),
            "speech_rate_wpm": round(speech_rate_wpm, 2),
            "pause_ratio": round(silence / max(total_time, 0.001), 3),
            "filler_ratio": round((classic_total + repeated_count) /
                                  max(len(words), 1), 3)
        }
'''
'''
import os
import subprocess
import librosa
import soundfile as sf
import numpy as np
import torch
from transformers import Wav2Vec2ForCTC, Wav2Vec2Processor


CLASSIC_FILLERS = ["um", "uh", "uhh", "uhm", "erm", "hmm", "mmm"]


# -----------------------------
# 🔥 LOAD SILERO VAD (Only once)
# -----------------------------
silero_model, utils = torch.hub.load(
    repo_or_dir='snakers4/silero-vad',
    model='silero_vad',
    force_reload=False
)

(get_speech_timestamps,
 save_audio,
 read_audio,
 VADIterator,
 collect_chunks) = utils


# ---------------------------------------------------
# 🔥 FILTER: Keep only participant long speech parts
# ---------------------------------------------------
def filter_participant_speech(audio, sr, min_segment_duration=1.5):
    """
    Silero VAD filters out interviewer (short segments) 
    and keeps participant (long segments).
    """
    # Normalize floating audio → int16 PCM
    int_audio = (audio * 32767).astype(np.int16)

    timestamps = get_speech_timestamps(int_audio, silero_model, sampling_rate=sr)

    participant_chunks = []

    for ts in timestamps:
        start, end = ts["start"], ts["end"]
        duration = (end - start) / sr

        # Participant speaks longer → keep only these segments
        if duration >= min_segment_duration:
            participant_chunks.append(int_audio[start:end])

    if len(participant_chunks) == 0:
        print("⚠️ No clear participant segments detected — using full audio")
        return audio  # fallback

    # Merge back & convert to float32
    merged = np.concatenate(participant_chunks).astype("float32") / 32767.0
    return merged


# ---------------------------------
# 🔥 Extract WAV using FFmpeg
# ---------------------------------
def extract_wav(video_path, output="temp_audio.wav"):
    temp_raw = "temp_raw_audio.wav"

    # Clean up old files
    for f in [temp_raw, output]:
        if os.path.exists(f):
            os.remove(f)

    # Step 1: Extract raw PCM audio
    cmd_extract = [
        "ffmpeg", "-y",
        "-i", video_path,
        "-vn",
        "-acodec", "pcm_s16le",
        "-ar", "16000",
        "-ac", "1",
        temp_raw
    ]

    subprocess.run(cmd_extract, stdout=subprocess.PIPE, stderr=subprocess.PIPE)

    if not os.path.exists(temp_raw):
        raise Exception("❌ FFmpeg failed to extract raw audio")

    # Step 2: Convert to clean WAV
    cmd_convert = [
        "ffmpeg", "-y",
        "-i", temp_raw,
        "-ac", "1",
        "-ar", "16000",
        output
    ]

    subprocess.run(cmd_convert, stdout=subprocess.PIPE, stderr=subprocess.PIPE)

    if not os.path.exists(output):
        raise Exception("❌ FFmpeg failed converting audio")

    return output


# -----------------------------
# 🔥 Silence Detector
# -----------------------------
class SilenceDetector:
    def __init__(self, threshold=0.015, frame_ms=20):
        self.threshold = threshold
        self.frame_ms = frame_ms

    def detect(self, audio, sr):
        frame_size = int((self.frame_ms / 1000) * sr)
        voiced = 0.0
        silence = 0.0

        for i in range(0, len(audio), frame_size):
            frame = audio[i:i + frame_size]
            if len(frame) == 0:
                continue

            rms = np.sqrt(np.mean(frame ** 2))

            if rms > self.threshold:
                voiced += frame_size / sr
            else:
                silence += frame_size / sr

        return voiced, silence


# -----------------------------
# 🔥 MAIN Speech/Filler Analyzer
# -----------------------------
class FillerDetector:
    def __init__(self):
        self.processor = Wav2Vec2Processor.from_pretrained("facebook/wav2vec2-base-960h")
        self.model = Wav2Vec2ForCTC.from_pretrained("facebook/wav2vec2-base-960h")
        self.silence = SilenceDetector()

    def analyze(self, audio_path):

        # 1️⃣ Load audio
        audio, sr = librosa.load(audio_path, sr=16000)

        # 2️⃣ Filter to only participant speech
        audio = filter_participant_speech(audio, sr)

        # 3️⃣ Run Wav2Vec2
        inputs = self.processor(audio, sampling_rate=16000, return_tensors="pt")

        with torch.no_grad():
            logits = self.model(inputs.input_values).logits

        predicted_ids = torch.argmax(logits, dim=-1)
        transcript = self.processor.decode(predicted_ids[0]).lower()
        words = transcript.split()

        # 4️⃣ Filler detection
        classic_counts = {f: transcript.count(f) for f in CLASSIC_FILLERS}
        classic_total = sum(classic_counts.values())

        # repeated words detection
        repeated_words = [
            words[i] for i in range(len(words) - 1) if words[i] == words[i + 1]
        ]
        repeated_total = len(repeated_words)

        # 5️⃣ Silence detection
        voiced, silence = self.silence.detect(audio, sr)
        total_time = voiced + silence

        speech_rate_wps = len(words) / max(voiced, 0.001)
        speech_rate_wpm = speech_rate_wps * 60

        # 6️⃣ Return JSON
        return {
            "transcript": transcript,
            "classic_fillers": classic_counts,
            "repeated_word_fillers": repeated_words,
            "total_fillers": classic_total + repeated_total,
            "voiced_time": round(voiced, 3),
            "silence_time": round(silence, 3),
            "total_time": round(total_time, 3),
            "speech_rate_wpm": round(speech_rate_wpm, 2),
            "pause_ratio": round(silence / max(total_time, 0.001), 3),
            "filler_ratio": round((classic_total + repeated_total) / max(len(words), 1), 3)
        }
        '''


'''
import os
import subprocess
import numpy as np
import librosa
import torch
from transformers import Wav2Vec2ForCTC, Wav2Vec2Processor
from difflib import SequenceMatcher
# -----------------------------
# CONSTANTS
# -----------------------------
LOCAL_MODEL = "models/wav2vec2-base-960h"
CLASSIC_FILLERS = ["um", "uh", "uhh", "uhm", "erm", "hmm", "mmm"]
HOST_EMB_PATH = os.path.join("models", "host_embedding.npy")

# -----------------------------
# SILERO VAD (lazy load)
# -----------------------------
_silero_model = None
_silero_utils = None


def get_silero_vad():
    global _silero_model, _silero_utils
    if _silero_model is None:
        _silero_model, _silero_utils = torch.hub.load(
            repo_or_dir="snakers4/silero-vad",
            model="silero_vad",
            force_reload=False,
        )
    return _silero_model, _silero_utils


# -----------------------------
# HOST EMBEDDING LOADER
# -----------------------------
_host_embedding = None


def load_host_embedding():
    global _host_embedding
    if _host_embedding is not None:
        return _host_embedding

    if not os.path.exists(HOST_EMB_PATH):
        print("⚠️ No host embedding found, using full audio (no host removal).")
        _host_embedding = None
        return _host_embedding

    arr = np.load(HOST_EMB_PATH)
    t = torch.tensor(arr, dtype=torch.float32)
    t = t / t.norm(p=2)  # just in case
    _host_embedding = t
    print("✅ Loaded host embedding from", HOST_EMB_PATH)
    return _host_embedding


def text_similarity(a: str, b: str) -> float:
    """
    Returns float 0.0–1.0 showing how similar two sentences are.
    """
    a = a.lower().strip()
    b = b.lower().strip()
    return SequenceMatcher(None, a, b).ratio()


def is_host_sentence(transcript: str, host_sentences: list, threshold: float = 0.65):
    """
    Compare transcript against each known host sentence.
    If similarity > threshold, consider it HOST.
    """
    transcript = transcript.lower().strip()

    for sent in host_sentences:
        sim = text_similarity(transcript, sent)
        if sim >= threshold:
            print(f"🛑 Host sentence matched: '{sent}' (sim={sim:.2f})")
            return True

    return False


def filter_participant_with_host(
    audio: np.ndarray,
    sr: int,
    processor: Wav2Vec2Processor,
    model: Wav2Vec2ForCTC,
    host_emb: torch.Tensor,
    host_sentences: list = None,
    min_segment_duration: float = 1.0,
    host_threshold: float = 0.7,
):
    silero_model, utils = get_silero_vad()
    (get_speech_timestamps,
     save_audio,
     read_audio,
     VADIterator,
     collect_chunks) = utils

    int_audio = (audio * 32767).astype(np.int16)
    timestamps = get_speech_timestamps(int_audio, silero_model, sampling_rate=sr)

    participant_chunks = []

    for ts in timestamps:
        start, end = ts["start"], ts["end"]
        dur = (end - start) / sr
        if dur < min_segment_duration:
            continue

        segment = audio[start:end]

        # -------------- ASR transcript for host sentence check --------------
        inputs = processor(segment, sampling_rate=sr, return_tensors="pt")
        with torch.no_grad():
            logits = model(inputs.input_values).logits

        pred_ids = torch.argmax(logits, dim=-1)
        transcript = processor.decode(pred_ids[0]).lower().strip()

        # -------------- Host text matching --------------
        if host_sentences:
            if is_host_sentence(transcript, host_sentences):
                print(f"🚫 Removed host text segment: {transcript}")
                continue

        # -------------- Embedding check --------------
        if host_emb is not None:
            seg_emb = compute_wav2vec_embedding(segment, sr, processor, model)
            sim = torch.dot(seg_emb, host_emb).item()
            if sim >= host_threshold:
                print(f"🚫 Removed host audio (embedding sim={sim:.2f})")
                continue

        # Passed both checks → participant
        participant_chunks.append(segment)

    if len(participant_chunks) == 0:
        print("⚠️ No participant segments found. Returning full audio.")
        return audio

    return np.concatenate(participant_chunks).astype(np.float32)

audio = filter_participant_with_host(
    audio,
    sr,
    self.processor,
    self.model,
    self.host_embedding,
    host_sentences=[
        "welcome to the interview",
        "please introduce yourself",
        "why are you interested in this role",
        "thank you for joining",
        # you can add more host sentences here
    ]
)

# -----------------------------
# WAV2VEC2 EMBEDDING
# -----------------------------
def compute_wav2vec_embedding(audio, sr, processor, model) -> torch.Tensor:
    """
    Compute a simple speaker embedding from Wav2Vec2 hidden states.
    """
    inputs = processor(audio, sampling_rate=sr, return_tensors="pt")
    with torch.no_grad():
        outputs = model(
            inputs.input_values,
            output_hidden_states=True
        )
    hidden = outputs.hidden_states[-1]  # [1, T, D]
    emb = hidden.mean(dim=1).squeeze(0)
    emb = emb / emb.norm(p=2)
    return emb


def filter_participant_with_host(
    audio: np.ndarray,
    sr: int,
    processor: Wav2Vec2Processor,
    model: Wav2Vec2ForCTC,
    host_emb: torch.Tensor,
    min_segment_duration: float = 1.0,
    host_threshold: float = 0.7,
):
    """
    Use Silero VAD to split audio into speech segments.
    For each segment, compute Wav2Vec2 embedding and compare with host embedding.
    If cosine similarity > host_threshold → treat as HOST and drop it.
    Keep only participant segments.
    """
    if host_emb is None:
        # No host embedding available → just return original audio
        return audio

    silero_model, utils = get_silero_vad()
    (get_speech_timestamps,
     save_audio,
     read_audio,
     VADIterator,
     collect_chunks) = utils

    # Silero expects int16 PCM
    int_audio = (audio * 32767).astype(np.int16)
    timestamps = get_speech_timestamps(int_audio, silero_model, sampling_rate=sr)

    participant_chunks = []

    for ts in timestamps:
        start, end = ts["start"], ts["end"]
        dur = (end - start) / sr
        if dur < min_segment_duration:
            continue

        segment = audio[start:end]
        if len(segment) == 0:
            continue

        seg_emb = compute_wav2vec_embedding(segment, sr, processor, model)
        sim = torch.dot(seg_emb, host_emb).item()  # cos similarity (both normalized)

        # If too similar to host → skip
        if sim >= host_threshold:
            # print(f"Skipping host-like segment (sim={sim:.2f})")
            continue

        participant_chunks.append(segment)

    if len(participant_chunks) == 0:
        print("⚠️ No participant-only segments detected, falling back to full audio.")
        return audio

    merged = np.concatenate(participant_chunks).astype(np.float32)
    return merged


# -----------------------------
# EXTRACT WAV FROM VIDEO (FFMPEG)
# -----------------------------
def extract_wav(video_path, output="temp_audio.wav"):
    temp_raw = "temp_raw_audio.wav"

    # Remove old files if they exist
    for f in [temp_raw, output]:
        if os.path.exists(f):
            os.remove(f)

    # 1) Extract raw PCM audio
    cmd_extract = [
        "ffmpeg", "-y",
        "-i", video_path,
        "-vn",
        "-acodec", "pcm_s16le",
        "-ar", "16000",
        "-ac", "1",
        temp_raw,
    ]
    subprocess.run(cmd_extract, stdout=subprocess.PIPE, stderr=subprocess.PIPE)

    if not os.path.exists(temp_raw):
        raise Exception("FFmpeg failed during audio extraction")

    # 2) Convert to final clean WAV
    cmd_convert = [
        "ffmpeg", "-y",
        "-i", temp_raw,
        "-ac", "1",
        "-ar", "16000",
        output,
    ]
    subprocess.run(cmd_convert, stdout=subprocess.PIPE, stderr=subprocess.PIPE)

    if not os.path.exists(output):
        raise Exception("FFmpeg failed during WAV conversion")

    return output


# -----------------------------
# SILENCE DETECTOR (for pause ratio, etc.)
# -----------------------------
class SilenceDetector:
    def __init__(self, threshold=0.015, frame_ms=20):
        self.threshold = threshold
        self.frame_ms = frame_ms

    def detect(self, audio, sr):
        frame_size = int((self.frame_ms / 1000) * sr)
        voiced = 0.0
        silence = 0.0

        for i in range(0, len(audio), frame_size):
            frame = audio[i:i + frame_size]
            if len(frame) == 0:
                continue

            rms = np.sqrt(np.mean(frame ** 2))

            if rms > self.threshold:
                voiced += frame_size / sr
            else:
                silence += frame_size / sr

        return voiced, silence


# -----------------------------
# MAIN FILLER / SPEECH ANALYZER
# -----------------------------
class FillerDetector:
    def __init__(self):
        print("📥 Loading Wav2Vec2 model for speech + speaker embeddings...")
        self.processor = Wav2Vec2Processor.from_pretrained(LOCAL_MODEL,local_files_only=True)
        self.model = Wav2Vec2ForCTC.from_pretrained(LOCAL_MODEL,local_files_only=True)
        self.silence = SilenceDetector()
        self.host_embedding = load_host_embedding()

    def analyze(self, audio_path):
        # 1) Load full audio
        audio, sr = librosa.load(audio_path, sr=16000)

        # 2) Remove host segments using embedding
        audio = filter_participant_with_host(
            audio,
            sr,
            self.processor,
            self.model,
            self.host_embedding,
        )

        # 3) Run Wav2Vec2 ASR on participant-only audio
        inputs = self.processor(audio, sampling_rate=16000, return_tensors="pt")

        with torch.no_grad():
            outputs = self.model(
                inputs.input_values,
                output_hidden_states=False
            )
            logits = outputs.logits

        predicted_ids = torch.argmax(logits, dim=-1)
        transcript = self.processor.decode(predicted_ids[0]).lower()
        words = transcript.split()

        # 4) Filler detection
        classic_counts = {f: transcript.count(f) for f in CLASSIC_FILLERS}
        classic_total = sum(classic_counts.values())

        repeated_words = []
        for i in range(len(words) - 1):
            if words[i] == words[i + 1]:
                repeated_words.append(words[i])
        repeated_total = len(repeated_words)

        # 5) Silence metrics
        voiced, silence = self.silence.detect(audio, sr)
        total_time = voiced + silence

        speech_rate_wps = len(words) / max(voiced, 0.001)
        speech_rate_wpm = speech_rate_wps * 60

        return {
            "transcript": transcript,
            "classic_fillers": classic_counts,
            "repeated_word_fillers": repeated_words,
            "total_fillers": classic_total + repeated_total,
            "voiced_time": round(voiced, 2),
            "silence_time": round(silence, 2),
            "total_time": round(total_time, 2),
            "speech_rate_wpm": round(speech_rate_wpm, 2),
            "pause_ratio": round(silence / max(total_time, 0.001), 3),
            "filler_ratio": round((classic_total + repeated_total) / max(len(words), 1), 3),
        }
'''






'''
import re
import os
import subprocess
import numpy as np
import librosa
import torch
import wordninja
from transformers import Wav2Vec2ForCTC, Wav2Vec2Processor
from difflib import SequenceMatcher

# -----------------------------
# CONSTANTS
# -----------------------------
LOCAL_MODEL = "models/wav2vec2-large-robust"#models/wav2vec2-base-960h
CLASSIC_FILLERS = ["um", "uh", "uhh", "uhm", "erm", "hmm", "mmm","ah","ahh"]
HOST_EMB_PATH = os.path.join("models", "host_embedding.npy")

# -----------------------------
# SILERO VAD (lazy load)
# -----------------------------
_silero_model = None
_silero_utils = None

from spellchecker import SpellChecker
spell = SpellChecker()


def fix_word_errors(text: str):
    """
    Fixes split words (up plying → applying),
    ASR errors (simis → semester),
    and common mis-hearings.
    """

    words = text.split()
    fixed = []

    # Custom replacements for ASR weaknesses
    CUSTOM_FIXES = {
        "simis": "semester",
        "simiss": "semester",
        "trole": "role",
        "upplying": "applying",
        "up plying": "applying",
        "expedience": "experience",
        "oncludes": "concludes",
        "on clude s": "concludes",
        "cut ent ly": "currently",
        "cut ently": "currently",
        "f i p": "fyp",  # domain-specific
    }

    # First pass: join split words like "up plying"
    joined = []
    skip = False
    for i in range(len(words)):
        if skip:
            skip = False
            continue

        two = words[i]
        if i + 1 < len(words):
            two = words[i] + words[i+1]
            if two.lower().replace(" ", "") in [k.replace(" ", "") for k in CUSTOM_FIXES]:
                joined.append(two)
                skip = True
                continue

        joined.append(words[i])

    # Second pass: apply custom corrections
    for w in joined:
        w_clean = w.lower().strip()

        found = False
        for bad, good in CUSTOM_FIXES.items():
            if w_clean.replace(" ", "") == bad.replace(" ", ""):
                fixed.append(good)
                found = True
                break

        if found:
            continue

        # Autocorrect only for words > 3 letters
        if len(w_clean) > 3:
            corrected = spell.correction(w_clean)
            fixed.append(corrected)
        else:
            fixed.append(w)

    return " ".join(fixed)

def get_silero_vad():
    global _silero_model, _silero_utils
    if _silero_model is None:
        _silero_model, _silero_utils = torch.hub.load(
            repo_or_dir="snakers4/silero-vad",
            model="silero_vad",
            force_reload=False,
        )
    return _silero_model, _silero_utils


# -----------------------------
# HOST EMBEDDING LOADER
# -----------------------------
_host_embedding = None

def load_host_embedding():
    global _host_embedding
    if _host_embedding is not None:
        return _host_embedding

    if not os.path.exists(HOST_EMB_PATH):
        print("⚠️ No host embedding found, using full audio (no host removal).")
        _host_embedding = None
        return _host_embedding

    arr = np.load(HOST_EMB_PATH)
    t = torch.tensor(arr, dtype=torch.float32)
    t = t / t.norm(p=2)
    _host_embedding = t
    print("✅ Loaded host embedding from", HOST_EMB_PATH)
    return _host_embedding
# -----------------------------
# TEXT SEPERATION HELPERS
# -----------------------------
def clean_spacing(text: str):
    """
    Ensures words are properly separated.
    Also fixes glued-together words like 'yeahthankyou'.
    """
    # Add spaces between letters and numbers if merged
    text = re.sub(r"(?<=[a-z])(?=[A-Z0-9])", " ", text)
    text = re.sub(r"(?<=[0-9])(?=[A-Za-z])", " ", text)

    # Add space between merged lowercase words (rare but common in CTC)
    text = re.sub(r"([a-z])([A-Z])", r"\1 \2", text)
    text = re.sub(r"([a-z])([A-Z])", r"\1 \2", text)

    # Replace multi-spaces → single space
    text = re.sub(r"\s+", " ", text)

    return text.strip()
# -----------------------------
# TEXT SIMILARITY HELPERS
# -----------------------------
def text_similarity(a: str, b: str) -> float:
    a = a.lower().strip()
    b = b.lower().strip()
    return SequenceMatcher(None, a, b).ratio()


def is_host_sentence(transcript: str, host_sentences: list, threshold: float = 0.65):
    transcript = transcript.lower().strip()

    for sent in host_sentences:
        sim = text_similarity(transcript, sent)
        if sim >= threshold:
            print(f"🛑 Host sentence matched: '{sent}' (sim={sim:.2f})")
            return True

    return False
COMMON_WORDS = set("""
thank you this is my name is for me to start applying now welcome please introduce yourself
""".split())

def smart_split(text):
    words = text.split()
    fixed = []
    
    for w in words:
        # Already normal, keep it
        if w in COMMON_WORDS:
            fixed.append(w)
            continue
        
        # Try greedy split
        found_split = False
        for cw in COMMON_WORDS:
            if w.startswith(cw.replace(" ", "")):  # thankyou
                fixed.extend(cw.split())
                rest = w[len(cw.replace(" ", "")):]
                if rest:
                    fixed.append(rest)
                found_split = True
                break

        if not found_split:
            fixed.append(w)

    return " ".join(fixed)
# -----------------------------
# WAV2VEC2 EMBEDDING
# -----------------------------
def compute_wav2vec_embedding(audio, sr, processor, model) -> torch.Tensor:
    inputs = processor(audio, sampling_rate=sr, return_tensors="pt")
    with torch.no_grad():
        outputs = model(inputs.input_values, output_hidden_states=True)

    hidden = outputs.hidden_states[-1]  # [1, T, D]
    emb = hidden.mean(dim=1).squeeze(0)
    emb = emb / emb.norm(p=2)
    return emb

def ninja_split(text: str):
    """
    Breaks glued words using statistical English segmentation.
    Example: "yeahthankyou" → ["yeah", "thank", "you"]
    """
    words = []
    for w in text.split():
        # If normal word, keep it
        if len(w) < 8:
            words.append(w)
            continue

        # If long & suspicious → split
        split_w = wordninja.split(w)
        if len(split_w) > 1:
            words.extend(split_w)
        else:
            words.append(w)
    return " ".join(words)


CUSTOM_FIXES = {
    "cut ent ly": "currently",
    "cutently": "currently",
    "cut ently": "currently",
    "on clue s": "concludes",
    "onclues": "concludes",
    "onclude": "conclude",
    "simis": "semester",
    "simiss": "semester",
    "upplying": "applying",
    "up plying": "applying",
    "expedience": "experience",
    "trole": "role",
}

def fix_transcript(text: str):
    words = text.split()

    # ---------- FIRST PASS: join split words ----------
    joined = []
    skip = False
    for i in range(len(words)):
        if skip:
            skip = False
            continue

        # attempt joining current + next
        if i + 1 < len(words):
            combined = words[i] + " " + words[i+1]
            combined_cmp = combined.replace(" ", "").lower()

            for bad in CUSTOM_FIXES:
                if combined_cmp == bad.replace(" ", ""):
                    joined.append(CUSTOM_FIXES[bad])
                    skip = True
                    break
            if skip:
                continue

        joined.append(words[i])

    # ---------- SECOND PASS: apply custom corrections ----------
    corrected = []
    for w in joined:
        w_clean = w.lower().strip()

        # direct mapping
        for bad, good in CUSTOM_FIXES.items():
            if w_clean.replace(" ", "") == bad.replace(" ", ""):
                corrected.append(good)
                break
        else:
            # leave names uncorrected (good heuristic)
            if w_clean in ["santa", "havoc", "havot", "hafeez", "sana"]:
                corrected.append(w)
                continue

            # autocorrect English words >3 letters
            if len(w_clean) > 3:
                corrected.append(spell.correction(w_clean))
            else:
                corrected.append(w)

    return " ".join(corrected)

def remove_host_phrases(final_text: str):
    """
    Removes leftover host phrases after transcript cleaning.
    Handles variations like:
    - thank youconcludes our interview
    - thank you conclude s our interview
    - thankyou concludes our interview
    - concludes our interview
    - conclude s our interview
    """
    t = final_text.lower()

    HOST_PATTERNS = [
        r"thank you.*?(conclude|concludes).*?our interview",
        r"thank you.*?our interview",
        r"(conclude|concludes).*?our interview",
        r"this concludes our interview",
        r"conclude s our interview",
        r"concludes our interview",
    ]

    for pattern in HOST_PATTERNS:
        t = re.sub(pattern, " ", t)

    # Clean spacing after removal
    t = re.sub(r"\s+", " ", t).strip()

    return t

def filter_participant_with_host(
    audio: np.ndarray,
    sr: int,
    processor: Wav2Vec2Processor,
    model: Wav2Vec2ForCTC,
    host_sentences: list,
    min_segment_duration: float = 0.2,
    host_text_threshold: float = 0.55,    # LOWER threshold = more aggressive removal
):
    silero_model, utils = get_silero_vad()
    get_speech_timestamps = utils[0]

    int_audio = (audio * 32767).astype(np.int16)
    timestamps = get_speech_timestamps(int_audio, silero_model, sampling_rate=sr)

    participant_segments = []

    for ts in timestamps:
        start, end = ts["start"], ts["end"]
        duration = (end - start) / sr
        if duration < min_segment_duration:
            continue

        segment = audio[start:end]

        # TRANSCRIBE SEGMENT
        inputs = processor(segment, sampling_rate=sr, return_tensors="pt")
        with torch.no_grad():
            logits = model(inputs.input_values).logits
        pred_ids = torch.argmax(logits, dim=-1)
        raw_text = processor.decode(pred_ids[0]).lower().strip()

        # CLEAN IT BEFORE HOST MATCHING
        cleaned = clean_spacing(raw_text)
        cleaned = ninja_split(cleaned)
        cleaned = smart_split(cleaned)

        # HOST PHRASE DETECTION
        for host_line in host_sentences:
            # fuzzy partial match
            if host_line in cleaned:
                print(f"🛑 Removed HOST segment (direct match): {cleaned}")
                break

            similarity = SequenceMatcher(None, cleaned, host_line).ratio()
            if similarity >= host_text_threshold:
                print(f"🛑 Removed HOST segment (fuzzy match): {cleaned}  (sim={similarity:.2f})")
                break

        else:
            participant_segments.append(segment)

    if len(participant_segments) == 0:
        print("⚠️ No participant speech detected — returning original audio.")
        return audio

    return np.concatenate(participant_segments).astype(np.float32)

# -----------------------------
# EXTRACT WAV FROM VIDEO
# -----------------------------
def extract_wav(video_path, output="temp_audio.wav"):
    temp_raw = "temp_raw_audio.wav"

    for f in [temp_raw, output]:
        if os.path.exists(f):
            os.remove(f)

    cmd_extract = [
        "ffmpeg", "-y",
        "-i", video_path,
        "-vn",
        "-acodec", "pcm_s16le",
        "-ar", "16000",
        "-ac", "1",
        temp_raw,
    ]
    subprocess.run(cmd_extract, stdout=subprocess.PIPE, stderr=subprocess.PIPE)

    if not os.path.exists(temp_raw):
        raise Exception("FFmpeg failed during audio extraction")

    cmd_convert = [
        "ffmpeg", "-y",
        "-i", temp_raw,
        "-ac", "1",
        "-ar", "16000",
        output,
    ]
    subprocess.run(cmd_convert, stdout=subprocess.PIPE, stderr=subprocess.PIPE)

    if not os.path.exists(output):
        raise Exception("FFmpeg failed during WAV conversion")

    return output


# -----------------------------
# SILENCE DETECTOR
# -----------------------------
class SilenceDetector:
    def __init__(self, threshold=0.015, frame_ms=20):
        self.threshold = threshold
        self.frame_ms = frame_ms

    def detect(self, audio, sr):
        frame_size = int((self.frame_ms / 1000) * sr)
        voiced = silence = 0.0

        for i in range(0, len(audio), frame_size):
            frame = audio[i:i+frame_size]
            if len(frame) == 0:
                continue

            rms = np.sqrt(np.mean(frame**2))
            if rms > self.threshold:
                voiced += frame_size / sr
            else:
                silence += frame_size / sr

        return voiced, silence


# -----------------------------
# MAIN ANALYZER
# -----------------------------
class FillerDetector:
    def __init__(self):
        print("📥 Loading Wav2Vec2 model...")
        self.processor = Wav2Vec2Processor.from_pretrained(LOCAL_MODEL,local_files_only=True)
        self.model = Wav2Vec2ForCTC.from_pretrained(LOCAL_MODEL,local_files_only=True)
        self.silence = SilenceDetector()
        self.host_embedding = load_host_embedding()

    def analyze(self, audio_path):
        audio, sr = librosa.load(audio_path, sr=16000)

        # Remove host
        
        audio = filter_participant_with_host(
            audio=audio,
            sr=sr,
            processor=self.processor,
            model=self.model,
            host_sentences=[
                "welcome to the interview",
                "please introduce yourself",
                "why are you interested in this role",
                "what relevant experience do you have?",
                "thank you this concludes our interview",
                #"thank you for joining",
            ],
            min_segment_duration=0.3,
            host_text_threshold=0.55,
        )

        # ASR on participant-only audio
        inputs = self.processor(audio, sampling_rate=16000, return_tensors="pt")

        with torch.no_grad():
            logits = self.model(inputs.input_values).logits

        pred_ids = torch.argmax(logits, dim=-1)
        transcript = self.processor.decode(pred_ids[0]).lower()
        transcript = clean_spacing(transcript)
        transcript = ninja_split(transcript)
        transcript = smart_split(transcript)
        transcript = fix_transcript(transcript)
        transcript = fix_word_errors(transcript)
        transcript = remove_host_phrases(transcript)
        #transcript = clean_spacing(transcript)
        words = transcript.split()

        classic_counts = {f: transcript.count(f) for f in CLASSIC_FILLERS}
        classic_total = sum(classic_counts.values())

        repeated_words = [
            words[i] for i in range(len(words)-1) if words[i] == words[i+1]
        ]
        repeated_total = len(repeated_words)

        voiced, silence = self.silence.detect(audio, sr)
        total_time = voiced + silence

        speech_rate_wps = len(words) / max(voiced, 0.001)
        speech_rate_wpm = speech_rate_wps * 60

        return {
            "transcript": transcript,
            "classic_fillers": classic_counts,
            "repeated_word_fillers": repeated_words,
            "total_fillers": classic_total + repeated_total,
            "voiced_time": round(voiced, 2),
            "silence_time": round(silence, 2),
            "total_time": round(total_time, 2),
            "speech_rate_wpm": round(speech_rate_wpm, 2),
            "pause_ratio": round(silence / max(total_time, 0.001), 3),
            "filler_ratio": round((classic_total + repeated_total) / max(len(words), 1), 3),
        }
'''




import re
import os
import subprocess
import numpy as np
import librosa
import torch
import wordninja
import supabase
from transformers import Wav2Vec2ForCTC, Wav2Vec2Processor
from difflib import SequenceMatcher

# -----------------------------
# CONSTANTS
# -----------------------------
LOCAL_MODEL = "facebook/wav2vec2-large-robust"#"models/wav2vec2-large-robust"#models/wav2vec2-base-960h
CLASSIC_FILLERS = ["um", "uh", "uhh", "uhm", "erm", "hmm", "mmm","ah","ahh"]
HOST_EMB_PATH = os.path.join("models", "host_embedding.npy")
BUCKET_NAME="video"
# -----------------------------
# SILERO VAD (lazy load)
# -----------------------------
_silero_model = None
_silero_utils = None

from spellchecker import SpellChecker
spell = SpellChecker()

def fix_word_errors(text: str):
    words = text.split()
    fixed = []

    CUSTOM_FIXES = {
        "simis": "semester",
        "simiss": "semester",
        "trole": "role",
        "upplying": "applying",
        "up plying": "applying",
        "expedience": "experience",
        "oncludes": "concludes",
        "on clude s": "concludes",
        "cut ent ly": "currently",
        "cut ently": "currently",
        "f i p": "fyp",
    }

    joined = []
    skip = False
    for i in range(len(words)):
        if skip:
            skip = False
            continue

        if i + 1 < len(words):
            two = words[i] + words[i+1]
            if two.lower().replace(" ", "") in [k.replace(" ", "") for k in CUSTOM_FIXES]:
                joined.append(two)
                skip = True
                continue

        joined.append(words[i])

    for w in joined:
        w_clean = w.lower().strip()

        # custom mapping
        if w_clean.replace(" ", "") in [k.replace(" ", "") for k in CUSTOM_FIXES]:
            fixed.append(CUSTOM_FIXES[w_clean.replace(" ", "")])
            continue

        # autocorrect safely
        if len(w_clean) > 3:
            corrected = spell.correction(w_clean)
            if corrected is None:          # ← SAFE FALLBACK
                fixed.append(w)
            else:
                fixed.append(corrected)
        else:
            fixed.append(w)

    return " ".join(fixed)

def get_supabase_titles():
    files = supabase.storage.from_(BUCKET_NAME).list()

    valid_extensions = (".mp4", ".mov", ".m4v", ".webm", ".avi")

    cleaned = []

    for file in files:
        name = file.get("name", "")
        size = file.get("metadata", {}).get("size")

        # Skip folders (folders have no size or size=0)
        if not size or size == 0:
            continue

        # Skip objects without valid extensions
        if not name.lower().endswith(valid_extensions):
            continue

        cleaned.append(name)

    return cleaned


def get_silero_vad():
    global _silero_model, _silero_utils
    if _silero_model is None:
        _silero_model, _silero_utils = torch.hub.load(
            repo_or_dir="snakers4/silero-vad",
            model="silero_vad",
            force_reload=False,
        )
    return _silero_model, _silero_utils


# -----------------------------
# HOST EMBEDDING LOADER
# -----------------------------
_host_embedding = None

def load_host_embedding():
    global _host_embedding
    if _host_embedding is not None:
        return _host_embedding

    if not os.path.exists(HOST_EMB_PATH):
        print("⚠️ No host embedding found, using full audio (no host removal).")
        _host_embedding = None
        return _host_embedding

    arr = np.load(HOST_EMB_PATH)
    t = torch.tensor(arr, dtype=torch.float32)
    t = t / t.norm(p=2)
    _host_embedding = t
    print("✅ Loaded host embedding from", HOST_EMB_PATH)
    return _host_embedding
# -----------------------------
# TEXT SEPERATION HELPERS
# -----------------------------
def clean_spacing(text: str):
    """
    Ensures words are properly separated.
    Also fixes glued-together words like 'yeahthankyou'.
    """
    # Add spaces between letters and numbers if merged
    text = re.sub(r"(?<=[a-z])(?=[A-Z0-9])", " ", text)
    text = re.sub(r"(?<=[0-9])(?=[A-Za-z])", " ", text)

    # Add space between merged lowercase words (rare but common in CTC)
    text = re.sub(r"([a-z])([A-Z])", r"\1 \2", text)
    text = re.sub(r"([a-z])([A-Z])", r"\1 \2", text)

    # Replace multi-spaces → single space
    text = re.sub(r"\s+", " ", text)

    return text.strip()
# -----------------------------
# TEXT SIMILARITY HELPERS
# -----------------------------
def text_similarity(a: str, b: str) -> float:
    a = a.lower().strip()
    b = b.lower().strip()
    return SequenceMatcher(None, a, b).ratio()


def is_host_sentence(transcript: str, host_sentences: list, threshold: float = 0.65):
    transcript = transcript.lower().strip()

    for sent in host_sentences:
        sim = text_similarity(transcript, sent)
        if sim >= threshold:
            print(f"🛑 Host sentence matched: '{sent}' (sim={sim:.2f})")
            return True

    return False
COMMON_WORDS = set("""
thank you this is my name is for me to start applying now welcome please introduce yourself
""".split())

def smart_split(text):
    words = text.split()
    fixed = []
    
    for w in words:
        # Already normal, keep it
        if w in COMMON_WORDS:
            fixed.append(w)
            continue
        
        # Try greedy split
        found_split = False
        for cw in COMMON_WORDS:
            if w.startswith(cw.replace(" ", "")):  # thankyou
                fixed.extend(cw.split())
                rest = w[len(cw.replace(" ", "")):]
                if rest:
                    fixed.append(rest)
                found_split = True
                break

        if not found_split:
            fixed.append(w)

    return " ".join(fixed)
# -----------------------------
# WAV2VEC2 EMBEDDING
# -----------------------------
def compute_wav2vec_embedding(audio, sr, processor, model) -> torch.Tensor:
    inputs = processor(audio, sampling_rate=sr, return_tensors="pt")
    with torch.no_grad():
        outputs = model(inputs.input_values, output_hidden_states=True)

    hidden = outputs.hidden_states[-1]  # [1, T, D]
    emb = hidden.mean(dim=1).squeeze(0)
    emb = emb / emb.norm(p=2)
    return emb

def ninja_split(text: str):
    """
    Breaks glued words using statistical English segmentation.
    Example: "yeahthankyou" → ["yeah", "thank", "you"]
    """
    words = []
    for w in text.split():
        # If normal word, keep it
        if len(w) < 8:
            words.append(w)
            continue

        # If long & suspicious → split
        split_w = wordninja.split(w)
        if len(split_w) > 1:
            words.extend(split_w)
        else:
            words.append(w)
    return " ".join(words)


CUSTOM_FIXES = {
    "cut ent ly": "currently",
    "cutently": "currently",
    "cut ently": "currently",
    "on clue s": "concludes",
    "onclues": "concludes",
    "onclude": "conclude",
    "simis": "semester",
    "simiss": "semester",
    "upplying": "applying",
    "up plying": "applying",
    "expedience": "experience",
    "trole": "role",
}

def fix_transcript(text: str):
    words = text.split()

    joined = []
    skip = False
    for i in range(len(words)):
        if skip:
            skip = False
            continue

        if i + 1 < len(words):
            combined = words[i] + " " + words[i+1]
            combined_cmp = combined.replace(" ", "").lower()

            for bad in CUSTOM_FIXES:
                if combined_cmp == bad.replace(" ", ""):
                    joined.append(CUSTOM_FIXES[bad])
                    skip = True
                    break
            if skip:
                continue

        joined.append(words[i])

    corrected = []
    for w in joined:
        w_clean = w.lower().strip()

        # direct custom match
        for bad, good in CUSTOM_FIXES.items():
            if w_clean.replace(" ", "") == bad.replace(" ", ""):
                corrected.append(good)
                break
        else:
            # words to skip autocorrect (names, etc.)
            if w_clean in ["santa", "havoc", "havot", "hafeez", "sana"]:
                corrected.append(w)
                continue

            # SAFE AUTOCORRECT
            if len(w_clean) > 3:
                fixed = spell.correction(w_clean)
                corrected.append(fixed if fixed is not None else w)
            else:
                corrected.append(w)

    # final join
    return " ".join(corrected)

def remove_host_phrases(final_text: str):
    """
    Removes leftover host phrases after transcript cleaning.
    Handles variations like:
    - thank youconcludes our interview
    - thank you conclude s our interview
    - thankyou concludes our interview
    - concludes our interview
    - conclude s our interview
    """
    t = final_text.lower()

    HOST_PATTERNS = [
        r"thank you.?(conclude|concludes).?our interview",
        r"thank you.*?our interview",
        r"(conclude|concludes).*?our interview",
        r"this concludes our interview",
        r"conclude s our interview",
        r"concludes our interview",
    ]

    for pattern in HOST_PATTERNS:
        t = re.sub(pattern, " ", t)

    # Clean spacing after removal
    t = re.sub(r"\s+", " ", t).strip()

    return t


def filter_participant_with_host(
    audio: np.ndarray,
    sr: int,
    processor: Wav2Vec2Processor,
    model: Wav2Vec2ForCTC,
    host_sentences: list,
    min_segment_duration: float = 0.2,
    host_text_threshold: float = 0.55,    # LOWER threshold = more aggressive removal
):
    silero_model, utils = get_silero_vad()
    get_speech_timestamps = utils[0]

    int_audio = (audio * 32767).astype(np.int16)
    timestamps = get_speech_timestamps(int_audio, silero_model, sampling_rate=sr)

    participant_segments = []

    for ts in timestamps:
        start, end = ts["start"], ts["end"]
        duration = (end - start) / sr
        if duration < min_segment_duration:
            continue

        segment = audio[start:end]

        # TRANSCRIBE SEGMENT
        inputs = processor(segment, sampling_rate=sr, return_tensors="pt")
        with torch.no_grad():
            logits = model(inputs.input_values).logits
        pred_ids = torch.argmax(logits, dim=-1)
        raw_text = processor.decode(pred_ids[0]).lower().strip()

        # CLEAN IT BEFORE HOST MATCHING
        cleaned = clean_spacing(raw_text)
        cleaned = ninja_split(cleaned)
        cleaned = smart_split(cleaned)

        # HOST PHRASE DETECTION
        for host_line in host_sentences:
            # fuzzy partial match
            if host_line in cleaned:
                print(f"🛑 Removed HOST segment (direct match): {cleaned}")
                break

            similarity = SequenceMatcher(None, cleaned, host_line).ratio()
            if similarity >= host_text_threshold:
                print(f"🛑 Removed HOST segment (fuzzy match): {cleaned}  (sim={similarity:.2f})")
                break

        else:
            participant_segments.append(segment)

    if len(participant_segments) == 0:
        print("⚠️ No participant speech detected — returning original audio.")
        return audio

    return np.concatenate(participant_segments).astype(np.float32)

# -----------------------------
# EXTRACT WAV FROM VIDEO
# -----------------------------
def extract_wav(video_path, output="temp_audio.wav"):
    temp_raw = "temp_raw_audio.wav"

    for f in [temp_raw, output]:
        if os.path.exists(f):
            os.remove(f)

    cmd_extract = [
        "ffmpeg", "-y",
        "-i", video_path,
        "-vn",
        "-acodec", "pcm_s16le",
        "-ar", "16000",
        "-ac", "1",
        temp_raw,
    ]
    subprocess.run(cmd_extract, stdout=subprocess.PIPE, stderr=subprocess.PIPE)

    if not os.path.exists(temp_raw):
        raise Exception("FFmpeg failed during audio extraction")

    cmd_convert = [
        "ffmpeg", "-y",
        "-i", temp_raw,
        "-ac", "1",
        "-ar", "16000",
        output,
    ]
    subprocess.run(cmd_convert, stdout=subprocess.PIPE, stderr=subprocess.PIPE)

    if not os.path.exists(output):
        raise Exception("FFmpeg failed during WAV conversion")

    return output

class SilenceDetector:
    def __init__(self, threshold=0.015, frame_ms=20):
        self.threshold = threshold
        self.frame_ms = frame_ms

    def detect(self, audio, sr):
        frame_size = int((self.frame_ms / 1000) * sr)
        voiced = silence = 0.0

        for i in range(0, len(audio), frame_size):
            frame = audio[i:i+frame_size]
            if len(frame) == 0:
                continue

            rms = np.sqrt(np.mean(frame**2))
            if rms > self.threshold:
                voiced += frame_size / sr
            else:
                silence += frame_size / sr

        return voiced, silence


# -----------------------------
# MAIN ANALYZER
# -----------------------------
class FillerDetector:
    def __init__(self):
        print("📥 Loading Wav2Vec2 model...")
        '''self.processor = Wav2Vec2Processor.from_pretrained(LOCAL_MODEL, local_files_only=True)
        self.model = Wav2Vec2ForCTC.from_pretrained(LOCAL_MODEL, local_files_only=True)'''
        self.processor = Wav2Vec2Processor.from_pretrained(LOCAL_MODEL, local_files_only=False)
        self.model = Wav2Vec2ForCTC.from_pretrained(LOCAL_MODEL, local_files_only=False)
        self.silence = SilenceDetector()
        self.host_embedding = load_host_embedding()

    def analyze(self, audio_path):
        audio, sr = librosa.load(audio_path, sr=16000)

        # Remove host
        audio = filter_participant_with_host(
            audio=audio,
            sr=sr,
            processor=self.processor,
            model=self.model,
            host_sentences=[
                "welcome to the interview",
                "please introduce yourself",
                "why are you interested in this role",
                "what relevant experience do you have?",
                "thank you this concludes our interview",
            ],
            min_segment_duration=0.3,
            host_text_threshold=0.55,
        )

        # ASR
        inputs = self.processor(audio, sampling_rate=16000, return_tensors="pt")

        with torch.no_grad():
            logits = self.model(inputs.input_values).logits

        pred_ids = torch.argmax(logits, dim=-1)
        transcript = self.processor.decode(pred_ids[0]).lower()

        # Cleaning pipeline
        transcript = clean_spacing(transcript)
        transcript = ninja_split(transcript)
        transcript = smart_split(transcript)
        transcript = fix_transcript(transcript)
        transcript = fix_word_errors(transcript)
        transcript = remove_host_phrases(transcript)

        words = transcript.split()

        # Filler counts
        classic_counts = {f: transcript.count(f) for f in CLASSIC_FILLERS}
        classic_total = sum(classic_counts.values())

        repeated_words = [
            words[i] for i in range(len(words)-1) if words[i] == words[i+1]
        ]
        repeated_total = len(repeated_words)

        voiced, silence = self.silence.detect(audio, sr)
        total_time = voiced + silence

        speech_rate_wps = len(words) / max(voiced, 0.001)
        speech_rate_wpm = speech_rate_wps * 60

        return {
            "transcript": transcript,
            "classic_fillers": classic_counts,
            "repeated_word_fillers": repeated_words,
            "total_fillers": classic_total + repeated_total,
            "voiced_time": round(voiced, 2),
            "silence_time": round(silence, 2),
            "total_time": round(total_time, 2),
            "speech_rate_wpm": round(speech_rate_wpm, 2),
            "pause_ratio": round(silence / max(total_time, 0.001), 3),
            "filler_ratio": round((classic_total + repeated_total) / max(len(words), 1), 3),
        }
