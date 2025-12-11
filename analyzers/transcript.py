import whisper
import re
from difflib import SequenceMatcher
# Load Whisper once
model = whisper.load_model("base")

def generate_transcript(audio_path):
    """
    Returns full transcript from Whisper & timestamps
    """
    result = model.transcribe(audio_path, fp16=False)
    return result.get("segments", [])
def fuzzy_match(text, host_questions, threshold=0.55):
    text = text.lower().strip()
    best_q = None
    best_sim = 0

    for q in host_questions:
        q_clean = q.lower().strip()
        sim = SequenceMatcher(None, text, q_clean).ratio()

        if sim > best_sim:
            best_sim = sim
            best_q = q_clean

    return best_q if best_sim >= threshold else None


def build_qa_pairs(segments, host_questions, threshold=0.55):
    host_q = [q.lower().strip() for q in host_questions]

    qa_pairs = []
    current_q = None
    current_a = []

    for seg in segments:
        text = seg["text"].lower().strip()

        matched_q = fuzzy_match(text, host_q, threshold)

        if matched_q:
            if current_q and current_a:
                qa_pairs.append({
                    "question": current_q,
                    "answer": " ".join(current_a).strip()
                })

            current_q = matched_q
            current_a = []
            continue

        if current_q:
            current_a.append(text)

    if current_q and current_a:
        qa_pairs.append({
            "question": current_q,
            "answer": " ".join(current_a).strip()
        })

    return qa_pairs
