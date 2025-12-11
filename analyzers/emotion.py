'''
import cv2
from deepface import DeepFace

FRAME_SKIP = 10
NEGATIVE_EMOTIONS = ["angry", "fear", "disgust", "sad"]

def analyze_emotions(video_path):

    cap = cv2.VideoCapture(video_path)

    emotion_counts = {
        "angry": 0, "disgust": 0, "fear": 0,
        "happy": 0, "sad": 0, "surprise": 0, "neutral": 0
    }

    negative_detected = set()
    frame_id = 0

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        # skip frames
        if frame_id % FRAME_SKIP != 0:
            frame_id += 1
            continue

        try:
            result = DeepFace.analyze(  
                img_path = frame ,                # <-- correct argument #img=frame,
                actions=["emotion"],
                enforce_detection=False
            )

            # DeepFace returns a list → take first element
            if isinstance(result, list):
                result = result[0]

            emo = result["emotion"]
            dominant = max(emo, key=emo.get)
            emotion_counts[dominant] += 1

            if dominant in NEGATIVE_EMOTIONS:
                negative_detected.add(dominant)

        except Exception as e:
            print("DeepFace Error:", e)
            pass

        frame_id += 1

    cap.release()

    # Fix: If everything is 0, return neutral instead of angry
    if all(v == 0 for v in emotion_counts.values()):
        most_common = "neutral"
    else:
        most_common = max(emotion_counts, key=emotion_counts.get)

    return {
        "most_common": most_common,
        "emotion_counts": emotion_counts,
        "negative_detected": list(negative_detected)
    }
'''
import cv2
from deepface import DeepFace

FRAME_SKIP = 10

# Emotion merge rules
EMOTION_MAP = {
    "angry": "angry",
    "disgust": "angry",   # merge disgust → angry
    "fear": "neutral",    # merge fear → neutral
    "sad": "neutral",     # merge sad → neutral
    "neutral": "neutral",
    "happy": "happy",
    "surprise": "surprise"
}

def analyze_emotions(video_path):

    cap = cv2.VideoCapture(video_path)

    # Final categories only
    emotion_counts = {
        "angry": 0,
        "neutral": 0,
        "happy": 0,
        "surprise": 0
    }

    frame_id = 0

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        # Skip frames
        if frame_id % FRAME_SKIP != 0:
            frame_id += 1
            continue

        try:
            result = DeepFace.analyze(
                img_path=frame,
                actions=["emotion"],
                enforce_detection=False
            )

            if isinstance(result, list):
                result = result[0]

            # Raw emotions from DeepFace
            emo_dict = result["emotion"]
            raw_dominant = max(emo_dict, key=emo_dict.get)

            # Apply custom mapping
            mapped = EMOTION_MAP[raw_dominant]

            # Count mapped emotion
            emotion_counts[mapped] += 1

        except Exception as e:
            print("DeepFace Error:", e)

        frame_id += 1

    cap.release()

    # Pick the most common emotion
    if all(v == 0 for v in emotion_counts.values()):
        most_common = "neutral"
    else:
        most_common = max(emotion_counts, key=emotion_counts.get)

    return {
        "most_common": most_common,
        "emotion_counts": emotion_counts
    }
