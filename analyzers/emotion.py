'''import cv2
from deepface import DeepFace
from tqdm import tqdm

FRAME_SKIP = 10
NEGATIVE_EMOTIONS = ["angry", "fear", "disgust", "sad"]

def analyze_emotions(video_path):
    cap = cv2.VideoCapture(video_path)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    emotion_counts = {
        "angry": 0, "disgust": 0, "fear": 0, "happy": 0,
        "sad": 0, "surprise": 0, "neutral": 0
    }

    negative_detected = set()
    frame_id = 0

    pbar = tqdm(total_frames, desc="Emotion Analysis")

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        pbar.update(1)

        if frame_id % FRAME_SKIP != 0:
            frame_id += 1
            continue

        try:
            result = DeepFace.analyze(
                frame,
                actions=["emotion"],
                enforce_detection=False
            )
            emo = result["emotion"]
            pred = max(emo, key=emo.get)
            emotion_counts[pred] += 1

            if pred in NEGATIVE_EMOTIONS:
                negative_detected.add(pred)

        except:
            pass

        frame_id += 1

    cap.release()
    pbar.close()

    return {
        "emotion_counts": emotion_counts,
        "negative_emotions_detected": list(negative_detected)
    }
'''
'''
import cv2
from deepface import DeepFace

# OpenCV fast face detector
FACE_CASCADE_PATH = cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
face_detector = cv2.CascadeClassifier(FACE_CASCADE_PATH)

# Process every Nth frame
FRAME_SKIP = 10

def analyze_emotions(video_path):
    cap = cv2.VideoCapture(video_path)

    emotion_counts = {
        "angry": 0, "disgust": 0, "fear": 0,
        "happy": 0, "sad": 0, "surprise": 0, "neutral": 0
    }

    negative_emotions = ["angry", "fear", "disgust", "sad"]
    negative_detected = set()

    frame_id = 0

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        if frame_id % FRAME_SKIP != 0:
            frame_id += 1
            continue

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = face_detector.detectMultiScale(gray, 1.3, 5)

        # If no face detected → skip safely
        if len(faces) == 0:
            frame_id += 1
            continue

        (x, y, w, h) = faces[0]  # use first face
        face_roi = frame[y:y+h, x:x+w]

        try:
            result = DeepFace.analyze(
                img_path=face_roi,
                actions=["emotion"],
                enforce_detection=False
            )

            pred_emotion = result["emotion"]
            dominant = max(pred_emotion, key=pred_emotion.get)
            emotion_counts[dominant] += 1

            if dominant in negative_emotions:
                negative_detected.add(dominant)

        except Exception as e:
            pass

        frame_id += 1

    cap.release()

    return {
        "emotion_counts": emotion_counts,
        "negative_emotions_detected": list(negative_detected)
    }'''
'''
import cv2
from deepface import DeepFace

FRAME_SKIP = 10

def analyze_emotions(video_path):
    cap = cv2.VideoCapture(video_path)

    emotion_counts = {
        "angry": 0, "disgust": 0, "fear": 0,
        "happy": 0, "sad": 0, "surprise": 0, "neutral": 0
    }

    negative_emotions = ["angry", "fear", "disgust", "sad"]
    negative_detected = set()

    frame_id = 0

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        if frame_id % FRAME_SKIP != 0:
            frame_id += 1
            continue

        try:
            result = DeepFace.analyze(
                img_path=frame,
                actions=["emotion"],
                enforce_detection=False
            )

            emo = result["emotion"]
            dominant = max(emo, key=emo.get)
            emotion_counts[dominant] += 1

            if dominant in negative_emotions:
                negative_detected.add(dominant)

        except Exception:
            pass

        frame_id += 1

    cap.release()

    return {
        "emotion_counts": emotion_counts,
        "negative_emotions_detected": list(negative_detected)
    }
'''

'''import cv2
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

        if frame_id % FRAME_SKIP != 0:
            frame_id += 1
            continue

        try:
            result = DeepFace.analyze(
                img_path=frame,
                actions=["emotion"],
                enforce_detection=False
            )

            emo = result["emotion"]
            dominant = max(emo, key=emo.get)
            emotion_counts[dominant] += 1

            if dominant in NEGATIVE_EMOTIONS:
                negative_detected.add(dominant)

        except:
            pass

        frame_id += 1

    cap.release()

    # Compute final dominant emotion
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
