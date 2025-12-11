'''import mediapipe as mp
import cv2
import numpy as np
import pandas as pd

mp_face_mesh = mp.solutions.face_mesh


def get_center(lm, idxs):
    x = np.mean([lm[i].x for i in idxs])
    y = np.mean([lm[i].y for i in idxs])
    return np.array([x, y])


def analyze_attention(video_path):

    cap = cv2.VideoCapture(video_path)
    results_list = []

    with mp_face_mesh.FaceMesh(
            static_image_mode=False,
            max_num_faces=1,
            refine_landmarks=True,
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5) as fm:

        fps = cap.get(cv2.CAP_PROP_FPS) or 30
        frame_duration = 1 / fps

        while True:
            ret, frame = cap.read()
            if not ret:
                break

            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            faces = fm.process(rgb)

            if faces.multi_face_landmarks:
                lm = faces.multi_face_landmarks[0].landmark
                left = get_center(lm, [33, 133, 160, 159])
                right = get_center(lm, [362, 263, 385, 386])
                nose = np.array([lm[1].x, lm[1].y])

                gaze = (left + right) / 2 - nose

                if abs(gaze[0]) < 0.15 and abs(gaze[1]) < 0.15:
                    status = "attentive"
                else:
                    status = "distracted"
            else:
                status = "no_face"

            results_list.append(status)

    cap.release()

    df = pd.DataFrame({"status": results_list})
    df["time"] = df.index * frame_duration

    # Count attention
    attentive_time = df[df.status == "attentive"].time.count() * frame_duration
    total_time = df.time.iloc[-1]
    score = (attentive_time / total_time) * 100

    return {
        "attention_score": round(score, 2),
        "total_time": round(total_time, 2),
        "attentive_time": round(attentive_time, 2),
        "timeline": df.to_dict("records")
    }
'''
import cv2
import numpy as np
import mediapipe as mp
import pandas as pd

mp_face_mesh = mp.solutions.face_mesh

def get_center(lm, idxs):
    x = np.mean([lm[i].x for i in idxs])
    y = np.mean([lm[i].y for i in idxs])
    return np.array([x, y])


def analyze_attention(video_path):

    cap = cv2.VideoCapture(video_path)
    results_list = []

    with mp_face_mesh.FaceMesh(
        max_num_faces=1,
        refine_landmarks=True,
        min_detection_confidence=0.5,
        min_tracking_confidence=0.5
    ) as fm:

        fps = cap.get(cv2.CAP_PROP_FPS) or 30
        frame_duration = 1 / fps

        while True:
            ret, frame = cap.read()
            if not ret:
                break

            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            faces = fm.process(rgb)

            if faces.multi_face_landmarks:
                lm = faces.multi_face_landmarks[0].landmark
                left = get_center(lm, [33, 133, 160, 159])
                right = get_center(lm, [362, 263, 385, 386])
                nose = np.array([lm[1].x, lm[1].y])

                gaze = (left + right) / 2 - nose

                status = "attentive" if abs(gaze[0]) < 0.25 and abs(gaze[1]) < 0.25 else "distracted"
            else:
                status = "no_face"

            results_list.append(status)

    cap.release()

    df = pd.DataFrame({"status": results_list})
    df["time"] = df.index * frame_duration

    attentive_time = df[df.status == "attentive"].time.count() * frame_duration
    total_time = df.time.iloc[-1]
    score = (attentive_time / total_time) * 100

    return {
        "attention_score": round(score, 2),
        "attentive_time": round(attentive_time, 2),
        "total_time": round(total_time, 2)
    }
