'''from services.emotion_service import analyze_emotions
from services.gaze_service import analyze_gaze

def analyze_video(video_url):
    emotions = analyze_emotions(video_url)
    gaze = analyze_gaze(video_url)

    return {
        "emotions": emotions,
        "gaze": gaze,
        "status": "processing completed"
    }
'''
import cv2
import requests
import numpy as np

def analyze_video(video_url: str):

    print("➡️ Step 1: Downloading video:", video_url)

    try:
        video_bytes = requests.get(video_url).content
    except Exception as e:
        print("❌ DOWNLOAD FAILED:", e)
        return {"error": "Download failed", "details": str(e)}

    print("➡️ Step 2: Saving video to disk")
    with open("temp_video.mp4", "wb") as f:
        f.write(video_bytes)

    print("➡️ Step 3: Opening video with OpenCV")
    cap = cv2.VideoCapture("temp_video.mp4")

    if not cap.isOpened():
        print("❌ OPENCV FAILED TO OPEN VIDEO")
        return {"error": "Could not open video file"}

    print("➡️ Step 4: Reading first frame")
    success, frame = cap.read()

    if not success:
        print("❌ FAILED TO READ FRAME")
        return {"error": "Could not read frame"}

    print("➡️ SUCCESS! VIDEO IS READ CORRECTLY")

    # TEMP RESPONSE
    return {
        "status": "Video downloaded and opened successfully!",
        "first_frame_shape": str(frame.shape)
    }
