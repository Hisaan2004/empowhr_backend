def analyze_emotions(video_url):
    from deepface import DeepFace  # import INSIDE function

    result = DeepFace.analyze(video_url, actions=['emotion'])
    return result