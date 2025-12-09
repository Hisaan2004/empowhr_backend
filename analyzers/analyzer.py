#from analyzers.emotion import analyze_emotions
#from analyzers.speech import extract_wav, FillerDetector
#from analyzers.attention import analyze_attention
'''

def analyze_video(video_path):
    print("➡️ Running Emotion Analysis...")
    emotions = analyze_emotions(video_path)

    print("➡️ Extracting Audio...")
    wav_path = extract_wav(video_path)

    print("➡️ Running Speech Analysis...")
    speech = FillerDetector().analyze(wav_path)

    print("➡️ Running Attention Analysis...")
    attention = analyze_attention(video_path)

    return {
        "emotions": emotions,
        "speech": speech,
        "attention": attention
    }
    '''
from analyzers.emotion import analyze_emotions
from analyzers.speech import extract_wav, FillerDetector
from analyzers.attention import analyze_attention

def analyze_video(video_path):
    print("➡️ Running Emotion Analysis...")
    emotions = analyze_emotions(video_path)

    print("➡️ Extracting Audio...")
    wav_path = extract_wav(video_path)

    print("➡️ Running Speech Analysis...")
    speech = FillerDetector().analyze(wav_path)

    print("➡️ Running Attention Analysis...")
    attention = analyze_attention(video_path)

    return {
        "emotions": emotions,
        "speech": speech,
        "attention": attention
    }

