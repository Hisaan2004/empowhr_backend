'''from fastapi import APIRouter
from services.video_service import analyze_video

router = APIRouter()

@router.get("/analyze")
def analyze(video_url: str):
    return analyze_video(video_url)'''
'''from fastapi import APIRouter, Query
from services.video_service import analyze_video

router = APIRouter()

@router.get("/analyze")
def analyze(video_url: str = Query(...)):
    try:
        result = analyze_video(video_url)
        return result
    except Exception as e:
        print("🔥 ERROR IN ANALYSIS:", e)
        return {"error": str(e)}
'''



'''
import requests
import os
from fastapi import APIRouter
from analyzers.analyzer import analyze_video


router = APIRouter()


@router.get("/analyze")
def analyze(video_url: str):

    try:
        print("Downloading video:", video_url)

        # Download video
        video_data = requests.get(video_url)
        os.makedirs("downloads", exist_ok=True)
        local_path = f"downloads/temp.mp4"

        with open(local_path, "wb") as f:
            f.write(video_data.content)

        return analyze_video(local_path)

    except Exception as e:
        return {"error": str(e)}

'''

# the most correct solution down one 
'''
from fastapi import APIRouter, Query
from services.supabase_client import supabase
from analyzers.analyzer import analyze_video
import requests

router = APIRouter()

@router.get("/analyze")
async def analyze(video_url: str = Query(...)):
    """
    Main API endpoint:
    - Downloads video from Supabase URL
    - Saves to disk as temp_video.mp4
    - Runs: emotions + speech + attention
    - Returns all results
    """

    try:
        print("➡️ Downloading video from:", video_url)
        video_bytes = requests.get(video_url).content

        with open("temp_video.mp4", "wb") as f:
            f.write(video_bytes)

        print("➡️ File saved as temp_video.mp4")

    except Exception as e:
        return {"error": "Video download failed", "details": str(e)}

    print("➡️ Running full analysis pipeline...")
    results = analyze_video("temp_video.mp4")

    return {
        "message": "Analysis completed successfully",
        "results": results
    }
'''


'''
from fastapi import APIRouter, Query
from analyzers.analyzer import analyze_video
import requests
import datetime
import os
from pymongo import MongoClient
from dotenv import load_dotenv
router = APIRouter()

# -----------------------------
# MONGODB CONNECTION (Python)
# -----------------------------
load_dotenv()
MONGO_URI = os.getenv("MONGODB_URI")

if not MONGO_URI:
    raise Exception("❌ MONGODB_URI not found in environment variables")

mongo_client = MongoClient(MONGO_URI)
db = mongo_client["EmpowHR_db1"]           # MUST match your Mongoose dbName
processed_collection = db["processed_videos"]   # collection name


# -----------------------------
# ANALYZE ENDPOINT
# -----------------------------
@router.get("/analyze")
async def analyze(video_url: str = Query(...), file_name: str = Query(None)):
    """
    - Downloads video from Supabase URL
    - Saves as temp_video.mp4
    - Runs emotions + attention + speech analysis
    - Saves results into MongoDB
    """

    try:
        print("➡️ Downloading video from:", video_url)
        video_bytes = requests.get(video_url).content

        with open("temp_video.mp4", "wb") as f:
            f.write(video_bytes)

        print("➡️ File saved as temp_video.mp4")

    except Exception as e:
        return {"error": "Video download failed", "details": str(e)}

    print("➡️ Running full analysis pipeline...")
    results = analyze_video("temp_video.mp4")

    # -----------------------------
    # SAVE TO MONGODB
    # -----------------------------
    document = {
        "fileName": file_name or video_url.split("/")[-1],  # fallback if not provided
        "videoUrl": video_url,
        "processed": True,
        "processedAt": datetime.datetime.utcnow(),
        "results": results
    }

    processed_collection.insert_one(document)

    print("💾 Saved analysis results to MongoDB for:", document["fileName"])

    return {
        "message": "Analysis completed successfully",
        "results": results,
        "saved_to_db": True
    }
'''
'''
from fastapi import APIRouter, Query
from services.supabase_client import supabase
from analyzers.analyzer import analyze_video
from database.mongo_client import processed_videos
import requests
import time

router = APIRouter()

@router.get("/analyze")
async def analyze(video_url: str = Query(...)):
    """
    - Downloads video
    - Runs emotion + speech + attention
    - Saves results in MongoDB
    - Returns results
    """

    try:
        print("➡️ Downloading video from:", video_url)
        video_bytes = requests.get(video_url).content

        with open("temp_video.mp4", "wb") as f:
            f.write(video_bytes)

        print("➡️ File saved as temp_video.mp4")

    except Exception as e:
        return {"error": "Video download failed", "details": str(e)}

    print("➡️ Running full analysis pipeline...")
    results = analyze_video("temp_video.mp4")

    # -----------------------------
    # ⭐ SAVE TO MONGODB ⭐
    # -----------------------------
    record = {
        "video_url": video_url,
        "timestamp": time.time(),
        "emotion_analysis": results["emotions"],
        "speech_analysis": results["speech"],
        "attention_analysis": results["attention"]
    }

    inserted = processed_videos.insert_one(record)

    return {
        "message": "Analysis completed successfully",
        "mongo_id": str(inserted.inserted_id),
        "results": results
    }
    '''


'''
from fastapi import APIRouter, Query
from services.supabase_client import supabase
from analyzers.analyzer import analyze_video
from database.mongo_client import processed_videos
import requests
import time

router = APIRouter()

@router.get("/analyze")
async def analyze(video_url: str = Query(...)):
    """
    - Downloads video
    - Runs emotion + speech + attention
    - Saves results in MongoDB
    - Returns results
    """

    try:
        print("➡️ Downloading video from:", video_url)
        video_bytes = requests.get(video_url).content

        with open("temp_video.mp4", "wb") as f:
            f.write(video_bytes)

        print("➡️ File saved as temp_video.mp4")

    except Exception as e:
        return {"error": "Video download failed", "details": str(e)}

    print("➡️ Running full analysis pipeline...")
    results = analyze_video("temp_video.mp4")

    # -----------------------------
    # ⭐ SAVE TO MONGODB ⭐
    # -----------------------------
    record = {
        "video_url": video_url,
        "timestamp": time.time(),
        "emotion_analysis": results["emotions"],
        "speech_analysis": results["speech"],
        "attention_analysis": results["attention"]
    }

    inserted = processed_videos.insert_one(record)

    return {
        "message": "Analysis completed successfully",
        "mongo_id": str(inserted.inserted_id),
        "results": results
    }

'''


'''
from fastapi import APIRouter, Query
from analyzers.analyzer import analyze_video
from pymongo import MongoClient
from pymongo.server_api import ServerApi
import requests
import os

router = APIRouter()

MONGO_URI = os.getenv("MONGO_URI")
client = MongoClient(MONGO_URI, server_api=ServerApi("1"))
db = client["EmpowHR_db1"]
processed_videos = db["processed_videos"]

@router.get("/analyze")
async def analyze(video_url: str = Query(...)):

    # Download video
    video_bytes = requests.get(video_url).content
    with open("temp_video.mp4", "wb") as f:
        f.write(video_bytes)

    # Run analysis
    results = analyze_video("temp_video.mp4")

    # Save to MongoDB
    doc = {
        "video_url": video_url,
        "emotions": results["emotions"],
        "speech": results["speech"],
        "attention": results["attention"]
    }

    inserted_id = processed_videos.insert_one(doc).inserted_id

    return {
        "message": "Analysis completed & saved ✔",
        "id": str(inserted_id),
        "results": results
    }'''

'''
from fastapi import APIRouter, Query
from analyzers.analyzer import analyze_video
from pymongo import MongoClient
from pymongo.server_api import ServerApi
import requests
import os
from dotenv import load_dotenv

router = APIRouter()

# Load environment variables
load_dotenv()

# Get Mongo URI
MONGO_URI = os.getenv("MONGO_URI")

if not MONGO_URI:
    raise Exception("❌ MONGO_URI not found in environment variables")

print("📌 Using Mongo URI:", MONGO_URI)  # Debug

# Create MongoDB client
client = MongoClient(MONGO_URI, server_api=ServerApi("1"))
db = client["EmpowHR_db1"]
processed_videos = db["processed_videos"]


@router.get("/analyze")
async def analyze(video_url: str = Query(...)):

    # Download video
    video_bytes = requests.get(video_url).content
    with open("temp_video.mp4", "wb") as f:
        f.write(video_bytes)

    # Run analysis
    results = analyze_video("temp_video.mp4")

    # Save to MongoDB
    doc = {
        "video_url": video_url,
        "emotions": results["emotions"],
        "speech": results["speech"],
        "attention": results["attention"]
    }

    inserted_id = processed_videos.insert_one(doc).inserted_id

    return {
        "message": "Analysis completed & saved ✔",
        "id": str(inserted_id),
        "results": results
    }

'''

'''
from fastapi import APIRouter, Query
from analyzers.analyzer import analyze_video
from pymongo import MongoClient
from pymongo.server_api import ServerApi
from supabase import create_client
import requests
import os
from dotenv import load_dotenv

router = APIRouter()

# -----------------------------
# Load environment variables
# -----------------------------
load_dotenv()

MONGO_URI = os.getenv("MONGO_URI")
SUPABASE_URL = os.getenv("SUPABASE_URL")
SUPABASE_KEY = os.getenv("SUPABASE_KEY")
BUCKET_NAME = "video"

if not MONGO_URI:
    raise Exception("❌ MONGO_URI not found in environment variables")

if not SUPABASE_URL or not SUPABASE_KEY:
    raise Exception("❌ SUPABASE credentials are missing")

print("📌 Using Mongo URI:", MONGO_URI)

# -----------------------------
# MongoDB Setup
# -----------------------------
client = MongoClient(MONGO_URI, server_api=ServerApi("1"))
db = client["EmpowHR_db1"]
processed_videos = db["processed_videos1"]

# -----------------------------
# Supabase Setup
# -----------------------------
supabase = create_client(SUPABASE_URL, SUPABASE_KEY)


# ================================================================
# 🔹 Filter Function — This removes folders like "videos"
# ================================================================
def list_valid_supabase_videos():
    files = supabase.storage.from_(BUCKET_NAME).list()

    valid_ext = (".mp4", ".mov", ".m4v", ".webm", ".avi")
    cleaned = []

    for f in files:
        # skip if null / invalid
        if not f or not isinstance(f, dict):
            continue

        name = f.get("name")
        metadata = f.get("metadata", {}) or {}
        size = metadata.get("size")

        # skip invalid entries
        if not name:
            continue

        # folders have no size
        if not size or size == 0:
            continue

        # skip non-video files
        if not name.lower().endswith(valid_ext):
            continue

        cleaned.append(name)

    return cleaned



# ================================================================
# 🔹 1. Manual Analyze Single Video
# ================================================================
@router.get("/analyze")
async def analyze(video_url: str = Query(...)):
    video_bytes = requests.get(video_url).content

    with open("temp_video.mp4", "wb") as f:
        f.write(video_bytes)

    results = analyze_video("temp_video.mp4")

    doc = {
        "video_url": video_url,
        "video_title": video_url.split("/")[-1],
        "emotions": results["emotions"],
        "speech": results["speech"],
        "attention": results["attention"]
    }

    inserted_id = processed_videos.insert_one(doc).inserted_id

    return {
        "message": "Analysis completed & saved ✔",
        "id": str(inserted_id),
        "results": results
    }


# ================================================================
# 🔹 2. Auto-Process Pending Videos
# ================================================================
'''
'''
@router.get("/process_pending")
async def process_pending():
    # fetch only VALID video files
    supabase_titles = set(list_valid_supabase_videos())

    # MongoDB titles that are already processed
    processed_titles = {
        x.get("video_title", x.get("video_url", "")).split("/")[-1]
        for x in processed_videos.find({})
    }

    # find which videos still need processing
    pending = supabase_titles - processed_titles

    if not pending:
        return {"message": "All videos already processed ✔"}

    print("\n⚠️ Unprocessed videos found:", pending)

    processed_count = 0

    for title in pending:
        video_url = f"{SUPABASE_URL}/storage/v1/object/public/{BUCKET_NAME}/{title}"

        print(f"\n➡️ Processing: {title}")
        print("URL:", video_url)

        # download
        video_bytes = requests.get(video_url).content
        with open("temp_video.mp4", "wb") as f:
            f.write(video_bytes)

        # analyze
        results = analyze_video("temp_video.mp4")

        # save
        doc = {
            "video_title": title,
            "video_url": video_url,
            "emotions": results["emotions"],
            "speech": results["speech"],
            "attention": results["attention"]
        }

        processed_videos.insert_one(doc)
        processed_count += 1
        print(f"✔ Saved {title}")

    return {
        "message": "Auto-processing complete",
        "processed_count": processed_count,
        "processed_files": list(pending)
    }
'''
'''
@router.get("/process_pending")
async def process_pending():

    supabase_titles = set(list_valid_supabase_videos())

    processed_titles = {
        x.get("video_title", x.get("video_url", "")).split("/")[-1]
        for x in processed_videos.find({})
    }

    pending = supabase_titles - processed_titles

    if not pending:
        return {"message": "All videos already processed ✔"}

    print("⚠️ Pending videos:", pending)

    jobapps = db["jobapplication"]
    processed_count = 0

    for title in pending:

        # 1️⃣ Find job application using video file name
        app_doc = jobapps.find_one({"video_title": title})

        if not app_doc:
            print(f"❌ No job application found for: {title}")
            continue

        # 2️⃣ Extract interview questions → HOST SENTENCES
        host_sentences = []
        qlist = app_doc.get("interview_questions", [])

        for q in qlist:
            if isinstance(q, dict) and "text" in q:
                host_sentences.append(q["text"].lower().strip())
            elif isinstance(q, str):
                host_sentences.append(q.lower().strip())

        if not host_sentences:
            print("⚠️ No questions found, using fallback")
            host_sentences = ["welcome to the interview"]

        print("📌 Using host sentences:", host_sentences)

        # 3️⃣ Download video
        video_url = f"{SUPABASE_URL}/storage/v1/object/public/{BUCKET_NAME}/{title}"
        video_bytes = requests.get(video_url).content
        with open("temp_video.mp4", "wb") as f:
            f.write(video_bytes)

        # 4️⃣ Extract audio
        wav_path = extract_wav("temp_video.mp4")

        # 5️⃣ Run speech analysis with DYNAMIC HOST REMOVAL
        detector = FillerDetector()
        speech = detector.analyze_with_custom_hosts(wav_path, host_sentences)

        # 6️⃣ Emotion + Attention
        emotions = analyze_emotions("temp_video.mp4")
        attention = analyze_attention("temp_video.mp4")

        results = {
            "video_title": title,
            "video_url": video_url,
            "emotions": emotions,
            "speech": speech,
            "attention": attention
        }

        # 7️⃣ Save results inside SAME job application
        jobapps.update_one(
            {"_id": app_doc["_id"]},
            {"$set": {"analysis": results}}
        )

        print(f"✔ Saved analysis → jobapplication for {title}")

        processed_count += 1

    return {
        "message": "Auto-processing done ✔",
        "processed_count": processed_count,
        "files": list(pending)
    }
'''
from fastapi import APIRouter, Query
from analyzers.analyzer import analyze_video
from analyzers.speech import extract_wav, FillerDetector   # ✅ NEW
from analyzers.emotion import analyze_emotions             # ✅ NEW
from analyzers.attention import analyze_attention          # ✅ NEW
from analyzers.transcript import generate_transcript, build_qa_pairs
from pymongo import MongoClient
from pymongo.server_api import ServerApi
from supabase import create_client
import requests
import os
from dotenv import load_dotenv

router = APIRouter()

# -----------------------------
# Load environment variables
# -----------------------------
load_dotenv()

MONGO_URI = os.getenv("MONGO_URI")
SUPABASE_URL = os.getenv("SUPABASE_URL")
SUPABASE_KEY = os.getenv("SUPABASE_KEY")
BUCKET_NAME = "video"

if not MONGO_URI:
    raise Exception("❌ MONGO_URI not found in environment variables")

if not SUPABASE_URL or not SUPABASE_KEY:
    raise Exception("❌ SUPABASE credentials are missing")

print("📌 Using Mongo URI:", MONGO_URI)

# -----------------------------
# MongoDB Setup
# -----------------------------
client = MongoClient(MONGO_URI, server_api=ServerApi("1"))
db = client["EmpowHR_db1"]
processed_videos = db["processed_videos4"]

# -----------------------------
# Supabase Setup
# -----------------------------
supabase = create_client(SUPABASE_URL, SUPABASE_KEY)


# ================================================================
# 🔹 Filter Function — This removes folders like "videos"
# ================================================================
def list_valid_supabase_videos():
    files = supabase.storage.from_(BUCKET_NAME).list()

    valid_ext = (".mp4", ".mov", ".m4v", ".webm", ".avi")
    cleaned = []

    for f in files:
        # skip if null / invalid
        if not f or not isinstance(f, dict):
            continue

        name = f.get("name")
        metadata = f.get("metadata", {}) or {}
        size = metadata.get("size")

        # skip invalid entries
        if not name:
            continue

        # folders have no size
        if not size or size == 0:
            continue

        # skip non-video files
        if not name.lower().endswith(valid_ext):
            continue

        cleaned.append(name)

    return cleaned


# ================================================================
# 🔹 1. Manual Analyze Single Video
# ================================================================
@router.get("/analyze")
async def analyze(video_url: str = Query(...)):
    video_bytes = requests.get(video_url).content

    with open("temp_video.mp4", "wb") as f:
        f.write(video_bytes)

    results = analyze_video("temp_video.mp4")

    doc = {
        "video_url": video_url,
        "video_title": video_url.split("/")[-1],
        "emotions": results["emotions"],
        "speech": results["speech"],
        "attention": results["attention"]
    }

    inserted_id = processed_videos.insert_one(doc).inserted_id

    return {
        "message": "Analysis completed & saved ✔",
        "id": str(inserted_id),
        "results": results
    }


# ================================================================
# 🔹 2. Auto-Process Pending Videos (dynamic host questions)
# ================================================================
'''
@router.get("/process_pending")
async def process_pending():

    supabase_titles = set(list_valid_supabase_videos())

    processed_titles = {
        x.get("video_title", x.get("video_url", "")).split("/")[-1]
        for x in processed_videos.find({})
    }

    pending = supabase_titles - processed_titles

    if not pending:
        return {"message": "All videos already processed ✔"}

    print("⚠️ Pending videos:", pending)

    jobapps = db["jobapplication"]
    processed_count = 0

    for title in pending:

        # 1️⃣ Find job application using video file name
        app_doc = jobapps.find_one({"video_title": title})

        if not app_doc:
            print(f"❌ No job application found for: {title}")
            continue

        # 2️⃣ Extract interview questions → HOST SENTENCES
        host_sentences = []
        qlist = app_doc.get("interview_questions", [])

        for q in qlist:
            if isinstance(q, dict) and "text" in q:
                host_sentences.append(q["text"].lower().strip())
            elif isinstance(q, str):
                host_sentences.append(q.lower().strip())

        if not host_sentences:
            print("⚠️ No questions found, using fallback")
            host_sentences = ["welcome to the interview"]

        print("📌 Using host sentences:", host_sentences)

        # 3️⃣ Download video
        video_url = f"{SUPABASE_URL}/storage/v1/object/public/{BUCKET_NAME}/{title}"
        video_bytes = requests.get(video_url).content
        with open("temp_video.mp4", "wb") as f:
            f.write(video_bytes)

        # 4️⃣ Extract audio
        wav_path = extract_wav("temp_video.mp4")

        # 5️⃣ Run speech analysis with DYNAMIC HOST REMOVAL
        detector = FillerDetector()
        speech = detector.analyze_with_custom_hosts(wav_path, host_sentences)

        # 6️⃣ Emotion + Attention
        emotions = analyze_emotions("temp_video.mp4")
        attention = analyze_attention("temp_video.mp4")

        results = {
            "video_title": title,
            "video_url": video_url,
            "emotions": emotions,
            "speech": speech,
            "attention": attention
        }

        # 7️⃣ Save results inside SAME job application
        jobapps.update_one(
            {"_id": app_doc["_id"]},
            {"$set": {"analysis": results}}
        )

        print(f"✔ Saved analysis → jobapplication for {title}")

        processed_count += 1

    return {
        "message": "Auto-processing done ✔",
        "processed_count": processed_count,
        "files": list(pending)
    }
'''
@router.get("/process_pending")
async def process_pending():

    # 1️⃣ Get all Supabase video filenames
    supabase_titles = set(list_valid_supabase_videos())

    # 2️⃣ Find already processed filenames
    processed_titles = {
        x.get("video_title", x.get("video_url", "")).split("/")[-1]
        for x in processed_videos.find({})
    }

    # 3️⃣ Pending = videos not yet processed
    pending = supabase_titles - processed_titles

    if not pending:
        return {"message": "All videos already processed ✔"}

    print("\n⚠️ Pending videos found:", pending)

    jobapps = db["jobApplications"]
    processed_count = 0

    for filename in pending:

        # ---------------------------------------------------
        # 4️⃣ Remove extension (.mp4, .mov, etc.)
        # ---------------------------------------------------
        import os
        video_key, ext = os.path.splitext(filename)   # → ("abc123", ".mp4")

        print(f"\n🎯 Looking for job application with video_title = '{video_key}'")

        # ---------------------------------------------------
        # 5️⃣ Find the corresponding job application
        # ---------------------------------------------------
        app_doc = jobapps.find_one({"applicationId": video_key})

        if not app_doc:
            print(f"❌ No job application found for: {filename} (search key: {video_key})")
            continue

        # ---------------------------------------------------
        # 6️⃣ Extract interview questions → host sentences
        # ---------------------------------------------------
        host_sentences = []
        question_list = app_doc.get("interviewQuestions", [])

        for q in question_list:
            if isinstance(q, dict) and "text" in q:
                host_sentences.append(q["text"].lower().strip())
            elif isinstance(q, str):
                host_sentences.append(q.lower().strip())

        if not host_sentences:
            print("⚠️ No interview questions found. Using fallback sentence.")
            host_sentences = ["welcome to the interview"]

        print("📌 Host sentences:", host_sentences)

        # ---------------------------------------------------
        # 7️⃣ Download the video file from Supabase
        # ---------------------------------------------------
        video_url = f"{SUPABASE_URL}/storage/v1/object/public/{BUCKET_NAME}/{filename}"
        print(f"⬇️ Downloading video: {video_url}")

        video_bytes = requests.get(video_url).content
        with open("temp_video.mp4", "wb") as f:
            f.write(video_bytes)

        # ---------------------------------------------------
        # 8️⃣ Extract WAV audio
        # ---------------------------------------------------
        wav_path = extract_wav("temp_video.mp4")

        # ---------------------------------------------------
        # 9️⃣ Run Speech Analysis with dynamic host-removal
        # ---------------------------------------------------
        print("🗣️ Running Speech Analysis (dynamic host removal)...")
        detector = FillerDetector()
        speech = detector.analyze_with_custom_hosts(wav_path, host_sentences)
       
        # -------------------------
        #  Whisper Transcript + Q/A
        # -------------------------
        segments = generate_transcript(wav_path)
        qa_pairs = build_qa_pairs(segments, host_sentences)
        # ---------------------------------------------------
        # 🔟 Emotion + Attention Analysis
        # ---------------------------------------------------
        print("😊 Running Emotion Analysis...")
        emotions = analyze_emotions("temp_video.mp4")

        print("👀 Running Attention Analysis...")
        attention = analyze_attention("temp_video.mp4")

        # Final result structure
        '''results = {
            "video_title": video_key,
            "video_url": video_url,
            "emotions": emotions,
            "speech": speech,
            "attention": attention
            
        }'''
        results = {
            "video_title": video_key,
            "video_url": video_url,
            "emotions": emotions,
            "speech": speech,
            "attention": attention,

            # NEW SECTION
            "transcript": {
                "raw_segments": segments,
                "qa_pairs": qa_pairs
            }
        }

        # ---------------------------------------------------
        # 1️⃣1️⃣ Save analysis inside the SAME job application
        # ---------------------------------------------------
        jobapps.update_one(
            {"_id": app_doc["_id"]},
            {"$set": {"analysis": results}}
        )

        print(f"✔ Analysis saved for applicant with video_title '{video_key}'")
        processed_videos.insert_one({
            "video_title": filename,   # with extension
            "video_key": video_key,    # without extension
            "video_url": video_url,
            "analysis": results,
            "processed": True
        })
        processed_count += 1

    # ---------------------------------------------------
    # 1️⃣2️⃣ Final response
    # ---------------------------------------------------
    return {
        "message": "Auto-processing completed successfully ✔",
        "processed_count": processed_count,
        "processed_files": list(pending)
    }

