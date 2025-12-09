'''from pymongo import MongoClient
import os

# Your MongoDB URI (from MongoDB Atlas or local MongoDB)
MONGO_URI = os.getenv("MONGO_URI", "mongodb://localhost:27017")

client = MongoClient(MONGO_URI)

db = client["EmpowHR_db1"]         # Database name
processed_videos = db["processed_videos"]  # Collection'''
from pymongo import MongoClient
from pymongo.server_api import ServerApi
import os
from dotenv import load_dotenv

# Load .env
load_dotenv()

MONGO_URI = os.getenv("MONGO_URI")

if not MONGO_URI:
    raise Exception("❌ MONGO_URI NOT FOUND — Check your .env file!")

print("✅ Using MONGO_URI:", MONGO_URI)  # DEBUG

client = MongoClient(MONGO_URI, server_api=ServerApi("1"))

db = client["EmpowHR_db1"]
processed_videos = db["processed_videos"]

