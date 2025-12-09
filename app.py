'''from fastapi import FastAPI
from routes.analysis_routes import router as analysis_router
from routes.upload_routes import router as upload_router

app = FastAPI()

app.include_router(analysis_router, prefix="/api")
app.include_router(upload_router, prefix="/api")

@app.get("/")
def home():
    return {"message": "Backend is running successfully!"}
'''

'''
from fastapi import FastAPI
from routes.analysis_routes import router as analysis_router

app = FastAPI()

@app.get("/")
def root():
    return {"message": "Backend running ✔"}

app.include_router(analysis_router, prefix="/api")'''
from fastapi import FastAPI
from routes.analysis_routes import router as analysis_router
from routes.analysis_routes import process_pending   # <-- IMPORT THE FUNCTION

app = FastAPI()

@app.get("/")
async def root():
    # Automatically start processing on homepage visit
    result = await process_pending()
    return {
        "message": "Auto-processing triggered ✔",
        "details": result
    }

app.include_router(analysis_router, prefix="/api")
