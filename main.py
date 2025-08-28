from fastapi import FastAPI, UploadFile, File
from fastapi.responses import JSONResponse
import os
from ocr_utils import extract_text_from_image, extract_text_from_pdf
from llm_utils import call_gemma_llm

UPLOAD_FOLDER = "uploads"
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

from fastapi.middleware.cors import CORSMiddleware

app = FastAPI(title="Marksheet Extractor with Gemma LLM")

# Allow all origins (for testing)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000"],  # or ["*"] for all origins
    allow_credentials=True,
    allow_methods=["*"], 
    allow_headers=["*"],
)

@app.get("/")
def home():
    return {"message": "Welcome! Use POST /extract_marksheet/ to upload your marksheet (PDF/Image)."}

@app.post("/extract_marksheet/")
async def extract_marksheet(file: UploadFile = File(...)):
    try:
        file_path = os.path.join(UPLOAD_FOLDER, file.filename)
        with open(file_path, "wb") as f:
            f.write(await file.read())

        # OCR
        if file.filename.lower().endswith(".pdf"):
            text, confidences, _ = extract_text_from_pdf(file_path)
        else:
            text, confidences, _ = extract_text_from_image(file_path)

        # Call Gemma LLM
        json_data = call_gemma_llm(text)

        # Add overall confidence
        json_data["confidence_overall"] = round(sum(confidences)/len(confidences), 2) if confidences else None

        # Optional: Remove uploaded file after processing
        os.remove(file_path)

        return JSONResponse(content=json_data)
    except Exception as e:
        return {"error": str(e)}
    