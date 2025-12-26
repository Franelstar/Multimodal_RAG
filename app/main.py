# app/main.py

from fastapi import FastAPI, UploadFile, File, Form
from fastapi.responses import JSONResponse
from pathlib import Path
import shutil
import uuid

from app.core.pdf_processor import PDFProcessor
from app.storage.chroma import get_or_create_collection
from app.models.clip import CLIPEmbedding
from app.rag.pipeline import ask
from app.utils.hashing import make_session_id

app = FastAPI(title="RAG Multimodal API")

# -----------------------------
# Initialisation globale
# -----------------------------

DATA_DIR = Path("data/raw")
PDF_DIR = DATA_DIR / "pdfs"
IMG_DIR = DATA_DIR / "images"
PDF_DIR.mkdir(parents=True, exist_ok=True)
IMG_DIR.mkdir(parents=True, exist_ok=True)

pdf_processor = PDFProcessor()
clip_model = CLIPEmbedding()
collection = get_or_create_collection(
    name="Multimodal_arts",
    embedding_function=clip_model
)

# -----------------------------
# Endpoints
# -----------------------------

@app.post("/upload/")
async def upload_file(
    pdf: UploadFile = File(...),
    image: UploadFile = File(...),
    user_id: str = Form(...)
):
    """
    Upload d'un PDF et d'une image.
    Génére IDs, processe et indexe dans Chroma.
    """
    pdf_path = PDF_DIR / pdf.filename
    image_path = IMG_DIR / image.filename

    with pdf_path.open("wb") as f:
        shutil.copyfileobj(pdf.file, f)
    with image_path.open("wb") as f:
        shutil.copyfileobj(image.file, f)

    result = pdf_processor.process(pdf_path, image_path)

    # Indexation Chroma
    collection.add(
        ids=result["clip_ids"],
        documents=result["clip_docs"],
        metadatas=result["clip_meta"]
    )

    return JSONResponse({
        "status": "ok",
        "pdf_id": result["pdf_id"],
        "image_id": result["image_id"]
    })


@app.post("/ask/")
async def ask_question(user_id: str = Form(...), question: str = Form(...)):
    """
    Pose une question à la pipeline RAG multimodal
    """
    # Création ou récupération session
    conversation_id = str(uuid.uuid4())
    session_id = make_session_id(user_id, conversation_id)

    # Appel pipeline RAG
    response = ask(question=question, session_id=session_id)

    return JSONResponse({
        "session_id": session_id,
        "question": question,
        "answer": response
    })


@app.get("/")
def read_root():
    return {"message": "RAG Multimodal API en ligne"}
