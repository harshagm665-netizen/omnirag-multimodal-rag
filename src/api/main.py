from fastapi import FastAPI, UploadFile, File, Form, HTTPException
from fastapi.responses import RedirectResponse
from pydantic import BaseModel
import shutil
import os
import logging
from typing import List

from src.ingestion.document_parser import DocumentParser
from src.embeddings.colpali_embedder import ColpaliEmbedder
from src.database.qdrant_client import QdrantStore
from src.api.rag_service import RAGService
from src.llm.key_rotator import KeyRotator

# Setup Logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(name)s: %(message)s")
logger = logging.getLogger(__name__)

# Initialize Components
parser = DocumentParser()
embedder = ColpaliEmbedder(mock=True) # Mocked for local dev without GPU
qdrant = QdrantStore(in_memory=False)
key_rotator = KeyRotator()
rag_service = RAGService(embedder=embedder, qdrant=qdrant, key_rotator=key_rotator)

app = FastAPI(
    title="Multimodal RAG API",
    description="Enterprise API for processing PDFs, Images, and Tables with ColPali & Qdrant.",
    version="1.0.0"
)

@app.get("/")
def read_root():
    return RedirectResponse(url="/docs")

class QueryRequest(BaseModel):
    query: str

class QueryResponse(BaseModel):
    answer: str
    citations: List[dict]

@app.post("/upload")
async def upload_document(file: UploadFile = File(...)):
    """
    Ingest a PDF or Image file.
    Parses it, embeds with ColPali, and stores in Qdrant.
    """
    if not file.filename:
        raise HTTPException(status_code=400, detail="No filename provided.")
        
    temp_path = os.path.join(parser.temp_dir, file.filename)
    with open(temp_path, "wb") as buffer:
        shutil.copyfileobj(file.file, buffer)
        
    try:
        # 1. Parse
        logger.info(f"Parsing {file.filename}...")
        pages = parser.parse_file(temp_path, source_filename=file.filename)
        
        # 2. Embed
        logger.info(f"Embedding {len(pages)} pages...")
        embedded_data = embedder.embed_pages(pages)
        
        # 3. Store
        logger.info(f"Storing into Qdrant...")
        qdrant.upsert_vectors(embedded_data)
        
        return {"message": f"Successfully ingested {file.filename} ({len(pages)} pages)."}
        
    except Exception as e:
        logger.error(f"Failed to process {file.filename}: {e}")
        raise HTTPException(status_code=500, detail=str(e))
    finally:
        if os.path.exists(temp_path):
            os.remove(temp_path)

@app.post("/query", response_model=QueryResponse)
async def query_rag(request: QueryRequest):
    """
    Query the Multi-modal RAG system.
    """
    logger.info(f"Received query: {request.query}")
    try:
        response = rag_service.query(request.query)
        return QueryResponse(answer=response["answer"], citations=response["citations"])
    except Exception as e:
        logger.error(f"Query failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/health")
def health_check():
    return {"status": "healthy", "components": ["ColPali", "Qdrant", "Groq"]}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
