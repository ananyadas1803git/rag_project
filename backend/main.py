"""
FastAPI backend for RAG chatbot.
Provides REST API endpoints for chat and document upload.
"""

import os
import logging
from typing import Optional
from fastapi import FastAPI, UploadFile, File, HTTPException, Form
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from pathlib import Path

from .config import settings
from .rag import RAGPipeline
from .ingest import DocumentIngestor

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Initialize FastAPI app
app = FastAPI(
    title="RAG Chatbot API",
    description="Retrieval-Augmented Generation chatbot API",
    version="1.0.0"
)

# Configure CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.cors_origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Global RAG pipeline instance
rag_pipeline: Optional[RAGPipeline] = None
ingestor: Optional[DocumentIngestor] = None


def initialize_rag_pipeline():
    """Initialize the RAG pipeline on startup."""
    global rag_pipeline, ingestor
    
    logger.info("Initializing RAG pipeline...")
    
    try:
        # Create uploads directory if it doesn't exist
        os.makedirs(settings.uploads_path, exist_ok=True)
        
        # Initialize RAG pipeline
        rag_pipeline = RAGPipeline()
        
        # Try to load existing vector store
        if os.path.exists(settings.vector_db_path):
            logger.info("Loading existing vector store")
            rag_pipeline.load_vector_store()
            rag_pipeline.create_qa_chain()
        else:
            logger.info("No existing vector store found")
        
        # Initialize ingestor
        ingestor = DocumentIngestor(rag_pipeline)
        
        logger.info("RAG pipeline initialized successfully")
        
    except Exception as e:
        logger.error(f"Error initializing RAG pipeline: {str(e)}")
        raise


# Pydantic models
class ChatRequest(BaseModel):
    """Request model for chat endpoint."""
    question: str
    session_id: Optional[str] = None


class ChatResponse(BaseModel):
    """Response model for chat endpoint."""
    answer: str
    sources: list
    num_sources: int
    session_id: Optional[str] = None


class UploadResponse(BaseModel):
    """Response model for upload endpoint."""
    status: str
    message: str
    filename: str
    num_pages: Optional[int] = None


class ClearResponse(BaseModel):
    """Response model for clear endpoint."""
    status: str
    message: str


class HealthResponse(BaseModel):
    """Response model for health endpoint."""
    status: str
    vector_store_loaded: bool
    llm_provider: str


@app.on_event("startup")
async def startup_event():
    """Initialize RAG pipeline on application startup."""
    initialize_rag_pipeline()


@app.get("/health", response_model=HealthResponse)
async def health_check():
    """
    Health check endpoint.
    
    Returns:
        HealthResponse: Status of the application
    """
    vector_store_loaded = rag_pipeline is not None and rag_pipeline.vector_store is not None
    
    return HealthResponse(
        status="healthy",
        vector_store_loaded=vector_store_loaded,
        llm_provider=settings.llm_provider
    )


@app.post("/chat", response_model=ChatResponse)
async def chat(request: ChatRequest):
    """
    Chat endpoint for processing user questions.
    
    Args:
        request: ChatRequest containing the question
        
    Returns:
        ChatResponse: Answer with source documents
    """
    if not rag_pipeline or not rag_pipeline.qa_chain:
        raise HTTPException(
            status_code=400,
            detail="Vector store not initialized. Please upload documents first."
        )
    
    try:
        # Query the RAG pipeline
        result = rag_pipeline.query(request.question)
        
        return ChatResponse(
            answer=result["answer"],
            sources=result["sources"],
            num_sources=result["num_sources"],
            session_id=request.session_id
        )
        
    except Exception as e:
        logger.error(f"Error processing chat request: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/upload", response_model=UploadResponse)
async def upload_document(file: UploadFile = File(...)):
    """
    Upload endpoint for PDF documents.
    
    Args:
        file: Uploaded PDF file
        
    Returns:
        UploadResponse: Upload status and metadata
    """
    if not ingestor:
        raise HTTPException(status_code=500, detail="Ingestor not initialized")
    
    # Validate file type
    if not file.filename.endswith('.pdf'):
        raise HTTPException(status_code=400, detail="Only PDF files are supported")
    
    try:
        # Save uploaded file
        file_path = os.path.join(settings.uploads_path, file.filename)
        
        with open(file_path, "wb") as buffer:
            content = await file.read()
            buffer.write(content)
        
        logger.info(f"File saved to: {file_path}")
        
        # Ingest document
        result = ingestor.ingest_document(file_path)
        
        if result["status"] == "error":
            raise HTTPException(status_code=500, detail=result["message"])
        
        return UploadResponse(
            status=result["status"],
            message=result["message"],
            filename=result["filename"],
            num_pages=result.get("num_pages")
        )
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error uploading document: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/clear", response_model=ClearResponse)
async def clear_vector_store():
    """
    Clear the vector store endpoint.
    
    Returns:
        ClearResponse: Clear operation status
    """
    if not rag_pipeline:
        raise HTTPException(status_code=500, detail="RAG pipeline not initialized")
    
    try:
        rag_pipeline.clear_vector_store()
        
        # Reinitialize empty pipeline
        initialize_rag_pipeline()
        
        return ClearResponse(
            status="success",
            message="Vector store cleared successfully"
        )
        
    except Exception as e:
        logger.error(f"Error clearing vector store: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/documents")
async def list_documents():
    """
    List uploaded documents endpoint.
    
    Returns:
        List of uploaded document filenames
    """
    try:
        if not os.path.exists(settings.uploads_path):
            return {"documents": []}
        
        documents = [f for f in os.listdir(settings.uploads_path) if f.endswith('.pdf')]
        return {"documents": documents}
        
    except Exception as e:
        logger.error(f"Error listing documents: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(
        "main:app",
        host=settings.api_host,
        port=settings.api_port,
        reload=True
    )
