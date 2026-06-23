"""
Document ingestion module for processing uploaded files.
Handles PDF loading and document chunking for the RAG pipeline.
"""

import os
import logging
from typing import List
from pathlib import Path
from langchain_community.document_loaders import PyPDFLoader
from langchain_core.documents import Document

from .config import settings
from .rag import RAGPipeline

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class DocumentIngestor:
    """Handles document ingestion and processing."""
    
    def __init__(self, rag_pipeline: RAGPipeline):
        """
        Initialize the document ingestor.
        
        Args:
            rag_pipeline: RAG pipeline instance
        """
        self.rag_pipeline = rag_pipeline
    
    def load_pdf(self, file_path: str) -> List[Document]:
        """
        Load a PDF file and extract text.
        
        Args:
            file_path: Path to the PDF file
            
        Returns:
            List of Document objects
        """
        logger.info(f"Loading PDF from: {file_path}")
        
        if not os.path.exists(file_path):
            raise FileNotFoundError(f"File not found: {file_path}")
        
        # Load PDF using LangChain
        loader = PyPDFLoader(file_path)
        documents = loader.load()
        
        logger.info(f"Loaded {len(documents)} pages from PDF")
        
        # Add file metadata to documents
        filename = Path(file_path).name
        for doc in documents:
            doc.metadata["source"] = filename
        
        return documents
    
    def ingest_document(self, file_path: str) -> dict:
        """
        Ingest a document into the vector store.
        
        Args:
            file_path: Path to the document file
            
        Returns:
            Dict with ingestion status and metadata
        """
        try:
            # Load document
            documents = self.load_pdf(file_path)
            
            # Create or update vector store
            if self.rag_pipeline.vector_store is None:
                # Create new vector store
                self.rag_pipeline.create_vector_store(documents)
            else:
                # Add to existing vector store
                logger.info("Adding documents to existing vector store")
                from langchain.text_splitter import RecursiveCharacterTextSplitter
                
                text_splitter = RecursiveCharacterTextSplitter(
                    chunk_size=settings.chunk_size,
                    chunk_overlap=settings.chunk_overlap,
                    length_function=len,
                    separators=["\n\n", "\n", " ", ""]
                )
                
                chunks = text_splitter.split_documents(documents)
                self.rag_pipeline.vector_store.add_documents(chunks)
            
            # Recreate QA chain with updated vector store
            self.rag_pipeline.create_qa_chain()
            
            filename = Path(file_path).name
            return {
                "status": "success",
                "message": f"Document '{filename}' ingested successfully",
                "filename": filename,
                "num_pages": len(documents)
            }
            
        except Exception as e:
            logger.error(f"Error ingesting document: {str(e)}")
            return {
                "status": "error",
                "message": f"Failed to ingest document: {str(e)}",
                "filename": Path(file_path).name
            }
    
    def ingest_directory(self, directory_path: str) -> dict:
        """
        Ingest all PDF files from a directory.
        
        Args:
            directory_path: Path to the directory containing PDFs
            
        Returns:
            Dict with ingestion summary
        """
        logger.info(f"Ingesting PDFs from directory: {directory_path}")
        
        if not os.path.exists(directory_path):
            raise FileNotFoundError(f"Directory not found: {directory_path}")
        
        # Find all PDF files
        pdf_files = list(Path(directory_path).glob("*.pdf"))
        
        if not pdf_files:
            logger.warning(f"No PDF files found in {directory_path}")
            return {
                "status": "success",
                "message": "No PDF files found",
                "num_files": 0
            }
        
        logger.info(f"Found {len(pdf_files)} PDF files")
        
        # Load all documents
        all_documents = []
        for pdf_file in pdf_files:
            try:
                documents = self.load_pdf(str(pdf_file))
                all_documents.extend(documents)
            except Exception as e:
                logger.error(f"Error loading {pdf_file.name}: {str(e)}")
        
        if not all_documents:
            return {
                "status": "error",
                "message": "Failed to load any documents",
                "num_files": 0
            }
        
        # Create vector store with all documents
        self.rag_pipeline.create_vector_store(all_documents)
        self.rag_pipeline.create_qa_chain()
        
        return {
            "status": "success",
            "message": f"Ingested {len(pdf_files)} PDF files",
            "num_files": len(pdf_files),
            "num_pages": len(all_documents)
        }


def main():
    """Main function to run document ingestion from command line."""
    import argparse
    
    parser = argparse.ArgumentParser(description="Ingest documents into the RAG pipeline")
    parser.add_argument(
        "--file",
        type=str,
        help="Path to a single PDF file to ingest"
    )
    parser.add_argument(
        "--dir",
        type=str,
        help="Path to a directory containing PDF files to ingest"
    )
    parser.add_argument(
        "--clear",
        action="store_true",
        help="Clear the existing vector store before ingestion"
    )
    
    args = parser.parse_args()
    
    # Initialize RAG pipeline
    rag_pipeline = RAGPipeline()
    
    # Clear vector store if requested
    if args.clear:
        rag_pipeline.clear_vector_store()
        print("Vector store cleared")
    
    # Initialize ingestor
    ingestor = DocumentIngestor(rag_pipeline)
    
    # Ingest documents
    if args.file:
        result = ingestor.ingest_document(args.file)
        print(result)
    elif args.dir:
        result = ingestor.ingest_directory(args.dir)
        print(result)
    else:
        # Default: ingest from uploads directory
        uploads_dir = settings.uploads_path
        if os.path.exists(uploads_dir):
            result = ingestor.ingest_directory(uploads_dir)
            print(result)
        else:
            print(f"Uploads directory not found: {uploads_dir}")


if __name__ == "__main__":
    main()
