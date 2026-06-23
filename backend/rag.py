"""
RAG (Retrieval-Augmented Generation) pipeline implementation.
Handles document retrieval and LLM-based response generation.
"""

import logging
from typing import List, Dict, Any, Optional, Union
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import Chroma
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_openai import OpenAIEmbeddings
from langchain_community.llms import OpenAI as LangChainOpenAI
from langchain_community.llms import Ollama
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.documents import Document

from .config import settings

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class RAGPipeline:
    """RAG pipeline for document retrieval and response generation."""
    
    def __init__(self):
        """Initialize the RAG pipeline with embeddings and vector store."""
        self.embeddings = self._initialize_embeddings()
        self.vector_store: Optional[Chroma] = None
        self.qa_chain = None
        self.llm = self._initialize_llm()
        
    def _initialize_embeddings(self) -> Union[OpenAIEmbeddings, HuggingFaceEmbeddings]:
        """
        Initialize embeddings based on provider configuration.
        
        Returns:
            OpenAIEmbeddings or HuggingFaceEmbeddings: Configured embedding model
        """
        provider = settings.embedding_provider.lower()
        
        if provider == "openai":
            if not settings.openai_api_key:
                raise ValueError("OPENAI_API_KEY is required when using OpenAI embeddings")
            logger.info(f"Loading OpenAI embedding model: {settings.embedding_model}")
            embeddings = OpenAIEmbeddings(
                model=settings.embedding_model,
                openai_api_key=settings.openai_api_key
            )
        else:
            logger.info(f"Loading HuggingFace embedding model: {settings.embedding_model}")
            embeddings = HuggingFaceEmbeddings(
                model_name=settings.embedding_model,
                model_kwargs={'device': 'cpu'},
                encode_kwargs={'normalize_embeddings': True}
            )
        
        return embeddings
    
    def _initialize_llm(self):
        """
        Initialize the LLM based on configuration.
        
        Returns:
            LLM instance (OpenAI or Ollama)
        """
        provider = settings.llm_provider.lower()
        
        if provider == "openai":
            if not settings.openai_api_key:
                raise ValueError("OPENAI_API_KEY is required when using OpenAI provider")
            logger.info(f"Initializing OpenAI LLM with model: {settings.openai_model}")
            llm = LangChainOpenAI(
                openai_api_key=settings.openai_api_key,
                model_name=settings.openai_model,
                temperature=0.7
            )
        elif provider == "ollama":
            logger.info(f"Initializing Ollama LLM with model: {settings.ollama_model}")
            llm = Ollama(
                model=settings.ollama_model,
                base_url=settings.ollama_base_url
            )
        else:
            raise ValueError(f"Unsupported LLM provider: {provider}")
        
        return llm
    
    def _create_prompt_template(self) -> str:
        """
        Create a prompt template for RAG.
        
        Returns:
            str: Prompt template string
        """
        template = """
        Use the following pieces of context to answer the question at the end.
        If you don't know the answer based on the context, just say that you don't know,
        don't try to make up an answer.
        
        Context:
        {context}
        
        Question: {question}
        
        Answer:
        """
        return template
    
    def create_vector_store(self, documents: List[Document]) -> Chroma:
        """
        Create and persist a vector store from documents.
        
        Args:
            documents: List of Document objects to index
            
        Returns:
            Chroma: Configured vector store
        """
        logger.info(f"Creating vector store from {len(documents)} documents")
        
        # Create text splitter
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=settings.chunk_size,
            chunk_overlap=settings.chunk_overlap,
            length_function=len,
            separators=["\n\n", "\n", " ", ""]
        )
        
        # Split documents into chunks
        chunks = text_splitter.split_documents(documents)
        logger.info(f"Split documents into {len(chunks)} chunks")
        
        # Create vector store
        vector_store = Chroma.from_documents(
            documents=chunks,
            embedding=self.embeddings,
            persist_directory=settings.vector_db_path
        )
        
        self.vector_store = vector_store
        logger.info(f"Vector store created and persisted to {settings.vector_db_path}")
        
        return vector_store
    
    def load_vector_store(self) -> Chroma:
        """
        Load an existing vector store from disk.
        
        Returns:
            Chroma: Loaded vector store
        """
        logger.info(f"Loading vector store from {settings.vector_db_path}")
        
        vector_store = Chroma(
            persist_directory=settings.vector_db_path,
            embedding_function=self.embeddings
        )
        
        self.vector_store = vector_store
        logger.info("Vector store loaded successfully")
        
        return vector_store
    
    def create_qa_chain(self):
        """
        Create a retriever for the RAG pipeline.
        
        Returns:
            Retriever: Configured retriever
        """
        if not self.vector_store:
            raise ValueError("Vector store not initialized. Call create_vector_store or load_vector_store first.")
        
        logger.info("Creating retriever")
        
        # Create retriever
        retriever = self.vector_store.as_retriever(
            search_kwargs={"k": settings.top_k_retrieval}
        )
        
        self.qa_chain = retriever
        logger.info("Retriever created successfully")
        
        return retriever
    
    def query(self, question: str) -> Dict[str, Any]:
        """
        Query the RAG pipeline with a question.
        
        Args:
            question: User question
            
        Returns:
            Dict containing answer and source documents
        """
        if not self.qa_chain:
            raise ValueError("Retriever not initialized. Call create_qa_chain first.")
        
        logger.info(f"Processing query: {question}")
        
        # Retrieve relevant documents
        source_documents = self.qa_chain.invoke(question)
        
        # Format context from retrieved documents
        context = "\n\n".join([doc.page_content for doc in source_documents])
        
        # Create prompt
        prompt_template = self._create_prompt_template()
        prompt = prompt_template.format(context=context, question=question)
        
        # Get answer from LLM
        answer = self.llm.invoke(prompt)
        
        # Format source information
        sources = []
        for doc in source_documents:
            source_info = {
                "content": doc.page_content,
                "metadata": doc.metadata
            }
            sources.append(source_info)
        
        response = {
            "answer": answer,
            "sources": sources,
            "num_sources": len(sources)
        }
        
        logger.info(f"Query processed successfully. Used {len(sources)} source chunks")
        
        return response
    
    def clear_vector_store(self) -> None:
        """Clear the vector store by deleting the persistence directory."""
        import shutil
        import os
        
        if os.path.exists(settings.vector_db_path):
            logger.info(f"Clearing vector store at {settings.vector_db_path}")
            shutil.rmtree(settings.vector_db_path)
            self.vector_store = None
            self.qa_chain = None
            logger.info("Vector store cleared")
        else:
            logger.warning("Vector store directory does not exist")
