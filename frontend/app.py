"""
Streamlit frontend for RAG chatbot.
Provides a modern chat interface with document upload capabilities.
"""

import os
import requests
import streamlit as st
from typing import List, Dict, Optional
from datetime import datetime

# Page configuration
st.set_page_config(
    page_title="RAG Chatbot",
    page_icon="🤖",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Configuration
API_URL = "http://localhost:8000"
UPLOAD_URL = f"{API_URL}/upload"
CHAT_URL = f"{API_URL}/chat"
CLEAR_URL = f"{API_URL}/clear"
HEALTH_URL = f"{API_URL}/health"
DOCS_URL = f"{API_URL}/documents"

# Custom CSS for modern styling
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        font-weight: bold;
        color: #1f77b4;
        margin-bottom: 1rem;
    }
    .chat-message {
        padding: 1rem;
        border-radius: 0.5rem;
        margin-bottom: 1rem;
        animation: fadeIn 0.3s ease-in-out;
    }
    .user-message {
        background-color: #e3f2fd;
        border-left: 4px solid #1f77b4;
    }
    .bot-message {
        background-color: #f5f5f5;
        border-left: 4px solid #4caf50;
    }
    .source-chunk {
        background-color: #fff3e0;
        padding: 0.75rem;
        border-radius: 0.25rem;
        margin-top: 0.5rem;
        font-size: 0.85rem;
        border-left: 3px solid #ff9800;
    }
    @keyframes fadeIn {
        from { opacity: 0; transform: translateY(10px); }
        to { opacity: 1; transform: translateY(0); }
    }
    .stButton>button {
        width: 100%;
    }
</style>
""", unsafe_allow_html=True)


def check_api_health() -> bool:
    """
    Check if the backend API is healthy.
    
    Returns:
        bool: True if API is healthy, False otherwise
    """
    try:
        response = requests.get(HEALTH_URL, timeout=5)
        return response.status_code == 200
    except Exception:
        return False


def upload_document(file) -> Dict:
    """
    Upload a PDF document to the backend.
    
    Args:
        file: Uploaded file object
        
    Returns:
        Dict: Upload response
    """
    try:
        files = {"file": (file.name, file, "application/pdf")}
        response = requests.post(UPLOAD_URL, files=files, timeout=60)
        response.raise_for_status()
        return response.json()
    except requests.exceptions.RequestException as e:
        return {"status": "error", "message": str(e)}


def send_chat_message(question: str, session_id: Optional[str] = None) -> Dict:
    """
    Send a chat message to the backend.
    
    Args:
        question: User question
        session_id: Optional session ID for conversation tracking
        
    Returns:
        Dict: Chat response
    """
    try:
        payload = {"question": question, "session_id": session_id}
        response = requests.post(CHAT_URL, json=payload, timeout=120)
        response.raise_for_status()
        return response.json()
    except requests.exceptions.RequestException as e:
        return {"answer": f"Error: {str(e)}", "sources": [], "num_sources": 0}


def clear_vector_store() -> Dict:
    """
    Clear the vector store.
    
    Returns:
        Dict: Clear response
    """
    try:
        response = requests.post(CLEAR_URL, timeout=30)
        response.raise_for_status()
        return response.json()
    except requests.exceptions.RequestException as e:
        return {"status": "error", "message": str(e)}


def list_documents() -> List[str]:
    """
    List uploaded documents.
    
    Returns:
        List of document filenames
    """
    try:
        response = requests.get(DOCS_URL, timeout=5)
        response.raise_for_status()
        data = response.json()
        return data.get("documents", [])
    except requests.exceptions.RequestException:
        return []


def initialize_session_state():
    """Initialize Streamlit session state variables."""
    if "messages" not in st.session_state:
        st.session_state.messages = []
    if "session_id" not in st.session_state:
        st.session_state.session_id = datetime.now().strftime("%Y%m%d_%H%M%S")


def display_chat_message(role: str, content: str, sources: Optional[List[Dict]] = None):
    """
    Display a chat message with optional sources.
    
    Args:
        role: Message role ('user' or 'assistant')
        content: Message content
        sources: Optional list of source documents
    """
    message_class = "user-message" if role == "user" else "bot-message"
    
    st.markdown(f"""
    <div class="chat-message {message_class}">
        <strong>{'You' if role == 'user' else '🤖 Assistant'}</strong>
        <p>{content}</p>
    </div>
    """, unsafe_allow_html=True)
    
    # Display sources if available
    if sources and role == "assistant":
        with st.expander(f"📚 View {len(sources)} Source Chunks", expanded=False):
            for i, source in enumerate(sources, 1):
                st.markdown(f"""
                <div class="source-chunk">
                    <strong>Source {i}</strong><br>
                    <em>File: {source.get('metadata', {}).get('source', 'Unknown')}</em><br>
                    <p>{source.get('content', '')[:500]}...</p>
                </div>
                """, unsafe_allow_html=True)


def main():
    """Main application function."""
    # Initialize session state
    initialize_session_state()
    
    # Header
    st.markdown('<h1 class="main-header">🤖 RAG Chatbot</h1>', unsafe_allow_html=True)
    
    # Sidebar
    with st.sidebar:
        st.header("⚙️ Settings")
        
        # API Health Check
        api_healthy = check_api_health()
        if api_healthy:
            st.success("✅ Backend API is online")
        else:
            st.error("❌ Backend API is offline")
            st.warning("Please start the backend with: uvicorn backend.main:app --reload")
        
        st.divider()
        
        # Document Upload
        st.header("📄 Upload Documents")
        uploaded_file = st.file_uploader(
            "Upload a PDF file",
            type=["pdf"],
            help="Upload PDF documents to add to the knowledge base"
        )
        
        if uploaded_file is not None:
            with st.spinner("Processing document..."):
                result = upload_document(uploaded_file)
                if result.get("status") == "success":
                    st.success(f"✅ {result.get('message')}")
                    st.rerun()
                else:
                    st.error(f"❌ {result.get('message')}")
        
        st.divider()
        
        # List Documents
        st.header("📚 Uploaded Documents")
        documents = list_documents()
        if documents:
            for doc in documents:
                st.text(f"📄 {doc}")
        else:
            st.info("No documents uploaded yet")
        
        st.divider()
        
        # Clear Vector Store
        st.header("🗑️ Clear Knowledge Base")
        if st.button("Clear All Documents", type="secondary"):
            with st.spinner("Clearing vector store..."):
                result = clear_vector_store()
                if result.get("status") == "success":
                    st.success("✅ Vector store cleared")
                    st.session_state.messages = []
                    st.rerun()
                else:
                    st.error(f"❌ {result.get('message')}")
        
        st.divider()
        
        # Session Info
        st.header("ℹ️ Session Info")
        st.text(f"Session ID: {st.session_state.session_id}")
        st.text(f"Messages: {len(st.session_state.messages)}")
    
    # Main chat interface
    st.subheader("💬 Chat")
    
    # Display chat history
    for message in st.session_state.messages:
        display_chat_message(
            role=message["role"],
            content=message["content"],
            sources=message.get("sources")
        )
    
    # Chat input
    if prompt := st.chat_input("Ask a question about your documents..."):
        if not api_healthy:
            st.error("Backend API is offline. Please start the backend server.")
            return
        
        # Add user message to chat history
        st.session_state.messages.append({
            "role": "user",
            "content": prompt
        })
        
        # Display user message
        display_chat_message(role="user", content=prompt)
        
        # Generate assistant response
        with st.spinner("Thinking..."):
            response = send_chat_message(
                question=prompt,
                session_id=st.session_state.session_id
            )
        
        # Add assistant response to chat history
        st.session_state.messages.append({
            "role": "assistant",
            "content": response.get("answer", ""),
            "sources": response.get("sources", [])
        })
        
        # Display assistant response
        display_chat_message(
            role="assistant",
            content=response.get("answer", ""),
            sources=response.get("sources", [])
        )
        
        # Rerun to display the new message
        st.rerun()


if __name__ == "__main__":
    main()
