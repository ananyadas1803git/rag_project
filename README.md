# RAG Chatbot

A complete full-stack Retrieval-Augmented Generation (RAG) chatbot application built with FastAPI, Streamlit, LangChain, and ChromaDB.

## Tech Stack

- **Frontend**: Streamlit
- **Backend**: FastAPI
- **LLM**: OpenAI (default) with optional Ollama support
- **Embeddings**: SentenceTransformers (all-MiniLM-L6-v2)
- **Vector Database**: ChromaDB
- **RAG Framework**: LangChain
- **Storage**: Local files (PDF support)

## Project Structure

```
rag-chatbot/
│
├── backend/
│   ├── main.py           # FastAPI application with REST endpoints
│   ├── rag.py            # RAG pipeline implementation
│   ├── ingest.py         # Document ingestion logic
│   ├── config.py         # Configuration and environment variables
│   ├── requirements.txt  # Backend dependencies
│   ├── vectordb/         # ChromaDB persistent storage
│   └── uploads/          # Uploaded PDF files
│
├── frontend/
│   └── app.py            # Streamlit chatbot UI
│
├── data/                 # Additional data storage
│
├── .env                  # Environment variables (create from .env.example)
├── .env.example          # Environment variables template
├── .gitignore            # Git ignore rules
├── requirements.txt      # Project dependencies
├── docker-compose.yml    # Docker orchestration
└── README.md             # This file
```

## Features

- **Multi-turn conversation** with session memory
- **PDF document upload** and processing
- **Retrieval from uploaded files** only
- **Source chunk display** for transparency
- **OpenAI and Ollama support** for LLM
- **Local vector database** with ChromaDB
- **Modern responsive UI** with Streamlit
- **Error handling** and logging
- **Modular architecture** with clean code

## Installation

### Prerequisites

- Python 3.11 or higher
- pip package manager
- OpenAI API key (for OpenAI LLM) OR Ollama installed (for Ollama LLM)

### Step 1: Clone or Navigate to Project

```bash
cd rag-chatbot
```

### Step 2: Create Virtual Environment (Recommended)

```bash
python -m venv venv

# On Windows
venv\Scripts\activate

# On macOS/Linux
source venv/bin/activate
```

### Step 3: Install Dependencies

```bash
pip install -r requirements.txt
```

### Step 4: Configure Environment Variables

Copy the example environment file and configure it:

```bash
cp .env.example .env
```

Edit `.env` with your configuration:

**For OpenAI:**
```env
LLM_PROVIDER=openai
OPENAI_API_KEY=your_actual_openai_api_key_here
OPENAI_MODEL=gpt-3.5-turbo
```

**For Ollama:**
```env
LLM_PROVIDER=ollama
OLLAMA_MODEL=llama3
OLLAMA_BASE_URL=http://localhost:11434
```

## Running the Application

### Option 1: Run Backend and Frontend Separately

#### Start the Backend

```bash
uvicorn backend.main:app --reload
```

The backend will start on `http://localhost:8000`

#### Start the Frontend

In a new terminal:

```bash
streamlit run frontend/app.py
```

The frontend will open in your browser at `http://localhost:8501`

### Option 2: Run with Docker Compose

```bash
docker-compose up
```

This will start both backend and frontend services.

## Usage

### Uploading PDFs

1. Open the Streamlit frontend at `http://localhost:8501`
2. In the sidebar, click "Browse files" under "Upload Documents"
3. Select a PDF file to upload
4. Wait for the document to be processed and indexed
5. The document will appear in the "Uploaded Documents" list

### Chatting with Documents

1. Type your question in the chat input field
2. Press Enter or click Send
3. The chatbot will retrieve relevant chunks from your documents
4. View the answer and expand "View Source Chunks" to see the retrieved context

### Clearing the Knowledge Base

1. In the sidebar, click "Clear All Documents"
2. This will remove all documents from the vector store
3. Upload new documents to start fresh

## Switching Between OpenAI and Ollama

### Using OpenAI

1. Set `LLM_PROVIDER=openai` in `.env`
2. Provide your `OPENAI_API_KEY`
3. Optionally set `OPENAI_MODEL` (default: gpt-3.5-turbo)
4. Restart the backend server

### Using Ollama

1. Install Ollama from https://ollama.ai
2. Pull your desired model (e.g., `ollama pull llama3`)
3. Set `LLM_PROVIDER=ollama` in `.env`
4. Set `OLLAMA_MODEL` to your model name
5. Optionally set `OLLAMA_BASE_URL` (default: http://localhost:11434)
6. Restart the backend server

## API Endpoints

### Health Check
```
GET /health
```
Returns API health status and configuration.

### Chat
```
POST /chat
Content-Type: application/json

{
  "question": "Your question here",
  "session_id": "optional_session_id"
}
```
Returns answer with source documents.

### Upload Document
```
POST /upload
Content-Type: multipart/form-data

file: <PDF file>
```
Uploads and processes a PDF document.

### Clear Vector Store
```
POST /clear
```
Clears all documents from the vector store.

### List Documents
```
GET /documents
```
Returns list of uploaded document filenames.

## Configuration Options

### RAG Parameters

- `CHUNK_SIZE`: Document chunk size (default: 500)
- `CHUNK_OVERLAP`: Chunk overlap for context (default: 100)
- `TOP_K_RETRIEVAL`: Number of chunks to retrieve (default: 3)
- `EMBEDDING_MODEL`: SentenceTransformers model (default: all-MiniLM-L6-v2)

### Storage Paths

- `VECTOR_DB_PATH`: Vector database storage path (default: backend/vectordb)
- `UPLOADS_PATH`: Uploaded files storage path (default: backend/uploads)

### API Configuration

- `API_HOST`: API server host (default: 0.0.0.0)
- `API_PORT`: API server port (default: 8000)
- `CORS_ORIGINS`: Allowed CORS origins (default: http://localhost:8501,http://localhost:8502)

## Troubleshooting

### Backend won't start

**Issue**: Port 8000 already in use
```bash
# Find process using port 8000
netstat -ano | findstr :8000  # Windows
lsof -i :8000                 # macOS/Linux

# Kill the process or change API_PORT in .env
```

**Issue**: Missing dependencies
```bash
pip install -r requirements.txt
```

### Frontend can't connect to backend

**Issue**: CORS error
- Check that `CORS_ORIGINS` in `.env` includes your frontend URL
- Ensure backend is running before starting frontend

**Issue**: Backend not responding
- Check backend logs for errors
- Verify API health at `http://localhost:8000/health`

### Document upload fails

**Issue**: File not a PDF
- Only PDF files are supported
- Check file extension is `.pdf`

**Issue**: Storage permission error
- Ensure `backend/uploads` directory exists and is writable
- Check file system permissions

### Chat returns no results

**Issue**: No documents uploaded
- Upload at least one PDF document before chatting

**Issue**: Vector store not initialized
- Check backend logs for initialization errors
- Try clearing and re-uploading documents

### OpenAI API errors

**Issue**: Invalid API key
- Verify `OPENAI_API_KEY` is correct in `.env`
- Check your OpenAI account has available credits

**Issue**: Rate limit exceeded
- Wait a few minutes before retrying
- Consider upgrading your OpenAI plan

### Ollama connection errors

**Issue**: Ollama not running
```bash
# Start Ollama
ollama serve
```

**Issue**: Model not found
```bash
# Pull the model
ollama pull llama3
```

**Issue**: Wrong base URL
- Verify `OLLAMA_BASE_URL` matches your Ollama installation
- Default is `http://localhost:11434`

### Memory issues

**Issue**: Out of memory with large documents
- Reduce `CHUNK_SIZE` in `.env`
- Process documents in smaller batches
- Increase system RAM or use a machine with more resources

## Development

### Running Tests

```bash
# Add test commands here when tests are implemented
pytest
```

### Code Style

The project follows PEP 8 style guidelines. Use linting tools:

```bash
pip install black flake8
black backend/ frontend/
flake8 backend/ frontend/
```

## License

This project is provided as-is for educational and development purposes.

## Contributing

Contributions are welcome! Please follow these steps:

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Submit a pull request

## Support

For issues and questions:
- Check the Troubleshooting section
- Review the API documentation at `http://localhost:8000/docs`
- Check backend logs for detailed error messages
