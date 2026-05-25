# OSS Onboarding & Contributor Agent 🚀

An intelligent agentic workspace built to onboard developers to new open-source codebases instantly. It recursively ingests any GitHub repository, creates visual architecture flowcharts, indexes key configurations, and answers contributor questions via a local SQLite RAG vector store.

---

## Key Features

* **Recursive Ingestion & Smart Pruning**: Automatically prunes noise (like `node_modules`, `.next`, venvs, lockfiles) and limits configuration hierarchy representation up to depth 3 with item count indicators for collapsed folders.
* **Mermaid.js System Architecture Flowcharts**: Generates an interactive system architecture flowchart. Features a fullscreen overlay mode that dynamically stretches vertically to fill the screen layout without horizontal distortion.
* **Conversational ReAct Agent**: Interactive chat workspace using a LangChain ReAct loop to call codebase tools dynamically, read files on-demand, and synthesize precise developer answers.
* **Real-Time SSE Streaming**: Streams LLM token responses and active tool-calling statuses (e.g. `Reading file: src/app/page.tsx`) in real-time using Server-Sent Events (SSE).
* **OpenAI Prompt Caching**: Restructures System & Human messages into a static container (repo structure, README, priority files) and a dynamic container (user query + RAG contexts) to maximize OpenAI Prompt Cache hits and minimize token costs.
* **Incremental Ingestion Cache (Content-Hashing)**: Computes SHA-256 hashes of repository files. Speeds up subsequent ingests to **under 0.1 seconds** and skips OpenAI embedding calls entirely for unchanged files.
* **SQLite Vector Database**: Pure Python cosine similarity calculation on top of SQLite, ensuring fast semantic searches without requiring heavy C++ packages or dependencies (like FAISS).

---

## Tech Stack

### Backend
* **Core**: [FastAPI](https://fastapi.tiangolo.com/) (Python 3.9+)
* **Agentic Framework**: [LangChain](https://www.langchain.com/) & [LangGraph](https://www.langchain.com/langgraph)
* **Database**: [SQLite](https://www.sqlite.org/) (handles persistency, code chunking, and vectors)
* **HTTP Client**: [HTTPX](https://www.python-httpx.org/) (concurrent GitHub API fetching with caching)

### Frontend
* **Core**: [Next.js](https://nextjs.org/) (React 19, TypeScript)
* **Styling**: Tailwind CSS
* **Formatting**: [react-markdown](https://github.com/remarkjs/react-markdown) + [remark-gfm](https://github.com/remarkjs/remark-gfm)
* **Icons**: Lucide React
* **Visualizations**: Mermaid.js

---

## Getting Started

### 1. Prerequisites & Environment Setup
Create a `.env` file at the root of the repository:
```env
GITHUB_TOKEN=your_github_personal_access_token
OPENAI_API_KEY=your_openai_api_key
```

### 2. Backend Setup
1. Navigate to the backend directory:
   ```bash
   cd backend
   ```
2. Create and activate a Python virtual environment:
   ```bash
   python3 -m venv venv
   source venv/bin/activate
   ```
3. Install Python dependencies:
   ```bash
   pip install -r ../requirements.txt
   ```
4. Start the FastAPI development server:
   ```bash
   uvicorn main:app --reload --port 8000
   ```
The backend API documentation will be available at `http://localhost:8000/docs`.

### 3. Frontend Setup
1. Navigate to the frontend directory:
   ```bash
   cd ../frontend
   ```
2. Install npm dependencies:
   ```bash
   npm install
   ```
3. Start the Next.js development server:
   ```bash
   npm run dev
   ```
Open `http://localhost:3000` in your browser to view the application workspace.

---

## Project Structure

```
├── backend/
│   ├── graph/             # LangGraph state definitions and agent nodes
│   ├── utils/
│   │   ├── http_cache.py  # SQLite cache for GitHub API requests
│   │   ├── session_db.py  # User active session manager
│   │   └── vector_db.py   # Text chunker, local similarity search, and hashing cache
│   ├── config.py          # Pydantic environment configurations
│   ├── main.py            # FastAPI main application endpoints
│   ├── onboard_github.py  # Concurrently fetches & formats repository trees
│   └── prompts.py         # Static system prompt templates
├── frontend/
│   ├── src/
│   │   ├── app/           # Next.js page layouts and global stylesheet
│   │   ├── components/    # Ingest, Sidebar, Chat, and Mermaid visual renderers
│   │   └── utils/         # Frontend API integration
│   └── package.json
└── requirements.txt       # Backend dependencies
```
