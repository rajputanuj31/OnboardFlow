import sqlite3
import json
import math
import os
from pathlib import Path
from langchain_openai import OpenAIEmbeddings
from config import settings

DB_DIR = Path(__file__).resolve().parent.parent / ".cache"
DB_PATH = DB_DIR / "sessions.db"

def init_vector_db():
    """Ensure sessions.db exists and create code_chunks table."""
    os.makedirs(DB_DIR, exist_ok=True)
    with sqlite3.connect(DB_PATH) as conn:
        conn.execute("""
            CREATE TABLE IF NOT EXISTS code_chunks (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                session_id TEXT,
                filepath TEXT,
                chunk_index INTEGER,
                content TEXT,
                embedding TEXT,
                file_hash TEXT,
                FOREIGN KEY (session_id) REFERENCES sessions (session_id) ON DELETE CASCADE
            )
        """)
        # Schema migration: Add file_hash column if it doesn't exist
        try:
            conn.execute("ALTER TABLE code_chunks ADD COLUMN file_hash TEXT")
        except sqlite3.OperationalError:
            pass  # Already exists
        conn.commit()

def chunk_text(text: str, chunk_size: int = 1000, overlap: int = 200) -> list[str]:
    """Line-aware chunker that splits text into chunks of ~chunk_size characters with overlap."""
    if not text or not text.strip():
        return []
        
    lines = text.splitlines(keepends=True)
    chunks = []
    current_chunk = []
    current_length = 0
    
    for line in lines:
        if current_length + len(line) > chunk_size and current_chunk:
            chunks.append("".join(current_chunk))
            # Keep overlap lines
            overlap_chunk = []
            overlap_len = 0
            for l in reversed(current_chunk):
                if overlap_len + len(l) <= overlap:
                    overlap_chunk.insert(0, l)
                    overlap_len += len(l)
                else:
                    break
            current_chunk = overlap_chunk
            current_length = overlap_len
            
        current_chunk.append(line)
        current_length += len(line)
        
    if current_chunk:
        chunks.append("".join(current_chunk))
    return chunks

def _embeddings_model(api_key: str = ""):
    key = api_key or settings.openai_api_key
    return OpenAIEmbeddings(
        model="text-embedding-3-small", 
        api_key=key
    )

async def save_repo_chunks(session_id: str, files: dict[str, str], readme: str = "", contributing: str = "", model_api_key: str = ""):
    """Chunk and embed all fetched repository files, saving to SQLite with content-hash caching."""
    init_vector_db()

    # Compile all current files
    all_files = {}
    if readme:
        all_files["README"] = readme
    if contributing:
        all_files["CONTRIBUTING"] = contributing
    all_files.update(files)

    # Compute hashes of current files
    import hashlib
    hashes = {}
    for filepath, content in all_files.items():
        if content:
            hashes[filepath] = hashlib.sha256(content.encode("utf-8")).hexdigest()

    # Retrieve existing hashes from the database for this session
    db_hashes = {}
    with sqlite3.connect(DB_PATH) as conn:
        cursor = conn.execute(
            "SELECT DISTINCT filepath, file_hash FROM code_chunks WHERE session_id = ?",
            (session_id,)
        )
        for filepath, f_hash in cursor.fetchall():
            if filepath and f_hash:
                db_hashes[filepath] = f_hash

    # Determine files to delete (deleted, or modified files whose hash has changed)
    files_to_delete = []
    for filepath in db_hashes:
        if filepath not in all_files or hashes.get(filepath) != db_hashes[filepath]:
            files_to_delete.append(filepath)

    if files_to_delete:
        with sqlite3.connect(DB_PATH) as conn:
            placeholders = ",".join("?" for _ in files_to_delete)
            conn.execute(
                f"DELETE FROM code_chunks WHERE session_id = ? AND filepath IN ({placeholders})",
                (session_id, *files_to_delete)
            )
            conn.commit()

    # Determine files to chunk and embed (new files, or modified files)
    files_to_embed = {}
    for filepath, content in all_files.items():
        if not content:
            continue
        if filepath not in db_hashes or hashes[filepath] != db_hashes[filepath]:
            files_to_embed[filepath] = content

    if not files_to_embed:
        # Cache hit: all files are unchanged
        return

    # Chunk and embed the new/modified files
    chunks_to_insert = []
    texts_to_embed = []

    for filepath, content in files_to_embed.items():
        chunks = chunk_text(content)
        for i, chunk in enumerate(chunks):
            texts_to_embed.append(chunk)
            chunks_to_insert.append((filepath, i, chunk, hashes[filepath]))

    if not texts_to_embed:
        return

    # Compute embeddings in batch
    emb_model = _embeddings_model(model_api_key)
    embeddings = await emb_model.aembed_documents(texts_to_embed)

    # Insert new chunks into DB
    with sqlite3.connect(DB_PATH) as conn:
        for (filepath, chunk_idx, chunk, file_hash), embedding in zip(chunks_to_insert, embeddings):
            embedding_json = json.dumps(embedding)
            conn.execute(
                "INSERT INTO code_chunks (session_id, filepath, chunk_index, content, embedding, file_hash) VALUES (?, ?, ?, ?, ?, ?)",
                (session_id, filepath, chunk_idx, chunk, embedding_json, file_hash)
            )
        conn.commit()

def cosine_similarity(v1: list[float], v2: list[float]) -> float:
    if not v1 or not v2 or len(v1) != len(v2):
        return 0.0
    dot_product = sum(x * y for x, y in zip(v1, v2))
    norm1 = math.sqrt(sum(x * x for x in v1))
    norm2 = math.sqrt(sum(y * y for y in v2))
    if norm1 == 0.0 or norm2 == 0.0:
        return 0.0
    return dot_product / (norm1 * norm2)

async def query_vector_db(session_id: str, query: str, top_k: int = 4, model_api_key: str = "") -> list[dict]:
    """Retrieve top K most similar chunks for the session."""
    if not query or not query.strip():
        return []

    # Get query embedding
    emb_model = _embeddings_model(model_api_key)
    query_embedding = await emb_model.aembed_query(query)

    # Retrieve all chunks for session
    init_vector_db()
    rows = []
    with sqlite3.connect(DB_PATH) as conn:
        cursor = conn.execute(
            "SELECT filepath, chunk_index, content, embedding FROM code_chunks WHERE session_id = ?",
            (session_id,)
        )
        rows = cursor.fetchall()

    if not rows:
        return []

    # Calculate similarities
    scored_chunks = []
    for filepath, chunk_idx, content, embedding_json in rows:
        try:
            embedding = json.loads(embedding_json)
            sim = cosine_similarity(query_embedding, embedding)
            scored_chunks.append({
                "filepath": filepath,
                "chunk_index": chunk_idx,
                "content": content,
                "similarity": sim
            })
        except Exception:
            continue

    # Sort and return top K
    scored_chunks.sort(key=lambda x: x["similarity"], reverse=True)
    return scored_chunks[:top_k]
