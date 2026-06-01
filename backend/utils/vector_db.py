import hashlib
from uuid import uuid4
from langchain_openai import OpenAIEmbeddings
from qdrant_client import QdrantClient
from qdrant_client.http import models as qmodels
from qdrant_client.http.exceptions import UnexpectedResponse
from config import settings

COLLECTION_NAME = "code_chunks"
VECTOR_SIZE = 1536  # text-embedding-3-small

_client: QdrantClient | None = None


def _get_client() -> QdrantClient:
    """Connect to Qdrant. `qdrant_url` may be a server URL (http://...) or a local
    filesystem path / ":memory:" for embedded mode (no Docker required)."""
    global _client
    if _client is None:
        url = settings.qdrant_url
        if url.startswith("http://") or url.startswith("https://"):
            _client = QdrantClient(url=url)
        else:
            _client = QdrantClient(path=url)
    return _client


def init_vector_db():
    """Ensure Qdrant collection + payload indexes exist. Idempotent."""
    client = _get_client()
    try:
        client.create_collection(
            collection_name=COLLECTION_NAME,
            vectors_config=qmodels.VectorParams(size=VECTOR_SIZE, distance=qmodels.Distance.COSINE),
        )
    except (UnexpectedResponse, ValueError):
        pass  # Already exists

    for field, schema in (("session_id", qmodels.PayloadSchemaType.KEYWORD),
                          ("filepath", qmodels.PayloadSchemaType.KEYWORD)):
        try:
            client.create_payload_index(
                collection_name=COLLECTION_NAME,
                field_name=field,
                field_schema=schema,
            )
        except (UnexpectedResponse, ValueError):
            pass


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
        api_key=key,
    )


def _session_filter(session_id: str) -> qmodels.Filter:
    return qmodels.Filter(
        must=[qmodels.FieldCondition(key="session_id", match=qmodels.MatchValue(value=session_id))]
    )


def _scroll_all(client: QdrantClient, scroll_filter: qmodels.Filter, with_payload: bool = True) -> list[qmodels.Record]:
    """Scroll through all points matching a filter, paginating as needed."""
    out: list[qmodels.Record] = []
    offset = None
    while True:
        points, offset = client.scroll(
            collection_name=COLLECTION_NAME,
            scroll_filter=scroll_filter,
            with_payload=with_payload,
            with_vectors=False,
            limit=256,
            offset=offset,
        )
        out.extend(points)
        if offset is None:
            break
    return out


async def save_repo_chunks(session_id: str, files: dict[str, str], readme: str = "", contributing: str = "", model_api_key: str = ""):
    """Chunk and embed all fetched repository files, upserting to Qdrant with content-hash caching."""
    init_vector_db()
    client = _get_client()

    all_files: dict[str, str] = {}
    if readme:
        all_files["README"] = readme
    if contributing:
        all_files["CONTRIBUTING"] = contributing
    all_files.update(files)

    hashes: dict[str, str] = {}
    for filepath, content in all_files.items():
        if content:
            hashes[filepath] = hashlib.sha256(content.encode("utf-8")).hexdigest()

    # Existing hashes for this session
    existing = _scroll_all(client, _session_filter(session_id))
    db_hashes: dict[str, str] = {}
    for rec in existing:
        payload = rec.payload or {}
        fp = payload.get("filepath")
        fh = payload.get("file_hash")
        if fp and fh:
            db_hashes[fp] = fh

    files_to_delete = [
        fp for fp in db_hashes
        if fp not in all_files or hashes.get(fp) != db_hashes[fp]
    ]

    if files_to_delete:
        client.delete(
            collection_name=COLLECTION_NAME,
            points_selector=qmodels.FilterSelector(
                filter=qmodels.Filter(
                    must=[
                        qmodels.FieldCondition(key="session_id", match=qmodels.MatchValue(value=session_id)),
                        qmodels.FieldCondition(key="filepath", match=qmodels.MatchAny(any=files_to_delete)),
                    ]
                )
            ),
        )

    files_to_embed = {
        fp: c for fp, c in all_files.items()
        if c and (fp not in db_hashes or hashes[fp] != db_hashes[fp])
    }

    if not files_to_embed:
        return  # Cache hit

    points_meta: list[tuple[str, int, str, str]] = []
    texts_to_embed: list[str] = []
    for filepath, content in files_to_embed.items():
        for i, chunk in enumerate(chunk_text(content)):
            texts_to_embed.append(chunk)
            points_meta.append((filepath, i, chunk, hashes[filepath]))

    if not texts_to_embed:
        return

    emb_model = _embeddings_model(model_api_key)
    embeddings = await emb_model.aembed_documents(texts_to_embed)

    points = [
        qmodels.PointStruct(
            id=str(uuid4()),
            vector=emb,
            payload={
                "session_id": session_id,
                "filepath": filepath,
                "chunk_index": chunk_idx,
                "content": content,
                "file_hash": file_hash,
            },
        )
        for (filepath, chunk_idx, content, file_hash), emb in zip(points_meta, embeddings)
    ]

    client.upsert(collection_name=COLLECTION_NAME, points=points)


async def query_vector_db(session_id: str, query: str, top_k: int = 4, model_api_key: str = "") -> list[dict]:
    """Retrieve top K most similar chunks for the session via Qdrant ANN."""
    if not query or not query.strip():
        return []

    init_vector_db()
    client = _get_client()

    emb_model = _embeddings_model(model_api_key)
    query_embedding = await emb_model.aembed_query(query)

    result = client.query_points(
        collection_name=COLLECTION_NAME,
        query=query_embedding,
        query_filter=_session_filter(session_id),
        limit=top_k,
        with_payload=True,
    )

    out: list[dict] = []
    for point in result.points:
        payload = point.payload or {}
        out.append({
            "filepath": payload.get("filepath", ""),
            "chunk_index": payload.get("chunk_index", 0),
            "content": payload.get("content", ""),
            "similarity": point.score,
        })
    return out


def read_file_chunks(session_id: str, filepath: str) -> str:
    """Reassemble all chunks for a single file under a session, ordered by chunk_index."""
    init_vector_db()
    client = _get_client()
    records = _scroll_all(
        client,
        qmodels.Filter(
            must=[
                qmodels.FieldCondition(key="session_id", match=qmodels.MatchValue(value=session_id)),
                qmodels.FieldCondition(key="filepath", match=qmodels.MatchValue(value=filepath)),
            ]
        ),
    )
    if not records:
        return ""
    ordered = sorted(records, key=lambda r: (r.payload or {}).get("chunk_index", 0))
    return "".join((r.payload or {}).get("content", "") for r in ordered)


def cleanup_session(session_id: str) -> None:
    """Delete all chunks for a given session. Used when a session is removed."""
    init_vector_db()
    client = _get_client()
    client.delete(
        collection_name=COLLECTION_NAME,
        points_selector=qmodels.FilterSelector(filter=_session_filter(session_id)),
    )
