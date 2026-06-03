"""Qdrant round-trip with mocked embeddings (no OpenAI calls)."""
from unittest.mock import patch

import pytest


class _FakeEmb:
    """Deterministic toy embeddings — first three dims encode a 'theme'."""

    async def aembed_documents(self, texts):
        out = []
        for t in texts:
            v = [0.0] * 1536
            v[3] = 0.5
            if "login" in t or "password" in t:
                v[0] = 1.0
            if "hash" in t:
                v[1] = 1.0
            out.append(v)
        return out

    async def aembed_query(self, text):
        v = [0.0] * 1536
        v[3] = 0.5
        if "login" in text or "auth" in text or "password" in text:
            v[0] = 1.0
        if "hash" in text:
            v[1] = 1.0
        return v


@pytest.fixture
def fake_embeddings():
    with patch("utils.vector_db._embeddings_model", return_value=_FakeEmb()):
        yield


@pytest.mark.asyncio
async def test_save_then_query_returns_relevant_chunk(fresh_qdrant_path, fake_embeddings):
    from utils.vector_db import query_vector_db, save_repo_chunks

    files = {
        "auth.py": "def login(user, password):\n    return password == 'secret'\n",
        "utils.py": "def hash_password(p):\n    import hashlib; return hashlib.sha256(p.encode()).hexdigest()\n",
    }
    await save_repo_chunks("s1", files)
    results = await query_vector_db("s1", "how does login work", top_k=2)

    assert results, "expected at least one chunk"
    assert results[0]["filepath"] == "auth.py"


@pytest.mark.asyncio
async def test_cache_hit_when_files_unchanged(fresh_qdrant_path, fake_embeddings):
    from utils.vector_db import COLLECTION_NAME, _get_client, save_repo_chunks

    files = {"a.py": "def login(): pass\n", "b.py": "def hash_password(): pass\n"}
    await save_repo_chunks("s1", files)
    before = _get_client().count(COLLECTION_NAME, exact=True).count
    await save_repo_chunks("s1", files)
    after = _get_client().count(COLLECTION_NAME, exact=True).count

    assert before == after and before > 0


@pytest.mark.asyncio
async def test_session_isolation(fresh_qdrant_path, fake_embeddings):
    from utils.vector_db import query_vector_db, save_repo_chunks

    await save_repo_chunks("s1", {"auth.py": "def login(): pass"})
    await save_repo_chunks("s2", {"other.py": "print(1)"})

    r1 = await query_vector_db("s1", "login", top_k=5)
    r2 = await query_vector_db("s2", "login", top_k=5)

    s1_files = {x["filepath"] for x in r1}
    s2_files = {x["filepath"] for x in r2}
    assert "auth.py" in s1_files
    assert "auth.py" not in s2_files


@pytest.mark.asyncio
async def test_cleanup_session_removes_all_chunks(fresh_qdrant_path, fake_embeddings):
    from utils.vector_db import cleanup_session, query_vector_db, save_repo_chunks

    await save_repo_chunks("s1", {"x.py": "def login(): pass"})
    assert (await query_vector_db("s1", "login", top_k=5))
    cleanup_session("s1")
    assert (await query_vector_db("s1", "login", top_k=5)) == []


@pytest.mark.asyncio
async def test_read_file_chunks_reassembles_in_order(fresh_qdrant_path, fake_embeddings):
    from utils.vector_db import read_file_chunks, save_repo_chunks

    long_file = "\n".join(f"line {i} login password" for i in range(120))
    await save_repo_chunks("s1", {"long.py": long_file})
    joined = read_file_chunks("s1", "long.py")

    assert "line 0" in joined and "line 119" in joined
    assert joined.index("line 0") < joined.index("line 119")


def test_chunk_text_basic():
    from utils.vector_db import chunk_text

    text = "\n".join(f"line {i}" for i in range(50))
    chunks = chunk_text(text, chunk_size=80, overlap=20)
    assert len(chunks) >= 2
    assert all(c for c in chunks)
