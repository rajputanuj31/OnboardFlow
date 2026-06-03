import sqlite3
import json
import logging
import time
import hashlib
import httpx
from pathlib import Path

logger = logging.getLogger(__name__)

CACHE_DB = Path(__file__).resolve().parent.parent / ".cache" / "http_cache.db"

class SQLiteCache:
    def __init__(self, db_path=CACHE_DB):
        self.db_path = db_path
        self._init_db()

    def _init_db(self):
        try:
            self.db_path.parent.mkdir(parents=True, exist_ok=True)
            with sqlite3.connect(self.db_path) as conn:
                conn.execute("""
                    CREATE TABLE IF NOT EXISTS http_cache (
                        key TEXT PRIMARY KEY,
                        value TEXT,
                        status_code INTEGER,
                        headers TEXT,
                        created_at REAL,
                        ttl REAL
                    )
                """)
                conn.commit()
        except Exception as e:
            logger.warning("HTTP cache init failed: %s", e)

    def get(self, key: str) -> tuple[str, int, dict] | None:
        try:
            with sqlite3.connect(self.db_path, timeout=5.0) as conn:
                cursor = conn.cursor()
                cursor.execute(
                    "SELECT value, status_code, headers, created_at, ttl FROM http_cache WHERE key = ?",
                    (key,)
                )
                row = cursor.fetchone()
                if not row:
                    return None
                value, status_code, headers_json, created_at, ttl = row
                if time.time() > created_at + ttl:
                    # Expired
                    cursor.execute("DELETE FROM http_cache WHERE key = ?", (key,))
                    conn.commit()
                    return None
                return value, status_code, json.loads(headers_json)
        except Exception as e:
            logger.warning("HTTP cache get error: %s", e)
            return None

    def set(self, key: str, value: str, status_code: int, headers: dict, ttl: float):
        try:
            with sqlite3.connect(self.db_path, timeout=5.0) as conn:
                conn.execute(
                    """
                    INSERT OR REPLACE INTO http_cache (key, value, status_code, headers, created_at, ttl)
                    VALUES (?, ?, ?, ?, ?, ?)
                    """,
                    (key, value, status_code, json.dumps(headers), time.time(), ttl)
                )
                conn.commit()
        except Exception as e:
            logger.warning("HTTP cache set error: %s", e)

_cache = SQLiteCache()

def make_cache_key(url: str, params: dict | None = None) -> str:
    if params:
        # Sort keys to ensure deterministic cache hits
        sorted_params = sorted((str(k), str(v)) for k, v in params.items())
        param_str = json.dumps(sorted_params)
    else:
        param_str = ""
    raw = f"{url}||{param_str}"
    return hashlib.sha256(raw.encode("utf-8")).hexdigest()

async def cached_get(
    client: httpx.AsyncClient,
    url: str,
    headers: dict | None = None,
    params: dict | None = None,
    ttl: float = 3600.0
) -> httpx.Response:
    """
    Perform an async GET request with cache check and fallback to live call on miss.
    Only caches status 200 responses.
    """
    request = client.build_request("GET", url, headers=headers, params=params)
    key = make_cache_key(url, params)
    cached = _cache.get(key)
    
    if cached is not None:
        value, status_code, cached_headers = cached
        # Remove compression/length headers to prevent httpx from attempting decompression
        cleaned_headers = {
            k: v for k, v in cached_headers.items()
            if k.lower() not in ("content-encoding", "transfer-encoding", "content-length")
        }
        return httpx.Response(
            status_code=status_code,
            headers=httpx.Headers(cleaned_headers),
            content=value.encode("utf-8"),
            request=request,
        )
    
    # Cache miss: request live content
    response = await client.send(request)
    
    if response.status_code == 200:
        _cache.set(
            key,
            response.text,
            response.status_code,
            dict(response.headers),
            ttl
        )
        
    return response
