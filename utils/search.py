"""
Semantic search over Milvus embeddings with PostgreSQL metadata.

Collections are created by the pipeline (see utils/aud_db_utils.py): VARCHAR ids,
768-d COSINE embeddings (all-mpnet-base-v2). Use QUERY_EMBEDDING_MODEL from config
when querying so vectors match insert-time embeddings.
"""
from __future__ import annotations

import json
import os
import re
import warnings
from typing import Any, List, Optional, Sequence, Tuple

import pandas as pd
import psycopg2
from pymilvus import Collection, connections
from sentence_transformers import SentenceTransformer

warnings.filterwarnings("ignore")

_IDENTIFIER = re.compile(r"^[a-zA-Z_][a-zA-Z0-9_]*$")

_model: Optional[SentenceTransformer] = None
_milvus_connected = False


def _get_embedding_model_name() -> str:
    try:
        import config

        return getattr(config, "QUERY_EMBEDDING_MODEL", config.EMBEDDING_MODEL)
    except Exception:
        return os.getenv("QUERY_EMBEDDING_MODEL", "sentence-transformers/all-mpnet-base-v2")


def get_embedding_model() -> SentenceTransformer:
    global _model
    if _model is None:
        _model = SentenceTransformer(_get_embedding_model_name())
    return _model


def _pg_settings() -> dict:
    """Prefer environment (e.g. .env); fall back to config.py."""
    if os.getenv("DB_NAME") or os.getenv("HOST"):
        return {
            "dbname": os.environ["DB_NAME"],
            "user": os.environ["DB_USER"],
            "password": os.environ["PASSWORD"],
            "host": os.environ["HOST"],
            "port": os.environ["PORT"],
        }
    import config

    return {
        "dbname": config.DB_NAME,
        "user": config.DB_USER,
        "password": config.PASSWORD,
        "host": config.HOST,
        "port": config.PORT,
    }


def _milvus_host_port() -> Tuple[str, str]:
    if os.getenv("MILVUS_HOST"):
        return os.environ["MILVUS_HOST"], os.environ["MILVUS_PORT"]
    import config

    return config.MILVUS_HOST, str(config.MILVUS_PORT)


def ensure_milvus() -> None:
    global _milvus_connected
    if not _milvus_connected:
        host, port = _milvus_host_port()
        connections.connect("default", host=host, port=port)
        _milvus_connected = True


def validate_sql_identifier(name: str) -> str:
    if not name or not _IDENTIFIER.match(name):
        raise ValueError(f"Invalid SQL identifier: {name!r} (use letters, digits, underscore)")
    return name


def get_pg_conn():
    return psycopg2.connect(**_pg_settings())


def search_milvus(
    collection_name: str,
    embedding: List[float],
    top_k: int = 5,
    output_fields: Optional[Sequence[str]] = None,
) -> List[Tuple[Any, float]]:
    """Return list of (id, distance) from Milvus, best-first."""
    ensure_milvus()
    validate_sql_identifier(collection_name)
    fields = list(output_fields) if output_fields else ["id"]
    collection = Collection(collection_name)
    collection.load()
    results = collection.search(
        data=[embedding],
        anns_field="embedding",
        param={"metric_type": "COSINE", "params": {"nprobe": 10}},
        limit=top_k,
        output_fields=fields,
    )
    out: List[Tuple[Any, float]] = []
    for hits in results:
        for hit in hits:
            entity_id = hit.id
            if entity_id is None and hit.entity:
                entity_id = hit.entity.get("id")
            out.append((entity_id, float(hit.distance)))
    return out


def fetch_metadata(ids: Sequence[Any], table: str) -> pd.DataFrame:
    if not ids:
        return pd.DataFrame()
    table_safe = validate_sql_identifier(table)
    conn = get_pg_conn()
    placeholders = ",".join(["%s"] * len(ids))
    q = f"SELECT * FROM {table_safe} WHERE id IN ({placeholders});"
    df = pd.read_sql(q, conn, params=list(ids))
    conn.close()
    return df


def semantic_search(
    query_text: str,
    collection_name: str,
    table_name: str,
    top_k: int = 10,
    model: Optional[SentenceTransformer] = None,
) -> dict:
    """
    Embed query_text, search Milvus, join rows from PostgreSQL `table_name` by id.

    Returns dict with keys: query, collection, table, hits (list of {id, distance, row}).
    """
    m = model or get_embedding_model()
    emb = m.encode(query_text).tolist()
    ranked = search_milvus(collection_name, emb, top_k=top_k)
    if not ranked:
        return {
            "query": query_text,
            "collection": collection_name,
            "table": table_name,
            "hits": [],
            "query_embedding_dim": len(emb),
        }
    ids = [r[0] for r in ranked]
    df = fetch_metadata(ids, table_name)
    by_id = {str(row["id"]): row.to_dict() for _, row in df.iterrows()}
    hits = []
    for eid, dist in ranked:
        key = str(eid)
        row = by_id.get(key)
        hits.append({"id": eid, "distance": dist, "row": row})
    return {
        "query": query_text,
        "collection": collection_name,
        "table": table_name,
        "hits": hits,
        "query_embedding_dim": len(emb),
    }


def multimodal_search(
    query_text: str,
    top_k: int = 5,
    audio_collection: str = "milvus_audio_table",
    video_collection: str = "milvus_frame_table",
    audio_table: str = "pg_audio_table",
    video_table: str = "pg_frame_table",
) -> dict:
    """Search default audio + video collection/table names (legacy helper)."""
    m = get_embedding_model()
    emb = m.encode(query_text).tolist()
    audio_ranked = search_milvus(audio_collection, emb, top_k=top_k)
    video_ranked = search_milvus(video_collection, emb, top_k=top_k)
    audio_ids = [x[0] for x in audio_ranked]
    video_ids = [x[0] for x in video_ranked]
    audio_df = fetch_metadata(audio_ids, audio_table)
    video_df = fetch_metadata(video_ids, video_table)
    return {
        "query_embedding_dim": len(emb),
        "audio_results": audio_df.to_dict(orient="records"),
        "video_results": video_df.to_dict(orient="records"),
    }


if __name__ == "__main__":
    while True:
        q = input("Enter search query (empty to exit): ").strip()
        if not q:
            break
        out = multimodal_search(q, top_k=2)
        print(json.dumps(out, indent=2, default=str))
