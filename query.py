#!/usr/bin/env python3
"""
CLI for semantic search over Milvus + PostgreSQL (see README "Query" section).

Example:
  python query.py "dialogue about family" \\
    --collection viacom18_movies_hindi_None_audio \\
    --table viacom18_movies_hindi_None_audio \\
    --top-k 5
"""
from __future__ import annotations

import argparse
import json
import sys

from utils.search import semantic_search


def main() -> int:
    p = argparse.ArgumentParser(description="Semantic search over embedded chunks (Milvus + Postgres).")
    p.add_argument("text", nargs="?", help="Natural-language query")
    p.add_argument(
        "--collection",
        "-c",
        required=True,
        help="Milvus collection name (must match pipeline insert, e.g. network_mediatype_lang_channel_audio)",
    )
    p.add_argument(
        "--table",
        "-t",
        required=True,
        help="PostgreSQL table with the same chunk ids as Milvus",
    )
    p.add_argument("--top-k", "-k", type=int, default=10, help="Number of hits (default: 10)")
    p.add_argument("--json", action="store_true", help="Print JSON only")
    args = p.parse_args()

    query = args.text
    if not query or not query.strip():
        query = sys.stdin.read().strip()
    if not query:
        p.error("Provide query text as an argument or on stdin")

    out = semantic_search(
        query.strip(),
        collection_name=args.collection,
        table_name=args.table,
        top_k=args.top_k,
    )

    if args.json:
        print(json.dumps(out, indent=2, default=str))
        return 0

    print(f"Query: {out['query']}")
    print(f"Collection: {out['collection']} | Table: {out['table']} | dim={out['query_embedding_dim']}\n")
    for i, hit in enumerate(out["hits"], 1):
        print(f"--- Hit {i} | id={hit['id']!r} | distance={hit['distance']:.4f}")
        row = hit.get("row")
        if row is None:
            print("  (no PostgreSQL row for this id)\n")
            continue
        summary = row.get("content_summary") or row.get("description") or ""
        if summary:
            preview = str(summary)[:500]
            print(f"  summary: {preview}{'...' if len(str(summary)) > 500 else ''}")
        tr = row.get("transcript_full_text")
        if tr:
            t = str(tr)[:400]
            print(f"  transcript: {t}{'...' if len(str(tr)) > 400 else ''}")
        print()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
