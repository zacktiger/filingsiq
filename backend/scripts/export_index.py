"""
Export the Qdrant server's collection to small files the deployed app loads.

    python backend/scripts/export_index.py            # -> data/index_export/

Why export instead of deploying Qdrant: the deployed app is one container with
no database server. It reads these files into an in-memory Qdrant at startup
(QDRANT_LOCAL_INDEX, see config.py). Nothing is re-embedded - the vectors are
copied exactly as the server holds them.

Vectors are stored as float16. That halves the file (~7.5 MB for 9,830 x 384)
so every file stays under Hugging Face's 10 MB no-LFS limit, and cosine
ranking is unaffected at that precision - check with the retrieval eval.
"""

import gzip
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from app.config import DATA_DIR, get_settings  # noqa: E402
from app.rag.store import assert_index_matches_model, get_client  # noqa: E402

OUT_DIR = DATA_DIR / "index_export"


def main() -> int:
    settings = get_settings()
    if settings.qdrant_local_index:
        raise SystemExit("Unset QDRANT_LOCAL_INDEX: export reads from the Qdrant server.")
    assert_index_matches_model()

    client = get_client()
    collection = settings.collection_name
    config = client.get_collection(collection).config.params.vectors
    # langchain-qdrant creates ONE named vector whose name is "" - a dict, not a
    # bare VectorParams. Recorded so the loader recreates exactly that shape;
    # QdrantVectorStore looks the vector up by this name.
    vector_name, params = next(iter(config.items())) if isinstance(config, dict) else ("", config)

    ids, payloads, vectors = [], [], []
    offset = None
    while True:
        points, offset = client.scroll(
            collection, limit=1000, offset=offset, with_payload=True, with_vectors=True
        )
        for point in points:
            ids.append(str(point.id))
            payloads.append(point.payload)
            vectors.append(point.vector)
        if offset is None:
            break

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    np.save(OUT_DIR / "vectors.npy", np.asarray(vectors, dtype=np.float16))
    with gzip.open(OUT_DIR / "points.jsonl.gz", "wt", encoding="utf-8") as fh:
        for point_id, payload in zip(ids, payloads):
            fh.write(json.dumps({"id": point_id, "payload": payload}, ensure_ascii=False) + "\n")
    (OUT_DIR / "meta.json").write_text(
        json.dumps(
            {
                "collection": collection,
                "embedding_model": settings.embedding_model,
                "vector_name": vector_name,
                "dim": params.size,
                "distance": params.distance.value,
                "count": len(ids),
            },
            indent=2,
        ),
        encoding="utf-8",
    )

    for path in sorted(OUT_DIR.iterdir()):
        print(f"  {path.name:18} {path.stat().st_size / 1e6:6.2f} MB")
    print(f"Exported {len(ids)} points from {collection!r} to {OUT_DIR}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
