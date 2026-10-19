"""
Dense retrieval against ChromaDB -- English side only, strictly scoped to
ONE standard's collection (standard isolation is enforced one layer below,
in db.get_collection()/get_standard_english_db(), never here).

Prints NOTHING. Dense retrieval is an internal candidate-generation step
per the spec -- the pipeline decides what's shown to the terminal, this
module just returns data.

SCORE COMPUTATION -- deliberately does NOT use Chroma's own "distance"
value. db.py's real collections (matching the actual ingestion script)
never set hnsw:space, so Chroma's internal metric is its default (squared
L2), not cosine -- and converting an L2 distance into a 0-1 similarity
score correctly requires knowing whether the stored vectors are
normalized, which isn't confirmed. Rather than guess, this module asks
Chroma to also return the raw stored vectors, then computes TRUE cosine
similarity directly in Python. This is correct regardless of which
internal metric Chroma's index is configured with, and regardless of
whether the stored vectors are normalized -- the formula below divides
by each vector's own norm explicitly, so it doesn't assume anything.
"""
import numpy as np
from sentence_transformers import SentenceTransformer

import db

EMBED_MODEL_NAME = "sentence-transformers/LaBSE"

_model = None


def _get_model():
    global _model
    if _model is None:
        _model = SentenceTransformer(EMBED_MODEL_NAME,device="cpu")
    return _model


def _cosine_similarity(a, b):
    """True cosine similarity between two raw vectors, no normalization
    assumed on either side -- divides by each vector's own norm explicitly."""
    a = np.asarray(a, dtype=np.float64)
    b = np.asarray(b, dtype=np.float64)
    denom = (np.linalg.norm(a) * np.linalg.norm(b)) + 1e-12  # avoid div-by-zero
    return float(np.dot(a, b) / denom)


def dense_search(english_query, standard, seed_k=10):
    """
    Retrieval-exclusion-avoidance flow, exactly as confirmed:
        1. dense search -> top seed_k candidates (true cosine similarity)
        2. collect every DISTINCT logical_section_id among those seeds
           (empty string "" is skipped -- that means "no section id was
           found for this record," not a real shared grouping; filtering
           on it would wrongly lump together every record with missing
           section data)
        3. Chroma metadata-filtered fetch: ALL chunks sharing ANY of
           those section ids, via collection.get(where={...,"$in":...})
        4. combine seed set + full section expansion, deduplicated by
           alignment_id (a seed chunk's own section expansion will
           re-include itself -- skipped, not recomputed)

    english_query: the query, already translated to English if needed.
    standard: 6, 7, or 8 -- selects which collection is searched.
    seed_k: how many top candidates seed the section expansion. 10,
            per the confirmed design -- not the old flat top_k=50.

    Returns the combined candidate pool: list of {"alignment_id",
    "source_id", "logical_section_id", "english", "score"}, ready for
    reranking. Pool size varies run to run -- it's seed_k plus however
    many additional chunks those sections' expansion pulls in.
    """
    collection = db.get_standard_english_db(standard)
    model = _get_model()
    q_emb = model.encode([english_query], convert_to_numpy=True,
                          normalize_embeddings=True)[0]

    results = collection.query(
        query_embeddings=[q_emb.tolist()],
        n_results=min(seed_k, collection.count()),
        include=["documents", "metadatas", "embeddings"],
    )

    candidates = {}
    section_ids = set()
    for doc, meta, emb in zip(results["documents"][0], results["metadatas"][0], results["embeddings"][0]):
        cos_sim = _cosine_similarity(q_emb, emb)
        sid = meta.get("logical_section_id", "")
        aid = meta["alignment_id"]
        candidates[aid] = {
            "alignment_id": aid,
            "source_id": meta["source_id"],
            "logical_section_id": sid,
            "english": doc,
            "score": max(0.0, cos_sim),
        }
        if sid:  # skip empty -- not a real section grouping
            section_ids.add(sid)

    if section_ids:
        expansion = collection.get(
            where={"logical_section_id": {"$in": list(section_ids)}},
            include=["documents", "metadatas", "embeddings"],
        )
        for doc, meta, emb in zip(expansion["documents"], expansion["metadatas"], expansion["embeddings"]):
            aid = meta["alignment_id"]
            if aid in candidates:
                continue  # already a seed -- don't duplicate or recompute
            cos_sim = _cosine_similarity(q_emb, emb)
            candidates[aid] = {
                "alignment_id": aid,
                "source_id": meta["source_id"],
                "logical_section_id": meta.get("logical_section_id", ""),
                "english": doc,
                "score": max(0.0, cos_sim),
            }

    return list(candidates.values())


def get_tamil_by_alignment_id(alignment_id, standard):
    """
    The stage-5 lookup: given an English result's alignment_id, fetch its
    Tamil counterpart from the SAME standard's Tamil collection, matched
    purely by alignment_id -- never by source_id, never by text similarity.

    Returns the Tamil text string, or None if no Tamil record exists for
    this alignment_id (a chunk whose Tamil side was missing at ingest time
    -- see ingest_from_json.py's tamil_embedding-is-None handling).
    """
    collection = db.get_standard_tamil_db(standard)
    result = collection.get(ids=[alignment_id], include=["documents"])
    if not result["documents"]:
        return None
    return result["documents"][0]
