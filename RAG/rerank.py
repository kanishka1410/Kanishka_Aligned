"""
Cross-encoder reranking -- UNCHANGED LOGIC from the original rerank.py,
only the dedup key changed from the old flat "id" field to "alignment_id"
(candidates now come from retrieve_chroma.py, which uses alignment_id as
the per-record identifier rather than the old file-based index's "id").

English-only cross-encoder, deliberately -- see the original reasoning:
mMARCO's Tamil coverage couldn't be confirmed, so the query is translated
to English upstream before reaching this stage.
"""
from sentence_transformers import CrossEncoder

RERANK_MODEL_NAME = "cross-encoder/ms-marco-MiniLM-L-6-v2"


class Reranker:
    def __init__(self):
        self.model = CrossEncoder(RERANK_MODEL_NAME,device="cpu")

    def rerank(self, query_en, candidates, top_k=10):
        """query_en: query, already translated to English if it originated in Tamil/Tanglish."""
        pairs = [(query_en, c["english"]) for c in candidates]
        scores = self.model.predict(pairs)  # raw relevance logits

        best = {}
        for c, s in zip(candidates, scores):
            s = float(s)
            if c["alignment_id"] not in best or s > best[c["alignment_id"]]["rerank_score"]:
                best[c["alignment_id"]] = {**c, "rerank_score": s}

        ranked = sorted(best.values(), key=lambda x: -x["rerank_score"])
        return ranked[:top_k]
