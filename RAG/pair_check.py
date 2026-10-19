"""
Stage 10 pair check without an LLM: LaBSE similarity between a chunk's
English and Tamil text. Thresholds from eval_models.py on 275 logged pairs:
0.74 = best SAME / not-SAME cut; 0.55 = mean of pairs Qwen called WRONG.
PARTIAL and WRONG are handled the same downstream.
"""
import retrieve_chroma

SAME_THRESHOLD = 0.74
PARTIAL_THRESHOLD = 0.55


def classify_pairs(pairs):
    """pairs: list of (english, tamil). Returns list of (verdict, similarity)."""
    if not pairs:
        return []
    model = retrieve_chroma._get_model()
    en = model.encode([p[0] for p in pairs], convert_to_numpy=True, normalize_embeddings=True)
    ta = model.encode([p[1] for p in pairs], convert_to_numpy=True, normalize_embeddings=True)
    results = []
    for e, t in zip(en, ta):
        sim = float((e * t).sum())
        if sim >= SAME_THRESHOLD:
            verdict = "SAME"
        elif sim >= PARTIAL_THRESHOLD:
            verdict = "PARTIAL"
        else:
            verdict = "WRONG"
        results.append((verdict, sim))
    return results