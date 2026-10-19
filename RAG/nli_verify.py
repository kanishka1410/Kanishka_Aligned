"""
NLI verification: for each generated claim, check whether its source chunk
actually ENTAILS it (vs NEUTRAL/unsupported or CONTRADICTION).

Uses a dedicated NLI cross-encoder, separate from the reranker -- reranking
measures relevance, NLI measures whether the claim is actually supported.
"""
from sentence_transformers import CrossEncoder

NLI_MODEL_NAME = "cross-encoder/nli-deberta-v3-base"
# this model's output order is [contradiction, entailment, neutral]
LABELS = ["contradiction", "entailment", "neutral"]


class NLIVerifier:
    
    def __init__(self):
        self.model = CrossEncoder(NLI_MODEL_NAME,device="cpu")

    def verify(self, claims):
        """
        claims: list of {"alignment_id": ..., "claim": ..., "source_english": ...}
        Returns the same list with "verdict" and "nli_scores" added. Keys
        other than "claim"/"source_english" are passed through unchanged
        (via **c), so this works regardless of what identifier field the
        caller uses -- no hardcoded dependency on "alignment_id" vs "id".
        """
        pairs = [(c["source_english"], c["claim"]) for c in claims]  # (premise, hypothesis)
        scores = self.model.predict(pairs)  # shape (N, 3), raw logits

        results = []
        for c, score_row in zip(claims, scores):
            probs = _softmax(score_row)
            verdict = LABELS[probs.argmax()]
            results.append({
                **c,
                "verdict": verdict,
                "nli_scores": {label: float(p) for label, p in zip(LABELS, probs)},
            })
        return results


def _softmax(x):
    import numpy as np
    e = np.exp(x - np.max(x))
    return e / e.sum()


if __name__ == "__main__":
    verifier = NLIVerifier()
    test_claims = [
        {"alignment_id": "A000001",
         "claim": "A series circuit stops working entirely if one bulb is damaged.",
         "source_english": "If any one of the bulbs is damaged or disconnected, the entire circuit will not work."},
        {"alignment_id": "A000002",
         "claim": "Parallel circuits are more expensive than series circuits.",
         "source_english": "If two or more bulbs are connected in parallel in a circuit, then that type of circuit is called parallel circuit."},
    ]
    for r in verifier.verify(test_claims):
        print(f"[{r['verdict'].upper()}] {r['claim']}")
        print(f"  scores: {r['nli_scores']}")
