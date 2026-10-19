"""
Confidence gate.

REUSED, UNCHANGED: the weighted formula in score_claim() -- same three
signals (retrieval, rerank, NLI), same weights. This is the "reuse the
existing confidence calculation, do not redesign it" part of the spec.

CHANGED: the decision logic. The original version had two different rules
for English (SHOW/SHOW_WITH_CAVEAT/FLAG_FOR_REVIEW at 0.75/0.50) and Tamil
(strict >0.90). One threshold is applied identically to accept/reject a
claim regardless of which language the original query was in --
accept_claim() replaces both of the old decision functions.

THRESHOLD: 0.60 (lowered from an earlier 0.90). At 0.90, even a claim
with a PERFECT NLI entailment verdict (nli_component=1.0) still needed
meaningfully strong retrieval+rerank scores on top of that just to clear
the bar, since 0.60*1.0 = 0.60 alone isn't enough -- 0.90 required the
other two signals to contribute a combined 0.30 even when NLI was
maximally confident. 0.60 means a claim with solid entailment and
reasonably positive retrieval/rerank signals can pass without needing
every signal to be near-perfect simultaneously.
"""

NLI_WEIGHT = {"entailment": 1.0, "neutral": 0.4, "contradiction": 0.0}

W_RETRIEVAL = 0.15
W_RERANK = 0.25
W_NLI = 0.60

ACCEPTANCE_THRESHOLD = 0.60


def score_claim(claim):
    """claim must have: score (cosine sim, 0-1), rerank_score (raw logit), verdict.
    UNCHANGED from the original -- same formula, same weights."""
    retrieval_component = min(max(claim["score"], 0), 1.0)
    rerank_component = 1 / (1 + pow(2.718281828, -claim["rerank_score"]))
    nli_component = NLI_WEIGHT[claim["verdict"]]

    confidence = (W_RETRIEVAL * retrieval_component
                  + W_RERANK * rerank_component
                  + W_NLI * nli_component)
    return round(confidence, 3)


def accept_claim(confidence, rerank_score=None, threshold=ACCEPTANCE_THRESHOLD):
    """Single accept/reject rule, used for BOTH English and Tamil query paths.

    RELEVANCE FLOOR (added): a negative rerank_score means the cross-encoder
    itself judged this chunk NOT relevant to the query. NLI entailment alone
    cannot override that -- entailment measures whether a claim is
    self-consistent with its source chunk, not whether that chunk actually
    answers the query. Without this floor, a claim built from an irrelevant
    chunk can still clear the confidence threshold on NLI weight alone,
    regardless of which query or language caused retrieval to drift
    off-topic. This generalizes across all queries; it is not specific to
    any one term or domain.
    """
    if rerank_score is not None and rerank_score < 0:
        return False
    return confidence >= threshold
