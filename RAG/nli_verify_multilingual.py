"""
Multilingual NLI verification -- Stage 11.5 (Tamil path only).

PURPOSE, precisely: Stage 7's NLI (nli_verify.py) checks whether the
GENERATED ENGLISH CLAIM is supported by its English source. It never
touches Tamil. Stage 10's check_pair_quality() checks whether the
ORIGINAL CORPUS ALIGNMENT (english vs tamil, as stored) was correct.
Neither of these verifies the actual Tamil text that reaches the
student -- the output of Stage 11. This module fills that gap: it
checks whether the FINAL GENERATED TAMIL CLAIM preserved the meaning of
whatever it was derived from.

MODEL CHOICE: MoritzLaurer/mDeBERTa-v3-base-xnli-multilingual-nli-2mil7
-- confirmed via web search to include real Tamil training data (105,000
examples, language code 'ta') in its fine-tuning set, unlike the plain
XNLI-only mDeBERTa variant (standard XNLI's 15 languages do NOT include
Tamil). This is a different, stronger model than the one used in
nli_verify.py -- that one stays English-only, deliberately, since it
never needs to handle Tamil (see nli_verify.py's own docstring).

LABEL ORDER: read DYNAMICALLY from the model's own config
(model.config.id2label) rather than hardcoded. This is a deliberate
improvement over nli_verify.py's approach, where the label order had to
be stated as an assumption needing manual verification. Reading it from
the model's own config eliminates that guessing risk entirely for this
module.

WHICH PAIR GETS COMPARED depends on the alignment verdict from Stage 10,
confirmed with the user as the correct design:
    SAME              -> (tamil_evidence, tamil_claim)
                         same-language comparison: did the "correction"
                         step drift from what the real evidence said?
    PARTIAL or WRONG  -> (english_claim, tamil_claim)
                         cross-lingual comparison: did the fresh
                         translation preserve the claim's meaning?
This module itself is agnostic to that choice -- it only exposes
verify_pair(premise, hypothesis) and leaves the caller (pipeline_v2.py)
to decide which two strings to pass in, based on the verdict.
"""
import numpy as np
from sentence_transformers import CrossEncoder

MULTILINGUAL_NLI_MODEL_NAME = "MoritzLaurer/mDeBERTa-v3-base-xnli-multilingual-nli-2mil7"


class MultilingualNLIVerifier:
    def __init__(self):
        self.model = CrossEncoder(MULTILINGUAL_NLI_MODEL_NAME,device="cpu")
        # Read the REAL label order from the model's own config -- not
        # hardcoded. CrossEncoder wraps AutoModelForSequenceClassification,
        # whose .config.id2label is the model author's own ground truth.
        id2label = self.model.model.config.id2label
        self.raw_labels = [id2label[i] for i in range(len(id2label))]

    def verify_pair(self, premise, hypothesis):
        """
        premise, hypothesis: two text strings (language depends on caller --
        see module docstring for which pair to pass for SAME vs PARTIAL/WRONG).

        Returns (verdict, scores_dict).
        verdict is normalized to exactly one of: "entailment", "neutral",
        "contradiction" -- regardless of how this specific model's own
        labels happen to be spelled/capitalized (different NLI models use
        slightly different label strings, e.g. "ENTAILMENT" vs "entailment").
        """
        score_row = self.model.predict([(premise, hypothesis)])[0]
        probs = _softmax(score_row)
        winning_raw_label = self.raw_labels[int(probs.argmax())]
        verdict = _normalize_label(winning_raw_label)
        scores = {_normalize_label(lbl): float(p) for lbl, p in zip(self.raw_labels, probs)}
        return verdict, scores


def _softmax(x):
    e = np.exp(x - np.max(x))
    return e / e.sum()


def _normalize_label(raw_label):
    """Maps whatever this specific model calls its labels onto our 3
    canonical strings, used consistently throughout the rest of the
    pipeline (matches nli_verify.py's LABELS strings exactly)."""
    low = raw_label.lower()
    if "entail" in low:
        return "entailment"
    if "contradict" in low:
        return "contradiction"
    return "neutral"


if __name__ == "__main__":
    verifier = MultilingualNLIVerifier()
    print("Detected label order from model config:", verifier.raw_labels)
    print()

    # SAME-style test: same-language Tamil-vs-Tamil comparison
    verdict, scores = verifier.verify_pair(
        "ஒரு மின்சுற்றில் இரண்டு அல்லது அதற்கு மேற்பட்ட மின்விளக்குகள் தொடரில் இணைக்கப்பட்டால், "
        "அந்த வகை மின்சுற்று தொடர் மின்சுற்று எனப்படும்.",
        "தொடர் மின்சுற்று என்பது இரண்டு அல்லது அதற்கு மேற்பட்ட மின்விளக்குகள் தொடரில் இணைக்கப்பட்ட மின்சுற்று ஆகும்."
    )
    print(f"SAME-style test (tamil vs tamil): [{verdict.upper()}] {scores}")

    # PARTIAL/WRONG-style test: cross-lingual English-vs-Tamil comparison
    verdict2, scores2 = verifier.verify_pair(
        "A series circuit is a circuit in which two or more bulbs are connected one after another.",
        "தொடர் மின்சுற்று என்பது இரண்டு அல்லது அதற்கு மேற்பட்ட மின்விளக்குகள் தொடரில் இணைக்கப்பட்ட மின்சுற்று ஆகும்."
    )
    print(f"PARTIAL/WRONG-style test (english vs tamil): [{verdict2.upper()}] {scores2}")
