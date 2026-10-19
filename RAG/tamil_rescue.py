"""
TAMIL CHECK (Tamil-script queries only).

For every Tamil question, the textbook's own Tamil text is compared with
the Tamil question by a multilingual reranker (BAAI/bge-reranker-v2-m3),
with no translation involved. Candidates = the translation search's
top-10 + a Tamil-side search + an English-side search.
Measured on 44 Tamil questions (eval_combined.py): 36/41 hits (35
before), 134 relevant lines (123 before), no extra off-topic lines,
both out-of-book questions still refused. About 6 s per question on CPU.
"""
import math

import numpy as np

import db
import re
import retrieve_chroma

RESCUE_MODEL_NAME = "BAAI/bge-reranker-v2-m3"
TAMIL_THRESHOLD = 0.7    # measured: best hits, no extra off-topic lines
POOL_K = 50              # Tamil-side top-50 + English-side top-50
MAX_LINES = 10           # answer length

_model = None
_cache = {}


def load():
    """Loads the reranker once (CPU). Called at app start, or lazily."""
    global _model
    if _model is None:
        from sentence_transformers import CrossEncoder
        _model = CrossEncoder(RESCUE_MODEL_NAME, device="cpu", max_length=256)
    return _model


def _norm(x):
    x = np.asarray(x, dtype=np.float32)
    return x / (np.linalg.norm(x, axis=1, keepdims=True) + 1e-12)


def _data(standard):
    if standard not in _cache:
        en = db.get_standard_english_db(standard).get(
            include=["documents", "metadatas", "embeddings"])
        ta = db.get_standard_tamil_db(standard).get(
            include=["documents", "embeddings"])
        _cache[standard] = {
            "en_ids": en["ids"], "en_emb": _norm(en["embeddings"]),
            "en_doc": dict(zip(en["ids"], en["documents"])),
            "en_meta": dict(zip(en["ids"], en["metadatas"])),
            "ta_ids": ta["ids"], "ta_emb": _norm(ta["embeddings"]),
            "ta_doc": dict(zip(ta["ids"], ta["documents"])),
        }
    return _cache[standard]


def _is_question(text):
    t = (text or "").strip()
    return t.endswith("?") or t.endswith("\uff1f")



# "Learning objectives" chunks (start of a chapter) list what the
# lesson teaches; they never answer a question.
_LO_EN = re.compile(r"learning objectives|students will be able to|"
                    r"after (studying|completing|learning) (this|the) "
                    r"(unit|lesson|chapter)", re.I)
_LO_TA = re.compile(r"கற்றல்\s*நோக்க")
# split-up objective lines: "To distinguish ...", "Understands ..."
_LO_START = re.compile(r"^(To [a-z]+|Understands|Recognizes|Recognises|"
                       r"Identifies|Explains|Describes|Knows|Learns)\b")
LO_MAX_INDEX = 15   # objective lines only occur in a chapter's first chunks


def _chunk_index(alignment_id):
    try:
        return int(str(alignment_id).rsplit("_", 1)[1])
    except (IndexError, ValueError):
        return None


def _lo_text(en, ta):
    """Objective wording, ignoring position."""
    if re.search(r"\bIntroduction\b", en):
        return False          # chunk also carries the chapter's real intro
    if _LO_EN.search(en):
        return True
    return en.startswith("To ") and bool(_LO_TA.search(ta or ""))


def is_learning_objectives(english, tamil=None, alignment_id=None):
    en = (english or "").strip()
    if _lo_text(en, tamil):
        return True
    idx = _chunk_index(alignment_id)
    if idx is None or idx > LO_MAX_INDEX or not _LO_START.match(en):
        return False
    if re.search(r"\bIntroduction\b", en):
        return False
    # "To ..." line: only an objective if every chunk from the chapter's
    # first one up to this one is an objective line too (one block).
    # So "To maintain good health, you should ..." later on is kept.
    try:
        d = _data(int(str(alignment_id).split("_", 1)[0]))
    except Exception:
        return False
    prefix = str(alignment_id).rsplit("_", 1)[0]
    width = len(str(alignment_id).rsplit("_", 1)[1])
    for j in range(1, idx):
        aid = f"{prefix}_{j:0{width}d}"
        pe = (d["en_doc"].get(aid) or "").strip()
        if not pe or re.search(r"\bIntroduction\b", pe):
            return False
        if not (_lo_text(pe, d["ta_doc"].get(aid)) or _LO_START.match(pe)):
            return False
    return True

def _top(ids, emb, text, k):
    q = retrieve_chroma._get_model().encode(
        [text], convert_to_numpy=True, normalize_embeddings=True)[0]
    sims = emb @ q
    order = np.argsort(-sims)[:k]
    return [(ids[i], float(sims[i])) for i in order]


def logit(p):
    p = min(max(p, 0.001), 0.999)
    return math.log(p / (1 - p))   # 0.7 -> +0.85, same scale as the gate


def score_candidates(tamil_query, query_en, standard, extra_ids=()):
    """Scores every candidate's Tamil text against the Tamil question.
    Returns chunk dicts (same shape as Stage 4/5 output) with
    "tamil_check" = probability 0..1, best first. Questions left out."""
    d = _data(standard)
    pool = dict(_top(d["ta_ids"], d["ta_emb"], tamil_query, POOL_K))
    for aid, sim in _top(d["en_ids"], d["en_emb"], query_en, POOL_K):
        pool.setdefault(aid, sim)
    ids = list(dict.fromkeys(list(extra_ids) + list(pool)))
    aids = [a for a in ids
            if a in d["ta_doc"] and a in d["en_doc"]
            and not _is_question(d["ta_doc"][a])
            and not _is_question(d["en_doc"][a])
            and not is_learning_objectives(d["en_doc"][a], d["ta_doc"][a], a)]
    if not aids:
        return []

    probs = load().predict([(tamil_query, d["ta_doc"][a]) for a in aids],
                           batch_size=16)
    out = []
    for aid, p in zip(aids, map(float, probs)):
        meta = d["en_meta"][aid]
        out.append({
            "alignment_id": aid,
            "source_id": meta["source_id"],
            "logical_section_id": meta.get("logical_section_id", ""),
            "english": d["en_doc"][aid],
            "tamil": d["ta_doc"][aid],
            "score": max(0.0, pool.get(aid, 0.0)),
            "rerank_score": logit(p),
            "tamil_check": p,
        })
    out.sort(key=lambda c: -c["tamil_check"])
    return out