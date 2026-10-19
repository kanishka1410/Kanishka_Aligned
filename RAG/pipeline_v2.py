"""
End-to-end pipeline implementing the 13-stage workflow (plus sub-stages
11A/11B/11C) for the bilingual Standard 6/7 Science RAG system.

Stages 1-11: identical for English and Tamil/Tanglish queries (11
            onward applies to Tamil/Tanglish only -- see each stage).
Stage 11.5: NEW -- multilingual NLI verification of the FINAL Tamil
            claim produced by Stage 11 (Tamil/Tanglish only).
Stage 12:   NEW -- strict accept/reject filter based on Stage 11.5.
Stage 13:   final answer + source citations (renamed from the former
            Stage 12 -- same function body, no logic change, just
            renumbered to make room for 11.5/12).

Usage:
    python pipeline_v2.py --query "7-what is series and parallel"
    python pipeline_v2.py --query "6-series circuit endral ena"
"""
import argparse
import json
import math
import db
import re
import retrieve_chroma
import rerank
import nli_verify
import nli_verify_multilingual
import llm_backend
import gate
import pair_check
import tamil_rescue
import neighbours


SEP = "=" * 60
LINE = "-" * 60


def _header(title):
    print(SEP)
    print(title)
    print(SEP)


# ------------------------------------------------------------------
# STAGE 1 -- QUERY PROCESSING
# ------------------------------------------------------------------
def parse_query(query):
    """
    "7-what is series and parallel" -> (7, "what is series and parallel")
    Splits on the FIRST "-" only, so a hyphen inside the actual query text
    (e.g. "7-what is a series-parallel circuit") doesn't break parsing.
    Raises ValueError with the spec's exact error text on anything invalid.
    """
    if "-" not in query:
        raise ValueError("Invalid standard.\nSupported standards: 6, 7, 8")
    standard_str, actual_query = query.split("-", 1)
    standard_str = standard_str.strip()
    actual_query = actual_query.strip()
    if not standard_str.isdigit() or int(standard_str) not in db.VALID_STANDARDS:
        raise ValueError("Invalid standard.\nSupported standards: 6, 7, 8")
    return int(standard_str), actual_query


def stage1_query_processing(query):
    standard, actual_query = parse_query(query)
    language = llm_backend.detect_language(actual_query)

    _header("STAGE 1 -- QUERY PROCESSING")
    print(f"Original Query:\n{query}")
    print(f"Extracted Standard:\n{standard}")
    print(f"Actual Query:\n{actual_query}")
    print(f"Detected Language:\n{language}")
    print()
    return standard, actual_query, language


# ------------------------------------------------------------------
# STAGE 2 -- QUERY TRANSLATION
# ------------------------------------------------------------------
def stage2_query_translation(actual_query, language):
    _header("STAGE 2 -- QUERY TRANSLATION")
    if language == "English":
        print("Query is already English.")
        print(f"English Query Used for Retrieval:\n{actual_query}")
        query_en = actual_query
    else:
        query_en = llm_backend.translate_query_to_english(actual_query, language)
        print(f"Original Query:\n{actual_query}")
        print(f"Translated English Query:\n{query_en}")
        print("English query will be used for retrieval.")
    print()
    return query_en


# ------------------------------------------------------------------
# STAGE 3 -- DATABASE SELECTION
# ------------------------------------------------------------------
def stage3_database_selection(standard):
    _header("STAGE 3 -- DATABASE SELECTION")
    print(f"Requested Standard:\n{standard}")
    print(f"Chroma Folder:\n{db.CHROMA_PATHS[standard]}")
    print(f"English Collection:\nenglish_{standard}")
    print(f"Tamil Collection:\ntamil_{standard}")
    print(f"Search Restriction:\nONLY STANDARD {standard}")
    print()


# ------------------------------------------------------------------
# DENSE RETRIEVAL -- silent, per spec (no stage header, no output)
# Now includes heading/section-context expansion: dense_search() seeds
# from the top 10, then pulls every chunk sharing a seed's
# logical_section_id, before handing the combined pool to reranking.
# ------------------------------------------------------------------
def run_dense_retrieval(query_en, standard):
    try:
        return retrieve_chroma.dense_search(query_en, standard, seed_k=100)
    except Exception as e:
        print(f"[retrieval error] {type(e).__name__}: {e}")
        # most commonly: this standard's collection was never ingested.
        # Degrade to "no results" rather than crash -- process_query()
        # already handles an empty candidate list with the proper
        # spec-mandated message.
        return []


# ------------------------------------------------------------------
# STAGE 4 -- TOP 10 CROSS-ENCODER RERANKED RESULTS
# ------------------------------------------------------------------
'''def stage4_rerank(query_en, standard, candidates, reranker):
    top10 = reranker.rerank(query_en, candidates, top_k=10)

    _header("STAGE 4 -- TOP 10 CROSS-ENCODER RERANKED RESULTS")
    print(f"Query:\n{query_en}")
    print(f"Standard:\n{standard}")
    print(LINE)
    for i, c in enumerate(top10, 1):
        print(f"RANK {i}")
        print(f"Cross-Encoder Score:\n{c['rerank_score']:.4f}")
        print(f"Alignment ID:\n{c['alignment_id']}")
        print(f"Chunk Text:\n{c['english']}")
        print(LINE)
    print(SEP)
    print("ONLY TOP 10 CROSS-ENCODER RESULTS ARE DISPLAYED")
    print(SEP)
    print()
    return top10'''
CONTEXT_POOL = 50      # best 50 by own score also get a context score
CONTEXT_MARGIN = 1.0   # context score must beat own score by > 1 to count


def stage4_rerank(query_en, standard, candidates, reranker):
    # Own score for every candidate (unchanged), then for the best 50 a
    # second score with the 2 chunks before it prepended, so a sentence
    # like "The plants which grow in this habitat..." is judged with the
    # text that says which habitat. The chunk itself is unchanged; only
    # its relevance score can go up.
    pool = reranker.rerank(query_en, candidates, top_k=CONTEXT_POOL)
    need = []
    for c in pool:
        ctx = neighbours.prev_context(c["alignment_id"])
        if ctx:
            need.append((c, ctx))
    if need:
        ctx_scores = reranker.model.predict(
            [(query_en, ctx + " " + c["english"]) for c, ctx in need])
        for (c, _), cs in zip(need, ctx_scores):
            c["own_score"] = c["rerank_score"]
            c["context_score"] = float(cs)
            c["rerank_score"] = max(c["rerank_score"], float(cs) - CONTEXT_MARGIN)
    top10 = sorted(pool, key=lambda x: -x["rerank_score"])[:10]

    _header("STAGE 4 -- TOP 10 CROSS-ENCODER RERANKED RESULTS")
    print(f"Query:\n{query_en}")
    print(f"Standard:\n{standard}")
    print(LINE)
    for i, c in enumerate(top10, 1):
        print(f"RANK {i}")
        print(f"Cross-Encoder Score:\n{c['rerank_score']:.4f}")
        if "context_score" in c and c["rerank_score"] > c["own_score"]:
            print(f"(raised by context: own {c['own_score']:.4f}, "
                  f"with previous chunks {c['context_score']:.4f})")
        print(f"Alignment ID:\n{c['alignment_id']}")
        print(f"Chunk Text:\n{c['english']}")
        print(LINE)
    print(SEP)
    print("ONLY TOP 10 CROSS-ENCODER RESULTS ARE DISPLAYED")
    print(SEP)
    print()
    return top10


# ------------------------------------------------------------------
# STAGE 5 -- ALIGNMENT LOOKUP
# ------------------------------------------------------------------
def stage5_alignment_lookup(top10, standard):
    _header("STAGE 5 -- ALIGNMENT LOOKUP")
    for i, c in enumerate(top10, 1):
        tamil_text = retrieve_chroma.get_tamil_by_alignment_id(c["alignment_id"], standard)
        c["tamil"] = tamil_text  # may be None -- handled downstream

        print(f"RESULT {i}")
        print(LINE)
        print(f"Alignment ID:\n{c['alignment_id']}")
        print(f"English Text:\n{c['english']}")
        if tamil_text:
            print(f"Tamil Text:\n{tamil_text}")
        else:
            print(f"Tamil Text:\n[MISSING -- no Tamil record found for alignment_id {c['alignment_id']}]")
        print()
    return top10


# ------------------------------------------------------------------
# STAGE 6 -- CLAIM GENERATION
# ------------------------------------------------------------------
FALLBACK_MIN_CE = 1.0   # only clearly relevant chunks get a fallback claim
_ACTIVITY_START = re.compile(
    r"^(activity|let us|let's|discuss|ask your|fill in|find out|try|"
    r"observe|take a|do you know|think|look at)\b", re.I)


def _sentences(text):
    parts = re.split(r"(?<=[.!?])\s+", text or "")
    return [p.strip() for p in parts if len(p.split()) >= 3]


def _fallback_claims(query_en, chunks, have_ids, reranker):
    """Qwen sometimes writes no claim for a clearly relevant chunk. For
    such a chunk (cross-encoder >= 1.0) the textbook's own best-matching
    sentence becomes the claim -- copied word for word, so Stage 7 marks
    it verbatim. Activity prompts and questions are never used."""
    out = []
    for c in chunks:
        if c["alignment_id"] in have_ids or c["rerank_score"] < FALLBACK_MIN_CE:
            continue
        if _is_question_text(c["english"]):
            continue
        sents = [x for x in _sentences(c["english"])
                 if not x.endswith("?") and not _ACTIVITY_START.match(x)]
        if not sents:
            continue
        if reranker is not None and len(sents) > 1:
            sc = reranker.model.predict([(query_en, x) for x in sents])
            best = max(zip(map(float, sc), sents))[1]
        else:
            best = sents[0]
        out.append({"alignment_id": c["alignment_id"], "claim": best,
                    "fallback": True})
    return out


def stage6_claim_generation(query_en, top10, reranker=None):
    claim_pairs = list(llm_backend.generate_claims(query_en, top10))
    have_ids = {cp.get("alignment_id") for cp in claim_pairs}
    claim_pairs += _fallback_claims(query_en, top10, have_ids, reranker)
    chunk_by_id = {c["alignment_id"]: c for c in top10}

    _header("STAGE 6 -- CLAIM GENERATION")
    claims = []
    result_num = 1
    for cp in claim_pairs:
        chunk = chunk_by_id.get(cp["alignment_id"])
        if chunk is None:
            continue
        if cp["claim"].strip().endswith("?") or _is_question_text(chunk["english"]):
            continue  # a question is not an answer
        claim_record = {
            "alignment_id": cp["alignment_id"],
            "claim": cp["claim"],
            "source_english": chunk["english"],
            "tamil": chunk["tamil"],
            "source_id": chunk["source_id"],
            "score": chunk["score"],
            "rerank_score": chunk["rerank_score"],
        }
        claims.append(claim_record)

        print(f"RESULT:\n{result_num}")
        print(f"Alignment ID:\n{cp['alignment_id']}")
        print(f"English Query:\n{query_en}")
        print(f"Evidence Chunk:\n{chunk['english']}")
        print(f"Generated Claim:\n{cp['claim']}")
        if cp.get("fallback"):
            print("(Qwen gave no claim for this chunk -- textbook sentence used)")
        print()
        result_num += 1
    return claims


# ------------------------------------------------------------------
# STAGE 7 -- NLI VERIFICATION
# ------------------------------------------------------------------
_ARTICLES = {"the", "a", "an"}
_STOP = ".?!:;"


def _is_verbatim(claim, chunk):
    """True if the claim is a whole sentence / list item copied word for
    word from the chunk (ignoring case, punctuation and a leading article).
    Such a claim cannot be made up, so NLI is not needed for it.

    The copy must START at a sentence / list-item start (chunk start, after
    . ? ! : ; or at a capitalised word) and END at a sentence / item end.
    So "heat is a form of matter" taken from "It is wrong to say that heat
    is a form of matter." does NOT count -- NLI still checks that one.
    """
    cw = [w.lower() for w in re.findall(r"\w+", claim or "")]
    while cw and cw[0] in _ARTICLES:
        cw = cw[1:]
    if len(cw) < 3:
        return False
    toks = list(re.finditer(r"\w+", chunk or ""))
    low = [t.group().lower() for t in toks]
    n = len(cw)
    for i in range(len(low)):
        j = i
        if low[j] in _ARTICLES and (j + 1 < len(low)) and low[j + 1] == cw[0]:
            j += 1   # chunk has "The image ...", claim has "image ..."
        if low[j:j + n] != cw:
            continue
        before = chunk[toks[i - 1].end():toks[i].start()] if i else "."
        starts = (any(ch in _STOP for ch in before)
                  or toks[i].group()[0].isupper())
        k = j + n
        after = chunk[toks[k - 1].end():toks[k].start()] if k < len(toks) else "."
        ends = (any(ch in _STOP for ch in after)
                or (k < len(toks) and toks[k].group()[0].isupper()))
        if starts and ends:
            return True
    return False


def stage7_nli_verification(claims, verifier):
    claims = verifier.verify(claims)

    _header("STAGE 7 -- NLI VERIFICATION")
    for c in claims:
        if _is_verbatim(c["claim"], c.get("source_english")):
            # copied word for word from the textbook -> entailed by definition
            c["nli_scores"]["entailment"] = 1.0
            c["verdict"] = "entailment"
            c["verbatim"] = True
        entailment_score = c["nli_scores"]["entailment"]
        print(f"Alignment ID:\n{c['alignment_id']}")
        print(f"Claim:\n{c['claim']}")
        print(f"NLI Entailment Score:\n{entailment_score:.4f}")
        if c.get("verbatim"):
            print("Verbatim from textbook:\nYES (NLI score set to 1.0)")
        print()
    return claims


# ------------------------------------------------------------------
# STAGE 8 -- CONFIDENCE CALCULATION
# ------------------------------------------------------------------
def stage8_confidence_calculation(claims):
    _header("STAGE 8 -- CONFIDENCE CALCULATION")
    for c in claims:
        c["confidence"] = gate.score_claim(c)
        decision = "ACCEPT" if gate.accept_claim(c["confidence"],c["rerank_score"]) else "REJECT"
        c["_stage8_decision"] = decision  # provisional; stage 9 is the authoritative filter

        print(f"Alignment ID:\n{c['alignment_id']}")
        print(f"Cross-Encoder Relevance Score:\n{c['rerank_score']:.4f}")
        print(f"NLI Entailment Score:\n{c['nli_scores']['entailment']:.4f}")
        print(f"Final Confidence:\n{c['confidence']:.4f}")
        print(f"Acceptance Threshold:\n{gate.ACCEPTANCE_THRESHOLD}")
        print(f"Decision:\n{decision}")
        print()
    return claims


# ------------------------------------------------------------------
# STAGE 9 -- CONFIDENCE FILTERING
# ------------------------------------------------------------------
def stage9_confidence_filtering(claims):
    _header("STAGE 9 -- CONFIDENCE FILTERING")
    accepted, rejected = [], []
    for c in claims:
        if gate.accept_claim(c["confidence"], c["rerank_score"]):
            accepted.append(c)
        else:
            rejected.append(c)
            print(f"Alignment ID:\n{c['alignment_id']}")
            print(f"Confidence:\n{c['confidence']:.4f}")
            print(f"Threshold:\n{gate.ACCEPTANCE_THRESHOLD}")
            print("Decision:\nREJECT")
            print("Action:\nClaim discarded. Corresponding Tamil result discarded.")
            print()
    if not rejected:
        print("No claims rejected -- all passed the confidence threshold.")
        print()
    return accepted, rejected


# ------------------------------------------------------------------
# STAGE 10 -- ALIGNMENT VERIFICATION  (Tamil/Tanglish path only)
# ------------------------------------------------------------------
'''def stage10_alignment_verification(accepted_claims):
    _header("STAGE 10 -- ALIGNMENT VERIFICATION")
    for c in accepted_claims:
        if c["tamil"] is None:
            # spec section 34: missing Tamil -- print the missing alignment_id,
            # do not invent content, treat as untrustworthy (same handling as WRONG)
            print(f"Alignment ID:\n{c['alignment_id']}")
            print("English Text:")
            print(c["source_english"])
            print("Tamil Text:\n[MISSING]")
            print("Alignment Classification:\nWRONG")
            print("Reason:\nNo Tamil record exists for this alignment_id.")
            print()
            c["alignment_verdict"] = "WRONG"
            continue

        verdict = llm_backend.check_pair_quality(c["source_english"], c["tamil"])
        c["alignment_verdict"] = verdict

        print(f"Alignment ID:\n{c['alignment_id']}")
        print(f"English Text:\n{c['source_english']}")
        print(f"Tamil Text:\n{c['tamil']}")
        print(f"Alignment Classification:\n{verdict}")
        print()
    return accepted_claims'''
def stage10_alignment_verification(accepted_claims):
    _header("STAGE 10 -- ALIGNMENT VERIFICATION")
    # All pairs scored together in one LaBSE call -- no LLM call per chunk.
    with_tamil = [c for c in accepted_claims if c["tamil"] is not None]
    results = pair_check.classify_pairs([(c["source_english"], c["tamil"]) for c in with_tamil])
    result_by_id = {c["alignment_id"]: r for c, r in zip(with_tamil, results)}

    for c in accepted_claims:
        if c["tamil"] is None:
            print(f"Alignment ID:\n{c['alignment_id']}")
            print("English Text:")
            print(c["source_english"])
            print("Tamil Text:\n[MISSING]")
            print("Alignment Classification:\nWRONG")
            print("Reason:\nNo Tamil record exists for this alignment_id.")
            print()
            c["alignment_verdict"] = "WRONG"
            continue

        verdict, sim = result_by_id[c["alignment_id"]]
        c["alignment_verdict"] = verdict

        print(f"Alignment ID:\n{c['alignment_id']}")
        print(f"English Text:\n{c['source_english']}")
        print(f"Tamil Text:\n{c['tamil']}")
        print(f"LaBSE Similarity:\n{sim:.3f}")
        print(f"Alignment Classification:\n{verdict}")
        print()
    return accepted_claims


# ------------------------------------------------------------------
# STAGE 11A / 11B / 11C -- per-verdict Tamil claim generation
# ------------------------------------------------------------------
def stage11_tamil_claims(tamil_query, accepted_claims):
    for c in accepted_claims:
        if c["alignment_verdict"] == "SAME":
            tamil_claim = llm_backend.generate_tamil_claim_for_same_alignment(
                tamil_query, c["claim"], c["tamil"]
            )
            c["tamil_claim"] = tamil_claim

            _header("STAGE 11A -- SAME ALIGNMENT -> TAMIL CLAIM")
            print(f"Alignment ID:\n{c['alignment_id']}")
            print(f"Tamil Query:\n{tamil_query}")
            print(f"English Claim:\n{c['claim']}")
            print(f"Tamil Textbook Evidence:\n{c['tamil']}")
            print(f"Generated Tamil Claim:\n{tamil_claim}")
            print()

        elif c["alignment_verdict"] == "PARTIAL":
            tamil_claim = llm_backend.translate_claim_to_tamil(c["claim"])
            c["tamil_claim"] = tamil_claim

            _header("STAGE 11B -- PARTIAL ALIGNMENT -> TAMIL TRANSLATION")
            print(f"Alignment ID:\n{c['alignment_id']}")
            print("Alignment:\nPARTIAL")
            print(f"English Claim:\n{c['claim']}")
            print("Action:\nTamil aligned text is not used as authoritative evidence.")
            print(f"Tamil Translation:\n{tamil_claim}")
            print()

        else:  # WRONG
            tamil_claim = llm_backend.translate_claim_to_tamil(c["claim"])
            c["tamil_claim"] = tamil_claim

            _header("STAGE 11C -- WRONG ALIGNMENT -> TAMIL TRANSLATION")
            print(f"Alignment ID:\n{c['alignment_id']}")
            print("Alignment:\nWRONG")
            print(f"English Claim:\n{c['claim']}")
            print("Action:\nIncorrect Tamil alignment ignored.")
            print(f"Tamil Translation:\n{tamil_claim}")
            print()
    return accepted_claims


# ------------------------------------------------------------------
# STAGE 11.5 -- MULTILINGUAL NLI VERIFICATION OF THE FINAL TAMIL CLAIM
#               (Tamil/Tanglish path only)
# ------------------------------------------------------------------
def stage11_5_multilingual_nli_verification(accepted_claims, multilingual_verifier):
    """
    Verifies the claim actually produced by Stage 11 -- not the English
    claim (Stage 7 already checked that), not the original corpus
    alignment (Stage 10 already checked that), but the FINAL Tamil text
    that would reach the student.

    Pairing, confirmed with the user as the correct design:
        alignment_verdict == SAME             -> (tamil_evidence, tamil_claim)
            same-language check: did "correcting" the real evidence drift
            from what it actually said?
        alignment_verdict == PARTIAL or WRONG  -> (english_claim, tamil_claim)
            cross-lingual check: did the fresh translation preserve the
            claim's meaning?
    """
    _header("STAGE 11.5 -- MULTILINGUAL NLI VERIFICATION (FINAL TAMIL CLAIM)")
    for c in accepted_claims:
        if c["alignment_verdict"] == "SAME":
            premise, hypothesis = c["tamil"], c["tamil_claim"]
            pairing_used = "tamil_evidence vs tamil_claim (same-language)"
        else:
            premise, hypothesis = c["claim"], c["tamil_claim"]
            pairing_used = "english_claim vs tamil_claim (cross-lingual)"

        verdict, scores = multilingual_verifier.verify_pair(premise, hypothesis)
        c["multilingual_nli_verdict"] = verdict
        c["multilingual_nli_scores"] = scores

        print(f"Alignment ID:\n{c['alignment_id']}")
        print(f"Alignment Verdict (Stage 10):\n{c['alignment_verdict']}")
        print(f"Pairing Used:\n{pairing_used}")
        print(f"Premise:\n{premise}")
        print(f"Hypothesis (Tamil Claim):\n{hypothesis}")
        print(f"Multilingual NLI Verdict:\n{verdict}")
        print()
    return accepted_claims


# ------------------------------------------------------------------
# STAGE 12 -- MULTILINGUAL NLI FILTERING  (Tamil/Tanglish path only)
# ------------------------------------------------------------------
def stage12_nli_based_filtering(claims):
    """
    Strict: a final Tamil claim is kept ONLY if Stage 11.5's verdict was
    "entailment". "neutral" or "contradiction" both mean the Tamil text
    did not clearly preserve its source's meaning -- discarded, not
    caveated, same philosophy as Stage 9's confidence filtering.

    Binary rather than a weighted score: this check has no retrieval/
    rerank signal to blend with (it's verifying generated text, not a
    retrieved chunk), so entailment-or-not is the honest criterion here,
    not a borrowed formula that doesn't actually apply.
    """
    _header("STAGE 12 -- MULTILINGUAL NLI FILTERING")
    kept, dropped = [], []
    for c in claims:
        if c["multilingual_nli_verdict"] == "entailment":
            kept.append(c)
        else:
            dropped.append(c)
            print(f"Alignment ID:\n{c['alignment_id']}")
            print(f"Multilingual NLI Verdict:\n{c['multilingual_nli_verdict']}")
            print("Decision:\nREJECT")
            print("Action:\nFinal Tamil claim discarded -- did not clearly preserve source meaning.")
            print()
    if not dropped:
        print("No claims rejected -- all Tamil claims passed multilingual NLI verification.")
        print()
    return kept


# ------------------------------------------------------------------
# STAGE 13 -- FINAL RESULT SUMMARY
# ------------------------------------------------------------------
def stage13_final_summary(final_answer, final_source_ids):
    """
    Prints ONLY the final answer and the source_id(s) behind it --
    nothing else. Renamed from stage12_final_summary to stage13 (pure
    rename, body unchanged) to make room for the new Stage 11.5/12
    multilingual NLI verification on the Tamil path.
    """
    _header("STAGE 13 -- FINAL RESULT SUMMARY")
    print("FINAL ANSWER:")
    print(final_answer)
    print()
    print("Source:")
    for sid in final_source_ids:
        print(sid)  # source_id denotes standard_term_chapter, e.g. "7_term2_ch1"
    print()


# ------------------------------------------------------------------
# STAGE 6A -- SAME-PATH CLAIM CONSTRUCTION (Tamil/Tanglish path only)
#
# NEW. Only used for chunks whose EARLY alignment check (see
# process_query() -- stage10_alignment_verification called right after
# Stage 5, not after Stage 9 as in the older flow) returned SAME.
#
# For these chunks, English claim generation (Stage 6) and English NLI
# (Stage 7) are skipped ENTIRELY -- the claim is generated directly from
# the trusted Tamil evidence, and verified with the multilingual NLI
# model (same-language comparison), not the English-only one.
#
# This does not modify stage6_claim_generation or stage7_nli_verification
# -- it is a parallel path, not a change to either.
# ------------------------------------------------------------------
'''def stage6a_same_path_claims(same_chunks, tamil_query, multilingual_verifier):
    _header("STAGE 6A -- SAME-PATH: DIRECT TAMIL CLAIM + MULTILINGUAL NLI")
    claims = []
    for c in same_chunks:
        tamil_claim = llm_backend.generate_tamil_claim_directly_from_evidence(
            tamil_query, c["tamil"]
        )
        if tamil_claim.strip().upper().startswith("NOT RELEVANT"):
            print(f"Alignment ID:\n{c['alignment_id']}")
            print("Result:\nSkipped -- evidence judged not relevant to the question.")
            print()
            continue

        verdict, scores = multilingual_verifier.verify_pair(c["tamil"], tamil_claim)

        print(f"Alignment ID:\n{c['alignment_id']}")
        print(f"Tamil Query:\n{tamil_query}")
        print(f"Tamil Textbook Evidence:\n{c['tamil']}")
        print(f"Generated Tamil Claim:\n{tamil_claim}")
        print(f"Multilingual NLI Verdict:\n{verdict}")
        print()

        claims.append({
            "alignment_id": c["alignment_id"],
            "claim": tamil_claim,  # no separate English claim exists on this path
            "source_english": c["english"],
            "tamil": c["tamil"],
            "tamil_claim": tamil_claim,  # already final -- Stage 11A's job, done here instead
            "alignment_verdict": "SAME",
            "source_id": c["source_id"],
            "score": c["score"],
            "rerank_score": c["rerank_score"],
            "verdict": verdict,  # feeds gate.score_claim() via NLI_WEIGHT, same as Stage 7's output does
            "nli_scores": scores,
        })
    return claims'''
MAX_SAME_CLAIMS = 10   # keep the answer short: only the best-ranked SAME chunks


def _is_question_text(text):
    """A textbook activity prompt / heading phrased as a question is not an answer."""
    t = text.strip()
    return t.endswith("?") or t.endswith("\uff1f")


def stage6a_same_path_claims(same_chunks, tamil_query, multilingual_verifier):
    _header("STAGE 6A -- SAME-PATH: DIRECT TAMIL CLAIM + MULTILINGUAL NLI")
    claims = []
    # Best-ranked first, so the MAX_SAME_CLAIMS cap keeps the most relevant.
    for c in sorted(same_chunks, key=lambda x: x["rerank_score"], reverse=True):
        if len(claims) >= MAX_SAME_CLAIMS:
            break
        # Stage 10 already confirmed this chunk's Tamil matches its English,
        # so the textbook's own Tamil is used AS-IS.
        tamil_claim = c["tamil"].strip()
        if _is_question_text(tamil_claim) or _is_question_text(c["english"]):
            print(f"Alignment ID:\n{c['alignment_id']}")
            print("Result:\nSkipped -- evidence is a question/activity prompt, not an answer.")
            print()
            continue

        # The claim IS the evidence text, so it is trivially entailed by it.
        verdict = "entailment"
        scores = {"entailment": 1.0, "neutral": 0.0, "contradiction": 0.0}

        print(f"Alignment ID:\n{c['alignment_id']}")
        print(f"Tamil Query:\n{tamil_query}")
        print(f"Tamil Textbook Evidence (used as-is):\n{c['tamil']}")
        print(f"Tamil Claim:\n{tamil_claim}")
        print(f"Multilingual NLI Verdict:\n{verdict}")
        print()

        claims.append({
            "alignment_id": c["alignment_id"],
            "claim": tamil_claim,
            "source_english": c["english"],
            "tamil": c["tamil"],
            "tamil_claim": tamil_claim,
            "alignment_verdict": "SAME",
            "source_id": c["source_id"],
            "score": c["score"],
            "rerank_score": c["rerank_score"],
            "verdict": verdict,
            "nli_scores": scores,
        })
    return claims

# ------------------------------------------------------------------
# STAGE 4R -- TAMIL RESCUE SEARCH (Tamil-script queries only)
# Runs only when no top-10 chunk is relevant to the translated query,
# i.e. the translation most likely failed. See tamil_rescue.py.
# ------------------------------------------------------------------
'''def stage4r_tamil_rescue(tamil_query, query_en, standard):
    _header("STAGE 4R -- TAMIL RESCUE SEARCH (NO TRANSLATION)")
    print("Reason:\nNo chunk was relevant to the translated query "
          f"\"{query_en}\" -- searching the Tamil textbook text directly.")
    rescued = tamil_rescue.rescue_search(tamil_query, query_en, standard)
    for i, c in enumerate(rescued, 1):
        print(f"RESCUE {i}")
        print(f"Reranker Probability:\n{c['rescue_probability']:.3f}")
        print(f"Alignment ID:\n{c['alignment_id']}")
        print(f"Tamil Text:\n{c['tamil']}")
        print(f"English Text:\n{c['english']}")
        print(LINE)
    if not rescued:
        print(f"Result:\nNothing scored >= {tamil_rescue.RESCUE_THRESHOLD}.")
    print()
    return rescued'''

# ------------------------------------------------------------------
# STAGE 4T -- TAMIL CHECK (Tamil-script queries only)
# Every Tamil question: the textbook's own Tamil text is judged against
# the Tamil question directly (no translation). A chunk is kept if the
# cross-encoder accepts it (> 0) OR the Tamil check scores >= 0.7.
# Chunks the Tamil check accepts are answered with their textbook Tamil
# as-is. See tamil_rescue.py for the measurement behind this.
# ------------------------------------------------------------------
def _sigmoid(x):
    return 1 / (1 + math.exp(-x))


def stage4t_tamil_check(tamil_query, query_en, standard, top10):
    _header("STAGE 4T -- TAMIL CHECK (TEXTBOOK TAMIL vs TAMIL QUESTION)")
    by_id = {c["alignment_id"]: c for c in top10}
    scored = tamil_rescue.score_candidates(
        tamil_query, query_en, standard, list(by_id))

    selected = []
    for rec in scored:
        p = rec["tamil_check"]
        c = by_id.get(rec["alignment_id"])
        ok_en = c is not None and c["rerank_score"] > 0
        ok_ta = p >= tamil_rescue.TAMIL_THRESHOLD
        if not (ok_en or ok_ta):
            continue
        if c is None:
            c = dict(rec)
        else:
            c = dict(c)
            c["tamil_check"] = p
            if not ok_en:
                c["rerank_score"] = rec["rerank_score"]
        c["tamil_ok"] = ok_ta
        c["rank_score"] = max(p, _sigmoid(c["rerank_score"]) if ok_en else 0)
        selected.append(c)

    selected.sort(key=lambda c: -c["rank_score"])
    selected = selected[:tamil_rescue.MAX_LINES]

    for i, c in enumerate(selected, 1):
        print(f"SELECTED {i}")
        print(f"Tamil Check:\n{c['tamil_check']:.3f}")
        print(f"Cross-Encoder Score:\n{c['rerank_score']:.4f}")
        print(f"Alignment ID:\n{c['alignment_id']}")
        print(f"Tamil Text:\n{c['tamil']}")
        print(f"English Text:\n{c['english']}")
        print(LINE)
    if not selected:
        print("Result:\nNo chunk passed either check.")
    print()
    return selected


# ------------------------------------------------------------------
# MAIN ORCHESTRATION
# ------------------------------------------------------------------
def process_query(query, reranker=None, nli_verifier=None, multilingual_verifier=None):
    try:
        standard, actual_query, language = stage1_query_processing(query)
    except ValueError as e:
        print(str(e))
        return

    query_en = stage2_query_translation(actual_query, language)
    stage3_database_selection(standard)

    candidates = run_dense_retrieval(query_en, standard)
    if not candidates:
        print(f"No relevant textbook content found in Standard {standard}.")
        return

    if reranker is None:
        reranker = rerank.Reranker()
    top10 = stage4_rerank(query_en, standard, candidates, reranker)

    top10 = stage5_alignment_lookup(top10, standard)
    # Learning-objectives chunks never answer a question (all languages).
    top10 = [c for c in top10
            if not tamil_rescue.is_learning_objectives(
                 c["english"], c.get("tamil"), c["alignment_id"])]

    insufficient = ("The available textbook evidence is insufficient to "
                    "answer this question confidently.")

    if language == "English":
        # ENGLISH PATH
        top10 = neighbours.expand_lists(top10)   # cut-off lists completed
        claims = stage6_claim_generation(query_en, top10, reranker)
        if not claims:
            print(f"No relevant textbook content found in Standard {standard}.")
            return

        verifier = nli_verifier if nli_verifier is not None else nli_verify.NLIVerifier()
        claims = stage7_nli_verification(claims, verifier)
        claims = stage8_confidence_calculation(claims)
        accepted, rejected = stage9_confidence_filtering(claims)

        if not accepted:
            stage13_final_summary(insufficient, [])
            return

        final_answer = llm_backend.format_final_english_answer(query_en, accepted)
        final_source_ids = sorted({c["source_id"] for c in accepted})
        stage13_final_summary(final_answer, final_source_ids)
        return

    # TAMIL / TANGLISH PATH
    if language == "Tamil":
        top10 = stage4t_tamil_check(actual_query, query_en, standard, top10)
        if not top10:
            stage13_final_summary(insufficient, [])
            return

       # cut-off lists completed
    top10 = neighbours.expand_lists(top10)   # cut-off lists completed
    for c in top10:
        c["source_english"] = c["english"]

    # Chunks the Tamil check accepted: their textbook Tamil is the verified
    # evidence, used as-is. Only the others get the LaBSE pair check.
    to_check = [c for c in top10 if not c.get("tamil_ok")]
    if to_check:
        stage10_alignment_verification(to_check)
    for c in top10:
        if c.get("tamil_ok"):
            c["alignment_verdict"] = "SAME"

    same_chunks = [c for c in top10 if c["alignment_verdict"] == "SAME"]
    other_chunks = [c for c in top10 if c["alignment_verdict"] != "SAME"]

    if multilingual_verifier is None:
        multilingual_verifier = nli_verify_multilingual.MultilingualNLIVerifier()
    same_path_claims = stage6a_same_path_claims(same_chunks, actual_query, multilingual_verifier)

    other_path_claims = []
    if other_chunks:
        other_path_claims = stage6_claim_generation(query_en, other_chunks, reranker)
        if other_path_claims:
            verifier = nli_verifier if nli_verifier is not None else nli_verify.NLIVerifier()
            other_path_claims = stage7_nli_verification(other_path_claims, verifier)
            verdict_by_id = {c["alignment_id"]: c["alignment_verdict"] for c in other_chunks}
            for claim in other_path_claims:
                claim["alignment_verdict"] = verdict_by_id.get(claim["alignment_id"], "WRONG")

    combined_claims = same_path_claims + other_path_claims
    if not combined_claims:
        stage13_final_summary(insufficient, [])
        return

    combined_claims = stage8_confidence_calculation(combined_claims)
    accepted, rejected = stage9_confidence_filtering(combined_claims)

    if not accepted:
        stage13_final_summary(insufficient, [])
        return

    same_survivors = [c for c in accepted if c["alignment_verdict"] == "SAME"]
    other_survivors = [c for c in accepted if c["alignment_verdict"] != "SAME"]

    final_other_survivors = []
    if other_survivors:
        other_survivors = stage11_tamil_claims(actual_query, other_survivors)
        other_survivors = stage11_5_multilingual_nli_verification(other_survivors, multilingual_verifier)
        final_other_survivors = stage12_nli_based_filtering(other_survivors)

    final_claims = same_survivors + final_other_survivors
    if not final_claims:
        stage13_final_summary(insufficient, [])
        return

    # Most relevant first (the order chosen in Stage 4 / Stage 4T).
    order = {c["alignment_id"]: i for i, c in enumerate(top10)}
    final_claims.sort(key=lambda c: order.get(c["alignment_id"], len(order)))

    tamil_claims_for_answer = [
        {"tamil_claim": c["tamil_claim"], "source_id": c["source_id"]} for c in final_claims
    ]
    final_answer = llm_backend.format_final_tamil_answer(actual_query, tamil_claims_for_answer)
    final_source_ids = sorted({c["source_id"] for c in final_claims})
    stage13_final_summary(final_answer, final_source_ids)


'''def verbatim_english_claims(chunks):
    """Rescue path: the textbook's own English sentence is the claim (no
    Qwen rewrite, which would use the failed translation). A sentence is
    trivially entailed by itself."""
    claims = []
    for c in chunks:
        if _is_question_text(c["english"]):
            continue
        claims.append({
            "alignment_id": c["alignment_id"],
            "claim": c["english"],
            "source_english": c["english"],
            "tamil": c["tamil"],
            "source_id": c["source_id"],
            "score": c["score"],
            "rerank_score": c["rerank_score"],
            "alignment_verdict": c["alignment_verdict"],
            "verdict": "entailment",
            "nli_scores": {"entailment": 1.0, "neutral": 0.0,
                           "contradiction": 0.0},
        })
    return claims

# ------------------------------------------------------------------
# MAIN ORCHESTRATION
# ------------------------------------------------------------------
def process_query(query, reranker=None, nli_verifier=None, multilingual_verifier=None):
    try:
        standard, actual_query, language = stage1_query_processing(query)
    except ValueError as e:
        print(str(e))
        return

    query_en = stage2_query_translation(actual_query, language)
    stage3_database_selection(standard)

    candidates = run_dense_retrieval(query_en, standard)
    if not candidates:
        print(f"No relevant textbook content found in Standard {standard}.")
        return

    if reranker is None:
        reranker = rerank.Reranker()    
    top10 = stage4_rerank(query_en, standard, candidates, reranker)

    top10 = stage5_alignment_lookup(top10, standard)
    rescued = False
    if language == "Tamil" and not any(c["rerank_score"] > 0 for c in top10):
        rescue_chunks = stage4r_tamil_rescue(actual_query, query_en, standard)
        if rescue_chunks:
            top10 = rescue_chunks
            rescued = True

    if language == "English":
        # ENGLISH PATH -- COMPLETELY UNCHANGED from the prior version.
        # No alignment branching, no Tamil processing of any kind.
        claims = stage6_claim_generation(query_en, top10)
        if not claims:
            print(f"No relevant textbook content found in Standard {standard}.")
            return

        verifier = nli_verifier if nli_verifier is not None else nli_verify.NLIVerifier()
        claims = stage7_nli_verification(claims, verifier)
        claims = stage8_confidence_calculation(claims)
        accepted, rejected = stage9_confidence_filtering(claims)

        if not accepted:
            final_answer = ("The available textbook evidence is insufficient to "
                             "answer this question confidently.")
            stage13_final_summary(final_answer, [])
            return

        final_answer = llm_backend.format_final_english_answer(query_en, accepted)
        final_source_ids = sorted({c["source_id"] for c in accepted})
        stage13_final_summary(final_answer, final_source_ids)
        return

    # TAMIL / TANGLISH PATH -- restructured per the confirmed design:
    # alignment check moved to immediately after Stage 5, SAME-verdict
    # chunks skip English generation entirely, PARTIAL/WRONG-verdict
    # chunks continue through the existing Stage 6/7 unchanged.

    # stage10_alignment_verification() reads c["source_english"] -- top10
    # items only have "english" at this point (set by retrieval/rerank,
    # confirmed in stage5_alignment_lookup's own code). Adding this alias
    # is new orchestration glue, not a change to any stage function body.
    for c in top10:
        c["source_english"] = c["english"]

    #top10 = stage10_alignment_verification(top10)  # called ONCE, early -- not called again later

    if rescued:
        # Rescue chunks were chosen by reading their TAMIL text against the
        # Tamil question, so that Tamil text is itself the verified
        # evidence -- use it as-is (the English pair check does not apply).
        for c in top10:
            c["alignment_verdict"] = "SAME"
    else:
        top10 = stage10_alignment_verification(top10)  # called ONCE, early -- not called again later

    same_chunks = [c for c in top10 if c["alignment_verdict"] == "SAME"]
    other_chunks = [c for c in top10 if c["alignment_verdict"] != "SAME"]

    if multilingual_verifier is None:
        multilingual_verifier = nli_verify_multilingual.MultilingualNLIVerifier()
    same_path_claims = stage6a_same_path_claims(same_chunks, actual_query, multilingual_verifier)

    other_path_claims = []
    if other_chunks and rescued:
        other_path_claims = verbatim_english_claims(other_chunks)
    elif other_chunks:
        other_path_claims = stage6_claim_generation(query_en, other_chunks)
        if other_path_claims:
            verifier = nli_verifier if nli_verifier is not None else nli_verify.NLIVerifier()
            other_path_claims = stage7_nli_verification(other_path_claims, verifier)
            # stage6_claim_generation doesn't carry alignment_verdict through --
            # restore it from the chunk it came from, needed by Stage 11 onward
            verdict_by_id = {c["alignment_id"]: c["alignment_verdict"] for c in other_chunks}
            for claim in other_path_claims:
                claim["alignment_verdict"] = verdict_by_id.get(claim["alignment_id"], "WRONG")

    combined_claims = same_path_claims + other_path_claims
    if not combined_claims:
        print(f"No relevant textbook content found in Standard {standard}.")
        return

    combined_claims = stage8_confidence_calculation(combined_claims)
    accepted, rejected = stage9_confidence_filtering(combined_claims)

    if not accepted:
        final_answer = ("The available textbook evidence is insufficient to "
                         "answer this question confidently.")
        stage13_final_summary(final_answer, [])
        return

    same_survivors = [c for c in accepted if c["alignment_verdict"] == "SAME"]
    other_survivors = [c for c in accepted if c["alignment_verdict"] != "SAME"]

    final_other_survivors = []
    if other_survivors:
        other_survivors = stage11_tamil_claims(actual_query, other_survivors)
        other_survivors = stage11_5_multilingual_nli_verification(other_survivors, multilingual_verifier)
        final_other_survivors = stage12_nli_based_filtering(other_survivors)

    final_claims = same_survivors + final_other_survivors
    if not final_claims:
        final_answer = ("The available textbook evidence is insufficient to "
                         "answer this question confidently.")
        stage13_final_summary(final_answer, [])
        return

    tamil_claims_for_answer = [
        {"tamil_claim": c["tamil_claim"], "source_id": c["source_id"]} for c in final_claims
    ]
    final_answer = llm_backend.format_final_tamil_answer(actual_query, tamil_claims_for_answer)
    final_source_ids = sorted({c["source_id"] for c in final_claims})
    stage13_final_summary(final_answer, final_source_ids)'''



if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--query", required=True,
                     help='e.g. "7-what is series and parallel" or "6-series circuit endral ena"')
    args = ap.parse_args()
    process_query(args.query)