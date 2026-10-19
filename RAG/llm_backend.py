"""
Local LLM calls via Ollama (https://ollama.com). One model, all roles --
same as before (see OLLAMA_MODEL below); Tamil-coverage caveats discussed
earlier in this project still apply to every Tamil-touching function here.

FUNCTIONS REUSED, UNCHANGED FROM THE ORIGINAL llm_backend.py:
    generate_claims(), translate_claim_to_tamil()

FUNCTIONS CHANGED:
    translate_tamil_to_english() -> generalized into
    translate_query_to_english(query, language), which handles Tamil AND
    Tanglish with one prompt (both need the same operation: get to
    English; the model is simply told which script it's reading).

    check_pair_quality() -> same logic, but now returns "WRONG" instead of
    "DIFFERENT" as the third verdict, to match the spec's SAME/PARTIAL/WRONG
    terminology exactly.

NEW FUNCTIONS:
    detect_language(text) -- English / Tamil / Tanglish classification.
    generate_tamil_claim_for_same_alignment() -- Stage 11A: for a SAME-
      verdict pair, produce the Tamil claim from the trustworthy aligned
      Tamil evidence (not a translation).
    format_final_english_answer() -- combines accepted claims into one
      answer, each claim's source_id attached as its citation.
    format_final_tamil_answer() -- combines per-claim Tamil claims
      (whether from generate_tamil_claim_for_same_alignment or
      translate_claim_to_tamil) into one coherent final answer, citations
      attached.
"""
import json
import re
import requests
import translate_nllb

OLLAMA_URL = "http://localhost:11434/api/generate"
OLLAMA_MODEL = "qwen2.5:7b-instruct"
NUM_CTX = 8192  # see earlier discussion: Ollama's own default is smaller
                 # and silently truncates -- this is set explicitly on
                 # every call below, never left to Ollama's default.


'''def _call_ollama(prompt, model=OLLAMA_MODEL, temperature=0.1):
    resp = requests.post(OLLAMA_URL, json={
        "model": model,
        "prompt": prompt,
        "stream": False,
        "options": {"temperature": temperature, "num_ctx": NUM_CTX},
    }, timeout=120)
    resp.raise_for_status()
    return resp.json()["response"].strip()'''

OLLAMA_TIMEOUT = 300  # raised from 120 -- on slower/CPU-bound machines
                       # (multiple models loaded at once: cross-encoder +
                       # NLI + multilingual NLI + this LLM) a single
                       # generation call can legitimately take longer
                       # than 120s without anything actually being broken.
OLLAMA_MAX_RETRIES = 2  # one retry after a timeout, in case it was a
                         # one-off stall (e.g. model being swapped back
                         # into memory), before giving up for real.


def _call_ollama(prompt, model=OLLAMA_MODEL, temperature=0.1):
    last_error = None
    for attempt in range(1, OLLAMA_MAX_RETRIES + 2):  # e.g. 3 total tries
        try:
            resp = requests.post(OLLAMA_URL, json={
                "model": model,
                "prompt": prompt,
                "stream": False,
                "options": {"temperature": temperature, "num_ctx": NUM_CTX},
            }, timeout=OLLAMA_TIMEOUT)
            if resp.status_code >= 400:
                raise RuntimeError(f"Ollama error {resp.status_code}: {resp.text[:300]}")
            return resp.json()["response"].strip()
        except requests.exceptions.ReadTimeout as e:
            last_error = e
            print(
                f"[llm_backend] Ollama call timed out "
                f"(attempt {attempt}/{OLLAMA_MAX_RETRIES + 1}, "
                f"timeout={OLLAMA_TIMEOUT}s) -- retrying..."
            )
    raise last_error


# Closed set of Tamil grammar / question words as typed in Latin letters.
# Deliberately contains NO science vocabulary, so it works for any topic.
TANGLISH_MARKERS = {
    "enna", "ena", "na", "endral", "entral", "enral", "endru", "yenna",
    "eppadi", "epdi", "epadi", "yaar", "yaaru", "yar", "edhu", "ethu", "evvalavu",
    "evlo", "ethanai", "enge", "engu", "eppothu", "epo", "yen", "ean",
    "ku", "kku", "la", "oda", "ode", "irukku", "irukkum", "iruku", "irukka",
    "irukkaa", "ulla", "illa", "illai", "vithyasam", "vidhyasam", "vithiyasam",
    "verupadu", "verupaadu", "seiyum", "seyum", "aagum", "agum", "aagudhu",
    "pannum", "panrathu", "pannunga", "solluga", "sollunga", "sollu",
    "vilakkam", "vilakkunga", "patri", "pathi", "thaan", "dhaan", "mudiyum",
    "kaattunga", "kaatunga", "kudunga", "kodunga", "eduthukkaatu",
}


def detect_language(text):
    """
    Returns exactly one of: "English", "Tamil", "Tanglish".

    1. Tamil script -> "Tamil" (character check, no LLM).
    2. Any Tamil grammar/question word from TANGLISH_MARKERS -> "Tanglish"
       (no LLM). The LLM alone misread mixed sentences such as
       "length mass time na enna" as English.
    3. Otherwise the LLM decides (fallback, same as before).
    """
    tamil_chars = sum(1 for c in text if '\u0B80' <= c <= '\u0BFF')
    if tamil_chars > len(text) * 0.3:
        return "Tamil"

    tokens = set(re.findall(r"[a-z]+", text.lower()))
    if tokens & TANGLISH_MARKERS:
        return "Tanglish"

    prompt = f"""Is the following text standard English, or is it Tamil words written
using English (Latin) letters -- sometimes called "Tanglish"?

Text: {text}

Answer with exactly one word: ENGLISH or TANGLISH."""
    raw = _call_ollama(prompt).strip().upper()
    return "Tanglish" if "TANGLISH" in raw else "English"

'''def translate_query_to_english(query, language):
    """
    query: the actual_query text (standard prefix already stripped).
    language: "Tamil" or "Tanglish", as returned by detect_language().
               Not called at all if language == "English".

    The translation must preserve exact meaning/intent -- not answer the
    question, not add or remove information, not reinterpret it. This
    instruction is in the prompt itself, not just a docstring promise.
    """
    script_note = (
        "written in Tamil script" if language == "Tamil"
        else "Tamil words written using English letters (Tanglish)"
    )
    prompt = f"""Translate the following {script_note} into English.
Preserve the EXACT meaning and intent. Do not answer the question. Do not
add information. Do not remove information. Do not reinterpret it.
Output ONLY the English translation, nothing else.

Text: {query}

English translation:"""
    return _call_ollama(prompt)'''

def translate_query_to_english(query, language):
    """
    query: the actual_query text (standard prefix already stripped).
    language: "Tamil" or "Tanglish", as returned by detect_language().
              Not called at all if language == "English".

    The translation must preserve exact meaning/intent -- not answer the
    question, not add or remove information, not reinterpret it.
    """
    if language == "Tamil" and translate_nllb.available():
        return translate_nllb.tamil_to_english(query)
    script_note = (
        "written in Tamil script" if language == "Tamil"
        else "Tamil words written using English letters (Tanglish)"
    )
    examples = """Example 1:
Tamil: தொடர் சுற்று என்றால் என்ன
English: What is a series circuit

Example 2:
Tamil: இணையான சுற்று மற்றும் பக்க இணைப்பு சுற்று வேறுபாடு என்ன
English: What is the difference between a parallel circuit and a series circuit

Example 3:
Tamil: ஒளிச்சேர்க்கை செயல்முறை என்ன
English: What is the process of photosynthesis

Example 4:
Tamil: ஒளியின் விலகல் என்றால் என்ன
English: What is refraction of light"""

    # Question-intent words (closed set, no science vocabulary). The model
    # was dropping these in Tanglish ("vithyasam" -> lost "difference").
    glossary = ""
    if language == "Tanglish":
        glossary = """
Tanglish question words (always keep their meaning in the translation):
enna / ena / na enna / endral enna = what is; eppadi = how; yaar = who;
ethanai = how many; yen = why; vithyasam / verupadu = difference;
gunangal / panbugal = properties; vagaigal / types = types;
kaaranam / kaaranangal = cause / causes; payan / payanbadu = use / uses;
ulla = between (in "X ku Y ku ulla vithyasam" = difference between X and Y);
velai seiyum = works / how it works; irukku = is present / are there.
"""

    prompt = f"""Translate the following {script_note} into English.
This text is a STUDENT'S SCIENCE TEXTBOOK QUESTION -- translate science
terms as the fixed technical term they are (e.g. "தொடர் சுற்று" is the
technical term "series circuit", not a literal word-by-word gloss),
not word-by-word. Preserve the EXACT meaning and intent. Do not answer
the question. Do not add information. Do not remove information. Do
not reinterpret it. Output ONLY the English translation, nothing else.

{examples}
{glossary}
Now translate this one:

Text: {query}

English translation:"""
    return _call_ollama(prompt, temperature=0.0)


def generate_claims(query_en, chunks):
    """
    UNCHANGED LOGIC from the original -- only the key the caller uses to
    identify each chunk changed (alignment_id instead of the old flat id);
    this function itself never hardcoded that key name, it just echoes
    back whatever "id" field is present in each chunk dict's reference.

    query_en: the query, always English by this point.
    chunks: list of {"alignment_id": ..., "english": ...} -- the top-10
            reranked chunks.

    Returns a list of {"alignment_id": ..., "claim": ...}.
    """
    numbered = "\n".join(f"{i+1}. [{c['alignment_id']}] {c['english']}" for i, c in enumerate(chunks))
    prompt = f"""You are helping answer a student's question using ONLY the passages below.
Question: {query_en}

For EACH numbered passage, extract ONE single, clean, atomic claim that is
directly supported by THAT passage's own text. Do not add outside knowledge.
Do not combine facts from different passages. If a passage is irrelevant or
just a heading with no real content, skip it.

Passages:
{numbered}

Output STRICT JSON only, a list of objects: [{{"alignment_id": "...", "claim": "..."}}]
No other text before or after the JSON."""
    raw = _call_ollama(prompt)
    raw = raw.strip().removeprefix("```json").removeprefix("```").removesuffix("```").strip()
    try:
        return json.loads(raw)
    except json.JSONDecodeError:
        raise ValueError(f"Claim generation did not return valid JSON:\n{raw}")


def check_pair_quality(english, tamil):
    """
    Same logic as the original -- compares a CORPUS-aligned (english, tamil)
    pair, not a generated claim against Tamil. Verdict renamed from
    "DIFFERENT" to "WRONG" to match the spec's exact SAME/PARTIAL/WRONG
    terminology.
    """
    prompt = f"""Do these two sentences express the same claim?

English: {english}
Tamil: {tamil}

Answer with exactly one word on the first line -- SAME, PARTIAL, or WRONG --
then a brief one-line reason on the second line. Nothing else."""
    raw = _call_ollama(prompt)
    first_line = raw.strip().splitlines()[0].strip().upper()
    if "SAME" in first_line:
        return "SAME"
    elif "PARTIAL" in first_line:
        return "PARTIAL"
    else:
        return "WRONG"


'''def translate_claim_to_tamil(claim_en):
    """
    UNCHANGED from the original. Used for PARTIAL and WRONG verdicts: the
    corpus's own aligned Tamil isn't trustworthy, so the verified English
    CLAIM (not the raw chunk) is translated directly instead.
    """
    prompt = f"""Translate the following English sentence into natural, correct Tamil.
Output ONLY the Tamil translation, nothing else.

English: {claim_en}

Tamil translation:"""
    return _call_ollama(prompt, temperature=0.2)'''

def translate_claim_to_tamil(claim_en):
    """
    Used for PARTIAL and WRONG verdicts: the corpus's own aligned Tamil
    isn't trustworthy, so the verified English CLAIM (not the raw chunk)
    is translated directly instead.

    CHANGED from the original: same few-shot/domain-anchoring treatment
    applied to translate_query_to_english() after it was observed live
    producing garbled/nonsensical Tamil for science-circuit claims (e.g.
    "the vector of a continuously rolled point..." for a voltage claim).
    Same root cause -- the model drifting away from the actual science
    terms without an example to anchor it -- so the same fix: a few
    worked examples in this exact domain, plus temperature 0.0 for
    consistency instead of 0.2.
    """
    if translate_nllb.available():
        return translate_nllb.english_to_tamil(claim_en)
    
    examples = """Example 1:
English: In a series circuit, the same current flows through all the components.
Tamil: தொடர் சுற்றில், அனைத்து பாகங்களிலும் ஒரே மின்னோட்டம் பாயும்.

Example 2:
English: The voltage across each parallel branch is the same and equals the source voltage.
Tamil: ஒவ்வொரு இணையான கிளையிலும் உள்ள மின்னழுத்தம் ஒன்றேயாகும், மேலும் அது மூல மின்னழுத்தத்திற்கு சமமாகும்."""

    prompt = f"""Translate the following English sentence into natural, correct Tamil.
This is a SCIENCE TEXTBOOK CLAIM about an electrical circuit -- translate
circuit/electricity terms (current, voltage, series, parallel, branch,
component) using their correct Tamil science-textbook vocabulary, not a
literal word-by-word gloss. Preserve the EXACT meaning. Output ONLY the
Tamil translation, nothing else.

{examples}

Now translate this one:

English: {claim_en}

Tamil translation:"""
    return _call_ollama(prompt, temperature=0.0)


def generate_tamil_claim_directly_from_evidence(tamil_query, tamil_evidence):
    """
    NEW -- used ONLY when check_pair_quality() returns SAME, and ONLY as
    part of the restructured flow where the alignment check now happens
    immediately after retrieval (before any English claim is generated
    at all, not after it like the older generate_tamil_claim_for_same_
    alignment() below, which still exists, still works, and is still
    used by the OTHER path -- see pipeline_v2.py for exactly which path
    calls which function).

    No english_claim parameter -- that is the entire point of this
    function existing. For a SAME-verdict chunk, English generation is
    skipped entirely; the Tamil claim is produced directly from the
    Tamil evidence, answering the Tamil question directly.
    """
    prompt = f"""You are answering a question using Tamil textbook evidence that has
been confirmed to correctly match its English source (so it can be
trusted directly, as-is).

Tamil question: {tamil_query}
Tamil textbook evidence (your source -- may contain OCR errors):
{tamil_evidence}

Extract ONE single, clean, atomic claim from this evidence that is
relevant to answering the question above. Ground it ONLY in this
evidence -- do not add outside knowledge. Fix any OCR errors or awkward
phrasing from the source, but do not change its meaning. If this
evidence is not actually relevant to the question, output exactly:
NOT RELEVANT

Output ONLY the Tamil claim (or "NOT RELEVANT"), nothing else."""
    return _call_ollama(prompt, temperature=0.2)


def generate_tamil_claim_for_same_alignment(tamil_query, english_claim, tamil_evidence):
    """
    Stage 11A -- used ONLY when check_pair_quality() returned SAME.

    Unlike translate_claim_to_tamil(), this does NOT translate the English
    claim from scratch. It uses the corpus's own tamil_evidence as the
    actual source, corrects OCR noise / awkward phrasing in it, and
    confirms it matches the english_claim and the original tamil_query --
    "do not blindly translate when the Tamil textbook evidence is
    correctly aligned" (spec section 24).
    """
    prompt = f"""You are producing a Tamil claim for a question, using CORRECTLY ALIGNED
Tamil textbook evidence as your source (not a translation from English).

Tamil question: {tamil_query}
English claim (for reference, confirms what this evidence supports): {english_claim}
Tamil textbook evidence (your actual source -- may contain OCR errors):
{tamil_evidence}

Output a clean, correct Tamil sentence conveying the same claim as the
evidence above, with OCR errors and awkward phrasing fixed, but the
meaning unchanged from the evidence. Output ONLY the Tamil sentence."""
    return _call_ollama(prompt, temperature=0.2)

def format_final_english_answer(query_en, accepted_claims):
    """
    NO LLM CALL. Lists only the verified claims, so nothing can be
    invented. Repeated sentences are listed once. The sources are shown
    once at the end by Stage 13, not after every line.
    """
    lines = [c['claim'].strip()
             for c in _drop_repeats(accepted_claims, "claim")]
    return "\n".join(lines)


def _norm(text):
    """Text with spaces / zero-width joiners removed, for repeat detection."""
    return re.sub(r"[\s\u200c\u200d]+", "", text).lower()


def _drop_repeats(items, key):
    """Keeps the first item; drops any later item whose text contains, or is
    contained in, an already kept one (the same textbook sentence stored in
    two chunks, e.g. once plain and once with a figure caption in front)."""
    kept, seen = [], []
    for it in items:
        n = _norm(it[key])
        if not n or any(n in s or s in n for s in seen):
            continue
        seen.append(n)
        kept.append(it)
    return kept



'''def format_final_tamil_answer(tamil_query, tamil_claims):
    """
    tamil_claims: list of {"tamil_claim": ..., "source_id": ...} -- each
    already in Tamil, whether produced by generate_tamil_claim_for_same_alignment
    (SAME) or translate_claim_to_tamil (PARTIAL/WRONG). This function's
    only job is composing them into ONE coherent natural-Tamil answer with
    citations -- it does not re-translate or re-verify anything.
    """
    claims_block = "\n".join(f"- {c['tamil_claim']} (ஆதாரம்: {c['source_id']})" for c in tamil_claims)
    prompt = f"""மாணவர் கேள்விக்கு பின்வரும் உறுதிப்படுத்தப்பட்ட தமிழ் உண்மைகளை மட்டும்
பயன்படுத்தி பதிலளிக்கவும். புதிய தகவல் சேர்க்க வேண்டாம். ஒவ்வொரு உண்மையையும்
பயன்படுத்திய பின் அதன் ஆதாரத்தை அடைப்புக்குறிக்குள் குறிப்பிடவும்.

கேள்வி: {tamil_query}

உறுதிப்படுத்தப்பட்ட உண்மைகள்:
{claims_block}

தெளிவான, இயல்பான தமிழில், ஒரு மாணவருக்கு ஏற்ற வகையில், ஆதாரங்களுடன் பதில் எழுதவும்."""
    return _call_ollama(prompt, temperature=0.3)'''

'''def format_final_tamil_answer(tamil_query, tamil_claims):
    """
    tamil_claims: list of {"tamil_claim": ..., "source_id": ...} -- each
    already in Tamil, whether produced by generate_tamil_claim_for_same_alignment
    (SAME) or translate_claim_to_tamil (PARTIAL/WRONG).

    FINAL VERSION -- no LLM call. Every previous version of this function
    (free-form "explain it", then a strict rule-list) failed live in a
    different way: inventing content not in any claim, dropping citations,
    or echoing its own instructions back as output. Those weren't prompt-
    wording problems, they were the structural risk of asking a small
    generative model to do a task that has no business being generative:
    listing already-verified facts with their already-known citations.
    Formatting is deterministic, so it's done deterministically here --
    this guarantees every citation is present and nothing is invented,
    every single time, with no dependence on model behavior at all.
    """
    #lines = [f"{c['tamil_claim']} (ஆதாரம்: {c['source_id']})" for c in tamil_claims]
    lines = [f"{c['tamil_claim'].strip()} (ஆதாரம்: {c['source_id']})"
             for c in _drop_repeats(tamil_claims, "tamil_claim")]
    return "\n".join(lines)'''

def format_final_tamil_answer(tamil_query, tamil_claims):
    """
    tamil_claims: list of {"tamil_claim": ..., "source_id": ...} -- each
    already in Tamil, whether produced by generate_tamil_claim_for_same_alignment
    (SAME) or translate_claim_to_tamil (PARTIAL/WRONG).

    No LLM call -- formatting is deterministic, so nothing is invented.
    Repeated sentences are listed once. The sources are shown once at
    the end by Stage 13, not after every line.
    """
    lines = [c['tamil_claim'].strip()
             for c in _drop_repeats(tamil_claims, "tamil_claim")]
    return "\n".join(lines)


if __name__ == "__main__":
    print("--- detect_language ---")
    print(detect_language("What is a series circuit?"))
    print(detect_language("series circuit endral ena"))
    print(detect_language("தொடர் மின்சுற்று என்றால் என்ன?"))

    print("\n--- translate_query_to_english (Tanglish) ---")
    print(translate_query_to_english("series circuit endral ena", "Tanglish"))

    print("\n--- check_pair_quality (should be SAME) ---")
    print(check_pair_quality(
        "If two or more bulbs are connected in series in a circuit, then that type of circuit is called series circuit.",
        "ஒரு மின்சுற்றில் இரண்டு அல்லது அதற்கு மேற்பட்ட மின்விளக்குகள் தொடரில் இணைக்கப்பட்டால், அந்த வகை மின்சுற்று தொடர் மின்சுற்று எனப்படும்."
    ))