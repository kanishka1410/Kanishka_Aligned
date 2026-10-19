"""
Dedicated translation model (NLLB-200-distilled-600M) for Tamil <-> English.
Tamil SCRIPT only -- NLLB cannot read Tanglish, which keeps using Qwen.
Loaded once, lazily, on CPU. If it cannot be loaded, available() returns
False and callers fall back to Qwen.
"""
NLLB_MODEL_NAME = "facebook/nllb-200-distilled-600M"

_state = {"tried": False, "ok": False}


def _load():
    if _state["tried"]:
        return _state["ok"]
    _state["tried"] = True
    try:
        import torch
        from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
        _state["torch"] = torch
        _state["tok_ta"] = AutoTokenizer.from_pretrained(NLLB_MODEL_NAME, src_lang="tam_Taml")
        _state["tok_en"] = AutoTokenizer.from_pretrained(NLLB_MODEL_NAME, src_lang="eng_Latn")
        _state["model"] = AutoModelForSeq2SeqLM.from_pretrained(NLLB_MODEL_NAME).eval()
        _state["ok"] = True
    except Exception as e:
        print(f"[translate_nllb] NLLB not available, falling back to Qwen: {type(e).__name__}: {e}")
        _state["ok"] = False
    return _state["ok"]


def available():
    return _load()


def _translate(text, tok, tgt_lang):
    torch = _state["torch"]
    enc = tok(text, return_tensors="pt", truncation=True, max_length=256)
    with torch.no_grad():
        out = _state["model"].generate(
            **enc,
            forced_bos_token_id=tok.convert_tokens_to_ids(tgt_lang),
            num_beams=4,
            no_repeat_ngram_size=6,   # stops loops like "கச்சா கச்சா கச்சா"
            max_new_tokens=200,
        )
    return tok.batch_decode(out, skip_special_tokens=True)[0].strip()


def tamil_to_english(text):
    return _translate(text, _state["tok_ta"], "eng_Latn")


def english_to_tamil(text):
    return _translate(text, _state["tok_en"], "tam_Taml")