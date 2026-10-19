import re
import sys


def parse_log(path):
    log = open(path, encoding="utf-8").read()
    parts = re.split(r"\n={78}\nQUERY (\d+): ", log)
    out = []
    for i in range(1, len(parts), 2):
        b = parts[i + 1]
        m_q = re.search(r"Actual Query:\n(.*)", b)
        m_t = re.search(r"Translated English Query:\n(.*)", b)
        s10 = ""
        if "STAGE 10 -- ALIGNMENT VERIFICATION" in b:
            s10 = b.split("STAGE 10 -- ALIGNMENT VERIFICATION", 1)[1].split("STAGE 6A", 1)[0]
        pairs = re.findall(
            r"English Text:\n(.*?)\nTamil Text:\n(.*?)\nAlignment Classification:\n(\w+)", s10, re.S)
        claims = re.findall(
            r"English Claim:\n(.*?)\nAction:\n.*?\nTamil Translation:\n(.*?)\n\n", b, re.S)
        out.append(dict(
            n=int(parts[i]),
            tamil_q=m_q.group(1).strip() if m_q else "",
            qwen_en=m_t.group(1).strip() if m_t else "",
            pairs=[(e.strip(), t.strip(), v) for e, t, v in pairs],
            claims=[(e.strip(), t.strip()) for e, t in claims],
        ))
    return out


def main():
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")
    path = sys.argv[1]
    data = parse_log(path)
    n_pairs = sum(len(d["pairs"]) for d in data)
    n_claims = sum(len(d["claims"]) for d in data)
    print(f"Parsed {len(data)} queries, {n_pairs} stage-10 pairs, {n_claims} stage-11 translations")
    if "--parse-only" in sys.argv:
        return

    import torch
    from sentence_transformers import SentenceTransformer, util
    from transformers import AutoTokenizer, AutoModelForSeq2SeqLM

    labse = SentenceTransformer("sentence-transformers/LaBSE", device="cpu")

    def sim(a, b):
        e = labse.encode([a, b], convert_to_tensor=True, normalize_embeddings=True)
        return float(util.cos_sim(e[0], e[1]))

    name = "facebook/nllb-200-distilled-600M"
    tok_ta = AutoTokenizer.from_pretrained(name, src_lang="tam_Taml")
    tok_en = AutoTokenizer.from_pretrained(name, src_lang="eng_Latn")
    mt = AutoModelForSeq2SeqLM.from_pretrained(name).eval()

    def nllb(text, tok, tgt_lang):
        enc = tok(text, return_tensors="pt", truncation=True, max_length=256)
        with torch.no_grad():
            out = mt.generate(**enc, forced_bos_token_id=tok.convert_tokens_to_ids(tgt_lang),
                              num_beams=4, max_new_tokens=200)
        return tok.batch_decode(out, skip_special_tokens=True)[0]

    print("\n" + "=" * 70)
    print("A) TAMIL QUERY -> ENGLISH  (LaBSE sim to the Tamil query; higher = closer in meaning)")
    print("=" * 70)
    nllb_wins = 0
    for d in data:
        if not d["tamil_q"]:
            continue
        cand = nllb(d["tamil_q"], tok_ta, "eng_Latn")
        s_q, s_n = sim(d["tamil_q"], d["qwen_en"]), sim(d["tamil_q"], cand)
        nllb_wins += s_n > s_q
        print(f"Q{d['n']:>2} {d['tamil_q']}")
        print(f"     Qwen : {s_q:.3f}  {d['qwen_en']}")
        print(f"     NLLB : {s_n:.3f}  {cand}")
    print(f"\nNLLB closer to the Tamil query in {nllb_wins} of {len(data)} queries")

    print("\n" + "=" * 70)
    print("B) ENGLISH CLAIM -> TAMIL  (LaBSE sim to the English claim)")
    print("=" * 70)
    k = nllb_wins_b = 0
    for d in data:
        for en, qwen_ta in d["claims"]:
            cand = nllb(en, tok_en, "tam_Taml")
            s_q, s_n = sim(en, qwen_ta), sim(en, cand)
            k += 1
            nllb_wins_b += s_n > s_q
            print(f"Q{d['n']:>2} EN : {en}")
            print(f"     Qwen : {s_q:.3f}  {qwen_ta}")
            print(f"     NLLB : {s_n:.3f}  {cand}")
    print(f"\nNLLB closer to the English claim in {nllb_wins_b} of {k} translations")

    print("\n" + "=" * 70)
    print("C) STAGE 10: LaBSE similarity of the corpus (English, Tamil) pair vs Qwen's verdict")
    print("=" * 70)
    seen, rows = set(), []
    for d in data:
        for en, ta, verdict in d["pairs"]:
            if (en, ta) in seen:
                continue
            seen.add((en, ta))
            rows.append((sim(en, ta), verdict, en, ta))
    for v in ("SAME", "PARTIAL", "WRONG"):
        xs = [r[0] for r in rows if r[1] == v]
        if xs:
            print(f"{v:8s} n={len(xs):3d}  mean={sum(xs)/len(xs):.3f}  min={min(xs):.3f}  max={max(xs):.3f}")
    best = (0, 0.0)
    for t in [x / 100 for x in range(40, 96)]:
        agree = sum((r[0] >= t) == (r[1] == "SAME") for r in rows)
        if agree > best[0]:
            best = (agree, t)
    print(f"\nBest single threshold for SAME vs not-SAME: {best[1]:.2f} "
          f"(agrees with Qwen on {best[0]} of {len(rows)} pairs)")
    print("\nLowest-similarity pairs Qwen called SAME (is the pair really mismatched?):")
    for s, v, en, ta in sorted([r for r in rows if r[1] == "SAME"])[:8]:
        print(f"  {s:.3f}  EN: {en[:70]}\n         TA: {ta[:70]}")
    print("\nHighest-similarity pairs Qwen called WRONG/PARTIAL (is the pair really fine?):")
    for s, v, en, ta in sorted([r for r in rows if r[1] != "SAME"], reverse=True)[:8]:
        print(f"  {s:.3f} [{v}] EN: {en[:70]}\n         TA: {ta[:70]}")


if __name__ == "__main__":
    main()