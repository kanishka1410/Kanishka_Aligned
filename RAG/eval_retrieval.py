import re
import sys
import time

import numpy as np

GOLD_EN = [
    "What are length, mass and time?",
    "What is zero error?",
    "What is the difference between contact force and non-contact force?",
    "Distinguish between solids, liquids and gases.",
    "What is a pure substance? What is a mixture?",
    "What is a balanced diet?",
    "What are deficiency diseases?",
    "What are the generations of computers?",
    "What is an electric circuit?",
    "What are solvent and solute?",
    "What are the components of air?",
    "What is the difference between unicellular and multicellular organisms?",
    "What are the major organ systems of the human body?",
    "What are the properties of a magnet?",
    "What are the types of pollution?",
    "What are fundamental units and derived units?",
    "What is the difference between distance and displacement?",
    "What is an atom? What is a molecule?",
    "How do plants reproduce?",
    "What are the causes of diseases?",
    "How to create a file and a folder in a computer?",
    "What is temperature?",
    "What are series and parallel circuits?",
    "What are physical change and chemical change?",
    "What are the differences between a plant cell and an animal cell?",
    "What is dichotomous classification?",
    "How is light reflected?",
    "What are the functions of ISRO?",
    "What are polymers?",
    "What is combustion? What are its types?",
]

TAMIL_SIDE_K = 100
BGE_MODEL = "BAAI/bge-reranker-v2-m3"


def parse_tamil_log(path):
    log = open(path, encoding="utf-8").read()
    parts = re.split(r"\n={78}\nQUERY (\d+): ", log)
    out = []
    for i in range(1, len(parts), 2):
        b = parts[i + 1]
        std = int(re.search(r"Extracted Standard:\n(\d+)", b).group(1))
        tq = re.search(r"Actual Query:\n(.*)", b).group(1).strip()
        m = re.search(r"Translated English Query:\n(.*)", b)
        out.append((int(parts[i]), std, tq, m.group(1).strip() if m else ""))
    return out


def main():
    if "--out" in sys.argv:
        sys.stdout = open(sys.argv[sys.argv.index("--out") + 1], "w", encoding="utf-8")
    elif hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")
    queries = parse_tamil_log(sys.argv[1])
    use_bge = "--no-bge" not in sys.argv
    print(f"Parsed {len(queries)} Tamil queries")

    import db
    import retrieve_chroma
    import rerank
    import translate_nllb

    labse = retrieve_chroma._get_model()
    ce = rerank.Reranker()
    if not translate_nllb.available():
        print("NLLB not available -- stopping.")
        return

    bge = None
    if use_bge:
        import torch
        from sentence_transformers import CrossEncoder
        device = "cuda" if torch.cuda.is_available() else "cpu"
        print(f"Loading {BGE_MODEL} on {device} ...")
        bge = CrossEncoder(BGE_MODEL, device=device, max_length=512)
        if device == "cuda":
            bge.model.half()

    cache = {}

    def std_data(std):
        if std not in cache:
            eng = db.get_standard_english_db(std).get(include=["documents", "metadatas"])
            tam = db.get_standard_tamil_db(std).get(include=["documents", "embeddings"])
            emb = np.asarray(tam["embeddings"], dtype=np.float32)
            emb /= (np.linalg.norm(emb, axis=1, keepdims=True) + 1e-12)
            cache[std] = {
                "en": {i: (d, m) for i, d, m in zip(eng["ids"], eng["documents"], eng["metadatas"])},
                "ta_ids": tam["ids"],
                "ta_docs": dict(zip(tam["ids"], tam["documents"])),
                "ta_emb": emb,
            }
        return cache[std]

    def english_side(query_en, std):
        return retrieve_chroma.dense_search(query_en, std, seed_k=100)

    def tamil_side(tamil_q, std):
        d = std_data(std)
        q = labse.encode([tamil_q], convert_to_numpy=True, normalize_embeddings=True)[0]
        sims = d["ta_emb"] @ q
        out = []
        for idx in np.argsort(-sims)[:TAMIL_SIDE_K]:
            aid = d["ta_ids"][idx]
            if aid not in d["en"]:
                continue
            doc, meta = d["en"][aid]
            out.append({"alignment_id": aid, "source_id": meta["source_id"],
                        "english": doc, "score": float(sims[idx])})
        return out

    def union(*lists):
        seen = {}
        for lst in lists:
            for c in lst:
                seen.setdefault(c["alignment_id"], c)
        return list(seen.values())

    def ce_scores(query_en, cands):
        if not cands:
            return {}
        s = ce.model.predict([(query_en, c["english"]) for c in cands])
        return {c["alignment_id"]: float(x) for c, x in zip(cands, s)}

    def top3(scores, floor):
        ranked = sorted(scores.items(), key=lambda kv: -kv[1])
        if floor:
            ranked = [kv for kv in ranked if kv[1] > 0]
        return ranked[:3]

    methods = ["A", "B", "C", "D"] + (["E", "F"] if bge else [])
    hits = {m: 0 for m in methods}
    secs = {m: 0.0 for m in methods}
    scored_queries = 0

    for (n, std, tq, qwen_en), gold_en in zip(queries, GOLD_EN):
        t0 = time.time()
        nllb_en = translate_nllb.tamil_to_english(tq)
        t_tr = time.time() - t0

        gold_scores = ce_scores(gold_en, english_side(gold_en, std))
        gold = [aid for aid, s in top3(gold_scores, floor=True)]

        print("\n" + "=" * 78)
        print(f"Q{n} (std {std}) {tq}")
        print(f"  gold EN : {gold_en}")
        print(f"  NLLB    : {nllb_en}")
        print(f"  Qwen    : {qwen_en}")
        d = std_data(std)
        for aid in gold:
            print(f"  GOLD    {aid} ({gold_scores[aid]:.2f}) {d['en'][aid][0][:90]}")
        if not gold:
            print("  (no gold chunk with a positive score -- query skipped in totals)")
            continue
        scored_queries += 1

        results = {}
        t = time.time(); cand_a = english_side(nllb_en, std)
        results["A"] = top3(ce_scores(nllb_en, cand_a), True); secs["A"] += time.time() - t + t_tr

        t = time.time(); cand_b = english_side(qwen_en, std) if qwen_en else []
        results["B"] = top3(ce_scores(qwen_en, cand_b), True) if qwen_en else []
        secs["B"] += time.time() - t

        t = time.time(); cand_ta = tamil_side(tq, std); cand_c = union(cand_a, cand_ta)
        results["C"] = top3(ce_scores(nllb_en, cand_c), True); secs["C"] += time.time() - t + t_tr

        t = time.time(); cand_d = union(cand_c, cand_b)
        s1, s2 = ce_scores(nllb_en, cand_d), (ce_scores(qwen_en, cand_d) if qwen_en else {})
        results["D"] = top3({a: max(s1[a], s2.get(a, -99)) for a in s1}, True)
        secs["D"] += time.time() - t + t_tr

        if bge:
            t = time.time()
            s = bge.predict([(tq, c["english"]) for c in cand_c], batch_size=32)
            results["E"] = top3({c["alignment_id"]: float(x) for c, x in zip(cand_c, s)}, False)
            secs["E"] += time.time() - t + t_tr
            t = time.time()
            s = bge.predict([(tq, d["ta_docs"].get(c["alignment_id"], "")) for c in cand_c], batch_size=32)
            results["F"] = top3({c["alignment_id"]: float(x) for c, x in zip(cand_c, s)}, False)
            secs["F"] += time.time() - t

        print(f"  candidates: A={len(cand_a)}  Tamil-side={len(cand_ta)}  C={len(cand_c)}  D={len(cand_d)}")
        for m in methods:
            ok = any(aid in gold for aid, _ in results[m])
            hits[m] += ok
            shown = "; ".join(f"{aid.split('_align_')[-1]}({s:.2f}){'*' if aid in gold else ''}"
                              for aid, s in results[m]) or "(nothing above floor)"
            print(f"  {m} {'HIT ' if ok else 'MISS'} {shown}")

    print("\n" + "=" * 78)
    print(f"SUMMARY over {scored_queries} queries with a gold chunk  (* = gold chunk)")
    labels = {
        "A": "NLLB translation, current retrieval (pipeline now)",
        "B": "Qwen translation, current retrieval (pipeline before)",
        "C": "A + Tamil-side candidates, cross-encoder on NLLB",
        "D": "C + Qwen candidates, best of both translations",
        "E": "C candidates, bge reranker (Tamil query, English chunk)",
        "F": "C candidates, bge reranker (Tamil query, Tamil chunk)",
    }
    for m in methods:
        print(f"  {m}: {hits[m]:2d}/{scored_queries} hits   avg {secs[m] / max(scored_queries, 1):5.1f}s   {labels[m]}")


if __name__ == "__main__":
    main()