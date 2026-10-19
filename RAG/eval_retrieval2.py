import sys
import time

import numpy as np

from eval_retrieval import GOLD_EN, parse_tamil_log

POOL_K = 50
THRESHOLDS = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]
BGE_MODEL = "BAAI/bge-reranker-v2-m3"


def main():
    if "--out" in sys.argv:
        out = sys.argv[sys.argv.index("--out") + 1]
        sys.stdout = open(out, "w", encoding="utf-8")
    queries = parse_tamil_log(sys.argv[1])

    import db
    import retrieve_chroma
    import rerank
    import translate_nllb
    from sentence_transformers import CrossEncoder

    labse = retrieve_chroma._get_model()
    ce = rerank.Reranker()
    translate_nllb.available()
    bge = CrossEncoder(BGE_MODEL, device="cpu", max_length=256)

    cache = {}

    def data(std):
        if std not in cache:
            en = db.get_standard_english_db(std).get(
                include=["documents", "metadatas", "embeddings"])
            ta = db.get_standard_tamil_db(std).get(
                include=["documents", "embeddings"])

            def norm(x):
                x = np.asarray(x, dtype=np.float32)
                return x / (np.linalg.norm(x, axis=1, keepdims=True) + 1e-12)

            cache[std] = {
                "en_ids": en["ids"], "en_emb": norm(en["embeddings"]),
                "en_doc": dict(zip(en["ids"], en["documents"])),
                "ta_ids": ta["ids"], "ta_emb": norm(ta["embeddings"]),
                "ta_doc": dict(zip(ta["ids"], ta["documents"])),
            }
        return cache[std]

    def top_ids(ids, emb, text, k):
        q = labse.encode([text], convert_to_numpy=True,
                         normalize_embeddings=True)[0]
        return [ids[i] for i in np.argsort(-(emb @ q))[:k]]

    def ce_map(query, aids, d):
        aids = [a for a in aids if a in d["en_doc"]]
        if not aids:
            return {}
        s = ce.model.predict([(query, d["en_doc"][a]) for a in aids])
        return dict(zip(aids, map(float, s)))

    hits_a = 0
    hits_f = {t: 0 for t in THRESHOLDS}
    hits_af = {t: 0 for t in THRESHOLDS}
    wrong_a = 0
    wrong_f = {t: 0 for t in THRESHOLDS}
    f_secs, n_scored = 0.0, 0

    for (n, std, tq, _), gold_en in zip(queries, GOLD_EN):
        d = data(std)
        nllb_en = translate_nllb.tamil_to_english(tq)

        gpool = retrieve_chroma.dense_search(gold_en, std, seed_k=100)
        gscore = ce_map(gold_en, [c["alignment_id"] for c in gpool], d)
        gold = [a for a, s in sorted(gscore.items(), key=lambda x: -x[1])
                if s > 0][:3]

        print("\n" + "=" * 70)
        print(f"Q{n} (std {std}) {tq}")
        print(f"  gold EN: {gold_en}   NLLB: {nllb_en}")
        if not gold:
            print("  no gold chunk -- skipped")
            continue
        n_scored += 1

        def relevant(aid):
            if aid not in gscore:
                gscore.update(ce_map(gold_en, [aid], d))
            return gscore.get(aid, -1) > 0

        apool = retrieve_chroma.dense_search(nllb_en, std, seed_k=100)
        amap = ce_map(nllb_en, [c["alignment_id"] for c in apool], d)
        a_acc = [a for a, s in sorted(amap.items(), key=lambda x: -x[1])
                 if s > 0][:3]
        a_hit = any(a in gold for a in a_acc)
        hits_a += a_hit
        wrong_a += sum(not relevant(a) for a in a_acc)

        t = time.time()
        pool = top_ids(d["ta_ids"], d["ta_emb"], tq, POOL_K)
        pool += [a for a in top_ids(d["en_ids"], d["en_emb"], nllb_en,
                                    POOL_K) if a not in pool]
        pool = [a for a in pool if a in d["ta_doc"] and a in d["en_doc"]]
        s = bge.predict([(tq, d["ta_doc"][a]) for a in pool], batch_size=16)
        f_secs += time.time() - t
        franked = sorted(zip(pool, map(float, s)), key=lambda x: -x[1])

        print(f"  A {'HIT ' if a_hit else 'MISS'} "
              + "; ".join(f"{a[-5:]}({amap[a]:.1f})" for a in a_acc))
        print(f"  F pool={len(pool)}  top-5:")
        for a, sc in franked[:5]:
            mark = "GOLD" if a in gold else ("rel " if relevant(a) else "NOT ")
            print(f"     {sc:.2f} {mark} {a[-5:]} {d['en_doc'][a][:70]}")

        for t_ in THRESHOLDS:
            f_acc = [a for a, sc in franked if sc >= t_][:3]
            f_hit = any(a in gold for a in f_acc)
            hits_f[t_] += f_hit
            hits_af[t_] += a_hit or f_hit
            wrong_f[t_] += sum(not relevant(a) for a in f_acc
                               if a not in a_acc)

    print("\n" + "=" * 70)
    print(f"SUMMARY over {n_scored} queries with a gold chunk")
    print(f"  A (pipeline now): hits {hits_a}, wrong accepted {wrong_a}")
    print(f"  F avg time per query: {f_secs / max(n_scored, 1):.1f}s (CPU)")
    print("  thr   F hits   A+F hits   extra wrong from F")
    for t_ in THRESHOLDS:
        print(f"  {t_:.1f}   {hits_f[t_]:5d}   {hits_af[t_]:8d}   "
              f"{wrong_f[t_]:5d}")


if __name__ == "__main__":
    main()