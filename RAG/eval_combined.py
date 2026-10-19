"""
One-off evaluation -- NOT part of the pipeline.
OLD = pipeline now; NEW = English search AND Tamil search always,
accept if cross-encoder > 0 OR Tamil check >= T, best 4 by relevance.
"""
import math
import sys
import time

from eval_retrieval import GOLD_EN, parse_tamil_log

NEW_QUESTIONS = [
    (6, "இடமாறு தோற்றப் பிழை என்றால் என்ன?", "What is parallax error?"),
    (6, "வேகம் என்றால் என்ன?", "What is speed?"),
    (6, "மின்கலன் என்றால் என்ன?", "What is an electric cell?"),
    (6, "காந்தப் பொருள்கள் என்றால் என்ன?", "What are magnetic materials?"),
    (6, "ஆக்சிஜனின் பயன்கள் யாவை?", "What are the uses of oxygen?"),
    (6, "மின் கடத்திகள் என்றால் என்ன?", "What are electrical conductors?"),
    (7, "திசைவேகம் என்றால் என்ன?", "What is velocity?"),
    (7, "முடுக்கம் என்றால் என்ன?", "What is acceleration?"),
    (7, "வெப்ப விரிவு என்றால் என்ன?", "What is thermal expansion?"),
    (7, "ஒளி எதிரொளிப்பு விதிகள் யாவை?",
     "What are the laws of reflection?"),
    (7, "செயற்கை இழைகள் என்றால் என்ன?", "What are synthetic fibres?"),
    (7, "தீயணைப்பான்களின் வகைகள் யாவை?",
     "What are the types of fire extinguishers?"),
]
REFUSE_QUESTIONS = [
    (6, "கருந்துளை என்றால் என்ன?"),
    (7, "அணுக்கரு பிளவு என்றால் என்ன?"),
]
THRESHOLDS = [0.6, 0.7, 0.8, 0.9]
MAX_LINES = 4


def sig(x):
    return 1 / (1 + math.exp(-x))


def main():
    out = None
    if "--out" in sys.argv:
        out = open(sys.argv[sys.argv.index("--out") + 1], "w",
                   encoding="utf-8")

    def say(s=""):
        (out or sys.stdout).write(s + "\n")
        (out or sys.stdout).flush()

    import retrieve_chroma
    import rerank
    import translate_nllb
    import tamil_rescue as tr

    ce = rerank.Reranker()
    translate_nllb.available()
    bge = tr.load()

    items = [(std, tq, gold)
             for (_, std, tq, _), gold
             in zip(parse_tamil_log(sys.argv[1]), GOLD_EN)]
    items += NEW_QUESTIONS
    items += [(std, tq, None) for std, tq in REFUSE_QUESTIONS]

    def ce_map(query, aids, d):
        aids = [a for a in aids if a in d["en_doc"]]
        if not aids:
            return {}
        s = ce.model.predict([(query, d["en_doc"][a]) for a in aids])
        return dict(zip(aids, map(float, s)))

    def usable(a, d):
        return (a in d["ta_doc"] and a in d["en_doc"]
                and not tr._is_question(d["ta_doc"][a])
                and not tr._is_question(d["en_doc"][a]))

    keys = ["OLD"] + [f"NEW{t}" for t in THRESHOLDS]
    tot = {k: {"hit": 0, "rel": 0, "off": 0} for k in keys}
    refuse_off = {k: 0 for k in keys}
    n_gold, bge_secs = 0, 0.0

    for qi, (std, tq, gold_en) in enumerate(items, 1):
        d = tr._data(std)
        nllb_en = translate_nllb.tamil_to_english(tq)
        say("=" * 70)
        say(f"Q{qi} (std {std}) {tq}")
        say(f"  NLLB: {nllb_en}   gold EN: {gold_en}")

        gscore, gold = {}, []
        if gold_en:
            gpool = retrieve_chroma.dense_search(gold_en, std, seed_k=100)
            gscore = ce_map(gold_en, [c["alignment_id"] for c in gpool], d)
            gold = [a for a, s in sorted(gscore.items(), key=lambda x: -x[1])
                    if s > 0 and usable(a, d)][:3]
            if not gold:
                say("  no gold chunk -- not counted")
        if gold:
            n_gold += 1

        def relevant(a):
            if not gold_en:
                return False
            if a not in gscore:
                gscore.update(ce_map(gold_en, [a], d))
            return gscore.get(a, -1) > 0

        apool = retrieve_chroma.dense_search(nllb_en, std, seed_k=100)
        amap = ce_map(nllb_en, [c["alignment_id"] for c in apool], d)
        top10 = [a for a, _ in sorted(amap.items(), key=lambda x: -x[1])][:10]

        pool = dict(tr._top(d["ta_ids"], d["ta_emb"], tq, tr.POOL_K))
        for a, s in tr._top(d["en_ids"], d["en_emb"], nllb_en, tr.POOL_K):
            pool.setdefault(a, s)
        cand = [a for a in dict.fromkeys(top10 + list(pool)) if usable(a, d)]
        t0 = time.time()
        probs = bge.predict([(tq, d["ta_doc"][a]) for a in cand],
                            batch_size=16)
        bge_secs += time.time() - t0
        bmap = dict(zip(cand, map(float, probs)))

        sel = {}
        acc = [a for a in top10 if amap[a] > 0 and usable(a, d)]
        if acc:
            sel["OLD"] = acc[:MAX_LINES]
        else:
            sel["OLD"] = [a for a, p in sorted(bmap.items(),
                                               key=lambda x: -x[1])
                          if p >= tr.RESCUE_THRESHOLD][:tr.MAX_RESCUE]
        for t in THRESHOLDS:
            score = {}
            for a in cand:
                ok_en = a in top10 and amap[a] > 0
                ok_ta = bmap[a] >= t
                if ok_en or ok_ta:
                    score[a] = max(bmap[a], sig(amap[a]) if ok_en else 0)
            sel[f"NEW{t}"] = [a for a, _ in sorted(score.items(),
                                                   key=lambda x: -x[1])
                              ][:MAX_LINES]

        for k in keys:
            ch = sel[k]
            rel = sum(relevant(a) for a in ch)
            off = len(ch) - rel
            hit = any(a in gold for a in ch)
            if gold:
                tot[k]["hit"] += hit
                tot[k]["rel"] += rel
                tot[k]["off"] += off
            if not gold_en:
                refuse_off[k] += len(ch)
            tag = "HIT " if hit else ("    " if gold else "--  ")
            say(f"  {k:7s} {tag} rel={rel} off={off}")
        say("  chunks chosen by NEW0.7 (Tamil check score, GOLD/rel/NOT):")
        for a in sel["NEW0.7"]:
            m = "GOLD" if a in gold else ("rel " if relevant(a) else "NOT ")
            en = amap.get(a)
            en_s = f"{en:5.1f}" if en is not None else "  -  "
            say(f"    ta={bmap[a]:.2f} en={en_s} {m} {d['ta_doc'][a][:60]}")

    say("=" * 70)
    say(f"SUMMARY: {n_gold} questions with a gold chunk, "
        f"{len(REFUSE_QUESTIONS)} should-refuse questions")
    say(f"Tamil check time: avg {bge_secs / len(items):.1f}s per question")
    say("method   hits  relevant-lines  off-topic-lines  refuse-violations")
    for k in keys:
        v = tot[k]
        say(f"{k:7s} {v['hit']:5d} {v['rel']:14d} {v['off']:16d} "
            f"{refuse_off[k]:18d}")


if __name__ == "__main__":
    main()