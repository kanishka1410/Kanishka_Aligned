"""
One-off measurement -- NOT part of the pipeline. Reads only, changes
nothing.

Part 1  DAMAGED TAMIL TEXT (for fix 3, the cleanup)
  Per chapter: how many Tamil chunks look damaged, measured three ways:
    labse   English vs Tamil similarity of the pair (low = mismatch/damage)
    bad     broken letter sequences (e.g. a vowel sign with no letter
            before it, double pulli "்்", a digit inside a word "ர௫")
    rare    share of words that appear only once in the whole book
            (garbled words are almost always unique)
  Then the "ரு written for டு" candidates: words like பயன்பருத்த whose
  டு form (பயன்படுத்த) also appears in the book more often.

Part 2  RELEVANCE GAP (for idea C)
  For 34 questions (correct English wording), every chunk the
  cross-encoder accepts (score > 0), with its gap below the best chunk.
  I read these to decide how far below the best is still relevant.

    python eval_damage.py --out eval_damage_out.txt
"""
import collections
import contextlib
import io
import re
import sys
import unicodedata

import numpy as np

GOLD_QUESTIONS = [
    (6, "What is the SI unit of mass?"),
    (6, "What is uniform motion?"),
    (6, "What are the methods of separating mixtures?"),
    (6, "What is health?"),
    (6, "What are the sources of electricity?"),
    (6, "What is a magnetic compass?"),
    (6, "What are input devices?"),
    (6, "What are the causes of air pollution?"),
    (6, "What is the digestive system?"),
    (6, "What are natural magnets?"),
    (7, "What are the fundamental particles of an atom?"),
    (7, "What is atomic number?"),
    (7, "What is mass number?"),
    (7, "What are the types of pollination?"),
    (7, "What are communicable diseases?"),
    (7, "What are insulators?"),
    (7, "What is an ammeter?"),
    (7, "What is binomial nomenclature?"),
    (7, "What are the properties of the image formed by a plane mirror?"),
    (7, "What are natural fibres?"),
    (7, "What are the units for measuring temperature?"),
    (7, "What is the structure of a flame?"),
]

CONS = "\u0B95-\u0BB9"          # Tamil consonants
SIGNS = "\u0BBE-\u0BCD\u0BD7"   # vowel signs + pulli
BAD = re.compile(
    rf"(?<![{CONS}])[{SIGNS}]"          # sign with no letter before it
    rf"|\u0BCD\u0BCD"                   # double pulli
    rf"|[{CONS}][\u0BE6-\u0BEF]")       # Tamil digit glued to a letter
WORD = re.compile(r"[\u0B80-\u0BFF]+")


def norm(t):
    t = unicodedata.normalize("NFC", t or "")
    return t.replace("\u200c", "").replace("\u200d", "")


def bad_count(t):
    return len(BAD.findall(norm(t)))


def words(t):
    return WORD.findall(norm(t))


def part1(say, tr):
    say("=" * 70)
    say("PART 1: DAMAGED TAMIL TEXT")
    for std in (6, 7):
        d = tr._data(std)
        en_ix = {a: i for i, a in enumerate(d["en_ids"])}
        ta_ix = {a: i for i, a in enumerate(d["ta_ids"])}
        aids = [a for a in d["en_ids"] if a in ta_ix]

        freq = collections.Counter()
        for a in aids:
            freq.update(words(d["ta_doc"][a]))

        rows = {}
        for a in aids:
            ta = d["ta_doc"][a]
            sim = float(d["en_emb"][en_ix[a]] @ d["ta_emb"][ta_ix[a]])
            ws = words(ta)
            rare = sum(freq[w] == 1 for w in ws) / max(len(ws), 1)
            rows[a] = (sim, bad_count(ta), rare)

        # ரு-for-டு candidates
        pairs = collections.Counter()
        for w, n in freq.items():
            if "ரு" not in w:
                continue
            for m in re.finditer("ரு", w):
                v = w[:m.start()] + "டு" + w[m.end():]
                if freq.get(v, 0) > n:
                    pairs[(w, v)] = n
        ru_words = {w for w, _ in pairs}

        say("")
        say(f"STANDARD {std}: {len(aids)} chunks")
        say("chapter              chunks  labse  %labse<0.6  bad/chunk"
            "  rare%  ru-hits")
        by_ch = collections.defaultdict(list)
        for a in aids:
            by_ch[d["en_meta"][a]["source_id"]].append(a)
        for ch in sorted(by_ch):
            ids = by_ch[ch]
            sims = [rows[a][0] for a in ids]
            low = 100 * sum(s < 0.6 for s in sims) / len(ids)
            bad = sum(rows[a][1] for a in ids) / len(ids)
            rare = 100 * sum(rows[a][2] for a in ids) / len(ids)
            ru = sum(sum(w in ru_words for w in words(d["ta_doc"][a]))
                     for a in ids)
            say(f"{ch:20s} {len(ids):6d}  {np.mean(sims):.2f}  "
                f"{low:9.0f}%  {bad:9.2f}  {rare:5.0f}  {ru:7d}")

        say("")
        say(f"'ரு for டு' candidates in standard {std} "
            f"({len(pairs)} word forms), most frequent first:")
        top = sorted(pairs.items(), key=lambda x: -x[1])[:40]
        for (w, v), n in top:
            say(f"  {w}  ->  {v}   (x{n}, correct form x{freq[v]})")

        say("")
        say(f"15 most damaged chunks in standard {std} (rare% x bad):")
        worst = sorted(aids, key=lambda a: -(rows[a][2] + 0.2 * rows[a][1]))
        for a in worst[:15]:
            sim, bad, rare = rows[a]
            say(f"  {a}  labse={sim:.2f} bad={bad} rare={100*rare:.0f}%")
            say(f"    TA: {d['ta_doc'][a][:150]}")
            say(f"    EN: {d['en_doc'][a][:150]}")


def part2(say, tr):
    import pipeline_v2 as pv
    import rerank

    say("")
    say("=" * 70)
    say("PART 2: RELEVANCE GAP (cross-encoder score below the best chunk)")
    ce = rerank.Reranker()
    qs = list(GOLD_QUESTIONS)
    try:
        from eval_combined import NEW_QUESTIONS
        qs += [(s, en) for s, _, en in NEW_QUESTIONS]
    except Exception:
        pass

    gaps = []
    for qi, (std, q) in enumerate(qs, 1):
        with contextlib.redirect_stdout(io.StringIO()):
            cands = pv.run_dense_retrieval(q, std)
            top10 = pv.stage4_rerank(q, std, cands, ce) if cands else []
            top10 = pv.stage5_alignment_lookup(top10, std)
        top10 = [c for c in top10
                 if not tr.is_learning_objectives(c["english"], c.get("tamil"))
                 and not tr._is_question(c["english"])]
        acc = [c for c in top10 if c["rerank_score"] > 0]
        say("-" * 70)
        say(f"Q{qi} (std {std}) {q}")
        if not acc:
            say("  no chunk with score > 0")
            continue
        best = max(c["rerank_score"] for c in acc)
        for c in acc:
            g = best - c["rerank_score"]
            gaps.append(g)
            say(f"  ce={c['rerank_score']:5.2f} gap={g:5.2f}  "
                f"{c['alignment_id'][-5:]}  {c['english'][:110]}")

    say("")
    say("gap range   accepted chunks")
    for lo, hi in [(0, 1), (1, 2), (2, 3), (3, 4), (4, 5), (5, 6),
                   (6, 8), (8, 99)]:
        say(f"{lo:2d}-{hi:<2d}       {sum(lo <= g < hi for g in gaps):5d}")


def main():
    out = sys.stdout
    if "--out" in sys.argv:
        out = open(sys.argv[sys.argv.index("--out") + 1], "w",
                   encoding="utf-8")

    def say(s=""):
        out.write(s + "\n")
        out.flush()

    import tamil_rescue as tr
    part1(say, tr)
    part2(say, tr)


if __name__ == "__main__":
    main()