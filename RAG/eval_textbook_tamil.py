"""
One-off evaluation -- NOT part of the pipeline. Changes nothing.

Part 1. For every Tamil question, it repeats the pipeline's selection
(NLLB translation -> search -> cross-encoder -> Stage 4T) and looks at
the chunks that would NOT use their textbook Tamil (Tamil check < 0.7
and LaBSE pair score < 0.74 -> the answer line comes from NLLB).
For each one it prints, side by side:
  - the textbook Tamil line and the NLLB Tamil line
  - LaBSE pair score (English chunk vs textbook Tamil)
  - Tamil check score of each line against the Tamil question
    (how well that line answers the question, 0..1)
So we can see whether using the textbook Tamil would be better, and
from which LaBSE score it is safe.

Part 2. Lists every "learning objectives" chunk in the books, so the
filter for them can be checked for mistakes before we add it.

    python eval_textbook_tamil.py --out eval_textbook_tamil_out.txt
or, to also include the old 30 questions:
    python eval_textbook_tamil.py "C:\\...\\pipeline_logs_all_30_queries.txt"
        --out eval_textbook_tamil_out.txt
"""
import contextlib
import io
import re
import sys

FRESH_QUESTIONS = [
    (6, "நிறையின் SI அலகு என்ன?"),
    (6, "சீரான இயக்கம் என்றால் என்ன?"),
    (6, "கலவையைப் பிரித்தெடுக்கும் முறைகள் யாவை?"),
    (6, "நலம் என்றால் என்ன?"),
    (6, "மின் மூலங்கள் என்றால் என்ன?"),
    (6, "காந்த திசைகாட்டி என்றால் என்ன?"),
    (6, "உள்ளீட்டுக் கருவிகள் யாவை?"),
    (6, "காற்று மாசுபாட்டிற்கான காரணங்கள் யாவை?"),
    (6, "செரிமான மண்டலம் என்றால் என்ன?"),
    (6, "இயற்கை காந்தங்கள் என்றால் என்ன?"),
    (7, "அணுவின் அடிப்படைத் துகள்கள் யாவை?"),
    (7, "அணு எண் என்றால் என்ன?"),
    (7, "நிறை எண் என்றால் என்ன?"),
    (7, "மகரந்தச் சேர்க்கையின் வகைகள் யாவை?"),
    (7, "தொற்று நோய்கள் என்றால் என்ன?"),
    (7, "மின்கடத்தாப் பொருள்கள் என்றால் என்ன?"),
    (7, "அம்மீட்டர் என்றால் என்ன?"),
    (7, "இருசொற் பெயரிடும் முறை என்றால் என்ன?"),
    (7, "சமதள ஆடியில் தோன்றும் பிம்பத்தின் பண்புகள் யாவை?"),
    (7, "இயற்கை இழைகள் என்றால் என்ன?"),
    (7, "வெப்பநிலையை அளக்கும் அலகுகள் யாவை?"),
    (7, "சுடரின் அமைப்பு என்ன?"),
]

LO_EN = re.compile(r"learning objectives|students will be able to|"
                   r"after (studying|completing|learning) this", re.I)
LO_TA = re.compile(r"கற்றல்\s*நோக்க|கற்றபின்|கற்ற\s*பின்")

LABSE_BUCKETS = [0.0, 0.5, 0.55, 0.6, 0.65, 0.7, 0.74]


def is_lo(en, ta):
    return bool(LO_EN.search(en or "")) or bool(LO_TA.search(ta or ""))


def main():
    out = sys.stdout
    if "--out" in sys.argv:
        out = open(sys.argv[sys.argv.index("--out") + 1], "w",
                   encoding="utf-8")

    def say(s=""):
        out.write(s + "\n")
        out.flush()

    import pipeline_v2 as pv
    import pair_check
    import rerank
    import tamil_rescue as tr
    import translate_nllb

    ce = rerank.Reranker()
    translate_nllb.available()
    bge = tr.load()

    items = list(FRESH_QUESTIONS)
    try:
        from eval_combined import NEW_QUESTIONS
        items += [(s, q) for s, q, _ in NEW_QUESTIONS]
    except Exception:
        pass
    args = [a for a in sys.argv[1:] if not a.startswith("--")]
    if "--out" in sys.argv:
        args = [a for a in args
                if a != sys.argv[sys.argv.index("--out") + 1]]
    if args:
        from eval_retrieval import parse_tamil_log
        items += [(s, q) for _, s, q, _ in parse_tamil_log(args[0])]

    rows = []   # (labse, p_textbook, p_nllb, is_lo)
    n_lines, n_tb_lines, n_lo_selected = 0, 0, 0

    for qi, (std, tq) in enumerate(items, 1):
        q_en = translate_nllb.tamil_to_english(tq)
        with contextlib.redirect_stdout(io.StringIO()):
            cands = pv.run_dense_retrieval(q_en, std)
            if not cands:
                continue
            top10 = pv.stage4_rerank(q_en, std, cands, ce)
            top10 = pv.stage5_alignment_lookup(top10, std)
            sel = pv.stage4t_tamil_check(tq, q_en, std, top10)

        say("=" * 70)
        say(f"Q{qi} (std {std}) {tq}")
        say(f"  NLLB question: {q_en}")
        if not sel:
            say("  nothing selected (answer = insufficient)")
            continue

        n_lines += len(sel)
        for c in sel:
            if is_lo(c["english"], c.get("tamil")):
                n_lo_selected += 1
                say(f"  LEARNING-OBJECTIVES chunk selected: "
                    f"{c['alignment_id']}")

        rest = [c for c in sel if not c.get("tamil_ok") and c.get("tamil")]
        n_tb_lines += len(sel) - len(rest)
        if not rest:
            say("  all selected lines already use textbook Tamil")
            continue
        verdicts = pair_check.classify_pairs(
            [(c["english"], c["tamil"]) for c in rest])
        for c, (v, sim) in zip(rest, verdicts):
            if v == "SAME":
                n_tb_lines += 1
                continue
            nllb_ta = translate_nllb.english_to_tamil(c["english"])
            p_tb, p_nl = map(float, bge.predict(
                [(tq, c["tamil"]), (tq, nllb_ta)]))
            lo = is_lo(c["english"], c["tamil"])
            rows.append((sim, p_tb, p_nl, lo))
            say("-" * 70)
            say(f"  {c['alignment_id']}  LaBSE={sim:.3f} ({v})"
                f"{'  [LEARNING OBJ]' if lo else ''}")
            say(f"  English : {c['english'][:300]}")
            say(f"  TEXTBOOK (check {p_tb:.2f}): {c['tamil'][:300]}")
            say(f"  NLLB     (check {p_nl:.2f}): {nllb_ta[:300]}")

    say("=" * 70)
    say("PART 1 SUMMARY")
    say(f"questions: {len(items)}   answer lines selected: {n_lines}")
    say(f"lines already using textbook Tamil: {n_tb_lines}")
    say(f"lines that would come from NLLB: {len(rows)}")
    say(f"learning-objectives chunks selected: {n_lo_selected}")
    say("LaBSE range     lines  textbook>=NLLB  textbook_check>=0.3")
    for lo_, hi in zip(LABSE_BUCKETS, LABSE_BUCKETS[1:] + [1.01]):
        b = [r for r in rows if lo_ <= r[0] < hi]
        if not b:
            continue
        better = sum(r[1] >= r[2] for r in b)
        good = sum(r[1] >= 0.3 for r in b)
        say(f"{lo_:.2f}-{hi:.2f}   {len(b):9d}  {better:14d}  {good:19d}")

    say("")
    say("=" * 70)
    say("PART 2: every learning-objectives chunk in the books")
    for std in (6, 7):
        d = tr._data(std)
        hits = [a for a in d["en_ids"]
                if is_lo(d["en_doc"][a], d["ta_doc"].get(a))]
        say(f"Standard {std}: {len(hits)} chunks")
        for a in hits:
            say(f"  {a}")
            say(f"    EN: {d['en_doc'][a][:160]}")
            say(f"    TA: {(d['ta_doc'].get(a) or '')[:160]}")


if __name__ == "__main__":
    main()