"""
Neighbouring-chunk helpers (read-only -- the dataset is NOT changed).

The textbook was split into small chunks, so a sentence often depends on
the chunk just before it ("The plants which grow in THIS habitat have
thick leaves..."), and a list is often cut after its first line ("The
important nutrients are: 1."). These helpers look up the neighbouring
chunks of the SAME chapter and section from the existing database, at
answer time.

  prev_context(aid)   text of up to 2 chunks just before aid (same section)
  expand_lists(chunks) a chunk that ends with a list start (": 1.", "... 3.")
                       gets its next chunks appended (English AND Tamil of
                       the same alignment ids, so the pair stays aligned)
"""
import re

import tamil_rescue

PREV_CHUNKS = 2       # context for scoring
LIST_NEXT = 3         # chunks appended to a cut-off list
LIST_END = re.compile(r"(:\s*(1\.)?|\b\d+\.)\s*$")


def _parts(aid):
    prefix, num = str(aid).rsplit("_", 1)
    return prefix, int(num), len(num)


def _std(aid):
    return int(str(aid).split("_", 1)[0])


def _section(d, aid):
    return (d["en_meta"].get(aid) or {}).get("logical_section_id")


def _same_section(d, a, b):
    sa, sb = _section(d, a), _section(d, b)
    return not sa or not sb or sa == sb


def _neighbour(aid, offset):
    prefix, n, w = _parts(aid)
    if n + offset < 1:
        return None
    return f"{prefix}_{n + offset:0{w}d}"


def prev_context(aid, n=PREV_CHUNKS):
    """English text of up to n chunks right before aid, oldest first."""
    try:
        d = tamil_rescue._data(_std(aid))
    except Exception:
        return ""
    texts = []
    for k in range(1, n + 1):
        p = _neighbour(aid, -k)
        if not p or p not in d["en_doc"] or not _same_section(d, p, aid):
            break
        texts.insert(0, d["en_doc"][p])
    return " ".join(texts)


def expand_lists(chunks):
    """Returns copies of chunks; a chunk whose English ends with a list
    start gets up to LIST_NEXT following chunks appended."""
    out = []
    for c in chunks:
        en = (c.get("english") or "").strip()
        if not LIST_END.search(en):
            out.append(c)
            continue
        try:
            d = tamil_rescue._data(_std(c["alignment_id"]))
        except Exception:
            out.append(c)
            continue
        c = dict(c)
        added = []
        for k in range(1, LIST_NEXT + 1):
            nxt = _neighbour(c["alignment_id"], k)
            if (not nxt or nxt not in d["en_doc"]
                    or not _same_section(d, nxt, c["alignment_id"])):
                break
            c["english"] = c["english"].rstrip() + " " + d["en_doc"][nxt]
            if c.get("tamil") is not None and d["ta_doc"].get(nxt):
                c["tamil"] = c["tamil"].rstrip() + " " + d["ta_doc"][nxt]
            added.append(nxt)
            if not LIST_END.search(d["en_doc"][nxt].strip()):
                break   # list continues only while chunks keep ending open
        if added:
            c["list_continued_with"] = added
            print(f"[list continued] {c['alignment_id']} + {', '.join(added)}")
        out.append(c)
    return out