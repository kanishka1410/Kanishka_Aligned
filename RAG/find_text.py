import sys

import db
import retrieve_chroma
import rerank

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

standard = int(sys.argv[1])
word = sys.argv[2].lower()
query = sys.argv[3]

eng = db.get_standard_english_db(standard)
tam = db.get_standard_tamil_db(standard)
everything = eng.get(include=["documents", "metadatas", "embeddings"])

model = retrieve_chroma._get_model()
q = model.encode([query], convert_to_numpy=True, normalize_embeddings=True)[0]

rows = []
for doc, meta, emb in zip(everything["documents"], everything["metadatas"], everything["embeddings"]):
    rows.append((retrieve_chroma._cosine_similarity(q, emb), meta["alignment_id"], doc, meta))
rows.sort(key=lambda r: -r[0])
dense_rank = {r[1]: i + 1 for i, r in enumerate(rows)}

matches = [r for r in rows if word in r[2].lower()]
print(f"{len(matches)} English chunks in standard {standard} contain '{word}' "
      f"(out of {len(rows)} total)\n")

ce = rerank.Reranker()
scored = ce.rerank(query, [{"alignment_id": r[1], "english": r[2]} for r in matches], top_k=len(matches))
for c in scored:
    aid = c["alignment_id"]
    t = tam.get(ids=[aid], include=["documents"])["documents"]
    print(f"{aid}  dense rank {dense_rank[aid]}  cross-encoder {c['rerank_score']:.2f}")
    print(f"  EN: {c['english']}")
    print(f"  TA: {t[0] if t else '(no Tamil)'}\n")