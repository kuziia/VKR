"""Rebuild BEIR queries.jsonl + qrels/test.tsv from queries_manual.json (v2 schema).

queries_manual.json format:
  {
    "queries": [
      {"qid": "...", "text": "...", "topic": "...", "kind": "kw|nl|kw_adv|nl_adv",
       "qrels": {"doc_id": 2, "doc_id2": 1, ...},
       "design": "optional design note"}
    ]
  }

Source corpus and metadata are not changed — only queries.jsonl and qrels/test.tsv
are rewritten under beir/.
"""
from __future__ import annotations

import io
import json
import sys
from collections import Counter
from pathlib import Path

sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")

ROOT = Path(__file__).parent
QUERIES_MANUAL = ROOT / "queries_manual.json"
BEIR_DIR = ROOT / "beir"


def main() -> None:
    if not QUERIES_MANUAL.exists():
        raise SystemExit(f"missing {QUERIES_MANUAL}")
    data = json.loads(QUERIES_MANUAL.read_text(encoding="utf-8"))
    queries = data["queries"]

    # Sanity: corpus must exist and qrels must reference real docs
    corpus_ids = set()
    for line in (BEIR_DIR / "corpus.jsonl").open(encoding="utf-8"):
        d = json.loads(line)
        corpus_ids.add(d["_id"])

    BEIR_DIR.mkdir(exist_ok=True)
    (BEIR_DIR / "qrels").mkdir(exist_ok=True)

    bad_refs = []
    score_counter = Counter()
    docs_used = set()
    queries_per_topic = Counter()
    queries_per_kind = Counter()

    with (BEIR_DIR / "queries.jsonl").open("w", encoding="utf-8") as fq, \
         (BEIR_DIR / "qrels" / "test.tsv").open("w", encoding="utf-8") as fr:
        fr.write("query-id\tcorpus-id\tscore\n")
        for q in queries:
            qid = q["qid"]
            qrec = {
                "_id": qid,
                "text": q["text"],
                "topic": q.get("topic", ""),
                "kind": q.get("kind", "kw"),
            }
            fq.write(json.dumps(qrec, ensure_ascii=False) + "\n")
            queries_per_topic[q.get("topic", "?")] += 1
            queries_per_kind[q.get("kind", "?")] += 1
            for doc_id, score in q.get("qrels", {}).items():
                if doc_id not in corpus_ids:
                    bad_refs.append((qid, doc_id))
                    continue
                fr.write(f"{qid}\t{doc_id}\t{int(score)}\n")
                score_counter[int(score)] += 1
                docs_used.add(doc_id)

    print(f"queries written: {len(queries)}")
    print(f"  by topic: {dict(queries_per_topic)}")
    print(f"  by kind:  {dict(queries_per_kind)}")
    print(f"qrels rows: {sum(score_counter.values())}")
    print(f"  score distribution: {dict(score_counter)}")
    print(f"docs touched by qrels: {len(docs_used)} / {len(corpus_ids)} corpus docs")
    if bad_refs:
        print(f"  WARNING: {len(bad_refs)} qrels reference unknown doc_ids:")
        for qid, doc_id in bad_refs[:10]:
            print(f"    {qid} -> {doc_id}")

    # Update README
    readme = BEIR_DIR / "README.md"
    readme.write_text(
        "# Cyberleninka mini IR dataset (BEIR format, v2)\n\n"
        "Built from cyberleninka.ru via topic-balanced sampling, with hand-crafted queries.\n\n"
        f"- documents: {len(corpus_ids)}\n"
        f"- topics: 8 (bio, cs, earth, econ, history, lang, law, med)\n"
        f"- queries: {len(queries)}\n"
        f"- queries by kind: {dict(queries_per_kind)}\n"
        f"- qrels rows: {sum(score_counter.values())}\n"
        f"- score distribution: {dict(score_counter)}\n\n"
        "## Files\n"
        "- `corpus.jsonl` — `{_id, title, text}` per article\n"
        "- `queries.jsonl` — `{_id, text, topic, kind}`. Kinds: `kw` (short keyword), "
        "`nl` (natural-language question), `kw_adv`/`nl_adv` (cross-topic adversarial)\n"
        "- `qrels/test.tsv` — `query-id`, `corpus-id`, `score` (header included)\n"
        "- `metadata.jsonl` — full article record (authors, journal, year, ISSN, keywords, abstract, URL)\n"
        "- PDFs (raw originals): `../pdfs/{doc_id}.pdf`\n\n"
        "## Relevance scheme\n"
        "- **2** — perfect match: the query is answered specifically by this document\n"
        "- **1** — partial match: the document covers a closely related theme but isn't the primary answer\n"
        "- **missing** — not relevant\n\n"
        "## Query design\n"
        "Per article: 1 short keyword query (`kw`) + 1 natural-language question (`nl`).\n"
        "Several `nl` queries deliberately have multi-doc relevance when the question genuinely "
        "covers more than one article in the topic.\n\n"
        "Adversarial queries (`*_adv`) are crafted to test cross-topic confusion: they share "
        "lexicon with documents in another topic but the correct answer remains within the "
        "designated topic. Example: `adv_03_kw` 'ответственность за налоговые правонарушения' "
        "should retrieve `law_006`, not the lexically-overlapping `econ_001`.\n",
        encoding="utf-8",
    )


if __name__ == "__main__":
    main()
