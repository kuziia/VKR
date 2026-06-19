# Cyberleninka mini IR dataset (BEIR format, v2)

Built from cyberleninka.ru via topic-balanced sampling, with hand-crafted queries.

- documents: 80
- topics: 8 (bio, cs, earth, econ, history, lang, law, med)
- queries: 208
- queries by kind: {'kw': 80, 'nl': 80, 'kw_adv': 17, 'nl_adv': 31}
- qrels rows: 293
- score distribution: {2: 208, 1: 85}

## Files
- `corpus.jsonl` — `{_id, title, text}` per article
- `queries.jsonl` — `{_id, text, topic, kind}`. Kinds: `kw` (short keyword), `nl` (natural-language question), `kw_adv`/`nl_adv` (cross-topic adversarial)
- `qrels/test.tsv` — `query-id`, `corpus-id`, `score` (header included)
- `metadata.jsonl` — full article record (authors, journal, year, ISSN, keywords, abstract, URL)
- PDFs (raw originals): `../pdfs/{doc_id}.pdf`

## Relevance scheme
- **2** — perfect match: the query is answered specifically by this document
- **1** — partial match: the document covers a closely related theme but isn't the primary answer
- **missing** — not relevant

## Query design
Per article: 1 short keyword query (`kw`) + 1 natural-language question (`nl`).
Several `nl` queries deliberately have multi-doc relevance when the question genuinely covers more than one article in the topic.

Adversarial queries (`*_adv`) are crafted to test cross-topic confusion: they share lexicon with documents in another topic but the correct answer remains within the designated topic. Example: `adv_03_kw` 'ответственность за налоговые правонарушения' should retrieve `law_006`, not the lexically-overlapping `econ_001`.
