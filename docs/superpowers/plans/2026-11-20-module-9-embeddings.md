# Module 9: Embedding Distillation and Search — Design Note

**Spec:** `docs/superpowers/specs/2026-11-19-kd-course-platform-design.md` (listed there as an optional module using pgvector).

## Decisions

- **Task:** search over the banking messages. The collection is the pool (about 9,000 messages); the queries are 1,000 test messages. Message types are used only to score results.
- **Models, both MIT-licensed:** teacher `BAAI/bge-large-en-v1.5` (335M weights, 1,024 numbers per text), student `BAAI/bge-small-en-v1.5` (33M, 384).
- **Method: match similarities, not vectors.** For each batch, the student's similarity for every pair of texts is pulled toward the teacher's. This needs no labels and no extra layer for the different lengths, and the student keeps its shorter output. Matching vectors through a projection is taught in lesson 9.2 as the alternative for keeping an existing index, and not implemented.
- **Baseline:** the student as published. It is already a good general model, so the lessons say plainly that the quality gain from distilling may be small and that speed and storage are the certain gains.
- **Scores:** precision at 10 and top-1 with labels; neighbour overlap with the teacher without labels (named fidelity, not quality); paired comparison on per-query precision.
- **Vector store:** `PgVectorStore` uses SQLAlchemy 2 with the pgvector extension (cosine distance, HNSW index). `MemoryStore` has the same methods and is used when `KD_COURSE_DATABASE_URL` is not set, so the lab does not require Postgres. Table names are validated; values travel as parameters.
- **Index recall is measured, not assumed.** Testing against real Postgres showed the default search effort can miss a noticeable share of true neighbours (78% found on 9,000 random 64-number vectors). The store exposes `search_effort` (pgvector's `hnsw.ef_search`), the lab reports recall at two settings, and recall is computed by distance so that duplicate texts are not counted as misses.
- **CI** runs the vector store tests against a real Postgres with pgvector, as a service in the `labs` job.

## Lessons

9.1 Embeddings and search (storage explorer) · 9.2 Distilling an embedding model · 9.3 Measuring search quality (neighbour overlap explorer) · 9.4 Vectors in Postgres with pgvector · 9.5 Lab

## Verified here

- Vector store tests ran against a local Postgres 16 with pgvector 0.6.
- The lab notebook's own code cells were executed on CPU with a tiny model and toy data swapped in, with the in-memory store and with Postgres.

## Known limits

- Not run on a GPU with the real models. Whether distillation improves search on this domain is unknown.
- Relevance is approximated by message type.
- The lab requires a database user allowed to create the `vector` extension, or the extension already present.
