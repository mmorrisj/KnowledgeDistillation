"""Store embeddings and search them: in Postgres with pgvector, or in memory.

Both stores have the same methods, so the lab runs with or without a database.
"""

from __future__ import annotations

import os
import re
from collections.abc import Sequence
from dataclasses import dataclass
from typing import Any

_TABLE_NAME = re.compile(r"^[a-z_][a-z0-9_]{0,62}$")


@dataclass(frozen=True)
class Hit:
    id: int
    text: str
    label: int
    distance: float  # cosine distance: 0 means pointing the same way


def _check_rows(
    dim: int,
    ids: Sequence[int],
    texts: Sequence[str],
    embeddings: Sequence[Sequence[float]],
    labels: Sequence[int],
) -> None:
    if not (len(ids) == len(texts) == len(embeddings) == len(labels)):
        raise ValueError("ids, texts, embeddings and labels differ in length.")
    for vector in embeddings:
        _check_vector(dim, vector)


def _check_effort(search_effort: Any) -> None:
    if search_effort is None:
        return
    if isinstance(search_effort, bool) or not isinstance(search_effort, int):
        raise ValueError("search_effort must be a whole number between 1 and 1000.")
    if not 1 <= search_effort <= 1000:
        raise ValueError("search_effort must be a whole number between 1 and 1000.")


def _check_vector(dim: int, vector: Sequence[float]) -> None:
    if len(vector) != dim:
        raise ValueError(f"This store holds vectors of {dim} numbers; got one of {len(vector)}.")


class MemoryStore:
    """Vectors kept in memory and searched exactly. Needs no database."""

    def __init__(self, dim: int):
        self.dim = dim
        self.create()

    def create(self) -> None:
        self._ids: list[int] = []
        self._texts: list[str] = []
        self._labels: list[int] = []
        self._vectors: list[list[float]] = []
        self._matrix = None  # unit-length rows, built on first search

    def drop(self) -> None:
        self.create()

    def add(self, ids, texts, embeddings, labels) -> None:
        _check_rows(self.dim, ids, texts, embeddings, labels)
        self._ids.extend(int(i) for i in ids)
        self._texts.extend(texts)
        self._labels.extend(int(y) for y in labels)
        self._vectors.extend([float(v) for v in row] for row in embeddings)
        self._matrix = None

    def create_index(self) -> None:
        """Nothing to build: this store always compares the query with every vector."""

    def count(self) -> int:
        return len(self._ids)

    def size_bytes(self) -> int:
        return len(self._ids) * self.dim * 4

    def search(
        self, query: Sequence[float], k: int = 10, search_effort: int | None = None
    ) -> list[Hit]:
        """Exact search. `search_effort` is accepted for compatibility and has no effect."""
        import numpy as np

        _check_vector(self.dim, query)
        _check_effort(search_effort)
        if not self._ids:
            return []
        if self._matrix is None:
            matrix = np.asarray(self._vectors, dtype=np.float32)
            lengths = np.linalg.norm(matrix, axis=1, keepdims=True)
            self._matrix = matrix / np.where(lengths == 0, 1.0, lengths)
        q = np.asarray(query, dtype=np.float32)
        q = q / (np.linalg.norm(q) or 1.0)
        distances = 1.0 - self._matrix @ q
        order = np.argsort(distances, kind="stable")[:k]
        return [
            Hit(self._ids[i], self._texts[i], self._labels[i], float(distances[i])) for i in order
        ]


class PgVectorStore:
    """Vectors in a Postgres table, using the pgvector extension, through SQLAlchemy."""

    def __init__(self, url: str, table: str = "kd_vectors", dim: int = 384):
        if not _TABLE_NAME.match(table):
            raise ValueError(
                f"'{table}' is not a safe table name. "
                "Use lowercase letters, digits and underscores."
            )
        import sqlalchemy as sa
        from pgvector.sqlalchemy import Vector

        self.dim = dim
        self._sa = sa
        self._engine = sa.create_engine(url)
        self._metadata = sa.MetaData()
        self._table = sa.Table(
            table,
            self._metadata,
            sa.Column("id", sa.BigInteger, primary_key=True),
            sa.Column("text", sa.Text, nullable=False),
            sa.Column("label", sa.Integer, nullable=False),
            sa.Column("embedding", Vector(dim), nullable=False),
        )
        # An approximate index: much faster on large tables, and may miss a few true neighbours.
        self._index = sa.Index(
            f"{table}_embedding_hnsw",
            self._table.c.embedding,
            postgresql_using="hnsw",
            postgresql_ops={"embedding": "vector_cosine_ops"},
        )

    def create(self) -> None:
        """Make an empty table, replacing any earlier one with the same name."""
        with self._engine.begin() as connection:
            connection.execute(self._sa.text("CREATE EXTENSION IF NOT EXISTS vector"))
            self._table.drop(connection, checkfirst=True)
            # The index is built later, by create_index, once the rows are in.
            connection.execute(self._sa.schema.CreateTable(self._table))

    def drop(self) -> None:
        with self._engine.begin() as connection:
            self._table.drop(connection, checkfirst=True)

    def add(self, ids, texts, embeddings, labels, chunk: int = 1000) -> None:
        _check_rows(self.dim, ids, texts, embeddings, labels)
        rows = [
            {"id": int(i), "text": t, "label": int(y), "embedding": [float(v) for v in e]}
            for i, t, y, e in zip(ids, texts, labels, embeddings, strict=True)
        ]
        with self._engine.begin() as connection:
            for start in range(0, len(rows), chunk):
                connection.execute(self._sa.insert(self._table), rows[start : start + chunk])

    def create_index(self) -> None:
        with self._engine.begin() as connection:
            self._index.create(connection, checkfirst=True)

    def count(self) -> int:
        sa = self._sa
        with self._engine.connect() as connection:
            return connection.execute(
                sa.select(sa.func.count()).select_from(self._table)
            ).scalar_one()

    def size_bytes(self) -> int:
        """Disk used by the table and its indexes."""
        sa = self._sa
        with self._engine.connect() as connection:
            return connection.execute(
                sa.select(sa.func.pg_total_relation_size(self._table.name))
            ).scalar_one()

    def search(
        self, query: Sequence[float], k: int = 10, search_effort: int | None = None
    ) -> list[Hit]:
        """The k nearest rows by cosine distance.

        `search_effort` sets how many candidates the approximate index examines (pgvector's
        `hnsw.ef_search`, default 40). Higher finds more of the true neighbours and is slower.
        It has no effect before the index is built.
        """
        _check_vector(self.dim, query)
        _check_effort(search_effort)
        sa, table = self._sa, self._table
        distance = table.c.embedding.cosine_distance([float(v) for v in query]).label("distance")
        statement = (
            sa.select(table.c.id, table.c.text, table.c.label, distance).order_by(distance).limit(k)
        )
        with self._engine.begin() as connection:
            if search_effort is not None:
                # SET takes no parameters; the value was checked to be a small whole number.
                connection.execute(sa.text(f"SET LOCAL hnsw.ef_search = {int(search_effort)}"))
            rows = connection.execute(statement)
            return [Hit(r.id, r.text, r.label, float(r.distance)) for r in rows]


def open_store(dim: int, table: str = "kd_vectors", url: str | None = None) -> Any:
    """A pgvector store if a database is configured, otherwise an in-memory one.

    The database address comes from `url` or the `KD_COURSE_DATABASE_URL` environment variable,
    for example `postgresql+psycopg://user:password@localhost/dbname`.
    """
    url = url or os.environ.get("KD_COURSE_DATABASE_URL")
    if url:
        return PgVectorStore(url, table=table, dim=dim)
    return MemoryStore(dim=dim)


def index_recall(
    exact: Sequence[Sequence[Hit]], found: Sequence[Sequence[Hit]], tolerance: float = 1e-6
) -> float:
    """Share of an index's results that are as close as the true nearest neighbours.

    `exact` comes from a search with no index; `found` from the same queries with one.
    A result counts if it is no further away than the furthest true neighbour, so a
    different row at exactly the same distance, such as a duplicate text, is not a miss.
    """
    if len(exact) != len(found):
        raise ValueError(f"The two result lists differ in length: {len(exact)} and {len(found)}.")
    good, total = 0, 0
    for true_hits, index_hits in zip(exact, found, strict=True):
        if not true_hits:
            continue
        furthest = true_hits[-1].distance + tolerance
        good += sum(hit.distance <= furthest for hit in index_hits)
        total += len(true_hits)
    return good / total if total else 0.0
