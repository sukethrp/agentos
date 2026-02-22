from __future__ import annotations
import os

from agentos.rag.base_store import BaseVectorStore
from agentos.rag.types import SearchResult


class ChromaStore(BaseVectorStore):
    def __init__(self, collection_name: str = "default", persist_directory: str | None = None):
        import chromadb
        self._collection_name = collection_name
        if persist_directory:
            self._client = chromadb.PersistentClient(path=persist_directory)
        else:
            self._client = chromadb.EphemeralClient()
        self._collection = self._client.get_or_create_collection(name=collection_name, metadata={"hnsw:space": "cosine"})

    def add(self, text: str, embedding: list[float], metadata: dict | None = None, doc_id: str = "") -> int:
        ids = self.add_batch([text], [embedding], [metadata or {}], [doc_id or ""])
        return ids[0] if ids else 0

    def add_batch(
        self,
        texts: list[str],
        embeddings: list[list[float]],
        metadatas: list[dict] | None = None,
        doc_ids: list[str] | None = None,
    ) -> list[int]:
        metadatas = metadatas or [{} for _ in texts]
        doc_ids = doc_ids or [f"doc_{i}" for i in range(len(texts))]
        safe_meta = []
        for m in metadatas:
            sm = {k: str(v) for k, v in m.items() if isinstance(v, (str, int, float, bool))}
            safe_meta.append(sm)
        ids = [f"{did}_{i}" for i, did in enumerate(doc_ids)]
        self._collection.add(ids=ids, embeddings=embeddings, documents=texts, metadatas=safe_meta)
        return list(range(self._collection.count() - len(texts), self._collection.count()))

    def search(self, query_embedding: list[float], top_k: int = 5, threshold: float = 0.0) -> list[SearchResult]:
        cnt = self._collection.count()
        if cnt == 0:
            return []
        out = self._collection.query(query_embeddings=[query_embedding], n_results=min(top_k, cnt), include=["documents", "metadatas", "distances"])
        if not out or not out["documents"] or not out["documents"][0]:
            return []
        docs = out["documents"][0]
        dists = out.get("distances", [[]])[0]
        metas = out.get("metadatas", [[]])[0] or [{}] * len(docs)
        ids = out.get("ids", [[]])[0] or [""] * len(docs)
        results = []
        for i, (doc, meta, did) in enumerate(zip(docs, metas, ids)):
            dist = dists[i] if i < len(dists) else 0.0
            sim = 1.0 - float(dist) if dist is not None else 0.0
            if sim >= threshold:
                results.append(SearchResult(text=doc, score=sim, metadata=meta or {}, doc_id=did, index=i))
        return results[:top_k]

    @property
    def size(self) -> int:
        return self._collection.count()


class PineconeStore(BaseVectorStore):
    def __init__(self, index_name: str = "agentos", api_key: str | None = None, environment: str | None = None):
        from pinecone import Pinecone
        self._index_name = index_name
        pc = Pinecone(api_key=api_key or os.environ.get("PINECONE_API_KEY", ""), environment=environment or os.environ.get("PINECONE_ENV", "us-east1-gcp"))
        self._index = pc.Index(index_name)

    def add(self, text: str, embedding: list[float], metadata: dict | None = None, doc_id: str = "") -> int:
        ids = self.add_batch([text], [embedding], [metadata or {}], [doc_id or ""])
        return ids[0] if ids else 0

    def add_batch(
        self,
        texts: list[str],
        embeddings: list[list[float]],
        metadatas: list[dict] | None = None,
        doc_ids: list[str] | None = None,
    ) -> list[int]:
        metadatas = metadatas or [{} for _ in texts]
        doc_ids = doc_ids or [f"doc_{i}" for i in range(len(texts))]
        vectors = []
        for i, (emb, text, meta, did) in enumerate(zip(embeddings, texts, metadatas, doc_ids)):
            vid = f"{did}_{i}" if did else f"vec_{i}"
            m = {k: v for k, v in meta.items() if isinstance(v, (str, int, float, bool))}
            m["text"] = text[:40000]
            vectors.append({"id": vid, "values": emb, "metadata": m})
        self._index.upsert(vectors=vectors)
        return list(range(len(vectors)))

    def search(self, query_embedding: list[float], top_k: int = 5, threshold: float = 0.0) -> list[SearchResult]:
        r = self._index.query(vector=query_embedding, top_k=top_k, include_metadata=True)
        results = []
        for m in (r.matches or []):
            meta = m.metadata or {}
            text = meta.get("text", "")
            score = float(m.score or 0.0)
            if score >= threshold:
                results.append(SearchResult(text=text, score=score, metadata=meta, doc_id=m.id or "", index=len(results)))
        return results

    @property
    def size(self) -> int:
        return self._index.describe_index_stats().total_vector_count or 0


class PgVectorStore(BaseVectorStore):
    def __init__(self, collection_name: str = "default", connection_string: str | None = None):
        import psycopg
        from pgvector.psycopg import register_vector
        self._collection = collection_name
        self._conn_str = connection_string or os.environ.get("DATABASE_URL", "postgresql://localhost/agentos")
        self._conn = psycopg.connect(self._conn_str)
        register_vector(self._conn)
        self._conn.execute("CREATE EXTENSION IF NOT EXISTS vector")
        self._conn.execute(f"""
            CREATE TABLE IF NOT EXISTS {collection_name} (
                id SERIAL PRIMARY KEY,
                doc_id TEXT,
                text TEXT,
                embedding vector(1536),
                metadata JSONB
            )
        """)
        self._conn.commit()

    def add(self, text: str, embedding: list[float], metadata: dict | None = None, doc_id: str = "") -> int:
        ids = self.add_batch([text], [embedding], [metadata or {}], [doc_id or ""])
        return ids[0] if ids else 0

    def add_batch(
        self,
        texts: list[str],
        embeddings: list[list[float]],
        metadatas: list[dict] | None = None,
        doc_ids: list[str] | None = None,
    ) -> list[int]:
        import json
        metadatas = metadatas or [{} for _ in texts]
        doc_ids = doc_ids or ["" for _ in texts]
        from pgvector.psycopg import Vector
        cur = self._conn.cursor()
        indices = []
        for i, (text, emb, meta, did) in enumerate(zip(texts, embeddings, metadatas, doc_ids)):
            cur.execute(
                f"INSERT INTO {self._collection} (doc_id, text, embedding, metadata) VALUES (%s, %s, %s, %s) RETURNING id",
                (did, text, Vector(emb), json.dumps(meta)),
            )
            indices.append(cur.fetchone()[0])
        self._conn.commit()
        return indices

    def search(self, query_embedding: list[float], top_k: int = 5, threshold: float = 0.0) -> list[SearchResult]:
        import json
        from pgvector.psycopg import Vector
        cur = self._conn.cursor()
        cur.execute(
            f"SELECT id, doc_id, text, metadata, 1 - (embedding <=> %s) as score FROM {self._collection} ORDER BY embedding <=> %s LIMIT %s",
            (Vector(query_embedding), Vector(query_embedding), top_k),
        )
        rows = cur.fetchall()
        results = []
        for r in rows:
            score = float(r[4]) if r[4] is not None else 0.0
            if score >= threshold:
                meta = json.loads(r[3]) if isinstance(r[3], str) else (r[3] or {})
                results.append(SearchResult(text=r[2] or "", score=score, metadata=meta, doc_id=r[1] or "", index=r[0]))
        return results

    @property
    def size(self) -> int:
        cur = self._conn.cursor()
        cur.execute(f"SELECT COUNT(*) FROM {self._collection}")
        return cur.fetchone()[0] or 0
