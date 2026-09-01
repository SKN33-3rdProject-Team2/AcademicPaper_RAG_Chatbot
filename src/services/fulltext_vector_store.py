"""추출된 논문 본문을 섹션별 청크로 색인하는 검색 저장소."""

from __future__ import annotations

import hashlib
import sqlite3
from pathlib import Path
from typing import Any

from . import PROJECT_ROOT
from .summary_vector_store import STORAGE_CONFIG


EXTRACT_DB_PATH = PROJECT_ROOT / "data" / "paper_extract" / "extracted_papers.db"
COLLECTION_NAME = "paper_fulltext_chunks"
SECTION_COLUMNS = (
    "abstract", "introduction", "related_work", "method", "experiment",
    "result", "conclusion", "others",
)


class FullTextStoreError(RuntimeError):
    """본문 색인 또는 검색에 실패했을 때 발생한다."""


def split_text(text: str, *, chunk_size: int = 1200, overlap: int = 180) -> list[str]:
    """문단을 우선 보존하며 겹치는 검색용 청크로 나눈다."""
    if chunk_size < 200 or not 0 <= overlap < chunk_size:
        raise ValueError("chunk_size와 overlap 값이 올바르지 않습니다.")
    chunks: list[str] = []
    for paragraph in (part.strip() for part in text.split("\n\n") if part.strip()):
        start = 0
        while start < len(paragraph):
            end = min(start + chunk_size, len(paragraph))
            chunks.append(paragraph[start:end])
            if end == len(paragraph):
                break
            start = end - overlap
    return chunks


class ChromaFullTextStore:
    """추출 SQLite DB를 Chroma에 동기화하고 관련 본문 청크를 반환한다."""

    def __init__(self, *, db_path: str | Path = EXTRACT_DB_PATH, directory: str | Path | None = None) -> None:
        self.db_path = Path(db_path)
        self.directory = Path(directory or STORAGE_CONFIG["directory"])
        self._client: Any | None = None
        self._model_instance: Any | None = None

    def _collection(self):
        try:
            import chromadb
        except ImportError as exc:
            raise FullTextStoreError("chromadb가 설치되어 있지 않습니다.") from exc
        self.directory.mkdir(parents=True, exist_ok=True)
        if self._client is None:
            self._client = chromadb.PersistentClient(path=str(self.directory))
        return self._client.get_or_create_collection(
            name=COLLECTION_NAME, metadata={"hnsw:space": "cosine"}
        )

    def _model(self):
        if self._model_instance is None:
            try:
                from sentence_transformers import SentenceTransformer
            except ImportError as exc:
                raise FullTextStoreError("sentence-transformers가 설치되어 있지 않습니다.") from exc
            self._model_instance = SentenceTransformer(
                str(STORAGE_CONFIG["embedding_model"]),
                device=str(STORAGE_CONFIG["device"]),
            )
        return self._model_instance

    def _read_papers(self) -> list[tuple[str, str, list[tuple[str, str]]]]:
        if not self.db_path.exists():
            raise FullTextStoreError(f"논문 원본 DB를 찾을 수 없습니다: {self.db_path}")
        columns = ", ".join(("id", "title", *SECTION_COLUMNS, "content"))
        try:
            with sqlite3.connect(self.db_path) as conn:
                conn.row_factory = sqlite3.Row
                rows = conn.execute(f"SELECT {columns} FROM extracted").fetchall()
        except sqlite3.Error as exc:
            raise FullTextStoreError("논문 원본 DB를 읽지 못했습니다.") from exc
        papers = []
        for row in rows:
            sections = [(key, str(row[key] or "").strip()) for key in SECTION_COLUMNS if str(row[key] or "").strip()]
            if not sections and str(row["content"] or "").strip():
                sections = [("content", str(row["content"]).strip())]
            if sections:
                papers.append((str(row["id"]), str(row["title"] or ""), sections))
        return papers

    def ensure_index(self) -> int:
        collection = self._collection()
        added = 0
        for paper_id, title, sections in self._read_papers():
            source_hash = hashlib.sha256("\n".join(text for _, text in sections).encode()).hexdigest()
            existing = collection.get(where={"paper_id": paper_id}, include=["metadatas"])
            existing_metadata = existing.get("metadatas") or []
            if existing_metadata and all(item.get("source_hash") == source_hash for item in existing_metadata):
                continue
            if existing.get("ids"):
                collection.delete(where={"paper_id": paper_id})
            ids: list[str] = []
            documents: list[str] = []
            metadata: list[dict[str, Any]] = []
            for section, text in sections:
                for index, chunk in enumerate(split_text(text)):
                    ids.append(f"{paper_id}:{section}:{index}")
                    documents.append(f"{title}\n\n{section}\n{chunk}")
                    metadata.append({"paper_id": paper_id, "title": title, "section": section, "chunk_index": index, "source_hash": source_hash})
            if documents:
                embeddings = self._model().encode(documents, normalize_embeddings=bool(STORAGE_CONFIG["normalize_embeddings"]), show_progress_bar=False)
                collection.upsert(ids=ids, documents=documents, embeddings=embeddings.tolist(), metadatas=metadata)
                added += len(documents)
        return added

    def search(self, query: str, *, limit: int = 5, paper_id: str | None = None) -> list[dict[str, object]]:
        if not query.strip():
            raise ValueError("본문 검색어가 비어 있습니다.")
        self.ensure_index()
        collection = self._collection()
        where = {"paper_id": paper_id} if paper_id else None
        available = len(collection.get(where=where).get("ids", [])) if where else collection.count()
        if available == 0:
            return []
        embedding = self._model().encode([query], normalize_embeddings=bool(STORAGE_CONFIG["normalize_embeddings"]), show_progress_bar=False).tolist()[0]
        result = collection.query(query_embeddings=[embedding], n_results=min(limit, available), where=where, include=["documents", "metadatas", "distances"])
        return [
            {"id": item_id, "document": document, "metadata": metadata, "distance": distance}
            for item_id, document, metadata, distance in zip(result["ids"][0], result["documents"][0], result["metadatas"][0], result["distances"][0])
        ]
