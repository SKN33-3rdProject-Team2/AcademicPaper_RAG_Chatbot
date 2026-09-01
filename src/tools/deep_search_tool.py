# src/tools/deep_search_tool.py
import json
import sqlite3
from typing import Dict, Any, List

from pydantic import BaseModel, Field
from langchain_core.tools import tool
from langchain_openai import OpenAIEmbeddings

from tools import PROJECT_DIR

# ==========================================
# 1. 동적 경로 매핑
# ==========================================
BASE_DIR = PROJECT_DIR / "data" / "paper_extract"
JSON_LIST_PATH = BASE_DIR / "extracted_papers.json"
DB_PATH = BASE_DIR / "extracted_papers.db"
REF_DB_PATH = BASE_DIR / "extracted_papers_ref.db"

# 인메모리 캐싱 (디스크 I/O 최적화 및 벡터 스토어 재사용)
_PAPER_LIST_CACHE: Dict[str, Any] = {}
_VECTOR_STORE = None


class SearchPaperListInput(BaseModel):
    keyword: str = Field(
        default="",
        description="검색할 키워드. 특정 논문을 찾을 때 사용하며, 전체 리스트를 원할 경우 반드시 빈 문자열('')을 입력하세요."
    )


class GetPaperDetailsInput(BaseModel):
    paper_id: str = Field(..., description="상세 내용을 조회할 논문의 고유 ID")


class SearchPaperPassagesInput(BaseModel):
    question: str = Field(..., min_length=1, description="논문 본문에서 근거를 찾을 질문")
    paper_id: str | None = Field(default=None, description="검색 범위를 제한할 논문 ID")
    limit: int = Field(default=5, ge=1, le=10, description="반환할 근거 청크 수")


def _load_json_with_cache() -> Dict[str, Any]:
    """JSON 리스트 캐싱"""
    global _PAPER_LIST_CACHE
    if not _PAPER_LIST_CACHE:
        if not JSON_LIST_PATH.exists():
            raise FileNotFoundError(f"논문 리스트 파일 누락: {JSON_LIST_PATH}")
        with open(JSON_LIST_PATH, 'r', encoding='utf-8') as f:
            _PAPER_LIST_CACHE = json.load(f)
    return _PAPER_LIST_CACHE


def _get_vector_store():
    """논문 제목을 임베딩하여 Chroma 벡터 스토어로 변환 (최초 1회만 실행)"""
    global _VECTOR_STORE
    if _VECTOR_STORE is None:
        try:
            from langchain_chroma import Chroma
        except ImportError:
            from langchain_community.vectorstores import Chroma
        from langchain_core.documents import Document

        data = _load_json_with_cache()
        docs = [
            Document(page_content=info.get("title", ""), metadata={"id": pid})
            for pid, info in data.items()
        ]

        # OpenAI 임베딩을 활용한 벡터화
        embeddings = OpenAIEmbeddings()
        _VECTOR_STORE = Chroma.from_documents(docs, embeddings)
    return _VECTOR_STORE


@tool("search_local_paper_list", args_schema=SearchPaperListInput)
def search_local_paper_list(keyword: str = "") -> str:
    """저장된 논문 리스트를 반환합니다. 키워드가 없으면 전체를, 있으면 벡터 유사도 기반으로 검색합니다."""
    try:
        data = _load_json_with_cache()

        # [1] 전체 리스트 요청 처리 (키워드가 없을 때)
        if not keyword or keyword.strip() == "":
            results = [{"id": pid, "title": info.get("title", "")} for pid, info in data.items()]
            return json.dumps({"search_type": "all", "total_count": len(results), "results": results},
                              ensure_ascii=False)

        # [2] 벡터 유사도 검색 (키워드가 있을 때)
        vector_store = _get_vector_store()
        # 가장 관련성 높은 논문 상위 5개 추출
        docs = vector_store.similarity_search(keyword, k=5)
        results = [{"id": doc.metadata["id"], "title": doc.page_content} for doc in docs]

        if not results:
            return f"'{keyword}'와 유사한 논문을 벡터 DB에서 찾을 수 없습니다."
        return json.dumps({"search_type": "vector_similarity", "keyword": keyword, "results": results},
                          ensure_ascii=False)

    except Exception as e:
        return json.dumps({"error": f"논문 검색 실패: {e}"})


@tool("get_local_paper_details", args_schema=GetPaperDetailsInput)
def get_local_paper_details(paper_id: str) -> str:
    """논문 ID를 바탕으로 SQLite DB에서 논문의 초록, 본문, 참고문헌을 반환합니다."""
    if not DB_PATH.exists():
        return json.dumps({"error": f"논문 원본 DB 누락: {DB_PATH}"})

    details: Dict[str, Any] = {"paper_id": paper_id}

    try:
        with sqlite3.connect(DB_PATH) as conn:
            conn.row_factory = sqlite3.Row
            row = conn.execute("SELECT title, abstract, content FROM extracted WHERE id = ?", (paper_id,)).fetchone()
            if row:
                details["title"] = row["title"]
                details["abstract"] = row["abstract"]
                details["content"] = row["content"]
            else:
                return json.dumps({"error": f"ID '{paper_id}' 논문 없음."})
    except sqlite3.Error as e:
        return json.dumps({"error": f"DB 조회 오류: {e}"})

    if REF_DB_PATH.exists():
        try:
            with sqlite3.connect(REF_DB_PATH) as conn:
                conn.row_factory = sqlite3.Row
                ref_rows = conn.execute(
                    "SELECT ref_index, reference_text FROM extracted_ref WHERE paper_id = ? ORDER BY ref_index",
                    (paper_id,)).fetchall()
                details["references"] = [f"[{r['ref_index']}] {r['reference_text']}" for r in ref_rows]
        except sqlite3.Error:
            details["references"] = ["레퍼런스 DB 파싱 오류 발생"]
    else:
        details["references"] = ["레퍼런스 DB 누락됨"]

    return json.dumps(details, ensure_ascii=False)


@tool("search_local_paper_passages", args_schema=SearchPaperPassagesInput)
def search_local_paper_passages(
    question: str, paper_id: str | None = None, limit: int = 5
) -> str:
    """추출 논문 본문을 섹션별 청크로 검색해 답변 근거를 반환합니다."""
    try:
        from services.fulltext_vector_store import ChromaFullTextStore

        results = ChromaFullTextStore().search(
            question, paper_id=paper_id, limit=limit
        )
        return json.dumps({"question": question, "results": results}, ensure_ascii=False)
    except Exception as exc:
        return json.dumps({"error": f"논문 본문 검색 실패: {exc}"}, ensure_ascii=False)
