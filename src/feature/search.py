# =====================================================================
# 1. 패키지 임포트 (모든 임포트를 최상단에 배치)
# =====================================================================
import os
import arxiv
import json
import sqlite3
from typing import List
from dotenv import load_dotenv

from langchain_openai import ChatOpenAI
from langchain_core.tools import tool
from pydantic import BaseModel, Field

# =====================================================================
# 2. 고정 변수 (상수) 및 LLM 설정
# =====================================================================

# .env 파일에서 환경변수(OPENAI_API_KEY 등)를 불러옵니다.
load_dotenv()

DATA_DIR = "/data"
DB_FILE = os.path.join(DATA_DIR, "saved_papers.db")
JSON_FILE = os.path.join(DATA_DIR, "saved_papers.json")

# 💡 사용할 LLM 모델명 설정 (반드시 'gpt-4o', 'gpt-3.5-turbo' 등 실제 존재하는 모델명 사용)
OPENAI_CHAT_MODEL = "gpt-5.6-luna"

# API 키가 .env 또는 시스템 환경변수(OPENAI_API_KEY)에 등록되어 있어야 작동합니다.
llm = ChatOpenAI(model=OPENAI_CHAT_MODEL, temperature=0)


# =====================================================================
# 🚨🚨🚨 [키워드 확장 툴 연결 위치] 🚨🚨🚨
# =====================================================================

class KeywordList(BaseModel):
    """LLM이 키워드 확장을 위해 반환할 데이터 스키마"""
    keywords: List[str] = Field(description="검색에 사용할 확장된 파생 키워드 리스트 (원본 단어 포함)")

@tool
def generate_keywords_tool(base_keyword: str) -> List[str]:
    """
    [LLM 기반 키워드 생성 툴] - LLM이 실시간으로 연관 학술 키워드를 생성합니다.
    """
    print(f"\n[Tool] 🧠 '{base_keyword}'에 대한 학술 키워드를 LLM이 생각 중입니다...")

    structured_llm = llm.with_structured_output(KeywordList)
    prompt = (
        f"'{base_keyword}'와 관련된 논문을 ArXiv에서 검색하려고 해. "
        f"검색 정확도를 높일 수 있는 동의어나 핵심 학술 키워드를 3~5개 정도 추천해줘. "
        f"반드시 원본 단어를 포함해서 영어로 작성해줘."
    )

    try:
        result = structured_llm.invoke(prompt)
        return result.keywords
    except Exception as e:
        print(f"[Error] 키워드 생성 중 오류가 발생했습니다: {e}")
        return [base_keyword]


# =====================================================================
# 3. 데이터베이스, 검색 툴, 그리고 일반 저장 함수
# =====================================================================

def init_db():
    """DB 초기화 및 기존 테이블 업데이트(자동 마이그레이션)"""
    os.makedirs(DATA_DIR, exist_ok=True)
    conn = sqlite3.connect(DB_FILE)
    cursor = conn.cursor()

    # 1. 테이블이 아예 없으면 처음부터 created_at을 포함해서 생성
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS papers (
            id      TEXT PRIMARY KEY,
            title   TEXT,
            authors TEXT,
            summary TEXT,
            pdf_url TEXT,
            created_at DATETIME DEFAULT CURRENT_TIMESTAMP
        )
    ''')

    # 🌟 [자동 업데이트 로직] 2. 테이블 정보를 읽어와서 created_at 컬럼이 있는지 확인
    cursor.execute("PRAGMA table_info(papers)")
    columns = [info[1] for info in cursor.fetchall()]

    if 'created_at' not in columns:
        # 3. SQLite의 제약으로 ALTER TABLE 시 기본값을 지정할 수 없으므로,
        # 일단 빈 컬럼을 추가한 뒤 UPDATE 구문으로 일괄 적용합니다.
        cursor.execute("ALTER TABLE papers ADD COLUMN created_at DATETIME")
        cursor.execute("UPDATE papers SET created_at = CURRENT_TIMESTAMP")
        print("\n[System] 🛠️ 기존 DB에 'created_at' 컬럼을 성공적으로 업데이트(추가)했습니다.")

    conn.commit()
    conn.close()


@tool
def search_arxiv_papers_tool(final_query: str, sort_by: str = 'r', max_results: int = 10) -> List[dict]:
    """ArXiv 논문 검색 툴"""
    client = arxiv.Client()

    sort_criterion = arxiv.SortCriterion.SubmittedDate if sort_by == 'n' else arxiv.SortCriterion.Relevance
    sort_name = "최신순" if sort_by == 'n' else "관련도(영향력)순"

    print(f"\n[System] 🔍 ArXiv 검색 중... (조건: '{final_query}', 정렬: {sort_name}, 최대 {max_results}개)")

    search = arxiv.Search(
        query=final_query,
        max_results=max_results,
        sort_by=sort_criterion,
        sort_order=arxiv.SortOrder.Descending
    )

    results = []
    try:
        for paper in client.results(search):
            results.append({
                "id": paper.get_short_id(),
                "title": paper.title,
                "authors": ", ".join([author.name for author in paper.authors]),
                "summary": paper.summary.replace('\n', ' '),
                "pdf_url": paper.pdf_url
            })
    except Exception as e:
        print(f"[Error] 검색 중 오류가 발생했습니다: {e}")

    return results


def save_papers(selected_papers: List[dict]) -> str:
    """선택된 논문을 저장하고 1000개 초과 시 오래된 것부터 삭제하는 함수"""
    if not selected_papers:
        return "저장할 논문이 없습니다."

    os.makedirs(DATA_DIR, exist_ok=True)
    conn = sqlite3.connect(DB_FILE)
    cursor = conn.cursor()

    # 1. 새 논문 추가 (CURRENT_TIMESTAMP 직접 주입으로 중복/에러 방지)
    for paper in selected_papers:
        paper_id = paper['id']
        try:
            cursor.execute('''
                INSERT OR IGNORE INTO papers (id, title, authors, summary, pdf_url, created_at)
                VALUES (?, ?, ?, ?, ?, CURRENT_TIMESTAMP)
            ''', (paper_id, paper['title'], paper['authors'], paper['summary'], paper['pdf_url']))
        except Exception:
            pass

    # 2. 1000개 초과 시 최신 1000개를 제외한 나머지 오래된 데이터 삭제 (FIFO)
    cursor.execute('''
        DELETE FROM papers
        WHERE id NOT IN (
            SELECT id FROM papers
            ORDER BY created_at DESC
            LIMIT 1000
        )
    ''')

    # 3. DB와 JSON 동기화를 위해 최신 1000개 가져오기
    cursor.execute('SELECT id, title, authors, summary, pdf_url FROM papers ORDER BY created_at DESC')
    rows = cursor.fetchall()

    conn.commit()
    conn.close()

    json_data = {}
    for row in rows:
        json_data[row[0]] = {
            "id": row[0],
            "title": row[1],
            "authors": row[2],
            "summary": row[3],
            "pdf_url": row[4]
        }

    with open(JSON_FILE, 'w', encoding='utf-8') as f:
        json.dump(json_data, f, ensure_ascii=False, indent=4)

    return f"{len(selected_papers)}개의 논문이 추가되었습니다. (현재 보관된 총 논문 수: {len(json_data)}/1000개)"


# =====================================================================
# 4. 사용자 의도 분석 스키마
# =====================================================================
class SearchIntent(BaseModel):
    query: str = Field(description="검색할 영문 키워드나 논문 제목", default=None)
    search_type: str = Field(description="제목 검색이면 't', 키워드/주제 검색이면 'k'", default=None)
    sort_by: str = Field(description="영향력/관련도 순이면 'r', 최신순이면 'n'", default=None)
    max_results: int = Field(description="가져올 논문의 개수 (예: 20가지 -> 20)", default=None)


def parse_intent_with_langchain(user_input: str) -> dict:
    structured_llm = llm.with_structured_output(SearchIntent)
    result = structured_llm.invoke(f"다음 사용자의 요청에서 검색 파라미터를 추출해줘: {user_input}")
    # 💡 Pydantic v2 권장 사항에 맞춰 dict() 대신 model_dump() 사용
    return result.model_dump()


# =====================================================================
# 5. 메인 오케스트레이션 루프
# =====================================================================
def main():
    # 여기서 기존 DB를 알아서 점검하고 업데이트합니다!
    init_db()
    print("=" * 50)
    print(f"🤖 LangChain 기반 ArXiv 검색 챗봇 (Model: {OPENAI_CHAT_MODEL})")
    print("=" * 50)

    while True:
        user_input = input("\n무엇을 찾아드릴까요? (종료하려면 '그만', '종료', 'q' 등 입력): ")

        if not user_input.strip():
            continue

        # 자연어 강제 종료 의도 파악
        stop_keywords = ["종료", "그만", "중지", "멈춰", "q", "quit", "exit"]
        if any(keyword in user_input.lower() for keyword in stop_keywords):
            print("\n[System] 사용자의 요청으로 챗봇을 즉시 종료합니다. 이용해 주셔서 감사합니다!")
            break

        print("[System] LLM이 사용자 의도를 분석 중입니다...")
        params = parse_intent_with_langchain(user_input)

        # 누락된 정보 묻기
        if params["query"] is None:
            params["query"] = input("❓ 검색할 단어(영문)가 빠져있습니다. 무엇으로 검색할까요?: ")
        if params["search_type"] is None:
            ans = input("❓ 제목으로 검색할까요, 키워드로 검색할까요? (t: 제목 / k: 키워드 / 엔터: 키워드): ").strip().lower()
            params["search_type"] = 't' if ans == 't' else 'k'
        if params["sort_by"] is None:
            ans = input("❓ 어떻게 정렬할까요? (r: 영향력순 / n: 최신순 / 엔터: 영향력순): ").strip().lower()
            params["sort_by"] = 'n' if ans == 'n' else 'r'
        if params["max_results"] is None:
            ans = input("❓ 몇 개의 논문을 찾을까요? (숫자 입력 / 엔터: 10개): ").strip()
            params["max_results"] = int(ans) if ans.isdigit() else 10

        # 분기 처리 및 HITL (Human-in-the-Loop)
        if params["search_type"] == 't':
            final_query = f'ti:"{params["query"]}"'
        else:
            # LLM 기반 툴 호출로 키워드 생성
            expanded_keywords = generate_keywords_tool.invoke({"base_keyword": params["query"]})

            print(f"\n[Human-in-the-Loop] 👀 다음 파생 키워드들이 생성되었습니다:")
            print(f"👉 {expanded_keywords}")
            hitl_ans = input("이 키워드들로 검색을 진행할까요? (y: 예 / n: 원본 단어만 / e: 직접 수정 / 엔터: 예): ").strip().lower()

            if hitl_ans == 'n':
                final_keywords = [params["query"]]
            elif hitl_ans == 'e':
                custom_kw = input("검색에 사용할 키워드를 쉼표(,)로 구분하여 입력하세요: ")
                final_keywords = [kw.strip() for kw in custom_kw.split(",") if kw.strip()]
            else:
                final_keywords = expanded_keywords

            final_query = " OR ".join([f'all:"{kw}"' for kw in final_keywords])

        # 논문 검색 툴 호출
        papers = search_arxiv_papers_tool.invoke({
            "final_query": final_query,
            "sort_by": params["sort_by"],
            "max_results": params["max_results"]
        })

        if not papers:
            print("조건에 맞는 논문을 찾지 못했습니다.")
            continue

        print("\n[검색 결과]")
        for idx, p in enumerate(papers):
            print(f"{idx + 1}. [{p['id']}] {p['title']}")
            print(f"   - 저자: {p['authors']}")
            print(f"   - 요약: {p['summary']}")
            print("-" * 60)

        # 결과 저장 (툴이 아닌 일반 함수로 처리)
        selections = input("\n저장할 논문의 번호를 쉼표로 구분하여 입력하세요 (예: 1,3 / 넘기려면 엔터): ")
        if selections.strip():
            selected_indices = [int(x.strip()) - 1 for x in selections.split(',') if x.strip().isdigit()]
            papers_to_save = [papers[i] for i in selected_indices if 0 <= i < len(papers)]

            save_msg = save_papers(papers_to_save)
            print(f"\n[System] 💾 {save_msg}")


if __name__ == "__main__":
    main()