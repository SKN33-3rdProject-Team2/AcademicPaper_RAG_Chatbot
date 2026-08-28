# src/feature/search_list.py
import os
import json
import sqlite3
import requests
from typing import List
from pathlib import Path
from dotenv import load_dotenv

from langchain_openai import ChatOpenAI
from pydantic import BaseModel, Field

# 루트 기준 data/paper_list 및 data/paper_save 경로 설정
ROOT_DIR = Path(__file__).resolve().parent.parent.parent
DATA_LIST_DIR = os.path.join(ROOT_DIR, "data", "paper_list")
PDF_SAVE_DIR = os.path.join(ROOT_DIR, "data", "paper_save")

DB_FILE = os.path.join(DATA_LIST_DIR, "saved_papers.db")
JSON_FILE = os.path.join(DATA_LIST_DIR, "saved_papers.json")
DOWNLOADED_PDF_JSON = os.path.join(PDF_SAVE_DIR, "downloaded_pdfs.json")

load_dotenv()
OPENAI_CHAT_MODEL = "gpt-5.6-luna"
llm = ChatOpenAI(model=OPENAI_CHAT_MODEL, temperature=0)


class InitialIntent(BaseModel):
    intent: str = Field(description="'search'(검색), 'show_all'(전체보기), 'download_all'(전체 다운로드), 'exit'(종료) 중 하나",
                        default="search")
    search_keyword: str = Field(description="검색 키워드 또는 주제", default="")
    wants_download: bool = Field(description="검색이나 전체보기와 함께 곧바로 다운로드를 원하는지 여부", default=False)


def parse_initial_intent(user_input: str) -> dict:
    structured_llm = ChatOpenAI(model=OPENAI_CHAT_MODEL, temperature=0).with_structured_output(InitialIntent)
    prompt = (
        f"다음 사용자의 응답에서 의도를 분석해줘.\n"
        f"- 'intent': 'search'(특정 키워드 검색), 'show_all'(저장된 모든 리스트 보기), 'download_all'(현재 존재하는 모든 논문 다운), 'exit'(종료)\n"
        f"- 'search_keyword': 검색이나 관련 논문을 찾을 때의 핵심 키워드\n"
        f"- 'wants_download': 사용자가 당장 PDF 다운로드를 함께 원하면 True, 아니면 False\n"
        f"응답: {user_input}"
    )
    try:
        return structured_llm.invoke(prompt).model_dump()
    except Exception:
        return {"intent": "search", "search_keyword": user_input, "wants_download": False}


class UserInteractionIntent(BaseModel):
    action: str = Field(description="'download'(PDF 다운로드), 'translate'(초록 번역), 'cancel'(취소/종료) 중 하나", default="cancel")
    selected_numbers: List[int] = Field(description="대상 논문의 번호 리스트", default_factory=list)


def parse_user_interaction(user_input: str, total_count: int) -> dict:
    structured_llm = ChatOpenAI(model=OPENAI_CHAT_MODEL, temperature=0).with_structured_output(UserInteractionIntent)
    prompt = (f"다음 응답에서 의도를 분석해줘. 논문은 총 {total_count}개야. "
              f"전부/다 라고 하면 1부터 {total_count}까지 채움\n응답: {user_input}")
    try:
        return structured_llm.invoke(prompt).model_dump()
    except Exception:
        return {"action": "cancel", "selected_numbers": []}


class LocalLibraryBot:
    def __init__(self):
        os.makedirs(PDF_SAVE_DIR, exist_ok=True)
        os.makedirs(DATA_LIST_DIR, exist_ok=True)

    def get_all_json_ids(self) -> List[str]:
        if not os.path.exists(JSON_FILE): return []
        with open(JSON_FILE, 'r', encoding='utf-8') as f:
            try:
                return list(json.load(f).keys())
            except json.JSONDecodeError:
                return []

    def search_json(self, query: str) -> List[str]:
        if not os.path.exists(JSON_FILE): return []
        with open(JSON_FILE, 'r', encoding='utf-8') as f:
            try:
                json_data = json.load(f)
            except json.JSONDecodeError:
                return []
        matched_ids = []
        keywords = query.lower().split()
        for pid, pdata in json_data.items():
            if all(kw in pdata.get("title", "").lower() for kw in keywords if len(kw) > 1):
                matched_ids.append(pid)
        return matched_ids

    def fetch_full_data_from_db(self, paper_ids: List[str]) -> List[dict]:
        conn = sqlite3.connect(DB_FILE)
        cursor = conn.cursor()
        rows = []
        for i in range(0, len(paper_ids), 900):
            chunk = paper_ids[i:i + 900]
            cursor.execute(
                f"SELECT id, title, authors, summary, pdf_url FROM papers WHERE id IN ({','.join('?' for _ in chunk)})",
                chunk)
            rows.extend(cursor.fetchall())
        conn.close()
        row_dict = {r[0]: r for r in rows}
        return [{"id": r[0], "title": r[1], "authors": r[2], "summary": r[3], "pdf_url": r[4]} for pid in paper_ids if
                pid in row_dict for r in [row_dict[pid]]]

    def update_downloaded_pdf_json(self, paper_title: str, filepath: str):
        downloaded_data = {}
        if os.path.exists(DOWNLOADED_PDF_JSON):
            with open(DOWNLOADED_PDF_JSON, 'r', encoding='utf-8') as f:
                try:
                    downloaded_data = json.load(f)
                except json.JSONDecodeError:
                    pass
        downloaded_data[os.path.basename(filepath)] = {"title": paper_title, "filepath": filepath}
        with open(DOWNLOADED_PDF_JSON, 'w', encoding='utf-8') as f:
            json.dump(downloaded_data, f, ensure_ascii=False, indent=4)

    def download_pdf(self, paper: dict) -> str:
        url = paper.get("pdf_url")
        if not url: return "no_url"
        pdf_url = url.replace("abs", "pdf") + ".pdf" if not url.endswith(".pdf") else url
        safe_title = "".join(c for c in paper['title'] if c.isalnum() or c in " _-").rstrip()
        filepath = os.path.join(PDF_SAVE_DIR, f"{safe_title[:60]}.pdf")

        if os.path.exists(filepath): return "exists"
        print(f"\n[System] 📥 '{safe_title[:30]}...' 다운로드 중...")
        try:
            response = requests.get(pdf_url, stream=True, timeout=10)
            response.raise_for_status()
            with open(filepath, 'wb') as f:
                for chunk in response.iter_content(chunk_size=8192): f.write(chunk)
            print(f"✅ 다운로드 완료! 저장 위치: {filepath}")
            self.update_downloaded_pdf_json(paper['title'], filepath)
            return "success"
        except Exception as e:
            print(f"❌ 다운로드 실패: {e}")
            return "error"

    def process_downloads(self, papers: List[dict]):
        if not papers:
            print("\n[System] 📭 다운로드할 논문이 없습니다.")
            return

        existing_papers, valid_to_download = [], []
        for idx, target in enumerate(papers):
            safe_title = "".join(c for c in target['title'] if c.isalnum() or c in " _-").rstrip()
            if os.path.exists(os.path.join(PDF_SAVE_DIR, f"{safe_title[:60]}.pdf")):
                existing_papers.append((idx + 1, target))
            else:
                valid_to_download.append((idx + 1, target))

        if existing_papers:
            exist_nums_str = ", ".join([str(p[0]) for p in existing_papers])
            print(f"\n⚠️ [안내] 대상 논문 중 {exist_nums_str}번은 이미 'data/paper_save' 폴더에 존재합니다.")

            if valid_to_download:
                valid_nums_str = ", ".join([str(p[0]) for p in valid_to_download])
                choice = input(f"👉 이미 존재하는 논문을 제외하고 나머지({valid_nums_str}번)만 다운로드할까요? (예 / 아니오): ").strip().lower()
            else:
                choice = input("👉 선택하신 모든 논문이 이미 존재합니다. 다시 다운로드(덮어쓰기) 하시겠습니까? (예 / 아니오): ").strip().lower()

            positive_keywords = ["예", "응", "그래", "어", "ㅇㅇ", "맞아", "제외", "해", "해줘", "진행", "다운", "네", "y", "yes", "오케이",
                                 "ok"]
            is_positive = any(w in choice for w in positive_keywords) or any(choice == w for w in positive_keywords)

            if any(neg in choice for neg in ["아니", "취소", "그만", "stop", "no"]):
                print("[System] 다운로드를 취소합니다.")
                return

            if is_positive or len(choice.strip()) > 0:
                if not valid_to_download and "제외" not in choice:
                    valid_to_download = existing_papers
            else:
                print("[System] 다운로드를 취소합니다.")
                return

        for num, paper in valid_to_download:
            if self.download_pdf(paper) == "exists":
                print(f"ℹ️ {num}번 '{paper['title'][:30]}...' 파일이 이미 존재하여 건너뜁니다.")

    def translate_summary(self, summary: str) -> str:
        try:
            return llm.invoke(f"다음 영문 논문 초록을 자연스럽고 전문적인 한국어로 번역해줘:\n\n{summary}").content
        except Exception as e:
            return f"번역 중 오류가 발생했습니다: {e}"

    def run(self):
        print("=" * 50)
        print(f"📚 내 서재(Local JSON/DB) 관리 챗봇 (Model: {OPENAI_CHAT_MODEL})")
        print("=" * 50)

        while True:
            try:
                user_input = input("\n[내 서재] 무엇을 도와드릴까요?\n(예: '저장된 리스트 보여줘', 'Attention 논문 찾아줘', '현재 있는거 다 다운해줘'): ")
            except KeyboardInterrupt:
                print("\n\n[System] 프로그램을 안전하게 종료합니다.")
                break

            if not user_input.strip(): continue

            user_lower = user_input.lower()
            is_explicit_exit = any(k == user_lower.strip() for k in ["종료", "그만", "q", "quit", "exit"]) or any(
                k in user_lower for k in ["종료해", "프로그램 종료", "그만", "중지", "멈춰", "q", "quit", "exit", "안해", "그만할래"])
            if is_explicit_exit and not any(w in user_lower for w in ["다운", "받", "저장", "번", "모두", "다", "찾아"]):
                print("\n[System] 서재 챗봇을 종료합니다.")
                break

            intent_data = parse_initial_intent(user_input)
            intent = intent_data.get("intent")
            keyword = intent_data.get("search_keyword", "")
            wants_download = intent_data.get("wants_download", False)

            if intent == "exit": print("\n[System] 서재 챗봇을 종료합니다."); break

            matched_ids = self.get_all_json_ids() if intent in ["show_all", "download_all"] else self.search_json(
                keyword if keyword.strip() else user_input)

            if not matched_ids:
                print(f"\n[System] ❌ 내 서재에 일치하는 논문이 없습니다.\n💡 안내: 외부 검색 봇('search.py')을 통해 먼저 검색/저장을 진행해 주세요.")
                continue

            papers = self.fetch_full_data_from_db(matched_ids)

            if wants_download or intent == "download_all":
                print(f"\n[System] 📚 조건에 맞는 총 {len(papers)}개의 논문을 바로 다운로드합니다.")
                self.process_downloads(papers)
                continue

            while True:
                print(f"\n[System] 📚 내 서재에서 총 {len(papers)}개의 논문을 찾았습니다.")
                print("-" * 60)
                for idx, p in enumerate(papers):
                    if intent == "show_all" and not keyword.strip():
                        print(f"{idx + 1}. [{p['id']}] {p['title']}")
                    else:
                        print(f"{idx + 1}. [{p['id']}] {p['title']}\n   - 저자: {p['authors']}\n   - 요약: {p['summary']}")
                    print("-" * 60)

                try:
                    ans = input(
                        "\n[선택]\n - PDF 다운로드 (예: '1번 다운', '모두 다운')\n - 초록 번역 (예: '2번 요약 번역')\n - 종료/뒤로가기 (엔터)\n> ")
                except KeyboardInterrupt:
                    print("\n[System] 이전 메뉴로 돌아갑니다.");
                    break

                if not ans.strip(): break
                action_data = parse_user_interaction(ans, len(papers))

                if action_data.get("action") == "translate" and action_data.get("selected_numbers"):
                    for num in action_data["selected_numbers"]:
                        if 0 <= num - 1 < len(papers):
                            print(f"\n✨ 번역 중...\n{self.translate_summary(papers[num - 1]['summary'])}\n{'=' * 60}")
                    input("\n[안내] 번역을 확인하셨습니다. 엔터를 누르면 리스트로 돌아갑니다.")
                elif action_data.get("action") == "download" and action_data.get("selected_numbers"):
                    self.process_downloads(
                        [papers[n - 1] for n in action_data["selected_numbers"] if 0 <= n - 1 < len(papers)])
                    break
                else:
                    print("\n[System] 작업을 건너뜁니다.");
                    break


if __name__ == "__main__":
    LocalLibraryBot().run()