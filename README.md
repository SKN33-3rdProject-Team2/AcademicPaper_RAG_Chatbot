# Project-Team-2 / Academic Paper RAG Chatbot

arXiv와 PDF 학술 논문을 수집·파싱·인덱싱하여 논문 검색, 번역, 요약, 심층 질의응답 및 논문 간 연관성 분석을 제공하는 RAG(Retrieval-Augmented Generation) 챗봇 프로젝트입니다.

## 프로젝트 목표

- **비용 최소화:** 상시 GPU 서버 대신 로컬 CPU 임베딩과 외부 파싱 API를 조합하여 고정 인프라 비용을 최소화합니다.
- **선택적 논문 처리:** 검색 결과의 초록을 먼저 확인한 뒤 사용자가 선택한 논문만 다운로드하고 인덱싱합니다.
- **신뢰도 높은 답변:** 검색 결과와 논문 본문을 근거로 답변하여 환각을 줄입니다.
- **단계별 사용자 개입:** 번역과 요약 이후 사용자가 원하는 후속 분석을 선택할 수 있도록 Human-in-the-Loop 흐름을 적용합니다.
- **기능별 에이전트 분리:** LangGraph 기반 에이전트가 검색, 수집, 번역, 요약, 유사도 분석 및 심층 질의응답을 나누어 처리합니다.

## 주요 기능

1. 사용자 의도와 검색 조건 확인
2. arXiv API를 이용한 논문 메타데이터 및 초록 검색
3. 사용자가 선택한 논문만 다운로드 및 파싱
4. 텍스트 정제, 청킹, 임베딩 및 Vector DB 저장
5. 학술 논문 전문 번역과 4단 구조 요약
6. 유사 논문 및 핵심 참고문헌 추천
7. 논문 본문과 수식에 대한 심층 질의응답

## 시스템 흐름

```text
✅ 사용자 질의
↓
✅ 의도 파악 및 검색 조건 확인
↓
✅ arXiv 논문 검색 및 초록 목록 제공
↓
✅ 사용자가 처리할 논문 선택
↓
🔜 PDF 다운로드 및 하이브리드 파싱
↓
🔜 텍스트 정제 → 청킹 → 임베딩 → ChromaDB 저장
↓
🔜 전문 번역 및 구조화 요약
↓
❌ 사용자 선택에 따라 유사 논문 탐색 또는 심층 질의응답
```

## 예상 기술 스택

| 구분 | 기술 |
| --- | --- |
| Language | Python |
| Agent Workflow | LangGraph |
| Paper Search | arXiv API |
| PDF Parsing | PyMuPDF, NVIDIA Build API |
| Local LLM | Ollama (`qwen2.5:3b`, `translategemma:4b`, `gemma3:4b`) |
| Embedding | Hugging Face `BAAI/bge-m3` |
| Vector DB | ChromaDB |
| Architecture | RAG, Human-in-the-Loop, Multi-Agent |

> 기술 스택은 구현 및 검증 과정에서 변경될 수 있습니다.

## 프로젝트 구조

```text
AcademicPaper_RAG_Chatbot/
├── .env.sample
├── .gitignore
├── main.py
├── web_app.py
├── build_evaluation_corpus.py
├── evaluate_langsmith.py
├── evaluate_rag_langsmith.py
├── requirements.txt
├── requirements-orchestration.txt
├── apps/
│   ├── __init__.py
│   ├── ui.py
│   ├── search_app.py
│   ├── paper_list_app.py
│   ├── translation_summary_app.py
│   └── deep_search_app.py
├── data/
│   ├── paper_extract/
│   │   ├── extracted_papers.db
│   │   ├── extracted_papers.json
│   │   └── extracted_papers_ref.db
│   ├── paper_list/
│   │   ├── saved_papers.db
│   │   └── saved_papers.json
│   └── paper_save/
│       └── downloaded_pdfs.json
├── log/
│   ├── __init__.py
│   ├── app_logger.py
│   ├── log_codes.py
│   └── log_messages.py
├── src/
│   ├── config/
│   │   ├── model_config.yaml
│   │   ├── nvidia_config.yaml
│   │   └── ollama_config.yaml
│   ├── feature/
│   │   ├── supervisor_chatbot.py
│   │   ├── search.py
│   │   ├── search_list.py
│   │   ├── paper_extractor.py
│   │   └── deep_research.py
│   ├── orchestration/
│   │   ├── __init__.py
│   │   ├── state.py
│   │   ├── routing.py
│   │   ├── graph.py
│   │   ├── adapters.py
│   │   ├── rag_chain.py
│   │   └── evaluation.py
│   ├── services/
│   │   ├── __init__.py
│   │   ├── generation_options.py
│   │   ├── model_config_service.py
│   │   ├── nvidia_service.py
│   │   ├── ollama_service.py
│   │   ├── translation_service.py
│   │   ├── translation_checkpoint_service.py
│   │   ├── translation_markdown_service.py
│   │   ├── summary_checkpoint_service.py
│   │   ├── summary_markdown_store.py
│   │   └── summary_vector_store.py
│   └── tools/
│       ├── __init__.py
│       ├── keyword_tool.py
│       ├── translation_tool.py
│       ├── summary_tool.py
│       └── deep_search_tool.py
└── tests/
    ├── __init__.py
    ├── test_keyword_tool.py
    ├── test_translation_tool.py
    ├── test_summary_tool.py
    ├── test_paper_extractor.py
    ├── test_deep_research.py
    └── test_orchestration.py
```

### 디렉터리 및 파일 설명

- `main.py`: Supervisor 기반 LangGraph 챗봇의 CLI 진입점입니다.
- `web_app.py`: Streamlit 웹 애플리케이션 진입점입니다.
- `apps/`: 검색, 논문 목록, 번역·요약, Deep Research 화면을 구성합니다.
- `data/`: 논문 메타데이터, 본문 추출 결과와 다운로드 기록을 저장합니다.
- `log/`: 공통 로거, 로그 코드와 메시지를 관리합니다.
- `src/config/`: Ollama, NVIDIA 및 기능별 모델 설정을 관리합니다.
- `src/feature/`: 논문 검색·추출·Deep Research와 Supervisor 챗봇 기능을 구현합니다.
- `src/orchestration/`: LangGraph 상태, 라우팅, RAG 체인과 평가 로직을 관리합니다.
- `src/services/`: 모델 호출, 체크포인트, Markdown 및 Vector DB 저장 기능을 제공합니다.
- `src/tools/`: 키워드 생성, 번역, 요약과 Deep Research용 도구를 제공합니다.
- `tests/`: 각 기능과 오케스트레이션의 자동화 테스트를 포함합니다.
- `build_evaluation_corpus.py`: RAG 평가용 논문 코퍼스를 생성합니다.
- `evaluate_langsmith.py`: Supervisor 라우팅을 LangSmith에서 평가합니다.
- `evaluate_rag_langsmith.py`: 검색·답변·인용 품질을 LangSmith에서 평가합니다.
- `.env.sample`: 프로젝트 실행에 필요한 환경변수의 예시를 제공합니다.
- `requirements*.txt`: 기본 및 LangGraph 관련 Python 의존성을 관리합니다.

## 실행 방법

### 1. 환경변수 설정

저장소를 내려받은 뒤 `.env.sample` 파일을 복사하여 `.env` 파일을 생성합니다.

#### macOS / Linux

```bash
cp .env.sample .env
```

#### Windows PowerShell

```powershell
Copy-Item .env.sample .env
```

`.env.sample` 파일에는 OpenAI, NVIDIA 및 LangSmith 연결에 필요한 환경변수 이름이 정의되어 있습니다. 각 값을 본인의 키로 변경합니다.

```dotenv
OPENAI_API_KEY=your_openai_api_key
NVIDIA_API_KEY=your_nvidia_build_key
LANGSMITH_API_KEY=your_langsmith_api_key
```

Ollama 서버 주소와 기능별 모델은 `src/config/model_config.yaml`에서 관리합니다.

> 실제 API 키, 비밀번호 등 민감한 정보는 `.env.sample`이나 소스코드에 작성하지 않습니다.

### 2. Ollama 설치 및 실행

Ollama는 기본적으로 `http://localhost:11434`에서 실행됩니다. 운영체제에 맞는 터미널에서 아래 명령어를 실행합니다.

#### macOS

Terminal에서 공식 설치 스크립트를 실행합니다.

```bash
curl -fsSL https://ollama.com/install.sh | sh
```

설치 후 Ollama 앱이 실행되지 않았다면 다음 명령어로 실행합니다.

```bash
open -a Ollama
```

GUI 앱을 사용하지 않고 현재 Terminal에서 직접 서버를 실행하려면 다음 명령어를 사용합니다.

```bash
ollama serve
```

> `open -a Ollama`로 앱이 이미 실행 중이면 `ollama serve`를 다시 실행할 필요가 없습니다.

#### Windows

PowerShell에서 공식 설치 스크립트를 실행합니다.

```powershell
irm https://ollama.com/install.ps1 | iex
```

설치가 완료되면 PowerShell을 닫았다가 다시 실행합니다. Ollama는 일반적으로 백그라운드에서 자동 실행됩니다. 실행되지 않았다면 시작 메뉴에서 **Ollama**를 실행하거나 PowerShell에서 다음 명령어를 실행합니다.

```powershell
ollama serve
```

> Ollama가 이미 백그라운드에서 실행 중이면 `ollama serve`를 다시 실행할 필요가 없습니다.

### 3. 프로젝트용 Ollama 모델 다운로드

이 프로젝트는 기능별로 다음 세 모델을 사용합니다.

| 기능 | 모델 |
| --- | --- |
| arXiv 검색 키워드 생성 | `qwen2.5:3b` |
| 논문 번역 | `translategemma:4b` |
| 논문 요약 | `gemma3:4b` |

macOS Terminal 또는 Windows PowerShell에서 아래 명령어를 차례대로 실행합니다.

```bash
ollama pull qwen2.5:3b
ollama pull translategemma:4b
ollama pull gemma3:4b
```

모델 이름은 [`src/config/model_config.yaml`](src/config/model_config.yaml)의 설정과 동일해야 합니다.

### 4. Ollama 실행 확인

설치된 모델 목록을 확인합니다.

```bash
ollama list
```

목록에 `qwen2.5:3b`, `translategemma:4b`, `gemma3:4b`가 모두 표시되어야 합니다.

모델 응답을 직접 확인합니다.

```bash
ollama run qwen2.5:3b "retrieval augmented generation 검색 키워드를 만들어줘"
```

Ollama API 서버의 실행 상태를 확인합니다.

#### macOS

```bash
curl http://localhost:11434/api/tags
```

#### Windows PowerShell

```powershell
Invoke-RestMethod http://localhost:11434/api/tags
```

모델 목록이 JSON 형태로 반환되면 프로젝트에서 Ollama를 사용할 준비가 완료된 것입니다.

### 5. Supervisor 챗봇 실행

프로젝트 루트에서 다음 명령어를 실행한 뒤 질문을 입력합니다.

#### macOS

```bash
python3 main.py
```

#### Windows PowerShell

```powershell
python main.py
```

질문을 명령어에 바로 전달할 수도 있습니다.

#### macOS

```bash
python3 main.py "arXiv에서 RAG 관련 논문을 검색해줘"
```

#### Windows PowerShell

```powershell
python main.py "arXiv에서 RAG 관련 논문을 검색해줘"
```

> 실행 전에 프로젝트 Python 의존성과 `.env` 설정을 완료해야 합니다. LangGraph 관련 의존성은 `requirements-orchestration.txt`에 정의되어 있습니다.

Ollama 설치 관련 세부 사항은 [macOS 공식 문서](https://docs.ollama.com/macos)와 [Windows 공식 문서](https://docs.ollama.com/windows)를 참고합니다.

---

# Git & GitHub 협업 규칙

## 권장 작업 흐름

> **최신 `main` 확인 → 브랜치 생성 → 작업 및 커밋 → `main` 동기화 및 충돌 해결 → Push → PR 생성 → 코드 리뷰 → Squash Merge → 브랜치 삭제**

## 1. 브랜치 규칙

- **브랜치 생성:** 항상 최신 `main` 브랜치에서 생성합니다.
- **네이밍 규칙:** `작업종류/이니셜/작업명`
  - 예시: `feat/JHD/arxiv-search`
  - 작업명은 영문 소문자와 하이픈(`-`)만 사용합니다.
- **작업 단위:** `1 브랜치 = 1 목적` 원칙을 지킵니다.
- **사후 관리:** Merge가 완료된 브랜치는 로컬과 원격에서 모두 삭제합니다.

## 2. 커밋 메시지 규칙

- **메시지 형식:** `[이니셜] 타입: 변경 내용`
  - 예시: `[JHD] feat: arXiv 논문 검색 기능 추가`
- **작성 원칙:**
  - 변경 내용을 구체적으로 명시하고 끝에는 마침표를 붙이지 않습니다.
  - `수정`, `작업` 등 의미가 모호한 단어만으로 작성하지 않습니다.
  - 전체 파일 추가(`git add .`) 대신 변경된 파일을 확인하여 선택적으로 추가하고 커밋합니다.

### 주요 타입(Type)

| 타입 | 설명 |
| --- | --- |
| `feat` | 새로운 기능 추가 |
| `fix` | 버그 및 오류 수정 |
| `docs` | 문서 변경 |
| `refactor` | 기능 변경 없는 코드 구조 개선 |
| `test` | 테스트 코드 추가 및 수정 |
| `chore` | 환경 설정, 패키지 및 기타 작업 |
| `style` | 화면 디자인(UI) 및 스타일 변경 |

## 3. Pull Request 및 병합 규칙

- **PR 대상:** 항상 `main` 브랜치를 대상으로 PR을 생성합니다.
- **사전 작업:** PR 생성 전에 원격 `main`의 최신 변경 사항을 작업 브랜치에 반영하고 충돌을 해결합니다.
- **PR 내용:** 작업 목적, 주요 변경 사항, 실행 방법 및 테스트 결과를 상세히 작성합니다.
- **미완성 작업:** 아직 완료되지 않은 작업은 Draft PR을 활용합니다.
- **Merge 조건:**
  - 최소 1명 이상의 팀원 리뷰와 승인이 필요합니다.
  - 본인의 PR을 직접 병합하는 Self-merge는 금지합니다.
- **Merge 방식:** `Squash and merge`로 통일합니다.

## 4. 프로젝트 구조 및 데이터 관리

### 디렉터리 구조

- 메인 파일(`README.md`, `.env` 등)을 제외한 코드는 `src/feature/`와 `src/tools/`에 집중하여 최소 구조를 유지합니다.
- 새로운 디렉터리가 필요하면 팀원과 먼저 논의한 후 추가합니다.

### 데이터 관리

- 원본 데이터는 수정하지 않고 원형을 보존합니다.
- 개인정보, 비밀키가 포함된 `.env`, 대용량 파일은 저장소에 커밋하지 않습니다.
- 외부 자료를 활용할 때는 출처와 라이선스를 `README.md`에 명시합니다.

### 환경 관리

- 패키지와 버전은 `requirements.txt` 등의 의존성 파일로 관리합니다.
- 의존성을 추가하거나 변경하면 관련 파일도 함께 갱신합니다.

---

## 절대 금지 사항 (Don'ts)

- `main` 브랜치에 직접 Commit 또는 Push하기
- `main` 브랜치나 공유 브랜치에 Force Push(`-f`)하기
- 다른 팀원의 브랜치에 직접 Commit 또는 Force Push하기
- 다른 팀원의 브랜치를 동의 없이 병합 대상으로 삼기
- 충돌 해결 과정에서 다른 팀원의 코드를 임의로 삭제하거나 동의 없이 수정하기
