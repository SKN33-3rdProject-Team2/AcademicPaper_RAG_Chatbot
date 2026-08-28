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
사용자 질의
  ↓
의도 파악 및 검색 조건 확인
  ↓
arXiv 논문 검색 및 초록 목록 제공
  ↓
사용자가 처리할 논문 선택
  ↓
PDF 다운로드 및 하이브리드 파싱
  ↓
텍스트 정제 → 청킹 → 임베딩 → ChromaDB 저장
  ↓
전문 번역 및 구조화 요약
  ↓
사용자 선택에 따라 유사 논문 탐색 또는 심층 질의응답
```

## 예상 기술 스택

| 구분 | 기술 |
| --- | --- |
| Language | Python |
| Agent Workflow | LangGraph |
| Paper Search | arXiv API |
| PDF Parsing | PyMuPDF, NVIDIA Build API |
| Local LLM | Ollama, `qwen2.5:3b` |
| Embedding | Hugging Face `BAAI/bge-m3` |
| Vector DB | ChromaDB |
| Architecture | RAG, Human-in-the-Loop, Multi-Agent |

> 기술 스택은 구현 및 검증 과정에서 변경될 수 있습니다.

## 프로젝트 구조

```text
Project-Team-2/
├── .env                 # 환경변수 파일 (Git 제외)
├── .env.example         # 환경변수 예시 템플릿
├── README.md            # 프로젝트 설명서
└── src/                 # 소스코드 최상위 폴더
    ├── feature/         # 피처 생성 및 가공 관련 기능을 구현할 폴더
    └── tools/           # 공통 도구 및 유틸리티 기능을 구현할 폴더
```

### 디렉터리 및 파일 설명

- `.env`: API 키 등 외부에 공개하면 안 되는 로컬 환경변수를 관리합니다. Git에는 포함하지 않습니다.
- `.env.example`: 프로젝트 실행에 필요한 환경변수의 이름과 예시 값을 공유합니다.
- `README.md`: 프로젝트 개요, 설정 방법, 실행 방법 및 협업 규칙을 기록합니다.
- `src/feature/`: 피처 생성 및 가공과 관련된 기능과 함수를 구현합니다.
- `src/tools/`: 여러 기능에서 공통으로 사용하는 도구와 유틸리티 함수를 구현합니다.

## 실행 방법

### 1. 환경변수 설정

저장소를 내려받은 뒤 `.env.example` 파일을 복사하여 `.env` 파일을 생성합니다.

#### macOS / Linux

```bash
cp .env.example .env
```

#### Windows PowerShell

```powershell
Copy-Item .env.example .env
```

`.env.example` 파일은 다음과 같이 구성합니다.

```dotenv
# Ollama
OLLAMA_BASE_URL=http://localhost:11434
OLLAMA_MODEL=qwen2.5:3b
```

> 실제 API 키, 비밀번호 등 민감한 정보는 `.env.example`이나 소스코드에 작성하지 않습니다.

### 2. Ollama 설치

Windows PowerShell에서 다음 명령어를 실행합니다.

```powershell
irm https://ollama.com/install.ps1 | iex
```

> 위 명령어는 Python 명령어가 아닌 Windows PowerShell 명령어입니다.

설치가 완료되면 PowerShell을 종료한 후 다시 실행합니다.

macOS 또는 Linux 사용자는 [Ollama 공식 다운로드 페이지](https://ollama.com/download)에서 운영체제에 맞는 설치 방법을 확인합니다.

### 3. Qwen2.5 3B 모델 다운로드

터미널 또는 PowerShell에서 다음 명령어를 실행합니다.

```bash
ollama pull qwen2.5:3b
```

### 4. 설치 확인

설치된 모델 목록을 확인합니다.

```bash
ollama list
```

모델을 직접 실행하여 정상 작동 여부를 확인합니다.

```bash
ollama run qwen2.5:3b
```

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
