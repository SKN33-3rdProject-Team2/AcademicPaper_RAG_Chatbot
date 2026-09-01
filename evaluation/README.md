# 평가 실행 안내

평가셋 `v2`는 논문 10편과 총 46개 사례로 구성됩니다.

- 번역·요약 산출물 평가: 10개
- RAG 질문: 논문별 질문 20개와 답변 거절 질문 3개
- Deep Research 질문: 5개
- 전체 파이프라인: 8개

## 로컬 산출물 평가

외부 모델이나 네트워크 호출 없이 현재 저장된 번역·요약 파일을 평가합니다.

```bash
python evaluation/run_evaluation.py --suite artifacts --mode local
```

결과를 JSON 파일로 저장하려면 `--output`을 지정합니다.

```bash
python evaluation/run_evaluation.py \
  --suite artifacts \
  --mode local \
  --output output/artifact-evaluation.json
```

## RAG·Deep Research·전체 파이프라인 평가

다음 평가는 실제 모델, Vector DB 또는 외부 서비스가 필요할 수 있습니다.

```bash
python evaluation/run_evaluation.py --suite rag --mode local --allow-live
python evaluation/run_evaluation.py --suite deep_research --mode local --allow-live
python evaluation/run_evaluation.py --suite pipeline --mode local --allow-live
```

의미적 번역 품질, 요약 품질 및 인용-근거 일치도를 LLM으로 평가하려면 `--llm-judge`를 추가합니다.

```bash
python evaluation/run_evaluation.py --suite rag --mode local --allow-live --llm-judge
```

## LangSmith 평가

`.env`에 `LANGSMITH_API_KEY`를 설정한 후 실행합니다. 같은 버전의 데이터셋이 이미 있어도 새 사례를 추가하고 변경된 사례를 갱신합니다.

```bash
python evaluation/run_evaluation.py --suite artifacts --mode langsmith
python evaluation/run_evaluation.py --suite rag --mode langsmith --allow-live --llm-judge
python evaluation/run_evaluation.py --suite deep_research --mode langsmith --allow-live --llm-judge
python evaluation/run_evaluation.py --suite pipeline --mode langsmith --allow-live
```

실제 모델 평가에는 비용과 실행 시간이 발생할 수 있습니다.

## 네 가지 평가 도구

평가 결과는 다음 역할로 나누어 사용합니다.

| 도구 | 역할 |
| --- | --- |
| LangSmith | LangGraph 실행 추적, 데이터셋과 실험 관리 |
| RAGAS | 검색 문맥 정밀도·재현율, 답변 충실도·관련성 |
| DeepEval | RAG 답변 품질, 에이전트 도구 선택과 순서 |
| Promptfoo | 대표 질문 회귀 테스트와 모델·프롬프트 비교 |

RAGAS와 DeepEval의 의존성은 메인 실행 환경과 OpenAI 클라이언트 버전이
달라질 수 있으므로 별도 평가 환경에 설치합니다.

```bash
conda create -n evaluation_env python=3.12 -y
conda activate evaluation_env
pip install -r evaluation/requirements.txt
```

### RAGAS

기존 RAG 평가셋을 실제로 실행한 뒤 다음 네 지표를 계산합니다.

- ID 기반 Context Precision
- ID 기반 Context Recall
- Faithfulness
- Answer Relevancy

먼저 2건만 확인하려면 다음과 같이 실행합니다.

```bash
python -m evaluation.ragas_evaluation --max-cases 2
```

전체 결과를 파일로 저장하려면 `--output`을 지정합니다.

```bash
python -m evaluation.ragas_evaluation \
  --output tmp/ragas-results.json
```

### DeepEval

RAG는 Answer Relevancy와 Faithfulness를 평가합니다. Deep Research와 전체
파이프라인은 Answer Relevancy와 LangGraph 노드 선택·순서를 평가합니다.

```bash
python -m evaluation.deepeval_evaluation --suite rag --max-cases 2
python -m evaluation.deepeval_evaluation --suite deep_research --max-cases 2
python -m evaluation.deepeval_evaluation --suite pipeline --max-cases 2
```

### Promptfoo

Promptfoo는 `evaluation/promptfoo/`의 Python provider를 통해 실제
`SupervisorChatbot`을 호출합니다. 기본 평가셋은 Deep Research 5건, 근거
없는 질문 3건, 대표 파이프라인 2건으로 구성됩니다.

```bash
cd evaluation/promptfoo
PROMPTFOO_PYTHON="$(which python)" npx promptfoo@latest eval \
  --output ../../tmp/promptfoo-results.json
```

모델 또는 프롬프트를 변경한 뒤 같은 명령을 다시 실행하면 회귀 결과를
비교할 수 있습니다. RAGAS, DeepEval, Promptfoo는 실제 모델 호출 비용과
실행 시간이 발생할 수 있으므로 처음에는 `--max-cases 2`처럼 작게 실행합니다.
