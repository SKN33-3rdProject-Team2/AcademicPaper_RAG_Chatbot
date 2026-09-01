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
