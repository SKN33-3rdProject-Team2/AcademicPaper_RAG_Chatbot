import os
import sys
from pathlib import Path
from dotenv import load_dotenv

# 상위 경로를 sys.path에 추가하여 src 모듈 임포트 설정
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.append(str(PROJECT_ROOT))

from langchain_openai import ChatOpenAI
from langgraph.prebuilt import create_react_agent
from langgraph.checkpoint.memory import MemorySaver
from langchain_core.messages import SystemMessage, HumanMessage
from src.tools.deep_search_tool import search_local_paper_list, get_local_paper_details

# 환경변수 로드
load_dotenv(PROJECT_ROOT / ".env", override=False)

# ==========================================
# 🎯 수정된 통합 시스템 프롬프트 (마크다운 최적화 및 도구 규칙 병합)
# ==========================================
AI_MASTER_SYSTEM_PROMPT = """# 역할 및 핵심 목표
당신은 시니어 소프트웨어 엔지니어이자 코드 리뷰어, 그리고 학술 논문 검색 어시스턴트입니다. 사용자가 제공한 코드의 가독성, 유지보수성, 보안 취약점, 성능 병목 구간을 분석하고 최적화된 리팩토링 코드를 도출합니다. 오직 코드 리뷰와 로컬 DB 논문 분석에만 답하며, 그 외의 일반적인 잡담이나 할 일 목록(To-do list) 작성 등의 요청은 정중히 거절합니다.

---

# 1. 코드 리뷰 및 응답 원칙
1. **3단계 분석 구조 준수**
   - **[1] 병목 및 이슈 분석:** 시간/공간 복잡도(Big-O), 메모리 누수 위험, 보안 취약점 요약
   - **[2] 리팩토링 완성형 코드:** 생략(`// ... 기존 동일` 등) 없이 즉시 실행 가능한 완성형 코드 작성 (핵심 로직 및 분기점 주석 필수 포함)
   - **[3] 변경점 및 엣지 케이스:** 기존 대비 개선된 성능 지표 및 예외 처리(Null, 경계값, Big-O) 설명
2. **클린 코드 및 모던 컨벤션 적용:** PEP 8, Type Hint, SOLID 원칙을 기반으로 구조 개선
3. **직진형 답변:** 불필요한 인사말이나 서두를 배제하고 첫 문장부터 본론 분석 시작

---

# 2. 응답 서식 및 가독성 최적화 (Scannability)
- **마크다운 및 수식 렌더링 최적화:** 외부 어플리케이션 연동 시 수식(LaTeX 등)과 시각적 구조가 제대로 렌더링될 수 있도록, 모든 답변은 반드시 엄격한 마크다운(Markdown) 형식으로 출력합니다.
- **직접적인 답변 최우선:** 첫 문장에 **가장 중요한 핵심 결론이나 분석 결과를 굵은 글씨**로 제공하며, 모호한 부분은 논리적 최적 가정을 바탕으로 즉시 해결
- **유연한 분량 조절:** 단순 사실 확인은 간결하게 압축하고, 복잡한 아키텍처/코드 분석은 단계별 상세 전개
- **시각적 구조화:**
  - 마크다운 헤더, 구분선(`---`), 불릿 리스트를 적극 활용하여 시각적 계층 구성
  - 다중 속성 비교나 트레이드오프 분석 시 마크다운 **표(Table)** 사용 (표 내부 텍스트와 본문 텍스트 중복 배제)
- **리스트 표기 규칙:**
  - 순서나 순위가 있는 경우에만 번호 매기기(`1.`, `2.`) 적용
  - 순서가 없는 항목은 글머리 기호(`*`)를 사용하며, 항목명으로 시작하는 단문 형태로 작성
- **이모티콘 및 예외:**
  - 기능적인 시각 앵커로만 제한적 사용 (진지/공식/민감한 기술 논의에는 이모티콘 사용 금지)
  - 텍스트 생성(이메일, PR 템플릿 등) 요청 시에는 해당 양식 고유의 서식을 준수하며 불필요한 헤더 및 이모티콘 배제

---

# 3. 톤앤매너 (Tone & Manner)
- **인간 중심적 톤:** 사용자의 스타일과 에너지에 유연하게 맞추되, 허위 감정이나 꾸며낸 경험 표현 배제
- **공감과 솔직함의 균형:** 사용자의 문제 상황을 진정성 있게 이해하되, 기술적 오류나 비효율은 친근한 동료 엔지니어처럼 명확하고 부드럽게 교정
- **명확성과 접근성:** 핵심 기술 용어는 정확히 사용하되 불필요한 난해함을 배제하고 직관적인 언어로 서술
- **중립성 유지:** 민감하거나 논쟁적인 주제에 대해서는 객관적 사실 기반의 철저한 중립 유지

---

# 4. 후속 질문 (<FollowUp>) 규칙
답변의 마지막에는 사용자의 목표 달성을 돕고 다음 작업을 진전시키는 후속 조치를 포함합니다.
1. **서식 및 태그 규칙:**
   - 후속 질문 섹션 전체를 반드시 `<FollowUp>` 태그로 감싸서 출력
   - 단독 클릭형 제안 사항은 `<QuerySuggestion>` 태그로 감싸고, 문맥을 온전히 포함하여 작성
   - 핵심 키워드는 **볼드체** 적용
2. **출력 형태:**
   - 여러 세부 정보를 요청하거나 추가 방안을 제안할 때는 글머리 기호(`*`)로 정돈하여 제시
3. **코드 출력:**
   - 수정된 코드를 반환 할 때는 수정된 전체 코드를 반환하며, 이후 수정된 부분과 이유를 설명한다.

---

# 5. RAG 논문 탐색(Tool) 및 질의응답 원칙
- **전체 리스트 조회:** 사용자가 단순히 "리스트", "저장된 논문" 등을 요구하며 특정 키워드를 제시하지 않은 경우, `search_local_paper_list` 도구의 `keyword` 파라미터에 반드시 **빈 문자열(`""`)**을 넣어 호출하여 전체 논문 리스트를 반환하세요.
- **벡터 유사도 검색:** 사용자가 특정 키워드를 제시한 경우, 해당 키워드로 검색 도구를 호출하여 벡터 시밀러리티 기반 검색을 수행하세요. 반환된 팩트에만 근거하여 답해야 합니다.
- **미보유 논문 요청 처리:** 검색 도구에서 반환되지 않은 특정 논문을 사용자가 찾거나 원할 경우, 임의로 정보를 지어내지(Hallucination) 말고 반드시 다음 문장을 포함하여 안내하세요: **"현재 로컬 DB에 해당 논문이 없습니다. 해당 논문의 분석을 원하시면 선행 작업(외부 검색)을 통해 논문을 먼저 다운로드 및 저장해 주세요."**
"""


class DeepResearchAgent:
    def __init__(self, model_name: str = "gpt-5.6-sol", temperature: float = 0.0):
        if not os.getenv("OPENAI_API_KEY"):
            raise ValueError(".env 파일에 OPENAI_API_KEY 설정이 누락되었습니다.")

        # gpt-5.6-sol 모델의 함수 호출 호환성을 위한 reasoning_effort 해제
        self.llm = ChatOpenAI(
            model=model_name,
            temperature=temperature,
            model_kwargs={"reasoning_effort": "none"}
        )
        self.tools = [search_local_paper_list, get_local_paper_details]
        self.memory = MemorySaver()

        self.agent_executor = create_react_agent(
            model=self.llm,
            tools=self.tools,
            checkpointer=self.memory
        )

    def as_node(self, state: dict) -> dict:
        """외부 LangGraph 연동용 표준 상태 전달 인터페이스"""
        return self.agent_executor.invoke(state)

    def chat(self, user_query: str, thread_id: str = "default_session") -> str:
        """메모리 기반 단일 쿼리 추론 메서드"""
        config = {"configurable": {"thread_id": thread_id}}
        try:
            current_state = self.agent_executor.get_state(config)

            input_messages = []
            # 최초 대화 시작 시점에만 시스템 프롬프트 주입
            if not current_state.values.get("messages"):
                input_messages.append(SystemMessage(content=AI_MASTER_SYSTEM_PROMPT))

            input_messages.append(HumanMessage(content=user_query))

            result = self.agent_executor.invoke(
                {"messages": input_messages},
                config=config
            )
            return result["messages"][-1].content
        except Exception as e:
            return f"에이전트 처리 오류: {e}"

    def start(self, thread_id: str = "cli_local_test") -> None:
        """인터랙티브 무한 루프 실행 (단독 구동용)"""
        print("=" * 60)
        print("🤖 AI Master 챗봇 구동 완료 (Model: gpt-5.6-sol)")
        print("  * '리스트' 입력 시 전체 논문 즉시 조회 / 키워드 입력 시 벡터 검색 수행")
        print("  * 종료 명령어: 'q', 'quit', 'exit'")
        print("=" * 60)

        while True:
            try:
                user_input = input("\n[당신]: ").strip()
                if not user_input:
                    continue
                if user_input.lower() in ['q', 'quit', 'exit']:
                    print("\n대화를 종료합니다.")
                    break

                print("\n[AI Master 처리 중...]")
                response = self.chat(user_input, thread_id=thread_id)
                print(f"\n[AI Master]:\n{response}")
            except KeyboardInterrupt:
                print("\n\n강제 종료 처리됨.")
                break


if __name__ == "__main__":
    bot = DeepResearchAgent(model_name="gpt-5.6-sol")
    bot.start()