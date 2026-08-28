# paper_extract — 논문 본문 추출 결과

`src/feature/paper_extractor.py` 가 만들어 내는 폴더다. `data/paper_save` 의 PDF 중
사용자가 고른 논문을 가공해 여기에 떨어뜨린다.

다음 단계(심층 질의응답, 번역, 요약, 벡터 색인)는 **이 폴더만 보면 된다.** 원본 PDF를
다시 열 필요가 없다.

```bash
python src/feature/paper_extractor.py     # 목록에서 번호를 골라 추출
```

---

## 1. 세 가지 파일에 뭐가 담기나

논문 한 편을 추출하면 세 가지가 생긴다. **같은 내용을 셋 다 담는 게 아니라, 쓰임이
다르다.**

| 파일 | 담기는 것 | 누가 쓰나 |
|---|---|---|
| `extracted_papers.db` | 대단원별로 쪼갠 본문 + 전문 | **프로그램** (심층 질의응답의 주 입력) |
| `extracted_papers.json` | 논문 제목 목록 + 참고문헌 | **프로그램** (DB를 열지 않고 훑을 때) |
| `<논문id>.md` | 가공한 전문 하나로 이어 붙인 것 | **사람** (눈으로 확인할 때) |

### `extracted_papers.db` — SQLite, `extracted` 표 하나

논문 한 편이 한 행이다. 대단원별로 이미 쪼개져 있으므로, 질의응답이 "이 논문 실험
방법이 뭐야" 같은 질문을 받으면 `method` 컬럼만 꺼내 쓰면 된다. 전문을 통째로
프롬프트에 넣을 필요가 없다.

### `extracted_papers.json` — 논문 목록

DB를 열지 않고도 "지금 뭐가 추출돼 있나"를 알 수 있게 하는 색인이다.

```json
{
    "2312.04649v1": {
        "id": "2312.04649v1",
        "title": "PyThaiNLP: Thai Natural Language Processing in Python",
        "reference_pdf": [
            "Rami Al-Rfou. 2015. Polyglot. Available at https://pypi.org/project/polyglot/.",
            "Dimo Angelov. 2020. Top2Vec: Distributed representations of topics.",
            "..."
        ]
    }
}
```

`reference_pdf` 는 그 논문이 **인용한 문헌 목록**이다. 인용 논문을 따라 들어가는
기능을 붙일 때 쓰라고 모아 둔 것이다. 원본 PDF 파일명은 여기 없다 — DB의
`source_pdf` 컬럼과 마크다운 머리말에 있다.

### `<논문id>.md` — 사람이 읽는 전문

**절 나누기 전, 추출 직후에 쓴다.** 뒤이은 분류는 모델을 부르므로 실패하거나 오래
걸릴 수 있는데, 추출한 본문 자체는 그것과 무관하게 이미 완성돼 있기 때문이다.
분류가 이상하게 됐을 때 원본과 대조하는 용도이기도 하다.

수식은 LaTeX(`$...$`), 표는 마크다운 표로 들어 있다.

---

## 2. DB 스키마

```sql
CREATE TABLE extracted (
    id            TEXT PRIMARY KEY,   -- arXiv id. 예: "2312.04649v1"
    title         TEXT,               -- 논문 제목
    source_pdf    TEXT,               -- 원본 PDF 파일명

    -- ↓ 표준 골격(IMRaD)으로 쪼갠 본문. 논문마다 비는 칸이 생긴다.
    abstract      TEXT,
    introduction  TEXT,
    related_work  TEXT,
    method        TEXT,
    experiment    TEXT,
    result        TEXT,
    conclusion    TEXT,
    others        TEXT,               -- JSON dict. {"절 제목": "본문", ...}

    reference_pdf TEXT,               -- JSON 배열. 인용 문헌 문자열 목록
    content       TEXT,               -- 가공한 전문 (참고문헌 포함)
    n_pages       INTEGER,
    n_chars       INTEGER,            -- len(content)
    extractor     TEXT,               -- "vision" | "pymupdf" | "mixed(11/12)"
    created_at    DATETIME
)
```

### 컬럼을 읽을 때 알아야 할 것

**`content` 와 IMRaD 컬럼들의 합은 다르다.** `content` 는 참고문헌까지 포함한
전문이고, 컬럼들은 참고문헌을 뺀 본문만 나눠 담은 것이다. 실측 예: `content`
49,134자 vs 컬럼 합계 33,026자 — 차이는 참고문헌과 표지다.

**`extractor` 로 품질을 가늠할 수 있다.** `vision` 이면 전 쪽을 비전 모델이 읽은
것이고, `mixed(11/12)` 는 한 쪽이 실패해 PyMuPDF 원본으로 대체됐다는 뜻이다.
대체된 쪽은 2단 조판이 뒤섞여 문장이 끊길 수 있다.

**`others` 와 `reference_pdf` 는 TEXT 에 담긴 JSON 이다.** 꺼내 쓸 때
`json.loads()` 가 필요하다.

```python
import json
from feature.paper_extractor import PaperExtractor

record = PaperExtractor().get("2312.04649v1")
method = record["method"]                            # 그냥 문자열
others = json.loads(record["others"])                # {"절 제목": "본문"}
refs   = json.loads(record["reference_pdf"])         # ["인용1", "인용2", ...]
```

편의 메서드도 있다.

```python
extractor.get_part("2312.04649v1", "method")   # 한 컬럼
extractor.get_parts("2312.04649v1")            # IMRaD 컬럼 전부를 dict 로
extractor.get_others("2312.04649v1")           # others 를 dict 로 (json.loads 완료)
```

---

## 3. 어떻게 분류하나

세 단계다. **1단계는 규칙, 2단계는 모델, 3단계는 배분.**

### 1단계 — 대단원 찾기 (규칙)

가공된 본문에서 헤딩 줄을 모으고, 그중 무엇이 대단원인지 가린다. 비전 모델이 매기는
헤딩 레벨은 한 논문 안에서도 들쭉날쭉해서(`### 1 Introduction` 과
`## 3 Original Beam Search` 가 섞인다) **레벨로는 못 가린다.** 그래서 번호 체계를
본다.

- `_numbering_style()` 이 문서 전체를 보고 `arabic` 인지 `roman` 인지 먼저 정한다.
  `II`·`III` 처럼 두 글자 이상 로마 숫자가 하나라도 있으면 그 논문은 로마 체계다.
- 로마 체계에서는 번호를 **순서**로 본다. `C` 는 로마 숫자 100 이라 라벨만 봐서는
  소단원 글자와 구별되지 않지만, 대단원은 I·II·III·IV 로 이어지므로 앞 대단원
  다음 값인지 보면 갈린다. → `V. RESULTS` 는 살리고 `C. Tasks` 는 거른다.
- `References`·`Bibliography` 를 만나면 **거기서 멈추고 뒤는 담지 않는다.**

### 2단계 — 표준 골격에 맞추기 (모델 1회 호출)

논문마다 절 이름을 자기 주제에 맞게 짓는다. `Original Beam Search` 는 이름만 봐서는
모르지만 실제로는 method 다. 이걸 규칙으로 잡을 수 없어서 모델에게 맡긴다.

- 논문 한 편당 **호출 한 번**이다. 절 제목과 **첫 120자**만 보낸다. 전문을 보내지
  않으므로 싸고 빠르다.
- 지시문은 `SECTION_CLASSIFY_SYSTEM_PROMPT` (영어). 판단 대상(영어 절 제목)과
  출력(영어 식별자)이 모두 영어라 영어 지시가 낫다.
- 모델이 답한 절 이름에 번호가 붙어 있을 수 있어(`"3 Original Beam Search"`),
  `_normalize_label()` 로 번호를 떼고 맞춘다.
- **모델 호출이 실패하면 `SECTION_KEYWORDS` 로 대체한다.** 절 이름에
  `introduction`·`method`·`conclusion` 같은 낱말이 있으면 그 칸에 넣는 단순 규칙이다.
  로그 코드 `5403 SECTION_CLASSIFY_FALLBACK` 이 남는다.

### 3단계 — 컬럼에 배분

분류 결과대로 컬럼에 담고, 표준 8칸 중 어디에도 안 맞는 절은 `others` 에
`{"절 제목": "본문"}` 으로 모은다. 참고문헌은 어느 쪽에도 담지 않는다.

---

## 4. 다음 사람(심층 질의응답)이 알아야 할 한계

**하나. 컬럼이 비어 있는 건 정상일 수 있다.** 논문마다 구성이 다르다. PyThaiNLP는
라이브러리 소개 논문이라 실험 절 자체가 없어서 `experiment` 가 비었다. **빈 칸을
추출 실패로 단정하면 안 된다.** 그 논문에 그 절이 원래 없을 수 있다.

**둘. 분류는 매번 똑같지 않다.** 2단계가 모델 판단이라, 같은 논문을 다시 돌리면
경계에 있는 절이 다른 칸으로 갈 수 있다. 실제로 `5 PyThaiNLP in the Wild` 가 한
번은 `result`, 한 번은 `others` 로 갔다. **특정 컬럼에 뭐가 있을 거라 가정하지 말고,
비었으면 `others` 와 `content` 를 함께 보는 편이 안전하다.**

**셋. 쪽 번호가 DB에 없다.** 추출 중에는 블록마다 쪽 번호를 달고 다니지만, 컬럼에
담을 때 텍스트만 남는다. **"3페이지의 수식" 같은 질문에 근거를 짚어 주려면 지금
구조로는 안 된다.** 필요하면 `Section.pages` 를 저장하도록 스키마를 늘려야 한다.

**넷. `others` 의 키는 논문마다 다르다.** 원 논문의 절 제목을 그대로 쓰기 때문에
고정된 키를 기대할 수 없다. 순회해서 쓰는 것을 전제로 짜야 한다.

**다섯. 긴 논문에는 표와 그림이 없다.** 참고문헌을 뺀 본문이 8쪽을 넘으면 표와 그림
캡션을 모두 버리고 본문만 남긴다(`MAX_PAGES_FOR_TABLES`). 분량과 처리 시간 때문이다.
표를 근거로 답해야 하는 질문은 짧은 논문에서만 가능하다.
