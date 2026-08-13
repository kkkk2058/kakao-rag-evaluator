# -*- coding: utf-8 -*-
"""평가기_팀8.ipynb 생성기.

evaluator_core.py 를 셀 단위로 쪼개고 노트북 전용 셀(입출력·검증)을 붙인다.
코어 로직을 고쳤으면 이 스크립트를 다시 돌려 노트북을 재생성한다.

실행: python3 build_notebook.py
"""

import json
import os
import re

HERE = os.path.dirname(os.path.abspath(__file__))
CORE = open(os.path.join(HERE, "evaluator_core.py"), encoding="utf-8").read()


def section(start_marker, end_marker=None):
    """evaluator_core.py 에서 '# N. 제목' 구획을 잘라 온다."""
    start = CORE.index(start_marker)
    end = CORE.index(end_marker) if end_marker else len(CORE)
    return CORE[start:end].rstrip() + "\n"


HEAD = """# =====================================================================================
# 0. 채점식 상수
# ====================================================================================="""
S1 = "# 1. 정규화 유틸"
S2 = "# 2. 입력 로더"
S3 = "# 3. 지표 A"
S4 = "# 4. 지표 B"
S5 = "# 5. 지표 C"
S5B = "# 5b. 판정 4축의 실제 채점"
S6 = "# 6. 한 팀 채점"
S7 = "# 7. 교차평가"
S7B = '# 7b. 제출 파일 형식 검사'


def block(a, b=None):
    a_full = "# =====================================================================================\n" + a
    b_full = ("# =====================================================================================\n" + b) if b else None
    return section(a_full, b_full)


# evaluator_core.py의 import 문을 손으로 다시 베끼지 않고 원본에서 그대로 잘라 온다.
# 손으로 유지하면 코어에 import를 추가했을 때 노트북 쪽 갱신을 잊기 쉽다
# (실제로 math를 추가했다가 이 방식으로 바꾸기 전까지 노트북에서만 NameError가 났다).
# os는 evaluator_core.py 자체는 안 쓰지만 9·10절(경로 표시, API 키 환경변수) 노트북
# 전용 코드가 쓰므로 따로 추가한다 — CORE 추출만으로는 못 잡는다.
_CORE_IMPORTS = CORE[CORE.index("from __future__ import annotations"):CORE.index(HEAD)].strip()
IMPORTS = (
    "# -*- coding: utf-8 -*-\n"
    "# 표준 라이브러리만 쓴다. 새 Colab 런타임에서 설치 없이 바로 돈다.\n"
    "# (5b의 Gemini 판정만 예외로 google-genai가 필요하며, 그 설치는 해당 셀에서 한다.)\n"
    + _CORE_IMPORTS + "\n"
    "import os  # 노트북 9·10절 전용(경로 표시, API 키 환경변수) — 코어 자체는 안 씀\n"
)

CELLS = []


def md(text):
    CELLS.append({"cell_type": "markdown", "metadata": {}, "source": text.strip("\n").splitlines(True)})


def code(text):
    CELLS.append({"cell_type": "code", "execution_count": None, "metadata": {},
                  "outputs": [], "source": text.rstrip("\n").splitlines(True)})


# ─────────────────────────────────────────────────────────────────────────────────────
md("""
# KTB RAG 예선 — 팀 8 평가기

팀별 답변 파일을 골드셋과 맞대어 **총점(0~100)** 을 매기고, 교차평가 결과를
`eval_8.json` 으로 저장한다.

## 채점식

```
총점 = 0.20 × (MRR × 100)
     + 0.30 × (키팩트 F1 × 100)
     + 0.50 × (판정 점수 0~100)
```

| 지표 | 무엇을 보나 | 어떻게 재나 |
|---|---|---|
| **MRR** | 근거 조항을 몇 번째로 맞혔나 | 정답 조항이 처음 나온 순위의 역수. `gold_articles` 중 하나만 맞아도 정답 |
| **키팩트 F1** | 답변 내용이 정답과 얼마나 겹치나 | 답변 전체를 토큰 **집합**으로 보고 `key_facts` 전체와 F1 |
| **판정** | 사람이 읽었을 때 좋은 답변인가 | **Gemini 3.5 Flash** 가 accuracy·grounding·completeness·clarity 4축을 0~10으로 채점, 가중합 |

판정 4축 가중치는 `accuracy 0.40 / grounding 0.25 / completeness 0.20 / clarity 0.15` —
운영진 기준(정확성·근거성·완결성·명료성)을 그대로 따른다.

운영진 지침: **"Gemini 3.5 Flash로 공개 10문항과 비공개 30문항을 같은 코드로 처리하고,
익명 5팀 모두에 같은 기준을 적용해야 합니다."** 이 노트북은 판정 축을 실제로 Gemini
3.5 Flash API로 채점한다(5b, 9절). 규칙 기반 대리 심판(`judge_rule_based`, 5절)은
API 키 없이 도는 자체 검증(8절)과, Gemini 호출이 끝내 실패한 문항의 폴백으로만 쓴다.

## 골드셋에서 읽는 필드

`questions[].id`, `questions[].question`,
`questions[].gold_articles[].{doc, article, citation}`, `questions[].key_facts` — 이 넷뿐이다.

`ptype`·`difficulty` 같은 추가 필드는 **있어도 쓰지 않고 없어도 돈다.**
문항 id와 문항 수도 코드에 박지 않고 입력 파일에서 읽는다. 그래서 공개 10문항과
비공개 30문항에 같은 노트북을 그대로 쓴다.

## 이 노트북의 구성

| 절 | 내용 |
|---|---|
| 1~4 | 정규화, 입력 로더, MRR, 키팩트 F1 — 결정론적 지표 |
| 5 | 판정 4축 규칙 기반 대리 심판 — **자체 검증·폴백 전용**, 실채점 아님 |
| 5b | 판정 4축 **Gemini 3.5 Flash 실채점** — 실제 제출에 쓰는 것 |
| 6~7 | 한 팀 채점 → 교차평가 (BLIND01~05, status는 항상 completed) |
| 7b | 제출 파일 형식 검사 — 운영진 검사기와 동일 기준 |
| 8 | 자체 검증 (API 키 불필요, 5절 규칙 기반으로 배선만 확인) |
| 9 | Gemini API 키 설정 + 1문항 스모크 테스트 |
| 10 | 실행 — 업로드 → `eval_8.json` 생성 → 7b로 즉시 `[제출 가능]` 확인 |
| 11 | 문항별 들여다보기 |

> 다른 채점 방식을 쓰고 싶으면 `evaluate_team(..., judge_fn=내함수)` 로 갈아 끼우면 된다.
> `judge_fn(question, answer, key_facts, citations, retrieved, gold_keys)` 가
> `({축: 0~10점}, {축: 사유})` 를 돌려주면 된다. 운영진도 "팀이 중요하다고 판단한
> 기준이 있으면 새로 설계해도 된다"고 했으므로, 축 구성 자체를 바꾸는 것도 자유다 —
> 다만 Gemini 3.5 Flash로 공개·비공개 문항과 5팀 전원을 같은 코드로 처리한다는
> 제약은 유지해야 한다.
""")

code(IMPORTS + "\n\n" + block(HEAD.splitlines()[1], S1))
md("""
## 1. 정규화 — 표기가 흔들려도 같은 것은 같게 본다

남의 팀 파일을 채점하므로 표기 차이로 억울한 감점이 나면 안 된다.

- 조번호: `10`, `"10"`, `"제10조"`, `"제 10 조"` 를 모두 10으로 읽는다.
- 문서명: 공백을 지우고 비교해 `카카오 통합 약관` 과 `카카오통합약관` 을 같게 본다.
  (`카카오 통합서비스약관` 과 `카카오 통합 약관` 은 공백을 지워도 다르므로 안전하다.)
- 토큰: `\\w+` 로 끊는다. 가운뎃점(`수집·이용·제공`)은 경계가 되고, `제16조` 같은
  덩어리는 한 토큰으로 남는다.
""")
code(block(S1, S2))

md("""
## 2. 입력 읽기

골드셋은 **필수 필드가 없으면 즉시 중단**한다(내 채점 기준이 깨진 것이므로).
반대로 답변 파일은 **어떻게 깨져 있어도 중단하지 않는다** — 남의 팀 파일이고,
한 팀이 잘못 냈다고 교차평가 전체가 멈추면 안 되기 때문이다. 깨진 부분은
문제로 기록하고 해당 문항만 0점 처리한다.
""")
code(block(S2, S3))

md("""
## 3. 지표 A — MRR (가중치 0.20)

`retrieved` 를 앞에서부터 훑어 정답 조항이 처음 나온 순위의 역수를 준다.
1순위면 1.0, 2순위면 0.5, 못 맞히면 0.

`gold_articles` 가 여러 개인 문항이 있다. 같은 규정이 여러 약관에 중복 수록된
경우인데, 이때는 **그중 아무거나 맞으면 정답**이다.
""")
code(block(S3, S4))

md("""
## 4. 지표 B — 키팩트 F1 (가중치 0.30)

답변 전체를 토큰 **집합** 하나로 보고, `key_facts` 를 전부 이어붙인 것과 F1을 잰다.
문장별로 맞춰 보는 게 아니라 통짜 비교다. 그래서 두 방향으로 깎인다.

- 정답에 있는데 답변에 없는 토큰 → **재현율** 손해
- 답변에만 있는 군더더기 토큰 → **정밀도** 손해

즉 **빠뜨려도 깎이고 덧붙여도 깎인다.** 질문을 되풀이하거나 안 물어본 조항을
끌어오면 그만큼 점수가 내려간다.

`score_keyfact_recall` 은 팩트를 하나씩 보고 반영 여부를 세는 별도 지표다.
F1은 통짜 비교라 '어느 팩트가 통째로 빠졌는지'를 구분하지 못하는데, 이쪽이
그걸 잡아 뒤의 completeness 축 근거가 된다.
""")
code(block(S4, S5))

md("""
## 5. 지표 C — 판정 4축, 규칙 기반 대리 심판 (자체 검증·폴백 전용)

**이 절의 `judge_rule_based`는 실채점에 쓰지 않는다.** 운영진 지침대로 실채점은
Gemini 3.5 Flash가 한다(5b). 이 규칙 기반 버전은 (1) API 키 없이 몇 초 안에 배선을
검증하는 8절 자체 검증과, (2) Gemini 호출이 끝내 실패한 개별 문항의 폴백으로만 쓰인다.

같은 입력에 항상 같은 점수를 내는 결정론적 함수라 자체 검증·폴백 용도로 적합하다.
축별 규칙과 임계값은 공식 심판이 매긴 축 점수 30건(3개 팀 × 공개 10문항)에 맞춰
정했다 — 그 30건에서 나온 사실이 설계를 결정했다.

| 축 | 가중치 | 무엇을 보나 | 깎는 조건 |
|---|---|---|---|
| accuracy | 0.40 | 틀리게 말한 게 있나 | 정답과 **단위는 같은데 숫자가 다름**(`6개월`을 `3개월`이라 함), 명시적 예/아니오 뒤집힘, 정답과 거의 안 겹침 |
| grounding | 0.25 | 근거가 정답 조항인가 | 정답 조항 순위(1위 10점 → 2위 8 → 3위 7 → 4위 이하 6 → 없음 2), 근거 미제시 0 |
| completeness | 0.20 | 키팩트를 빠짐없이 담았나 | 팩트별 토큰 겹침이 40% 넘으면 '반영됨'. 반영 비율을 완만한 곡선으로 점수화 |
| clarity | 0.15 | 읽히는 답변인가 | 정답 길이의 4배 초과, 같은 말 과도 반복 |

**누락은 completeness가 보고, accuracy는 '틀리게 말한 것'만 본다.** 두 축이 같은 것을
중복해서 깎지 않도록 역할을 갈랐다. 감점은 completeness에 쏠려 있었다(공식 심판
30건 중 16회 감점 중 10회). accuracy·grounding·clarity에서 길이·부정어 같은 얕은
신호로 깎아 보니 공식 점수와의 상관이 음수가 나와, 확실한 증거가 있을 때만 깎도록
좁혔다. 이 보정으로 공식 심판과의 상관이 r = 0.19 → 0.72 로 올랐다 — Gemini를 못
쓰는 상황(폴백)에서도 등급 없는 0점보다는 훨씬 신뢰할 수 있는 대체값이라는 뜻이다.
""")
code(block(S5, S5B))

md("""
## 5b. 판정 4축 — Gemini 3.5 Flash 실채점

**여기가 실제 제출에 쓰는 채점 로직이다.** `make_gemini_judge(api_key)` 가
`evaluate_team(..., judge_fn=...)` 에 꽂을 수 있는 채점 함수를 만들어 준다.

- **모델**: `gemini-3.5-flash` 고정. SDK는 `google-genai`(`pip install -U google-genai`).
  `generate_content`는 문서상 "레거시"지만 전면 지원되고 구조화 출력이 안정적으로
  문서화돼 있어 이 경로를 택했다. 신설 Interactions API는 2026년 6월 GA라 사례가
  아직 적어 제외했다.
- **문항 하나당 API 호출 1회** — 4축을 한 번에 JSON으로 받는다(축마다 나눠 부르지 않음).
- **temperature=0** — 완벽한 결정론은 아니지만 "같은 기준을 같은 코드로 5팀에 적용"
  요구에 맞춰 변동을 최소화한다.
- **구조화 출력**: `response_mime_type="application/json"` + JSON 스키마로 형식을
  강제한다. 점수 범위(0~10)는 스키마에 넣지 않았다 — Gemini의 스키마 서브셋이
  `minimum`/`maximum`을 항상 지원한다는 보장이 없어서다. 대신 프롬프트로 지시하고
  `_clamp_axis_score()` 로 클라이언트에서 한 번 더 강제한다.
- **재시도·폴백**: 네트워크 오류·429·JSON 파싱 실패가 `max_retries`(기본 3)번 반복되면
  **그 문항 하나만** 5절의 규칙 기반 심판으로 대체한다. 한 문항의 일시적 실패로
  나머지 수십 문항, 다른 팀의 채점까지 멈추지 않게 하기 위함이다.
- **속도 제한**: 성공 호출마다 `request_interval_s`(기본 4.5초) 만큼 쉰다 — 무료
  등급의 분당 요청 수 제한을 넘기지 않기 위한 보수적 기본값이다. 유료 등급이거나
  429가 안 뜨면 줄여도 된다.
- **stats**: `{"calls", "gemini_ok", "fallback"}` 을 실시간으로 채운다. 실행이 끝난 뒤
  `fallback` 이 크면 API 키·쿼터를 점검하고 다시 돌려야 한다 — 폴백이 많이 섞인
  결과는 "Gemini로 처리"라는 요구 조건을 만족하지 못한다.
""")
code(block(S5B, S6))

md("""
## 6. 한 팀 채점

**골드셋의 문항을 기준으로 돈다.** 답변 파일에 없는 문항은 0점 + 사유 기록.
답변 파일에만 있고 골드셋에 없는 문항은 그냥 무시한다. 문항 수·id는 전부
골드셋에서 오므로 10문항이든 30문항이든 코드는 그대로다.

**`qid`가 골드셋 `id`와 전혀 다른 라벨을 쓰는 경우** — 예를 들어 골드셋은
`P01`인데 답변 파일은 `BLIND01`처럼 완전히 다른 체계를 쓰는 경우 —
`_resolve_qid_mapping()` 이 **제출 순서를 골드셋 순서에 그대로 대응**시킨다.
실제로 이런 사례가 있었다: 팀 결과기가 낸 정상 qid를 운영진이 답변 파일을
재배포하며 통째로 다시 붙이는 경우, 문자열 매칭만 하면 내용은 멀쩡한 답변이
전부 "미응답 0점"으로 잘못 채점된다.

**개수가 같고 하나도 안 겹칠 때만** 이 폴백이 켜진다. 하나라도 겹치면(부분
일치) 켜지지 않는다 — 그런 애매한 경우는 정말 일부 문항이 빠진 것일 수 있고,
그때 위치로 임의 매칭하면 엉뚱한 답을 엉뚱한 문항에 채점하는 더 나쁜 실패로
이어지기 때문이다. 폴백이 켜지면 `report["qid_mapping_note"]` 에 사유가 남고,
`evaluate_submissions` 를 통하면 `details[blind_id]["problems"]` 로도 보인다.
""")
code(block(S6, S7))

md("""
## 7. 교차평가 → `eval_8.json`

`blind_id` 는 이 순서로 정한다: 파일 안 `blind_id` 필드 → 파일명의 `BLIND\\d+`(숫자는
항상 `BLIND01`처럼 2자리로 맞춤) → 파일명 나머지 → `team` 필드 → 순번.

**`status` 는 항상 `"completed"` 다.** 운영진이 배포한 "교차 평가 결과 파일 검사기"의
규칙 때문이다 — *"일부 문항 실패·평가 전체 실패·후보 중복·누락이 하나라도 있으면
그 평가 결과 파일 전체가 순위 산정에서 제외됩니다. 실패한 후보에게 0점을 주는 방식은
사용하지 않습니다."* 즉 `partial`/`failed` 는 그 후보만 감점되는 게 아니라 **내 제출
파일 전체를 무효화**시킨다. 그래서 후보 파일이 아예 안 읽혀도 빈 답변(`{}`)으로
채점을 강행해 실제 점수(대개 0점에 가까움)를 매기고 `completed` 로 마감한다 —
`load_answers`/`evaluate_team` 이 이미 빈 입력을 정상적인 0점 만점 채점으로 처리하도록
돼 있어(§2, §6) 가능하다. 채점 중 무슨 문제가 있었는지는 제출 파일이 아니라
`details[blind_id]["problems"]` 에만 남는다(진단용).

`check_blind_id_coverage()` 는 제출 직전에 `BLIND01`~`BLIND05` 가 정확히 한 번씩만
있는지 확인한다 — 운영진 검사기가 "후보 중복"·"누락"으로 잡는 것과 같은 조건이다.
""")
code(block(S7, S7B))

md("""
## 7b. 제출 파일 형식 검사 — 운영진 검사기와 동일 기준

운영진이 배포한 "[학생용] 교차 평가 결과 파일 검사기"를 그대로 옮겼다. 재구현하지
않고 원본을 옮긴 이유는, 다시 구현하는 과정에서 조건 하나라도 다르게 해석하면
**"내 검사는 통과했는데 실제 제출은 반려"** 되는 상황이 생기기 때문이다 —
`schema_version`/`rank` 금지, `BLIND01~05` 정확히 5개, `status` 전부 `completed`,
`total` 범위·null 규칙까지 운영진 코드와 1:1로 같다.

10절에서 `eval_8.json` 을 만들자마자 이 검사를 바로 돌려 `[제출 가능]` 을 확인한다.
운영진이 새 검사기를 배포하면 이 셀 내용을 그걸로 통째로 교체하면 된다.
""")
code(block(S7B))

# ─── 노트북 전용: 자체 검증 ───────────────────────────────────────────────────────────
md("""
## 8. 자체 검증 (API 키 불필요)

채점을 돌리기 전에 이 셀로 평가기 배선이 제대로 도는지 확인한다.
가짜 골드셋(id `B001`~`B030`)과 일부러 깨뜨린 답변 파일들을 만들어
만점·영점·깨진 파일 처리, 그리고 **다섯 후보가 파일 상태와 무관하게 항상
`completed`로 마감되는지**(7절 참고)까지 확인한다. 마지막엔 운영진 검사기 기준
(`validate_eval_document`, 7b절)으로도 문제 0건인지 직접 검사한다.

**id 체계와 문항 수가 공개셋과 다른 골드셋으로 도는 것**을 여기서 확인하므로,
비공개 30문항 골드셋을 받아도 그대로 쓸 수 있다는 근거가 된다.

이 셀은 5절의 규칙 기반 심판으로 동작한다 — Gemini API 키가 아직 없어도 실행할 수
있게 하기 위해서다. 점수 자체(축 보정)가 아니라 **채점 파이프라인의 배선**을
확인하는 것이 목적이며, 9절에서 Gemini 연결도 별도로 스모크 테스트한다.
""")
code(r'''
def _self_test():
    """가짜 입력으로 평가기 동작을 확인한다. 실패하면 목록을 찍는다."""
    ok, bad = [], []

    def check(name, cond, detail=""):
        if cond:
            ok.append(name)
        else:
            bad.append(name + ("  <- " + detail if detail else ""))

    # --- 정규화 ---
    check("조번호 '제10조' -> 10", parse_article_no("제10조") == 10)
    check("조번호 True는 무효", parse_article_no(True) is None)
    check("문서명 공백 무시", normalize_doc_name("카카오 통합 약관") == normalize_doc_name("카카오통합약관"))
    check("통합서비스약관/통합약관 구분",
          normalize_doc_name("카카오 통합서비스약관") != normalize_doc_name("카카오 통합 약관"))
    check("가운뎃점 분리", tokenize("수집·이용·제공") == ["수집", "이용", "제공"])

    # --- 지표 ---
    gk = [("카카오계정약관", 10), ("카카오통합약관", 13)]
    check("MRR 1순위", score_mrr([("카카오계정 약관", 10)], gk) == (1.0, 1))
    check("MRR 2순위", score_mrr([("카카오 통합 약관", 99), ("카카오계정 약관", 10)], gk)[0] == 0.5)
    check("MRR 복수 정답 중 하나", score_mrr([("카카오 통합 약관", 13)], gk) == (1.0, 1))
    check("MRR 실패", score_mrr([("카카오계정 약관", 99)], gk) == (0.0, 0))
    facts = ["담당자 1인만 이용할 수 있습니다.", "공유하는 것은 금지됩니다."]
    check("F1 정답 그대로면 1.0", abs(score_keyfact_f1(" ".join(facts), facts)[0] - 1.0) < 1e-9)
    check("F1 무관한 답변은 0", score_keyfact_f1("오늘 날씨가 좋습니다", facts)[0] == 0.0)
    check("F1 군더더기로 떨어짐",
          score_keyfact_f1(" ".join(facts) + " 관련 없는 말 " * 30, facts)[0]
          < score_keyfact_f1(" ".join(facts), facts)[0])

    # --- 공개셋과 다른 id 체계 / 다른 문항 수 ---
    fake = {"questions": [
        {"id": "B{:03d}".format(i), "question": "질문 {}".format(i),
         "gold_articles": [{"doc": "카카오계정 약관", "article": i, "citation": "제{}조".format(i)}],
         "key_facts": ["문항 {}의 핵심 사실 하나입니다.".format(i),
                       "문항 {}의 핵심 사실 둘입니다.".format(i)],
         "ptype": "basic", "difficulty": "hard", "운영진메모": "무시돼야 함"}
        for i in range(1, 31)]}
    gs = load_goldset(fake)
    check("골드셋 30문항 로드", len(gs) == 30)
    check("id 체계가 달라도 됨", gs.qids[0] == "B001" and gs.qids[-1] == "B030")
    check("추가 필드는 안 읽음",
          set(gs.questions[0]) == {"id", "question", "gold_keys", "citations", "key_facts"})

    def answers_of(items):
        return load_answers({"team": "T", "answers": items})[0]

    perfect = answers_of([{"qid": q["id"], "retrieved": [["카카오계정 약관", int(q["id"][1:])]],
                           "answer": " ".join(q["key_facts"])} for q in fake["questions"]])
    check("정답 그대로면 총점 100", abs(evaluate_team(gs, perfect)["total_0_100"] - 100.0) < 1e-6)

    empty = answers_of([{"qid": q["id"], "retrieved": [], "answer": ""} for q in fake["questions"]])
    check("전부 빈 답변이면 0점", evaluate_team(gs, empty)["total_0_100"] == 0.0)

    half = answers_of([{"qid": q["id"], "retrieved": [["카카오계정 약관", int(q["id"][1:])]],
                        "answer": " ".join(q["key_facts"])} for q in fake["questions"][:15]])
    rep_half = evaluate_team(gs, half)
    check("절반만 답하면 총점도 절반쯤", 45 < rep_half["total_0_100"] < 55)
    check("없는 문항은 0점 + 사유", any(p["note"] for p in rep_half["per_question"] if not p["answered"]))

    wrong = answers_of([{"qid": q["id"], "retrieved": [["카카오 위치정보 이용약관", 99]],
                         "answer": " ".join(q["key_facts"])} for q in fake["questions"]])
    rep_wrong = evaluate_team(gs, wrong)
    check("근거 틀리면 MRR 0", rep_wrong["objective"]["mrr"] == 0.0)
    check("근거 틀리면 grounding 감점",
          rep_wrong["per_question"][0]["judge_axes"]["grounding"] < 5)

    # --- 깨진 답변 파일 ---
    _, info = load_answers({"team": "x"})
    check("answers 없으면 fatal", info["fatal"])
    weird, winfo = load_answers({"team": "x", "answers": [
        {"qid": "B001", "retrieved": [{"doc": "카카오계정 약관", "article": "제1조"}], "answer": "정상"},
        {"qid": "B002", "retrieved": "목록아님", "answer": "그래도 채점됨"},
        {"qid": "B003", "retrieved": [["문서명만"]], "answer": "쌍이 깨짐"},
        {"qid": "", "answer": "qid 없음"},
        {"qid": "B004", "answer": None},
        "항목이 문자열",
    ]})
    check("이상한 항목이 섞여도 예외 없음", len(weird) == 4)
    check("dict 형태 retrieved 인정",
          score_mrr(weird["B001"]["retrieved"], [("카카오계정약관", 1)]) == (1.0, 1))
    check("answer가 null이면 빈 문자열", weird["B004"]["answer"] == "")

    # --- 교차평가 산출물: BLIND01~05 다섯 다 있어야 하고, status는 항상 completed ---
    # 운영진 검사기 규칙: 하나라도 completed가 아니면 제출 파일 전체가 순위 산정에서
    # 제외된다. 그래서 못 읽는 파일(BLIND02)·문항이 모자란 파일(BLIND03)도
    # failed/partial로 도망치지 않고 completed + 실제 점수(대개 낮음)로 마감해야 한다.
    out = evaluate_submissions(gs, [
        ("BLIND01", {"team": "1", "answers": [
            {"qid": q["id"], "retrieved": [["카카오계정 약관", int(q["id"][1:])]],
             "answer": " ".join(q["key_facts"])} for q in fake["questions"]]}),
        ("BLIND02", {"team": "2"}),                              # answers 필드 자체가 없음
        ("BLIND03", {"team": "3", "answers": [
            {"qid": q["id"], "retrieved": [], "answer": "x"} for q in fake["questions"][:5]]}),
        ("BLIND04", {"team": "4", "answers": []}),
        ("BLIND05", {"team": "5", "answers": [
            {"qid": q["id"], "retrieved": [["카카오계정 약관", int(q["id"][1:])]],
             "answer": " ".join(q["key_facts"])} for q in fake["questions"]]}),
    ])
    res = {r["blind_id"]: r for r in out["results"]}
    check("완주 -> completed, 100점", res["BLIND01"]["status"] == "completed" and res["BLIND01"]["total"] == 100.0)
    check("answers 필드가 아예 없어도 completed + 0점(failed 아님)",
          res["BLIND02"]["status"] == "completed" and res["BLIND02"]["total"] == 0.0)
    check("문항이 모자라도 completed(partial 아님)", res["BLIND03"]["status"] == "completed")
    check("빈 배열도 completed + 0점", res["BLIND04"]["status"] == "completed" and res["BLIND04"]["total"] == 0.0)
    check("다섯 결과 status가 전부 completed", all(r["status"] == "completed" for r in out["results"]))
    check("결과 키는 셋뿐", all(set(r) == {"blind_id", "total", "status"} for r in out["results"]))
    check("total은 0~100", all(0.0 <= r["total"] <= 100.0 for r in out["results"]))

    coverage_issues = check_blind_id_coverage(out["results"])
    check("BLIND01~05 커버리지 문제 없음", coverage_issues == [])

    validation_issues = validate_eval_document({"results": out["results"]})
    check("운영진 검사기 기준으로도 제출 가능(문제 0건)", validation_issues == [], str(validation_issues))

    # --- qid가 골드셋 id와 전혀 다른 라벨(예: BLIND01)이어도 위치로 매칭되는지 ---
    relabeled = {"team": "relabel-test", "answers": [
        {"qid": "ZZZ{:02d}".format(i + 1), "retrieved": [["카카오계정 약관", int(q["id"][1:])]],
         "answer": " ".join(q["key_facts"])} for i, q in enumerate(fake["questions"])]}
    rep_relabel = evaluate_team(gs, load_answers(relabeled)[0])
    check("qid가 골드셋 id와 완전히 달라도(개수는 같음) 위치로 매칭돼 만점",
          abs(rep_relabel["total_0_100"] - 100.0) < 1e-6, "총점 {}".format(rep_relabel["total_0_100"]))
    check("정상 qid에는 폴백 노트가 안 붙음", evaluate_team(gs, perfect)["qid_mapping_note"] is None)

    print("자체 검증: 통과 {} / 실패 {}".format(len(ok), len(bad)))
    for name in bad:
        print("  [실패] " + name)
    return not bad


_SELF_TEST_OK = _self_test()
''')

# ─── 노트북 전용: Gemini API 키 + 스모크 테스트 ───────────────────────────────────────
md("""
## 9. Gemini API 키 설정 + 스모크 테스트

10절에서 골드셋·답변 파일을 올리기 전에, 여기서 API 키와 모델 호출이 실제로 되는지
문항 1개로 먼저 확인한다. 200문항 가까이(5팀 × 최대 40문항) 처리하다가 중간에
키 오류를 발견하는 것보다 훨씬 싸게 고칠 수 있다.

**키 발급**: https://aistudio.google.com 에서 Google 계정으로 로그인 → API 키 생성.
`AIza` 로 시작하는 문자열이다. **코드에 직접 적지 말 것** — 아래 셀이 입력창으로
물어보거나(화면에 표시되지 않음), Colab 보안 비밀(왼쪽 열쇠 아이콘)에 `GEMINI_API_KEY`
로 저장해 두면 자동으로 읽는다.
""")
code(r'''
!pip install -q -U google-genai

import getpass


def _resolve_gemini_api_key():
    try:
        from google.colab import userdata
        key = userdata.get("GEMINI_API_KEY")
        if key:
            return key
    except Exception:
        pass
    env_key = os.environ.get("GEMINI_API_KEY")
    if env_key:
        return env_key
    return getpass.getpass("Gemini API 키를 입력하세요 (화면에 표시되지 않습니다): ").strip()


GEMINI_API_KEY = _resolve_gemini_api_key()
if not GEMINI_API_KEY:
    raise RuntimeError(
        "Gemini API 키가 없습니다. https://aistudio.google.com 에서 발급받아 "
        "다시 실행하며 입력하세요.")

judge_stats = {"calls": 0, "gemini_ok": 0, "fallback": 0}
gemini_judge = make_gemini_judge(GEMINI_API_KEY, stats=judge_stats)
print("Gemini 판정 준비 완료 (모델: {})".format(_JUDGE_MODEL_DEFAULT))

_smoke_axes, _smoke_reasons = gemini_judge(
    question="사업자/단체 카카오계정은 담당자 몇 명이 이용할 수 있나요?",
    answer="담당자 1인만 이용할 수 있으며, 다른 사람과 공유하는 것은 금지됩니다.",
    key_facts=["담당자 1인만 이용할 수 있습니다.", "다른 사람에게 공유하는 것은 금지됩니다."],
    citations=["카카오계정 약관 제10조"],
    retrieved=[["카카오계정 약관", 10]],
    gold_keys=[("카카오계정약관", 10)],
)
print("\n스모크 테스트 결과 (정답에 가까운 답변이므로 대체로 고득점이 정상):")
for axis, score in _smoke_axes.items():
    print("  {:<13s} {:>4.1f}   {}".format(axis, score, _smoke_reasons[axis]))

if judge_stats["fallback"] > 0:
    raise RuntimeError(
        "Gemini 호출이 실패해 규칙 기반으로 대체됐습니다. 위 사유를 확인하고 "
        "키·쿼터·네트워크를 점검한 뒤 이 셀을 다시 실행하세요.")
print("\nGemini 연결 정상 — 10절에서 실제 채점을 진행하면 됩니다.")
''')

# ─── 노트북 전용: 실행 ────────────────────────────────────────────────────────────────
md("""
## 10. 실행 — 골드셋과 답변 파일을 넣고 `eval_8.json` 만들기

**Colab**: 아래 셀을 실행하면 파일 선택 창이 뜬다. 골드셋 1개 + 채점할 답변 파일들을
한꺼번에 고르면 된다. 골드셋은 `questions` 가 들어 있는 파일로 자동 식별한다.

**로컬**: `USE_UPLOAD = False` 로 두고 `GOLD_PATH` / `ANSWER_PATHS` 에 경로를 적는다.

이 셀은 **두 가지 용도**로 그대로 쓴다. 답변 파일 개수로 자동 구분한다.

- **개발·연습(답변 파일 1개 이상, 5개가 아님)** — 결과기가 만든 자기 팀
  `answers_public_<팀>.json` 을 그대로 채점해 점수만 확인한다. BLIND01~05
  구성·제출 형식 검사는 **건너뛴다** — 아직 제출할 게 아니므로 당연하다.
- **본선 제출(답변 파일 정확히 5개)** — 익명 5팀 답변 파일을 채점해
  `eval_8.json` 을 만든다. 이때만 BLIND01~05 구성을 엄격히 검사하고, 저장 직후
  7b절 검사기로 `[제출 가능]`/`[확인 필요]` 를 바로 보여 준다.

문항마다 Gemini 호출 1회 + 속도 제한 대기가 들어가므로, 5팀 × 30문항 기준으로
**대략 10~20분** 걸린다고 보면 된다. 진행 중 문항마다 한 줄씩 진행 상황이 찍힌다.
""")
code(r'''
# ═══════════════════════════════════════════════════════════════
#  우리 팀 번호 — 결과 파일이 eval_<이 값>.json 으로 저장된다
TEAM = "8"
# ═══════════════════════════════════════════════════════════════

USE_UPLOAD = True          # Colab에서 파일을 골라 올릴지 여부
GOLD_PATH = ""             # USE_UPLOAD=False 일 때 쓰는 골드셋 경로
ANSWER_PATHS = []          # USE_UPLOAD=False 일 때 쓰는 답변 파일 경로 목록
OUT_DIR = "."


def _looks_like_goldset(path):
    """questions[] 를 가진 파일을 골드셋으로 본다."""
    try:
        with open(path, encoding="utf-8") as fp:
            payload = json.load(fp)
    except Exception:
        return False
    return isinstance(payload, dict) and isinstance(payload.get("questions"), list)


def collect_inputs():
    """골드셋 1개와 답변 파일 목록을 모은다."""
    if USE_UPLOAD:
        try:
            from google.colab import files
        except ImportError:
            raise RuntimeError(
                "Colab이 아닙니다. USE_UPLOAD = False 로 바꾸고 "
                "GOLD_PATH / ANSWER_PATHS 에 경로를 적어 주세요.")
        print("골드셋 1개와 채점할 답변 파일들을 한꺼번에 고르세요.")
        uploaded = files.upload()
        paths = list(uploaded.keys())
    else:
        paths = ([GOLD_PATH] if GOLD_PATH else []) + list(ANSWER_PATHS)

    if not paths:
        raise RuntimeError("입력 파일이 없습니다.")

    golds = [p for p in paths if _looks_like_goldset(p)]
    answers = [p for p in paths if p not in golds]
    if len(golds) != 1:
        raise RuntimeError(
            "골드셋(questions 배열이 있는 파일)이 정확히 1개여야 합니다. 지금 {}개: {}"
            .format(len(golds), golds))
    if not answers:
        raise RuntimeError("채점할 답변 파일이 없습니다.")
    return golds[0], sorted(answers)


gold_path, answer_paths = collect_inputs()
goldset = load_goldset(gold_path)

print("\n골드셋: {}  ({}문항)".format(os.path.basename(gold_path), len(goldset)))
print("문항 id: {}{}".format(", ".join(goldset.qids[:5]),
                             " ... " + goldset.qids[-1] if len(goldset) > 5 else ""))
print("채점 대상 {}건:".format(len(answer_paths)))
for p in answer_paths:
    print("  - " + os.path.basename(p))

_expected_calls = len(goldset) * len(answer_paths)
print("\nGemini 판정 예상 호출 수: 문항 {} × 제출물 {} = 최대 {}회".format(
    len(goldset), len(answer_paths), _expected_calls))


def _progress_gemini_judge(question, answer, key_facts, citations, retrieved, gold_keys):
    """gemini_judge를 그대로 호출하되, 문항마다 진행 상황을 한 줄씩 찍는다."""
    result = gemini_judge(question, answer, key_facts, citations, retrieved, gold_keys)
    print("  [{}/{}] 판정 완료 (성공 {} / 대체 {})".format(
        judge_stats["calls"], _expected_calls, judge_stats["gemini_ok"], judge_stats["fallback"]))
    return result


outcome = evaluate_submissions(goldset, answer_paths, judge_fn=_progress_gemini_judge)

# BLIND01~05 구성 검사는 "5팀 익명 제출을 만드는 중"일 때만 막아야 한다. 개발·연습
# 단계(자기 팀 답변 1개만 채점해 보는 것)까지 여기서 막으면 매일 쓰는 워크플로를
# 깨뜨린다 — 정확히 5개를 올렸을 때만 최종 제출 시도로 보고 엄격하게 검사한다.
if len(answer_paths) == 5:
    coverage_issues = check_blind_id_coverage(outcome["results"])
    if coverage_issues:
        print("\n[중단] BLIND01~BLIND05 구성에 문제가 있어 파일을 저장하지 않았습니다:")
        for text in coverage_issues:
            print("   · " + text)
        print("\n입력 파일의 이름·개수를 확인하고(정확히 5개, 각 파일명에 BLIND1~5가 "
              "식별 가능해야 함) 이 셀을 다시 실행하세요.")
        raise RuntimeError("BLIND01~BLIND05 구성 오류로 eval 파일을 생성하지 않았습니다.")
else:
    print("\n[안내] 답변 파일이 5개가 아니라({}개) BLIND01~05 최종 제출 형식 검사는 "
          "건너뜁니다 — 개발·연습 단계로 보고 점수만 계산합니다. 실제 제출 때는 "
          "익명 5팀 답변 파일을 정확히 5개 올려야 합니다.".format(len(answer_paths)))

saved = write_eval_file(outcome, team=TEAM, out_dir=OUT_DIR)

print("\nGemini 판정 요약: 총 {}회 호출 / 성공 {} / 규칙기반 대체 {}".format(
    judge_stats["calls"], judge_stats["gemini_ok"], judge_stats["fallback"]))
if judge_stats["fallback"] > 0:
    print("주의: 대체가 발생했습니다 — 위 사유를 확인하세요. 모든 문항을 Gemini로 "
          "처리해야 하므로, 키·쿼터를 점검하고 가능하면 다시 돌리는 것을 권장합니다.")

print("\n" + "=" * 62)
print("{:<14s} {:>10s}  {:<10s} {:>6s} {:>8s} {:>8s}".format(
    "blind_id", "총점", "status", "MRR", "F1", "판정"))
print("-" * 62)
for row in outcome["results"]:
    detail = outcome["details"].get(row["blind_id"], {})
    obj = detail.get("objective", {})
    print("{:<14s} {:>10.4f}  {:<10s} {:>6} {:>8} {:>8}".format(
        str(row["blind_id"]), row["total"], row["status"],
        "{:.3f}".format(obj["mrr"]) if obj else "-",
        "{:.3f}".format(obj["keyfact_f1"]) if obj else "-",
        "{:.1f}".format(detail["judge"]["score_0_100"]) if detail.get("judge") else "-"))
print("=" * 62)
print("저장: " + saved)

# 파일에 문제가 있었던 제출물은 따로 알려 준다 — 제출 파일 자체에는 안 들어가는 진단 정보
for blind_id, detail in outcome["details"].items():
    problems = detail.get("problems") or []
    if problems:
        print("\n[{}] 파일 문제 {}건 (해당 후보는 그래도 completed로 채점됨)".format(blind_id, len(problems)))
        for text in problems[:5]:
            print("   · " + text)

# 운영진 "교차 평가 결과 파일 검사기"와 같은 기준으로 방금 저장한 파일을 바로 검사한다.
# 답변 파일이 5개가 아니면(개발·연습 단계) 이 검사는 당연히 [확인 필요]가 뜬다 —
# BLIND01~05 5개가 아니니 정상이다. 그런 상황이면 결과가 아니라 이유를 알려 준다.
print("\n" + "=" * 62)
if len(answer_paths) != 5:
    print("[참고] 답변 파일이 5개가 아니므로 제출 형식 검사는 생략합니다 "
          "(개발·연습 단계에서는 정상). 위 표의 점수만 확인하면 됩니다.")
else:
    _submit_ok, _submit_issues = check_eval_file(saved)
    if _submit_ok:
        print("[제출 가능] {} — BLIND01~BLIND05가 모두 정상 완료됐습니다.".format(os.path.basename(saved)))
    else:
        print("[확인 필요] {} — 아래 문제를 해결한 뒤 이 셀을 다시 실행하세요.".format(os.path.basename(saved)))
        for text in _submit_issues:
            print("   · " + text)
''')

md("""
## 11. 문항별 들여다보기

한 제출물의 감점 지점을 확인한다. `TARGET` 을 위 표의 `blind_id` 로 바꿔 실행한다.
""")
code(r'''
TARGET = outcome["results"][0]["blind_id"]       # 보고 싶은 blind_id

detail = outcome["details"][TARGET]
if "per_question" not in detail:
    print("[{}] 채점되지 않았습니다: {}".format(TARGET, detail.get("problems")))
else:
    print("[{}] 총점 {:.4f}  (MRR {:.4f} / F1 {:.4f} / 판정 {:.2f})".format(
        TARGET, detail["total_0_100"], detail["objective"]["mrr"],
        detail["objective"]["keyfact_f1"], detail["judge"]["score_0_100"]))
    print("문항 {}개 중 {}개 응답\n".format(detail["n_questions"], detail["n_answered"]))

    print("{:<8s} {:>5s} {:>7s} {:>7s} {:>7s}  {:>6s} {:>6s} {:>6s} {:>6s}".format(
        "qid", "순위", "MRR", "F1", "정밀도", "정확", "근거", "완결", "명료"))
    print("-" * 72)
    for p in detail["per_question"]:
        if not p["answered"]:
            print("{:<8s}  (답변 없음)".format(p["qid"]))
            continue
        ax = p["judge_axes"]
        print("{:<8s} {:>5d} {:>7.3f} {:>7.3f} {:>7.3f}  {:>6.1f} {:>6.1f} {:>6.1f} {:>6.1f}".format(
            p["qid"], p["rank"], p["mrr"], p["keyfact_f1"], p["keyfact_precision"],
            ax["accuracy"], ax["grounding"], ax["completeness"], ax["clarity"]))

    # 점수가 낮은 문항의 사유
    worst = sorted((p for p in detail["per_question"] if p["answered"]),
                   key=lambda p: p["judge_total"])[:3]
    print("\n판정 점수가 낮은 문항:")
    for p in worst:
        print("\n  [{}] 판정 {:.2f}".format(p["qid"], p["judge_total"]))
        for axis, reason in p.get("judge_reasons", {}).items():
            print("    - {:<13s} {:>5.1f}  {}".format(axis, p["judge_axes"][axis], reason))
''')

# ─────────────────────────────────────────────────────────────────────────────────────
notebook = {
    "nbformat": 4,
    "nbformat_minor": 0,
    "metadata": {
        "colab": {"provenance": [], "toc_visible": True},
        "kernelspec": {"name": "python3", "display_name": "Python 3"},
        "language_info": {"name": "python"},
    },
    "cells": CELLS,
}

out_path = os.path.join(HERE, "평가기_팀8.ipynb")
with open(out_path, "w", encoding="utf-8") as fp:
    json.dump(notebook, fp, ensure_ascii=False, indent=1)
    fp.write("\n")

print("생성: {}  (셀 {}개: 마크다운 {}, 코드 {})".format(
    out_path, len(CELLS),
    sum(1 for c in CELLS if c["cell_type"] == "markdown"),
    sum(1 for c in CELLS if c["cell_type"] == "code")))
