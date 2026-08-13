# -*- coding: utf-8 -*-
"""KTB RAG 교차평가 평가기 — 코어 로직.

공개 10문항/비공개 30문항 골드셋과 팀별 답변 파일을 받아 총점을 계산한다.
노트북(평가기.ipynb)의 각 셀은 이 파일의 함수를 그대로 옮긴 것이다.

설계 원칙
  1. 골드셋에서 읽는 필드는 id / question / gold_articles[{doc,article,citation}] / key_facts 넷뿐이다.
     ptype·difficulty 같은 추가 필드는 있어도 무시하고, 없어도 동작한다.
  2. 문항 id·문항 수를 코드에 고정하지 않는다. 전부 입력 파일에서 읽는다.
  3. 남의 팀 파일을 채점하므로, 어떤 형태로 깨져 있어도 예외로 죽지 않고
     해당 문항 0점 + 사유 기록으로 넘어간다.
"""

from __future__ import annotations

import json
import math
import re
import time
import unicodedata
from collections import Counter

# =====================================================================================
# 0. 채점식 상수
# =====================================================================================
# 축 가중치: 총점 = 0.2*MRR(100) + 0.3*keyfact_f1(100) + 0.5*judge(100)
WEIGHTS = {"mrr": 0.20, "keyfact": 0.30, "judge": 0.50}

# 판정(judge) 4축 가중치. 축 점수는 0~10, judge_total = 10 * Σ w*s
JUDGE_AXIS_WEIGHTS = {
    "accuracy": 0.40,
    "grounding": 0.25,
    "completeness": 0.20,
    "clarity": 0.15,
}

# 근거 표기에 쓰이는 공식 문서명 4종. 채점에 직접 쓰지는 않고(골드셋의 doc이 기준),
# 답변 파일에 낯선 문서명이 나왔을 때 알려 주는 용도로만 둔다.
OFFICIAL_DOCUMENT_NAMES = (
    "카카오계정 약관",
    "카카오 위치정보 이용약관",
    "카카오 통합서비스약관",
    "카카오 통합 약관",
)


# =====================================================================================
# 1. 정규화 유틸 — 표기 흔들림을 흡수한다
# =====================================================================================
def nfkc(text) -> str:
    """유니코드 정규화. 전각/반각·호환문자 차이를 없앤다."""
    if text is None:
        return ""
    if not isinstance(text, str):
        text = str(text)
    return unicodedata.normalize("NFKC", text)


def tokenize(text) -> list:
    r"""채점용 토큰화.

    `\w+` 로 끊는다. 공백·구두점·가운뎃점(·)은 경계가 되고 한글/영문/숫자 덩어리만 남는다.
    예) "제16조 제2항에 근거하여" -> ['제16조', '제2항에', '근거하여']
    """
    return re.findall(r"\w+", nfkc(text))


def normalize_doc_name(name) -> str:
    """문서명 비교용 키. 공백을 모두 없애 '카카오 통합 약관'과 '카카오통합약관'을 같게 본다.

    '카카오 통합서비스약관'과 '카카오 통합 약관'은 공백 제거 후에도 서로 다르므로 안전하다.
    """
    return re.sub(r"\s+", "", nfkc(name))


def parse_article_no(value):
    """조번호를 정수로 뽑아낸다. 3 / "3" / "제3조" / "제 3 조" 모두 3으로 본다.

    숫자를 찾지 못하면 None.
    """
    if isinstance(value, bool):
        return None
    if isinstance(value, int):
        return value
    match = re.search(r"\d+", nfkc(value))
    return int(match.group()) if match else None


def article_key(doc, article):
    """(문서명, 조번호) 비교 키. 어느 한쪽이라도 못 읽으면 None."""
    doc_key = normalize_doc_name(doc)
    no = parse_article_no(article)
    if not doc_key or no is None:
        return None
    return (doc_key, no)


# =====================================================================================
# 2. 입력 로더 — 공통 필드만 읽고, 추가 필드에 의존하지 않는다
# =====================================================================================
class GoldSet:
    """골드셋 한 벌. questions[] 에서 공통 4필드만 뽑아 들고 있는다."""

    def __init__(self, questions, source=""):
        self.questions = questions          # [{id, question, gold_keys, citations, key_facts}]
        self.source = source
        self.qids = [q["id"] for q in questions]

    def __len__(self):
        return len(self.questions)


def load_goldset(path_or_payload) -> GoldSet:
    """골드셋 JSON을 읽는다.

    읽는 필드: questions[].id / .question / .gold_articles[{doc,article,citation}] / .key_facts
    그 밖의 필드(_meta, ptype, difficulty, tag ...)는 있으면 무시하고 없어도 된다.
    """
    if isinstance(path_or_payload, dict):
        payload, source = path_or_payload, "<payload>"
    else:
        source = str(path_or_payload)
        with open(path_or_payload, encoding="utf-8") as fp:
            payload = json.load(fp)

    raw_questions = payload.get("questions")
    if not isinstance(raw_questions, list) or not raw_questions:
        raise ValueError("골드셋에 questions 배열이 없습니다: {}".format(source))

    questions, seen = [], set()
    for index, item in enumerate(raw_questions, 1):
        if not isinstance(item, dict):
            raise ValueError("{}번째 문항을 읽을 수 없습니다.".format(index))

        qid = item.get("id")
        if not isinstance(qid, str) or not qid.strip():
            raise ValueError("{}번째 문항에 id가 없습니다.".format(index))
        qid = qid.strip()
        if qid in seen:
            raise ValueError("골드셋에 중복된 id가 있습니다: {}".format(qid))
        seen.add(qid)

        # gold_articles: 정답으로 인정하는 조항 목록(여러 개면 그중 하나만 맞아도 정답)
        gold_keys, citations = [], []
        for art in item.get("gold_articles") or []:
            if not isinstance(art, dict):
                continue
            key = article_key(art.get("doc"), art.get("article"))
            if key is not None:
                gold_keys.append(key)
            citation = art.get("citation")
            if isinstance(citation, str) and citation.strip():
                citations.append(citation.strip())

        key_facts = [f.strip() for f in (item.get("key_facts") or [])
                     if isinstance(f, str) and f.strip()]

        questions.append({
            "id": qid,
            "question": nfkc(item.get("question") or ""),
            "gold_keys": gold_keys,
            "citations": citations,
            "key_facts": key_facts,
        })

    return GoldSet(questions, source)


def load_answers(path_or_payload):
    """팀 답변 파일을 읽는다. 깨져 있어도 예외를 던지지 않고 문제를 기록해 돌려준다.

    반환: (answers_by_qid, info)
      answers_by_qid: {qid: {"answer": str, "retrieved": [(doc, article), ...]}}
      info: {"team":..., "blind_id":..., "problems":[...], "fatal": bool}
    """
    info = {"team": None, "blind_id": None, "problems": [], "fatal": False,
            "source": "<payload>", "n_answers": 0}

    if isinstance(path_or_payload, dict):
        payload = path_or_payload
    else:
        info["source"] = str(path_or_payload)
        try:
            with open(path_or_payload, encoding="utf-8") as fp:
                payload = json.load(fp)
        except Exception as exc:                       # 파일이 없거나 JSON이 깨진 경우
            info["problems"].append("파일을 읽을 수 없습니다: {}".format(exc))
            info["fatal"] = True
            return {}, info

    if not isinstance(payload, dict):
        info["problems"].append("최상위가 객체(JSON object)가 아닙니다.")
        info["fatal"] = True
        return {}, info

    team = payload.get("team")
    info["team"] = team.strip() if isinstance(team, str) and team.strip() else None
    blind = payload.get("blind_id")
    info["blind_id"] = blind.strip() if isinstance(blind, str) and blind.strip() else None

    raw_answers = payload.get("answers")
    if not isinstance(raw_answers, list) or not raw_answers:
        info["problems"].append("answers 배열이 비어 있거나 없습니다.")
        info["fatal"] = True
        return {}, info

    answers = {}
    for index, item in enumerate(raw_answers, 1):
        if not isinstance(item, dict):
            info["problems"].append("{}번째 답변 항목을 읽을 수 없습니다.".format(index))
            continue

        qid = item.get("qid")
        if not isinstance(qid, str) or not qid.strip():
            info["problems"].append("{}번째 답변에 qid가 없습니다.".format(index))
            continue
        qid = qid.strip()
        if qid in answers:
            info["problems"].append("qid가 중복되었습니다: {} (뒤엣것 무시)".format(qid))
            continue

        answer_text = item.get("answer")
        answer_text = answer_text if isinstance(answer_text, str) else ""

        retrieved = []
        raw_retrieved = item.get("retrieved")
        if isinstance(raw_retrieved, list):
            for pair in raw_retrieved:
                if isinstance(pair, (list, tuple)) and len(pair) == 2:
                    retrieved.append((pair[0], pair[1]))
                elif isinstance(pair, dict):           # {"doc":..,"article":..} 형태도 받아 준다
                    retrieved.append((pair.get("doc"), pair.get("article")))
        else:
            info["problems"].append("{}의 retrieved가 목록이 아닙니다.".format(qid))

        answers[qid] = {"answer": answer_text, "retrieved": retrieved}

    info["n_answers"] = len(answers)
    return answers, info


# =====================================================================================
# 3. 지표 A — 검색 정확도 MRR (가중치 0.20)
# =====================================================================================
def score_mrr(retrieved, gold_keys):
    """retrieved를 앞에서부터 훑어 정답 조항이 처음 나오는 순위의 역수를 준다.

    gold_articles가 여러 개면 그중 아무거나 맞으면 정답으로 본다(P02처럼 동일 규정이
    세 약관에 중복 수록된 경우가 있다).

    반환: (mrr_contrib, rank)  — 못 맞히면 (0.0, 0)
    """
    if not gold_keys:
        return 0.0, 0
    gold = set(gold_keys)
    for rank, (doc, article) in enumerate(retrieved, 1):
        key = article_key(doc, article)
        if key is not None and key in gold:
            return 1.0 / rank, rank
    return 0.0, 0


# =====================================================================================
# 4. 지표 B — 키팩트 F1 (가중치 0.30)
# =====================================================================================
def score_keyfact_f1(answer, key_facts):
    """답변 전체를 토큰 '집합'으로 보고 key_facts 전체와 F1을 잰다.

    핵심: 문장 단위가 아니라 통짜 집합 비교다. 그래서
      · 정답에 있는데 답변에 없는 토큰 -> 재현율 손해
      · 답변에만 있는 군더더기 토큰   -> 정밀도 손해
    누락과 군더더기가 같은 무게로 깎인다.

    반환: (f1, precision, recall)
    """
    if not key_facts:
        return 0.0, 0.0, 0.0
    answer_tokens = set(tokenize(answer))
    gold_tokens = set(tokenize(" ".join(key_facts)))
    if not answer_tokens or not gold_tokens:
        return 0.0, 0.0, 0.0

    overlap = len(answer_tokens & gold_tokens)
    if overlap == 0:
        return 0.0, 0.0, 0.0
    precision = overlap / len(answer_tokens)
    recall = overlap / len(gold_tokens)
    return 2 * precision * recall / (precision + recall), precision, recall


# key_fact 하나가 답변에 반영됐다고 볼 임계값(그 팩트의 토큰 중 몇 %가 답변에 있는가).
# 공식 심판의 completeness 축 30건에 맞춰 정한 값이다(§보정 참고).
KEYFACT_COVER_THRESHOLD = 0.40


def score_keyfact_recall(answer, key_facts):
    """key_fact를 하나씩 보고 '반영됐는가'를 0/1로 세어 비율을 낸다.

    F1이 못 보는 것을 본다. F1은 통짜 비교라 어느 팩트가 통째로 빠졌는지 구분하지
    못하는데, 이 지표는 팩트 단위 누락을 잡아 completeness 축의 근거가 된다.

    반환: (recall, [문항별 반영여부], [팩트별 커버율])
    """
    if not key_facts:
        return 0.0, [], []
    answer_tokens = set(tokenize(answer))
    covered, ratios = [], []
    for fact in key_facts:
        fact_tokens = set(tokenize(fact))
        if not fact_tokens:
            covered.append(False)
            ratios.append(0.0)
            continue
        ratio = len(fact_tokens & answer_tokens) / len(fact_tokens)
        ratios.append(ratio)
        covered.append(ratio >= KEYFACT_COVER_THRESHOLD)
    return sum(covered) / len(covered), covered, ratios


# =====================================================================================
# 5. 지표 C — 판정 4축 (가중치 0.50)
# =====================================================================================
# 공식 채점은 이 축을 LLM 심판이 매긴다. 교차평가에서 팀마다 다른 심판 모델을 쓰면
# 같은 답변에 다른 점수가 나와 공정성이 깨지므로, 같은 입력이면 항상 같은 점수가
# 나오는 규칙 기반 대리 심판을 기본으로 둔다. LLM 심판은 judge_fn 인자로 갈아 끼운다.
#
# 아래 상수는 공식 심판이 매긴 축 점수 30건(3개 팀 × 공개 10문항)에 맞춰 정했다.
# 그 30건에서 관찰된 사실이 설계를 결정했다:
#   · 감점은 completeness에 쏠려 있다(16회 중 10회). 팀을 가르는 축은 사실상 여기다.
#   · accuracy·grounding·clarity는 거의 만점이다. 이 축에서 어림짐작으로 깎으면
#     실제로 상관이 음수가 됐다. 그래서 '확실한 증거가 있을 때만' 깎는다.
_COMPLETENESS_EXPONENT = 0.40         # 반영 비율 -> 점수 곡선의 지수
                                      # 임계값은 KEYFACT_COVER_THRESHOLD 하나만 쓴다

# 조번호·기간·인원처럼 틀리면 곧바로 오답이 되는 수치 단위
_NUMERIC_UNIT_RE = re.compile(r"(\d+)\s*(시간|개월|년|일|분|인|명|세|조|항|가지|번|주|회)")

# 함정 문항에서 결론이 뒤집혔는지 보는 명시적 예/아니오 표현
_EXPLICIT_NO = ("아니오", "아니요", "아닙니다", "그렇지 않습니다")
_EXPLICIT_YES = ("예,", "예.", "네,", "네.", "그렇습니다", "맞습니다")


def _numeric_pairs(text):
    """(숫자, 단위) 쌍 집합. '2시간' -> ('2','시간'), '제16조' -> ('16','조')"""
    return set(_NUMERIC_UNIT_RE.findall(nfkc(text)))


def _has_any(text, patterns):
    flat = nfkc(text)
    return any(p in flat for p in patterns)


def judge_rule_based(question, answer, key_facts, citations, retrieved, gold_keys):
    """규칙 기반 대리 심판. 4축을 각각 0~10으로 매긴다.

    반환: ({축: 점수}, {축: 사유})
    """
    axes, reasons = {}, {}
    answer_tokens = set(tokenize(answer))
    gold_tokens = set(tokenize(" ".join(key_facts)))
    has_answer = bool(answer.strip())

    # --- accuracy: 사실관계가 어긋나지 않는가 ---------------------------------------
    # 누락은 completeness가 본다. 여기서는 '틀리게 말한 것'만 본다.
    if not has_answer:
        axes["accuracy"], reasons["accuracy"] = 0.0, "답변이 비어 있음"
    else:
        score, notes = 10.0, []

        # (1) 수치 모순 — 정답과 같은 단위인데 숫자가 다르다("6개월"을 "3개월"이라 함)
        #
        # 단, 그 단위의 '맞는 값'이 답변에 함께 있으면 모순으로 보지 않는다.
        # 답변 머리에 근거를 밝히는 "(제8조) …" 같은 표기가 흔한데, 정답 키팩트에
        # '제16조'가 들어 있으면 이 8조가 틀린 조번호로 잡히기 때문이다.
        # 근거 조항이 맞는지는 grounding 축이 따로 본다.
        gold_pairs = _numeric_pairs(" ".join(key_facts))
        answer_pairs = _numeric_pairs(answer)
        contradictions = set()
        for unit in {u for _, u in gold_pairs}:
            gold_values = {n for n, u in gold_pairs if u == unit}
            answer_values = {n for n, u in answer_pairs if u == unit}
            if answer_values and not (answer_values & gold_values):
                contradictions.update("{}{}".format(n, unit) for n in sorted(answer_values))
        if contradictions:
            score -= min(6.0, 3.0 * len(contradictions))
            notes.append("정답과 다른 수치: " + ", ".join(sorted(contradictions)[:4]))

        # (2) 명시적 결론 뒤집힘 — 정답이 '아니오'인데 답변이 '예'라고 단언한 경우만
        if _has_any(" ".join(key_facts), _EXPLICIT_NO) and _has_any(answer, _EXPLICIT_YES):
            score -= 4.0
            notes.append("정답은 '아니오'인데 '예'로 답함")
        elif _has_any(" ".join(key_facts), _EXPLICIT_YES) and _has_any(answer, _EXPLICIT_NO):
            score -= 4.0
            notes.append("정답은 '예'인데 '아니오'로 답함")

        # (3) 아예 딴 얘기 — 정답 토큰이 거의 안 겹치면 사실관계를 논할 수준이 아니다
        if gold_tokens:
            overlap = len(answer_tokens & gold_tokens) / len(gold_tokens)
            if overlap < 0.10:
                score = min(score, 2.0)
                notes.append("정답 내용과 겹치는 부분이 거의 없음")

        axes["accuracy"] = max(0.0, round(score, 2))
        reasons["accuracy"] = "; ".join(notes) if notes else "정답과 어긋나는 서술 없음"

    # --- grounding: 제시한 근거가 정답 조항인가 --------------------------------------
    _, rank = score_mrr(retrieved, gold_keys)
    if not has_answer:
        axes["grounding"], reasons["grounding"] = 0.0, "답변이 비어 있음"
    elif not retrieved:
        axes["grounding"], reasons["grounding"] = 0.0, "근거 조항을 제시하지 않음"
    elif rank == 1:
        axes["grounding"], reasons["grounding"] = 10.0, "1순위 근거가 정답 조항"
    elif rank == 2:
        axes["grounding"], reasons["grounding"] = 8.0, "정답 조항이 2순위"
    elif rank == 3:
        axes["grounding"], reasons["grounding"] = 7.0, "정답 조항이 3순위"
    elif rank >= 4:
        axes["grounding"], reasons["grounding"] = 6.0, "정답 조항이 {}순위".format(rank)
    else:
        axes["grounding"], reasons["grounding"] = 2.0, "제시한 근거 중 정답 조항이 없음"

    # --- completeness: 키팩트를 빠짐없이 담았는가 -----------------------------------
    # 팀을 가르는 축. 팩트 단위로 반영 여부를 세고, 비율을 완만한 곡선으로 점수화한다.
    if not key_facts:
        axes["completeness"], reasons["completeness"] = 10.0, "골드셋에 키팩트가 없어 만점 처리"
    else:
        _, covered, _ = score_keyfact_recall(answer, key_facts)
        fraction = sum(covered) / len(covered)
        axes["completeness"] = round(10.0 * (fraction ** _COMPLETENESS_EXPONENT), 2)
        missing_idx = [i + 1 for i, ok in enumerate(covered) if not ok]
        reasons["completeness"] = "키팩트 {}개 중 {}개 반영{}".format(
            len(key_facts), sum(covered),
            (", 누락 " + str(missing_idx)) if missing_idx else "")

    # --- clarity: 읽히는 답변인가 ----------------------------------------------------
    # 공식 심판은 30건 중 2건만 깎았다. 어설픈 길이 페널티는 상관을 떨어뜨렸으므로
    # 눈에 띄게 이상한 경우만 깎는다.
    if not has_answer:
        axes["clarity"], reasons["clarity"] = 0.0, "답변이 비어 있음"
    else:
        score, notes = 10.0, []
        gold_len = len(gold_tokens) or 1
        ratio = len(tokenize(answer)) / gold_len
        if ratio > 4.0:                      # 정답의 4배가 넘는 장광설
            score -= min(2.0, (ratio - 4.0) * 0.5)
            notes.append("정답 대비 길이 {:.1f}배".format(ratio))
        counts = Counter(tokenize(answer))
        repeated = sum(c - 1 for t, c in counts.items() if c > 3 and len(t) > 1)
        if repeated > 5:                     # 같은 말 반복
            score -= min(1.5, (repeated - 5) * 0.2)
            notes.append("중복 어절 {}회".format(repeated))
        axes["clarity"] = max(0.0, round(score, 2))
        reasons["clarity"] = "; ".join(notes) if notes else "군더더기 없음"

    return axes, reasons


def judge_total_0_100(axes):
    """4축 점수를 공식 가중치로 합쳐 0~100으로 만든다."""
    return 10.0 * sum(JUDGE_AXIS_WEIGHTS[k] * float(axes.get(k, 0.0)) for k in JUDGE_AXIS_WEIGHTS)


# =====================================================================================
# 5b. 판정 4축의 실제 채점 — Gemini 3.5 Flash
# =====================================================================================
# 운영진 지침: "Gemini 3.5 Flash로 공개 10문항과 비공개 30문항을 같은 코드로 처리하고,
# 익명 5팀 모두에 같은 기준을 적용해야 합니다." judge_rule_based는 API 키 없이 돌아가는
# 자체 검증·폴백용이고, 실제 제출 채점은 여기서 만드는 judge_fn을 써야 한다.
#
# SDK는 google-genai(`pip install -U google-genai`)를 쓴다. generateContent는 문서상
# "레거시"로 표시돼 있지만 여전히 전면 지원되고 구조화 출력(response_schema)이 안정적으로
# 문서화돼 있어 이 경로를 택했다 — 신설 Interactions API는 아직 사례가 적어 제외.

_JUDGE_MODEL_DEFAULT = "gemini-3.5-flash"

# score에 minimum/maximum을 넣지 않는다 — Gemini의 스키마 서브셋이 매 버전 그 필드를
# 지원한다는 보장이 없다. 범위는 프롬프트 지시 + _clamp_axis_score() 이중으로 강제한다.
_JUDGE_RESPONSE_SCHEMA = {
    "type": "object",
    "properties": {
        axis: {
            "type": "object",
            "properties": {
                "score": {"type": "integer"},
                "reason": {"type": "string"},
            },
            "required": ["score", "reason"],
        }
        for axis in JUDGE_AXIS_WEIGHTS
    },
    "required": list(JUDGE_AXIS_WEIGHTS),
}

_JUDGE_SYSTEM_PROMPT = """당신은 카카오 서비스 약관에 대한 RAG 답변을 채점하는 심사위원입니다.
아래 네 기준으로 참가자 답변을 각 0~10 정수로 채점하세요.

- accuracy(정확성): 답변에 사실과 다르거나 정답과 모순되는 내용이 있는가
- grounding(근거성): 참가자가 제시한 근거 조항이 실제 정답 조항과 일치하는가
- completeness(완결성): 정답 핵심 내용을 빠짐없이 담았는가
- clarity(명료성): 군더더기 없이 명확하고 읽기 쉬운가

각 기준마다 0~10 정수 점수와 한국어로 된 한두 문장의 근거를 반환하세요.
정답에 없는 내용을 답변이 지어냈다면 accuracy를 낮추고, 정답 조항과 다른 근거를
제시했다면 grounding을 낮추세요. 채점 기준은 모든 답변에 동일하게 적용하세요."""


def _format_retrieved_for_prompt(retrieved):
    if not retrieved:
        return "(제시한 근거 없음)"
    return "; ".join("{} {}".format(doc, article) for doc, article in retrieved)


def build_judge_prompt(question, answer, key_facts, citations, retrieved):
    """Gemini에게 보낼 채점 프롬프트 본문(시스템 지시 제외)을 만든다."""
    gold_citation_text = "; ".join(citations) if citations else "(표기 없음)"
    key_fact_text = "\n".join("- {}".format(f) for f in key_facts) if key_facts else "(없음)"
    return (
        "[질문]\n{question}\n\n"
        "[정답 근거 조항]\n{citation}\n\n"
        "[정답 핵심 내용]\n{facts}\n\n"
        "[참가자가 제시한 근거]\n{retrieved}\n\n"
        "[참가자 답변]\n{answer}\n"
    ).format(
        question=question or "(질문 없음)",
        citation=gold_citation_text,
        facts=key_fact_text,
        retrieved=_format_retrieved_for_prompt(retrieved),
        answer=answer.strip() if answer and answer.strip() else "(답변 없음)",
    )


def _clamp_axis_score(value):
    try:
        return max(0.0, min(10.0, float(value)))
    except (TypeError, ValueError):
        return 0.0


def _parse_judge_response(raw_text):
    """Gemini 응답 텍스트를 (axes, reasons)로 파싱한다. 형식이 안 맞으면 None."""
    text = (raw_text or "").strip()
    if text.startswith("```"):                              # JSON 모드에서도 가끔 코드펜스가 붙는다
        text = re.sub(r"^```[a-zA-Z]*\n?", "", text)
        text = re.sub(r"```$", "", text).strip()
    try:
        data = json.loads(text)
    except (json.JSONDecodeError, TypeError):
        return None
    if not isinstance(data, dict):
        return None

    axes, reasons = {}, {}
    for axis in JUDGE_AXIS_WEIGHTS:
        entry = data.get(axis)
        if not isinstance(entry, dict) or "score" not in entry:
            return None
        axes[axis] = _clamp_axis_score(entry.get("score"))
        reason = entry.get("reason")
        reasons[axis] = reason.strip() if isinstance(reason, str) and reason.strip() else "(사유 없음)"
    return axes, reasons


def make_gemini_judge(api_key, model=_JUDGE_MODEL_DEFAULT, max_retries=3,
                       retry_wait_s=8.0, request_interval_s=4.5, stats=None):
    """Gemini 3.5 Flash로 채점하는 judge_fn을 만든다.

    `evaluate_team(..., judge_fn=make_gemini_judge(키))` 형태로 꽂아 쓴다.

    한 문항에서 네트워크 오류·429·JSON 파싱 실패가 max_retries번 반복되면 그 문항만
    judge_rule_based로 대체하고 사유에 남긴다. 한 문항의 일시적 실패로 나머지 수십
    문항, 다른 팀의 채점까지 중단되는 것을 막기 위함이다. `stats`(dict)를 넘기면
    {"calls", "gemini_ok", "fallback"} 진행 상황이 실시간으로 쌓인다 — 채점이 끝난
    뒤 fallback이 크면 API 키·쿼터를 점검하고 다시 돌려야 한다는 신호다.
    """
    from google import genai
    from google.genai import errors as genai_errors

    client = genai.Client(api_key=api_key)
    if stats is None:
        stats = {}
    stats.setdefault("calls", 0)
    stats.setdefault("gemini_ok", 0)
    stats.setdefault("fallback", 0)

    def judge_fn(question, answer, key_facts, citations, retrieved, gold_keys):
        stats["calls"] += 1
        prompt = build_judge_prompt(question, answer, key_facts, citations, retrieved)
        last_error = "알 수 없는 오류"

        for attempt in range(max_retries):
            if attempt > 0:
                time.sleep(retry_wait_s * attempt)
            try:
                response = client.models.generate_content(
                    model=model,
                    contents=prompt,
                    config={
                        "system_instruction": _JUDGE_SYSTEM_PROMPT,
                        "response_mime_type": "application/json",
                        "response_schema": _JUDGE_RESPONSE_SCHEMA,
                        "temperature": 0,
                    },
                )
                parsed = _parse_judge_response(getattr(response, "text", None))
                if parsed is not None:
                    stats["gemini_ok"] += 1
                    time.sleep(request_interval_s)          # 무료 등급 분당 호출 수 방어
                    return parsed
                last_error = "응답을 정해진 JSON 형식으로 해석할 수 없음"
            except genai_errors.APIError as exc:
                last_error = "{} (code={})".format(exc, getattr(exc, "code", "?"))
            except Exception as exc:
                last_error = str(exc)

        stats["fallback"] += 1
        axes, reasons = judge_rule_based(question, answer, key_facts, citations, retrieved, gold_keys)
        reasons = {axis: "[Gemini {}회 실패, 규칙기반 대체: {}] {}".format(max_retries, last_error, text)
                   for axis, text in reasons.items()}
        return axes, reasons

    return judge_fn


def _resolve_qid_mapping(goldset, answers):
    """답변 qid가 골드셋 id와 하나도 안 겹치는데 개수는 같으면, 제출 순서를 골드셋
    순서에 그대로 대응시킨다(위치 기반 폴백).

    실제로 있었던 상황: 팀 결과기가 낸 정상 qid(P01..)를, 운영진이 답변 파일을
    익명화·재배포하는 과정에서 BLIND01..처럼 통째로 다시 붙이는 경우. 이때 문자열
    매칭만 하면 내용은 멀쩡한데 전부 "미응답 0점"으로 잘못 채점된다.

    조금이라도 겹치면(부분 일치) 폴백하지 않는다 — 애매하게 일부만 다른 경우는
    정말 일부 문항이 빠진 것일 수 있고, 그럴 때 위치로 임의 매칭하면 엉뚱한 답을
    엉뚱한 문항에 채점하는 더 나쁜 실패로 이어진다. 완전히 안 겹치고 개수까지
    맞을 때만 순서로 넘어간다. answers의 순서는 load_answers가 만든 딕셔너리라
    파이썬 3.7+ 삽입 순서 보장으로 원본 답변 파일 순서와 같다.

    반환: (실제로 조회에 쓸 answers 딕셔너리, 폴백을 썼다면 사유 문자열 또는 None)
    """
    gold_ids = [q["id"] for q in goldset.questions]
    answer_qids = list(answers.keys())
    if not answer_qids or (set(gold_ids) & set(answer_qids)):
        return answers, None
    if len(answer_qids) != len(gold_ids):
        return answers, None

    remapped = {gold_ids[i]: answers[answer_qids[i]] for i in range(len(gold_ids))}
    note = ("qid가 골드셋 id와 전혀 겹치지 않아(예: {!r} vs {!r}) 제출 순서를 골드셋 "
            "순서에 그대로 대응시켰습니다({}개 전부).").format(answer_qids[0], gold_ids[0], len(gold_ids))
    return remapped, note


# =====================================================================================
# 6. 한 팀 채점 — 문항별 점수 → 총점
# =====================================================================================
def evaluate_team(goldset, answers, judge_fn=judge_rule_based):
    """골드셋 한 벌과 팀 답변 하나를 맞대어 채점한다.

    골드셋의 문항을 기준으로 돈다. 답변에 없는 문항은 0점 처리하고 사유를 남긴다.
    문항 수·id는 전부 goldset에서 온다(코드에 고정된 값 없음).
    """
    answers, qid_note = _resolve_qid_mapping(goldset, answers)

    per_question = []
    for question in goldset.questions:
        qid = question["id"]
        entry = answers.get(qid)

        if entry is None:
            per_question.append({
                "qid": qid, "answered": False,
                "mrr": 0.0, "rank": 0,
                "keyfact_f1": 0.0, "keyfact_precision": 0.0, "keyfact_recall": 0.0,
                "judge_axes": {k: 0.0 for k in JUDGE_AXIS_WEIGHTS},
                "judge_total": 0.0,
                "note": "답변 파일에 이 문항이 없음",
            })
            continue

        answer, retrieved = entry["answer"], entry["retrieved"]
        mrr, rank = score_mrr(retrieved, question["gold_keys"])
        f1, precision, recall_tok = score_keyfact_f1(answer, question["key_facts"])
        axes, reasons = judge_fn(
            question=question["question"], answer=answer,
            key_facts=question["key_facts"], citations=question["citations"],
            retrieved=retrieved, gold_keys=question["gold_keys"],
        )
        per_question.append({
            "qid": qid, "answered": True,
            "mrr": round(mrr, 6), "rank": rank,
            "keyfact_f1": round(f1, 6),
            "keyfact_precision": round(precision, 6),
            "keyfact_recall": round(recall_tok, 6),
            "judge_axes": axes, "judge_reasons": reasons,
            "judge_total": round(judge_total_0_100(axes), 4),
            "note": "",
        })

    n = len(per_question) or 1
    mrr_mean = sum(p["mrr"] for p in per_question) / n
    f1_mean = sum(p["keyfact_f1"] for p in per_question) / n
    judge_mean = sum(p["judge_total"] for p in per_question) / n

    total = (WEIGHTS["mrr"] * mrr_mean * 100
             + WEIGHTS["keyfact"] * f1_mean * 100
             + WEIGHTS["judge"] * judge_mean)

    return {
        "objective": {"mrr": round(mrr_mean, 6), "keyfact_f1": round(f1_mean, 6)},
        "judge": {"score_0_100": round(judge_mean, 4)},
        "total_0_100": round(total, 4),
        "n_questions": len(per_question),
        "n_answered": sum(1 for p in per_question if p["answered"]),
        "per_question": per_question,
        "weights": dict(WEIGHTS),
        "qid_mapping_note": qid_note,
    }


# =====================================================================================
# 7. 교차평가 — 여러 팀 파일 → eval_<우리팀>.json
# =====================================================================================
# 운영진 검사기(교차 평가 결과 파일 검사기)의 규칙: results에는 BLIND01~BLIND05가
# 각각 정확히 한 번씩 있어야 하고, 다섯 결과의 status가 모두 completed여야 제출할 수
# 있다. "일부 문항 실패·평가 전체 실패·후보 중복·누락이 하나라도 있으면 그 평가 결과
# 파일 전체가 순위 산정에서 제외됩니다. 실패한 후보에게 0점을 주는 방식은 사용하지
# 않습니다." — 즉 후보 파일이 아무리 깨져 있어도 failed/partial로 도망치면 안 되고,
# 항상 실제 점수(깨졌으면 낮은 점수)를 매겨 completed로 마감해야 한다. 그래서
# evaluate_submissions는 파일이 아예 안 읽혀도 빈 답변({})으로 채점을 강행한다 —
# load_answers/evaluate_team이 이미 빈 입력을 0점 만점 정상 채점으로 처리하도록
# 설계돼 있어서 가능하다(§2, §6).
BLIND_IDS = tuple("BLIND{:02d}".format(i) for i in range(1, 6))

_BLIND_RE = re.compile(r"BLIND\s*[-_]?\s*(\d+)", re.IGNORECASE)


def resolve_blind_id(info, path, fallback_index):
    """블라인드 식별자를 정한다. 파일 안 blind_id > 파일명(basename) > team > 순번.

    숫자가 발견되면 항상 BLIND01처럼 2자리로 맞춘다 — 파일명이 BLIND1.json이든
    blind_01.json이든 운영진 검사기가 요구하는 BLIND01 형식으로 귀결시키기 위함이다.

    파일명(basename)만 본다 — 전체 경로를 보면 상위 폴더 이름에 숫자가 섞였을 때
    엉뚱한 값으로 오매칭될 수 있다(예: "blind5/answers_BLIND01.json" 같은 폴더
    구조에서 폴더명의 5를 먼저 집어 모든 파일이 BLIND05로 뭉개지는 사고가 실제로
    있었다).
    """
    basename = str(path).replace("\\", "/").rsplit("/", 1)[-1]
    for candidate in (info.get("blind_id"), basename):
        if not candidate:
            continue
        match = _BLIND_RE.search(str(candidate))
        if match:
            return "BLIND{:02d}".format(int(match.group(1)))
    stem = str(path).replace("\\", "/").rsplit("/", 1)[-1]
    stem = re.sub(r"\.json$", "", stem, flags=re.IGNORECASE)
    stem = re.sub(r"^answers[_-]?(public|blind|private)?[_-]?", "", stem, flags=re.IGNORECASE)
    if stem.strip():
        return stem.strip()
    if info.get("team"):
        return info["team"]
    return "ENTRY{:02d}".format(fallback_index)


def evaluate_submissions(goldset, submissions, judge_fn=judge_rule_based):
    """제출물 여러 개를 채점한다.

    submissions: [경로, ...] 또는 [(blind_id, 경로_또는_payload), ...]
    반환: {"results": [{blind_id, total, status}, ...], "details": {...}}

    status는 항상 "completed"다. 후보 파일을 전혀 읽을 수 없어도 빈 답변으로
    채점을 강행해 실제 점수를 매긴다 — "실패한 후보에게 0점을 주는 방식은
    사용하지 않는다"는 운영진 지침대로, failed/partial로 파일 전체를 무효화시킬
    위험을 피하기 위함이다. 문제가 있었다면 그 사유는 결과 파일이 아니라
    details[blind_id]["problems"]에만 남는다(진단용, 제출 파일에는 안 들어감).
    """
    results, details = [], {}

    for index, item in enumerate(submissions, 1):
        if isinstance(item, (tuple, list)) and len(item) == 2:
            given_id, source = item
        else:
            given_id, source = None, item

        answers, info = load_answers(source)
        blind_id = given_id or resolve_blind_id(info, source, index)

        try:
            report = evaluate_team(goldset, answers, judge_fn=judge_fn)
        except Exception as exc:
            info["problems"] = info["problems"] + ["채점 중 오류(0점 처리): {}".format(exc)]
            report = evaluate_team(goldset, {}, judge_fn=judge_fn)

        problems = info["problems"]
        if report.get("qid_mapping_note"):
            problems = problems + [report["qid_mapping_note"]]

        results.append({
            "blind_id": blind_id,
            "total": round(report["total_0_100"], 4),
            "status": "completed",
        })
        report["problems"] = problems
        report["source"] = info["source"]
        details[blind_id] = report

    results.sort(key=lambda r: str(r["blind_id"]))
    return {"results": results, "details": details}


def check_blind_id_coverage(results):
    """제출 직전 확인: BLIND01~BLIND05가 정확히 한 번씩만 있는가.

    운영진 검사기가 "후보 중복"·"누락"으로 잡는 것과 같은 조건이다. 문제 없으면 빈
    리스트, 있으면 사람이 읽을 문구 목록을 돌려준다.
    """
    issues = []
    seen_order = [str(r.get("blind_id")) for r in results]
    counts = Counter(seen_order)
    duplicates = sorted(bid for bid, n in counts.items() if n > 1)
    if duplicates:
        issues.append("blind_id가 중복되었습니다: {}".format(", ".join(duplicates)))
    unexpected = sorted(set(seen_order) - set(BLIND_IDS))
    if unexpected:
        issues.append("BLIND01~BLIND05가 아닌 값이 있습니다: {}".format(", ".join(unexpected)))
    missing = sorted(set(BLIND_IDS) - set(seen_order))
    if missing:
        issues.append("누락된 익명 후보가 있습니다: {}".format(", ".join(missing)))
    return issues


def write_eval_file(payload, team, out_dir="."):
    """eval_<팀>.json 을 규정 형식(results[{blind_id,total,status}])으로 저장한다."""
    path = "{}/eval_{}.json".format(out_dir.rstrip("/"), team)
    with open(path, "w", encoding="utf-8") as fp:
        json.dump({"results": payload["results"]}, fp, ensure_ascii=False, indent=2)
        fp.write("\n")
    return path


# =====================================================================================
# 7b. 제출 파일 형식 검사 — 운영진 "교차 평가 결과 파일 검사기"와 동일 기준
# =====================================================================================
# 운영진이 배포한 검사기 코드를 그대로 옮겼다(문항 검사기와 같은 방식). 직접 다시
# 구현하지 않고 원본을 옮긴 이유는, 재구현 과정에서 조건 하나라도 다르게 해석하면
# "내 검사는 통과했는데 실제 제출은 반려"되는 상황이 생기기 때문이다. 새 검사기가
# 배포되면 이 구획을 그걸로 통째로 교체하면 된다.

_EVAL_ROOT_KEYS = {"results"}
_EVAL_RESULT_KEYS = {"blind_id", "total", "status"}
_EVAL_STATUSES = {"completed", "partial", "failed"}
EVAL_FILENAME_RE = re.compile(r"^eval_([1-9]|1[0-7])\.json$")


def _eval_number(value):
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return None
    number = float(value)
    return number if math.isfinite(number) else None


def validate_eval_document(document):
    """운영진 취합기의 최소 계약과 같은 기준으로 검사하고 문제 문구 목록을 반환한다."""
    issues = []
    if not isinstance(document, dict):
        return ["최상위 값은 JSON 객체여야 합니다."]

    if "schema_version" in document:
        issues.append("schema_version은 학생 제출 파일에 저장하지 않습니다.")
    if "rank" in document:
        issues.append("rank는 저장하지 않습니다. 운영진이 반올림 전 total에서 계산합니다.")
    extra_root_keys = sorted(set(document) - _EVAL_ROOT_KEYS - {"schema_version", "rank"})
    if extra_root_keys:
        issues.append("최상위에는 results만 저장합니다. 허용되지 않은 항목: " + ", ".join(extra_root_keys))

    raw_results = document.get("results")
    if not isinstance(raw_results, list):
        issues.append("results는 배열이어야 합니다.")
        return issues

    seen = set()
    for index, item in enumerate(raw_results):
        at = "results[{}]".format(index)
        if not isinstance(item, dict):
            issues.append("{}는 JSON 객체여야 합니다.".format(at))
            continue
        if "rank" in item:
            issues.append("{}.rank는 저장하지 않습니다.".format(at))
        extra_result_keys = sorted(set(item) - _EVAL_RESULT_KEYS - {"rank"})
        if extra_result_keys:
            issues.append("{}에는 blind_id, total, status만 저장합니다. 허용되지 않은 항목: {}"
                          .format(at, ", ".join(extra_result_keys)))

        blind_id = str(item.get("blind_id") or "").strip()
        if blind_id not in BLIND_IDS:
            issues.append("{}.blind_id는 BLIND01~BLIND05 중 하나여야 합니다: {!r}".format(at, blind_id))
            continue
        if blind_id in seen:
            issues.append("{}가 두 번 이상 들어 있습니다.".format(blind_id))
            continue
        seen.add(blind_id)

        status = str(item.get("status") or "").strip()
        if status not in _EVAL_STATUSES:
            issues.append("{}.status는 completed·partial·failed 중 하나여야 합니다: {!r}".format(blind_id, status))
            continue

        raw_total = item.get("total")
        total = _eval_number(raw_total)
        if status == "failed":
            if raw_total is not None:
                issues.append("{}가 failed이면 total은 null이어야 합니다.".format(blind_id))
        elif total is None or not 0 <= total <= 100:
            issues.append("{}.total은 0~100 숫자여야 합니다.".format(blind_id))

        if status != "completed":
            issues.append("{} 상태가 {}라 평가 결과 파일 전체가 순위 산정에서 제외됩니다.".format(blind_id, status))

    missing = sorted(set(BLIND_IDS) - seen)
    if missing:
        issues.append("익명 후보가 누락됐습니다: {}".format(", ".join(missing)))
    if len(raw_results) != len(BLIND_IDS):
        issues.append("results는 정확히 5개여야 합니다: 현재 {}개".format(len(raw_results)))
    return issues


def check_eval_file(path):
    """eval_<팀>.json 파일을 읽어 (제출 가능 여부, 문제 문구 목록)을 반환한다."""
    name = str(path).replace("\\", "/").rsplit("/", 1)[-1]
    if not EVAL_FILENAME_RE.fullmatch(name):
        return False, ["파일명은 eval_1.json부터 eval_17.json까지의 형식이어야 합니다. 현재 파일명: {}".format(name)]
    try:
        with open(path, encoding="utf-8") as fp:
            document = json.load(fp)
    except (OSError, json.JSONDecodeError) as exc:
        return False, ["JSON 파일을 읽지 못했습니다: {}".format(exc)]
    issues = validate_eval_document(document)
    return not issues, issues
