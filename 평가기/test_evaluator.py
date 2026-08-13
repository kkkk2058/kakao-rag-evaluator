# -*- coding: utf-8 -*-
"""평가기 자체 검증. 남의 팀 파일이 어떤 식으로 깨져 있어도 죽지 않는지 확인한다.

실행: python3 test_evaluator.py
"""

import json
import os
import sys
import tempfile
import types

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import evaluator_core as ec  # noqa: E402
from evaluator_core import (  # noqa: E402
    BLIND_IDS, JUDGE_AXIS_WEIGHTS, evaluate_submissions, evaluate_team, load_answers,
    load_goldset, parse_article_no, normalize_doc_name, score_keyfact_f1,
    score_mrr, tokenize, write_eval_file,
)

PASS, FAIL = [], []


def check(name, condition, detail=""):
    (PASS if condition else FAIL).append(name)
    mark = "OK  " if condition else "실패"
    print("  [{}] {}{}".format(mark, name, ("  <- " + detail) if detail and not condition else ""))


# =====================================================================================
print("\n[1] 정규화 유틸")
# =====================================================================================
check("조번호: 정수", parse_article_no(10) == 10)
check("조번호: 문자열 '10'", parse_article_no("10") == 10)
check("조번호: '제10조'", parse_article_no("제10조") == 10)
check("조번호: '제 10 조'", parse_article_no("제 10 조") == 10)
check("조번호: 숫자 없음 -> None", parse_article_no("십조") is None)
check("조번호: True는 조번호가 아님", parse_article_no(True) is None)
check("문서명: 공백 무시", normalize_doc_name("카카오 통합 약관") == normalize_doc_name("카카오통합약관"))
check("문서명: 통합서비스약관과 통합약관은 구분",
      normalize_doc_name("카카오 통합서비스약관") != normalize_doc_name("카카오 통합 약관"))
check("토큰화: 가운뎃점 분리", tokenize("수집·이용·제공") == ["수집", "이용", "제공"])
check("토큰화: 조항 토큰 유지", "제16조" in tokenize("법률 제16조 제2항에"))

# =====================================================================================
print("\n[2] 지표 계산")
# =====================================================================================
gold_keys = [("카카오계정약관", 10), ("카카오통합약관", 13)]
check("MRR: 1순위 적중", score_mrr([("카카오계정 약관", 10)], gold_keys) == (1.0, 1))
check("MRR: 2순위 적중", score_mrr([("카카오 통합 약관", 99), ("카카오계정 약관", 10)], gold_keys)[0] == 0.5)
check("MRR: 복수 정답 중 하나", score_mrr([("카카오 통합 약관", 13)], gold_keys) == (1.0, 1))
check("MRR: 못 맞힘", score_mrr([("카카오계정 약관", 99)], gold_keys) == (0.0, 0))
check("MRR: 빈 목록", score_mrr([], gold_keys) == (0.0, 0))
check("MRR: '제10조' 표기도 인정", score_mrr([("카카오계정 약관", "제10조")], gold_keys) == (1.0, 1))

facts = ["담당자 1인만 이용할 수 있습니다.", "공유하는 것은 금지됩니다."]
f1_same, _, _ = score_keyfact_f1(" ".join(facts), facts)
check("F1: 정답 그대로면 1.0", abs(f1_same - 1.0) < 1e-9)
check("F1: 무관한 답변은 0", score_keyfact_f1("오늘 날씨가 좋습니다", facts)[0] == 0.0)
check("F1: 빈 답변은 0", score_keyfact_f1("", facts)[0] == 0.0)
f1_verbose, p_v, r_v = score_keyfact_f1(" ".join(facts) + " " * 1 + "관련 없는 말 " * 30, facts)
check("F1: 군더더기를 붙이면 떨어진다", f1_verbose < f1_same)

# =====================================================================================
print("\n[3] 골드셋 로더 — 문항 id·개수를 코드에 고정하지 않는다")
# =====================================================================================
tmp = tempfile.mkdtemp()

# 공개셋과 완전히 다른 id 체계 + 30문항 + 추가 필드 포함(비공개셋 모사)
blind_gold = {
    "_meta": {"name": "가짜 비공개셋"},
    "questions": [
        {
            "id": "B{:03d}".format(i),
            "question": "질문 {}".format(i),
            "gold_articles": [{"doc": "카카오계정 약관", "article": i, "citation": "제{}조".format(i)}],
            "key_facts": ["문항 {}의 핵심 사실 하나입니다.".format(i),
                          "문항 {}의 핵심 사실 둘입니다.".format(i)],
            # 비공개셋에만 있을 수 있는 추가 필드 — 평가기가 의존하면 안 된다
            "ptype": "basic", "difficulty": "hard", "tag": "x",
            "출제노트": "운영진 내부 메모", "새로운_필드": {"중첩": [1, 2, 3]},
        }
        for i in range(1, 31)
    ],
}
blind_path = os.path.join(tmp, "gold_blind30.json")
json.dump(blind_gold, open(blind_path, "w", encoding="utf-8"), ensure_ascii=False)

gs = load_goldset(blind_path)
check("골드셋: 30문항을 그대로 읽음", len(gs) == 30, "읽은 문항 수 {}".format(len(gs)))
check("골드셋: id 체계가 달라도 됨(B001..)", gs.qids[0] == "B001" and gs.qids[-1] == "B030")
check("골드셋: 추가 필드는 무시", set(gs.questions[0]) == {"id", "question", "gold_keys", "citations", "key_facts"})

# 필수 필드만 있는 최소 골드셋
minimal = {"questions": [{"id": "X1", "question": "q",
                          "gold_articles": [{"doc": "카카오계정 약관", "article": 1}],
                          "key_facts": ["사실"]}]}
check("골드셋: citation이 없어도 동작", len(load_goldset(minimal)) == 1)

try:
    load_goldset({"questions": []})
    check("골드셋: 빈 questions는 거부", False)
except ValueError:
    check("골드셋: 빈 questions는 거부", True)

try:
    load_goldset({"questions": [{"question": "id 없음"}]})
    check("골드셋: id 없으면 거부", False)
except ValueError:
    check("골드셋: id 없으면 거부", True)

# =====================================================================================
print("\n[4] 답변 로더 — 남의 팀 파일이 깨져 있어도 죽지 않는다")
# =====================================================================================
def write(name, payload, raw=None):
    path = os.path.join(tmp, name)
    with open(path, "w", encoding="utf-8") as fp:
        if raw is not None:
            fp.write(raw)
        else:
            json.dump(payload, fp, ensure_ascii=False)
    return path


perfect = {"team": "1", "answers": [
    {"qid": q["id"],
     "retrieved": [["카카오계정 약관", int(q["id"][1:])]],
     "answer": " ".join(q["key_facts"])}
    for q in blind_gold["questions"]]}
p_perfect = write("answers_blind_BLIND01.json", perfect)

broken_json = write("answers_blind_BLIND02.json", None, raw="{이건 JSON이 아님")
no_answers = write("answers_blind_BLIND03.json", {"team": "3"})
half = {"team": "4", "answers": perfect["answers"][:15]}          # 절반만 응답
p_half = write("answers_blind_BLIND04.json", half)
weird = {"team": "5", "answers": [
    {"qid": "B001", "retrieved": [{"doc": "카카오계정 약관", "article": "제1조"}], "answer": "정상"},
    {"qid": "B002", "retrieved": "목록아님", "answer": "그래도 채점됨"},
    {"qid": "B003", "retrieved": [["카카오계정 약관"]], "answer": "쌍이 깨짐"},
    {"qid": "B003", "retrieved": [], "answer": "중복 qid"},
    {"qid": "", "answer": "qid 없음"},
    {"qid": "B004", "answer": None},
    "항목이 문자열",
    {"qid": "ZZZ", "retrieved": [], "answer": "골드셋에 없는 문항"},
]}
p_weird = write("answers_blind_BLIND05.json", weird)

_, info = load_answers(broken_json)
check("답변: 깨진 JSON -> fatal", info["fatal"] and info["problems"])
_, info = load_answers(no_answers)
check("답변: answers 없음 -> fatal", info["fatal"])
ans_w, info_w = load_answers(p_weird)
check("답변: 이상한 항목이 섞여도 예외 없음", len(ans_w) >= 4, str(info_w["problems"]))
check("답변: 중복 qid는 앞엣것 유지", ans_w["B003"]["answer"] == "쌍이 깨짐")
check("답변: dict 형태 retrieved도 인정",
      score_mrr(ans_w["B001"]["retrieved"], [("카카오계정약관", 1)]) == (1.0, 1))
check("답변: answer가 null이면 빈 문자열", ans_w["B004"]["answer"] == "")

# =====================================================================================
print("\n[5] 채점 — 만점/영점/부분")
# =====================================================================================
rep = evaluate_team(gs, load_answers(p_perfect)[0])
check("채점: 정답 그대로면 총점 100", abs(rep["total_0_100"] - 100.0) < 1e-6,
      "총점 {}".format(rep["total_0_100"]))
check("채점: 문항 수는 골드셋에서 옴", rep["n_questions"] == 30)

empty = {"team": "9", "answers": [{"qid": q["id"], "retrieved": [], "answer": ""}
                                  for q in blind_gold["questions"]]}
rep_empty = evaluate_team(gs, load_answers(write("answers_empty.json", empty))[0])
check("채점: 전부 빈 답변이면 0점", rep_empty["total_0_100"] == 0.0,
      "총점 {}".format(rep_empty["total_0_100"]))

rep_half = evaluate_team(gs, load_answers(p_half)[0])
check("채점: 절반만 답하면 총점도 절반쯤", 45 < rep_half["total_0_100"] < 55,
      "총점 {}".format(rep_half["total_0_100"]))
check("채점: 없는 문항은 0점 + 사유 기록",
      any(p["note"] and not p["answered"] for p in rep_half["per_question"]))

wrong_doc = {"team": "9", "answers": [
    {"qid": q["id"], "retrieved": [["카카오 위치정보 이용약관", 99]], "answer": " ".join(q["key_facts"])}
    for q in blind_gold["questions"]]}
rep_wd = evaluate_team(gs, load_answers(write("answers_wrongdoc.json", wrong_doc))[0])
check("채점: 근거가 틀리면 MRR 0 + grounding 감점",
      rep_wd["objective"]["mrr"] == 0.0 and rep_wd["per_question"][0]["judge_axes"]["grounding"] < 5)
check("채점: 근거만 틀려도 총점은 남는다(내용은 맞음)", 60 < rep_wd["total_0_100"] < 90,
      "총점 {}".format(rep_wd["total_0_100"]))

# --- qid가 골드셋 id와 전혀 다른 라벨을 쓰는 경우: 위치 기반 폴백 ---
# 실제 사례: 팀 결과기가 낸 P01..qid를 운영진이 답변 파일을 재배포하며
# BLIND01..으로 통째로 다시 붙였다. 문자열 매칭만 하면 내용은 멀쩡한데 전부
# "미응답 0점"으로 잘못 채점된다.
relabeled = {"team": "9", "answers": [
    {"qid": "BLIND{:02d}".format(i + 1), "retrieved": [["카카오계정 약관", int(q["id"][1:])]],
     "answer": " ".join(q["key_facts"])}
    for i, q in enumerate(blind_gold["questions"])]}
rep_relabel = evaluate_team(gs, load_answers(write("answers_relabeled.json", relabeled))[0])
check("qid 완전 불일치 + 개수 일치 -> 위치로 매칭(0점 아님)",
      abs(rep_relabel["total_0_100"] - 100.0) < 1e-6, "총점 {}".format(rep_relabel["total_0_100"]))
check("qid 폴백 시 사유가 report에 남음", rep_relabel["qid_mapping_note"] is not None)

# 부분 일치(일부만 다른 라벨)는 애매한 상황이므로 폴백하지 않는다 — 정말 일부
# 문항이 빠진 것일 수 있고, 그때 위치로 임의 매칭하면 오답을 정답 자리에 채점하는
# 더 나쁜 실패로 이어진다.
partial_relabel = {"team": "9", "answers": (
    [{"qid": blind_gold["questions"][0]["id"], "retrieved": [], "answer": "x"}]
    + [{"qid": "WEIRD{:02d}".format(i), "retrieved": [], "answer": "y"} for i in range(1, 30)])}
rep_partial = evaluate_team(gs, load_answers(write("answers_partial_relabel.json", partial_relabel))[0])
check("qid 일부만 일치하면 폴백하지 않음(애매한 경우 임의 매칭 금지)",
      rep_partial["qid_mapping_note"] is None)

# 개수가 다르면(불완전 제출) 위치로 짜맞추지 않는다 — 순서를 보장할 수 없다.
short_relabel = {"team": "9", "answers": [
    {"qid": "BLIND{:02d}".format(i + 1), "retrieved": [], "answer": "x"}
    for i in range(len(blind_gold["questions"]) - 5)]}
rep_short = evaluate_team(gs, load_answers(write("answers_short_relabel.json", short_relabel))[0])
check("qid 불일치 + 개수도 다르면 폴백하지 않음(짜맞추기 금지)",
      rep_short["qid_mapping_note"] is None)
check("개수 달라 폴백 안 하면 전부 미응답 처리", rep_short["n_answered"] == 0)

# 정상 케이스(문자열 그대로 일치)에는 노트가 안 붙어야 한다 — 폴백 흔적이 정상
# 채점 결과에 섞여 들어가면 안 된다.
check("qid가 정상 일치하면 qid_mapping_note는 None(오염 없음)", rep["qid_mapping_note"] is None)

# =====================================================================================
print("\n[6] 교차평가 산출물 eval_<팀>.json")
# =====================================================================================
# 운영진 검사기 규칙: BLIND01~BLIND05 중 하나라도 status != completed면 제출 파일
# 전체가 순위 산정에서 제외된다("실패한 후보에게 0점을 주는 방식은 사용하지 않음").
# 그래서 파일이 아무리 깨져 있어도 evaluate_submissions는 항상 completed를 낸다 —
# broken_json/no_answers처럼 아예 못 읽는 파일도 0점짜리 completed가 되어야 한다.
out = evaluate_submissions(gs, [p_perfect, broken_json, no_answers, p_half, p_weird])
results = out["results"]
check("교차: 제출물 5건 모두 결과에 나옴", len(results) == 5)
check("교차: blind_id를 파일명에서 뽑음",
      [r["blind_id"] for r in results] == ["BLIND01", "BLIND02", "BLIND03", "BLIND04", "BLIND05"],
      str([r["blind_id"] for r in results]))
check("교차: status는 파일 상태와 무관하게 전부 completed(제출 자격 유지)",
      all(r["status"] == "completed" for r in results), str(results))
check("교차: 못 읽는 파일도 completed + 0점(failed로 도망치지 않음)",
      all(r["total"] == 0.0 for r in results if r["blind_id"] in ("BLIND02", "BLIND03")))
check("교차: 문항이 빠져도 completed, 빠진 문항만 0점 반영",
      45 < next(r for r in results if r["blind_id"] == "BLIND04")["total"] < 55)
check("교차: 완주하면 total 100", next(r for r in results if r["blind_id"] == "BLIND01")["total"] == 100.0)
check("교차: 결과 키는 blind_id/total/status 셋뿐",
      all(set(r) == {"blind_id", "total", "status"} for r in results))
check("교차: total은 0~100 실수",
      all(isinstance(r["total"], float) and 0.0 <= r["total"] <= 100.0 for r in results))

path = write_eval_file(out, team="8", out_dir=tmp)
saved = json.load(open(path, encoding="utf-8"))
check("교차: 파일명이 eval_8.json", os.path.basename(path) == "eval_8.json")
check("교차: 저장 형식이 예시와 같다(results 배열만)", set(saved) == {"results"})

# 명시적 blind_id 지정도 되는지
out2 = evaluate_submissions(gs, [("TEAM_X", p_perfect)])
check("교차: blind_id를 직접 지정할 수 있음", out2["results"][0]["blind_id"] == "TEAM_X")

# blind_id 정규화: 숫자 하나만 있어도 2자리로 맞춘다
_, info1 = load_answers(write("BLIND1.json", {"team": "1", "answers": []}))
_, info5 = load_answers(write("blind_05_final.json", {"team": "5", "answers": []}))
check("resolve_blind_id: BLIND1.json -> BLIND01",
      ec.resolve_blind_id(info1, tmp + "/BLIND1.json", 1) == "BLIND01")
check("resolve_blind_id: blind_05_final.json -> BLIND05",
      ec.resolve_blind_id(info5, tmp + "/blind_05_final.json", 1) == "BLIND05")

# 회귀 테스트: 상위 폴더 이름에 섞인 숫자를 파일명보다 먼저 집으면 안 된다.
# (blind5/answers_BLIND01.json 같은 실제 폴더 구조에서 전 파일이 BLIND05로
# 뭉개지는 사고가 있었다 — 전체 경로가 아니라 파일명(basename)만 봐야 한다.)
check("resolve_blind_id: 상위 폴더명의 숫자에 낚이지 않음(파일명만 봄)",
      ec.resolve_blind_id(info1, "/data/blind5/answers_BLIND02.json", 1) == "BLIND02")
_resolved_under_numbered_folder = [
    ec.resolve_blind_id(info1, "/data/blind5/answers_BLIND{:02d}.json".format(i), i)
    for i in range(1, 6)
]
check("resolve_blind_id: 폴더명이 blind5여도 5개 파일이 각자 다르게 resolve됨",
      _resolved_under_numbered_folder == list(BLIND_IDS),
      str(_resolved_under_numbered_folder))

# check_blind_id_coverage: 중복·누락·이물질 탐지
ok5 = [{"blind_id": b, "total": 1.0, "status": "completed"} for b in ec.BLIND_IDS]
check("커버리지 검사: 정상 5개는 문제 없음", ec.check_blind_id_coverage(ok5) == [])
dup = ok5[:4] + [ok5[0]]
check("커버리지 검사: 중복 잡음", any("중복" in m for m in ec.check_blind_id_coverage(dup)))
missing_one = ok5[:4]
check("커버리지 검사: 누락 잡음", any("누락" in m for m in ec.check_blind_id_coverage(missing_one)))
stray = ok5[:4] + [{"blind_id": "BLIND99", "total": 1.0, "status": "completed"}]
check("커버리지 검사: BLIND01~05 밖의 값 잡음",
      any("아닌 값" in m for m in ec.check_blind_id_coverage(stray)))

# 운영진 검사기(validate_eval_document/check_eval_file)를 내 산출물에 직접 적용
check("운영진 검사기: 정상 5팀 결과는 문제 없음(제출 가능)",
      ec.validate_eval_document(saved) == [], str(ec.validate_eval_document(saved)))
eval_ok, eval_issues = ec.check_eval_file(path)
check("운영진 검사기: check_eval_file도 제출 가능 판정", eval_ok, str(eval_issues))

bad_doc_extra_root = {"results": saved["results"], "schema_version": 1}
check("운영진 검사기: schema_version 있으면 반려",
      any("schema_version" in m for m in ec.validate_eval_document(bad_doc_extra_root)))
bad_doc_rank = json.loads(json.dumps(saved))
bad_doc_rank["results"][0]["rank"] = 1
check("운영진 검사기: 결과 안 rank 있으면 반려",
      any(".rank" in m for m in ec.validate_eval_document(bad_doc_rank)))
bad_doc_status = json.loads(json.dumps(saved))
bad_doc_status["results"][0]["status"] = "partial"
check("운영진 검사기: status가 partial이면 반려(파일 전체 무효화 취지)",
      any("순위 산정에서 제외" in m for m in ec.validate_eval_document(bad_doc_status)))
bad_doc_range = json.loads(json.dumps(saved))
bad_doc_range["results"][0]["total"] = 150.0
check("운영진 검사기: total이 범위를 벗어나면 반려",
      any("0~100" in m for m in ec.validate_eval_document(bad_doc_range)))
bad_doc_dup = json.loads(json.dumps(saved))
bad_doc_dup["results"][1]["blind_id"] = bad_doc_dup["results"][0]["blind_id"]
check("운영진 검사기: blind_id 중복이면 반려",
      any("두 번 이상" in m for m in ec.validate_eval_document(bad_doc_dup)))
check("운영진 검사기: 파일명이 eval_1~17 형식이 아니면 반려",
      ec.check_eval_file(write("eval_team8.json", saved))[0] is False)

# =====================================================================================
print("\n[7] Gemini 판정 — 실제 API 없이 가짜 google.genai로 배선만 검증")
# =====================================================================================
def install_fake_genai(responder):
    """client.models.generate_content 호출을 responder(호출순번, kwargs)로 가로챈다.

    make_gemini_judge는 함수 안에서 매번 `from google import genai`를 하므로(지연 임포트),
    reload 없이 sys.modules만 바꿔치기해도 다음 make_gemini_judge() 호출부터 바로 반영된다.
    """
    calls = []

    class FakeResponse:
        def __init__(self, text):
            self.text = text

    class FakeModels:
        def generate_content(self, **kwargs):
            calls.append(kwargs)
            result = responder(len(calls) - 1, kwargs)
            if isinstance(result, Exception):
                raise result
            return FakeResponse(result)

    class FakeClient:
        def __init__(self, api_key=None):
            self.api_key = api_key
            self.models = FakeModels()

    class FakeAPIError(Exception):
        def __init__(self, message, code=429):
            super().__init__(message)
            self.code = code

    google_mod = types.ModuleType("google")
    genai_mod = types.ModuleType("google.genai")
    genai_mod.Client = FakeClient
    errors_mod = types.ModuleType("google.genai.errors")
    errors_mod.APIError = FakeAPIError
    genai_mod.errors = errors_mod
    google_mod.genai = genai_mod
    sys.modules["google"] = google_mod
    sys.modules["google.genai"] = genai_mod
    sys.modules["google.genai.errors"] = errors_mod
    return calls, FakeAPIError


GOOD_JSON = json.dumps({
    "accuracy": {"score": 9, "reason": "대체로 정확함"},
    "grounding": {"score": 10, "reason": "정답 조항과 일치"},
    "completeness": {"score": 7, "reason": "일부 누락"},
    "clarity": {"score": 10, "reason": "명확함"},
}, ensure_ascii=False)

calls, FakeAPIError = install_fake_genai(lambda i, kw: GOOD_JSON)
stats = {}
judge = ec.make_gemini_judge("fake-key", stats=stats, request_interval_s=0)
axes, reasons = judge("질문", "답변입니다", ["키팩트1"], ["제1조"], [["카카오계정 약관", 1]], [("카카오계정약관", 1)])
check("정상 응답 파싱: accuracy=9.0", axes["accuracy"] == 9.0, str(axes))
check("정상 응답 파싱: completeness=7.0", axes["completeness"] == 7.0)
check("reason 텍스트 보존", reasons["grounding"] == "정답 조항과 일치")
check("stats.gemini_ok 증가, fallback=0", stats["gemini_ok"] == 1 and stats["fallback"] == 0, str(stats))
check("성공 시 재시도 없이 호출 1회", len(calls) == 1)

sent = calls[0]
check("model=gemini-3.5-flash", sent["model"] == "gemini-3.5-flash", sent["model"])
check("temperature=0", sent["config"]["temperature"] == 0)
check("response_mime_type=application/json", sent["config"]["response_mime_type"] == "application/json")
check("system_instruction에 4축 이름 포함",
      all(a in sent["config"]["system_instruction"] for a in JUDGE_AXIS_WEIGHTS))
check("프롬프트에 질문/답변/키팩트/근거가 실제로 들어감",
      all(s in sent["contents"] for s in ("질문", "답변입니다", "키팩트1", "제1조", "카카오계정 약관")))

install_fake_genai(lambda i, kw: "```json\n" + GOOD_JSON + "\n```")
judge_fenced = ec.make_gemini_judge("fake-key", stats={}, request_interval_s=0)
axes_f, _ = judge_fenced("q", "a", [], [], [], [])
check("코드펜스 섞인 JSON도 파싱", axes_f["accuracy"] == 9.0)

sequence = ["이건 JSON이 아님", GOOD_JSON]
calls_retry, _ = install_fake_genai(lambda i, kw: sequence[i])
stats_retry = {}
judge_retry = ec.make_gemini_judge("fake-key", stats=stats_retry, max_retries=3,
                                   retry_wait_s=0, request_interval_s=0)
axes_retry, _ = judge_retry("q", "a", [], [], [], [])
check("1차 실패 후 2차 성공으로 회복", axes_retry["accuracy"] == 9.0)
check("호출 2회, gemini_ok=1", len(calls_retry) == 2 and stats_retry["gemini_ok"] == 1)

install_fake_genai(lambda i, kw: "계속 깨진 응답")
stats_fail = {}
judge_fail = ec.make_gemini_judge("fake-key", stats=stats_fail, max_retries=2,
                                  retry_wait_s=0, request_interval_s=0)
axes_fail, reasons_fail = judge_fail("질문", "정답 그대로 씁니다", ["정답 그대로"], [],
                                     [["카카오계정 약관", 1]], [("카카오계정약관", 1)])
check("완전 실패해도 예외 없이 반환", set(axes_fail) == set(JUDGE_AXIS_WEIGHTS))
check("실패 시에도 축 값이 0~10 범위", all(0.0 <= v <= 10.0 for v in axes_fail.values()))
check("stats: fallback=1, gemini_ok=0", stats_fail["fallback"] == 1 and stats_fail["gemini_ok"] == 0)
check("사유에 규칙기반 대체 안내 포함", "규칙기반 대체" in reasons_fail["accuracy"])

seq_429 = [None, GOOD_JSON]


def responder_429(i, kw):
    return FakeAPIError("RESOURCE_EXHAUSTED", code=429) if seq_429[i] is None else seq_429[i]


install_fake_genai(responder_429)
judge_429 = ec.make_gemini_judge("fake-key", stats={}, max_retries=3, retry_wait_s=0, request_interval_s=0)
axes_429, _ = judge_429("q", "a", [], [], [], [])
check("429(APIError) 이후 재시도로 회복", axes_429["accuracy"] == 9.0)

oob_json = json.dumps({
    "accuracy": {"score": 15, "reason": "범위 초과"}, "grounding": {"score": -3, "reason": "음수"},
    "completeness": {"score": 5, "reason": "정상"}, "clarity": {"score": 5, "reason": "정상"},
}, ensure_ascii=False)
install_fake_genai(lambda i, kw: oob_json)
judge_oob = ec.make_gemini_judge("fake-key", stats={}, request_interval_s=0)
axes_oob, _ = judge_oob("q", "a", [], [], [], [])
check("범위 밖 점수는 0~10으로 clamp", axes_oob["accuracy"] == 10.0 and axes_oob["grounding"] == 0.0)

install_fake_genai(lambda i, kw: GOOD_JSON)
gemini_report = evaluate_team(gs, load_answers(p_perfect)[0],
                              judge_fn=ec.make_gemini_judge("fake-key", stats={}, request_interval_s=0))
check("evaluate_team이 gemini judge_fn을 그대로 받아 채점",
      gemini_report["per_question"][0]["judge_axes"]["accuracy"] == 9.0)

# =====================================================================================
print("\n[8] 하드코딩 검사 — 공개셋 id·문항수가 코드에 박혀 있지 않은가")
# =====================================================================================
source = open(os.path.join(os.path.dirname(os.path.abspath(__file__)),
                           "evaluator_core.py"), encoding="utf-8").read()
import re as _re
check("코드에 'P01' 같은 문항 id가 없다", not _re.search(r"['\"]P\d{2}['\"]", source))
check("코드에 문항 수 10/30이 상수로 없다",
      not _re.search(r"(PUBLIC_QUESTION_COUNT|n_questions\s*==\s*(10|30)|range\(1,\s*(11|31)\))", source))

# =====================================================================================
print("\n" + "=" * 60)
print("통과 {} / 실패 {}".format(len(PASS), len(FAIL)))
if FAIL:
    print("실패 목록:")
    for name in FAIL:
        print("  - " + name)
    sys.exit(1)
print("전부 통과")
