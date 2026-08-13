# -*- coding: utf-8 -*-
# =====================================================================================
#  함정/충돌 스트레스 테스트 — Colab 전용 (선택 사용)
# =====================================================================================
#  사용법
#    1. 1번 셀(결과기 구현)을 먼저 실행해 answer_question 이 정의된 상태로 만든다.
#    2. 이 코드를 새 셀에 붙여넣고 실행한다.
#    3. run_trap_stress_test() 를 호출한다.
#
#  운영진 공개 10문항이 아니라, 실제 원문에서 발견한 예외조항·문서간 충돌 지점을
#  근거로 직접 만든 질문 6개다. 정답은 원문 대조로 검증됨(하네스/trap_questions.json).
#  검색(retrieval)은 로컬 테스트에서 6문항 전부 1위 적중을 이미 확인했다 — 이 스크립트는
#  생성(답변 텍스트)이 trap을 실제로 통과하는지 보는 용도다.
# =====================================================================================
import json as _tj
import re as _tre
import time as _ttime
import unicodedata as _tuni

_TRAP_QUESTIONS = _tj.loads(r'''{"_meta":{"name":"자체 제작 함정/충돌 스트레스 테스트셋","설명":"운영진 공개 10문항이 아니라 실제 원문에서 발견한 예외조항·문서간 충돌 지점을 근거로 직접 만든 질문. 정답은 원문 대조로 검증됨.","n_questions":10},"questions":[{"id":"T01","question":"회사의 과실로 손해가 발생하면 회사가 예외 없이 항상 배상해주나요?","ptype":"trap","gold_articles":[{"doc":"카카오계정 약관","article":16,"citation":"카카오계정 약관 제16조(손해배상)"}],"key_facts":["회사의 과실로 손해를 입게 될 경우 배상하지만, 회사의 과실 없이 발생된 손해에 대해서는 책임을 부담하지 않습니다.","간접 손해, 특별 손해, 결과적 손해 등에 대해서도 책임을 부담하지 않습니다."]},{"id":"T02","question":"위치기반서비스 이용 목적을 달성하면 개인위치정보는 예외 없이 즉시 파기되나요?","ptype":"trap","gold_articles":[{"doc":"카카오 위치정보 이용약관","article":9,"citation":"카카오 위치정보 이용약관 제9조(개인위치정보의 보유 목적 및 보유기간)"}],"key_facts":["원칙적으로 목적을 달성한 때에는 지체없이 개인위치정보를 파기합니다.","다만, 게시물 또는 콘텐츠와 함께 위치정보가 저장되는 서비스의 경우 해당 게시물 또는 콘텐츠의 보관기간 동안 개인위치정보가 보관됩니다."]},{"id":"T03","question":"회사는 이용자가 게시한 모든 콘텐츠를 반드시 검토할 의무가 있나요?","ptype":"trap","gold_articles":[{"doc":"카카오 통합서비스약관","article":9,"citation":"카카오 통합서비스약관 제9조(권리의 귀속 및 저작물의 이용)"}],"key_facts":["아니오. 회사가 모든 콘텐츠를 검토할 의무가 있는 것은 아닙니다."]},{"id":"T04","question":"회원은 카카오계정 정보를 전부 자유롭게 수정할 수 있나요?","ptype":"trap","gold_articles":[{"doc":"카카오계정 약관","article":9,"citation":"카카오계정 약관 제9조(카카오계정 관리)"}],"key_facts":["카카오계정 웹사이트 또는 개별 서비스 내 설정 화면을 통해 정보를 열람하고 수정할 수 있습니다.","다만 카카오계정, 전화번호, 단말기 식별번호, 기타 본인확인정보 등 일부 정보는 수정이 불가능할 수 있습니다."]},{"id":"T05","question":"카카오 위치정보 이용약관에 따르면 위치정보 수집·이용·제공사실 확인자료는 정확히 몇 개월 보관되나요?","ptype":"exact_conflict","gold_articles":[{"doc":"카카오 위치정보 이용약관","article":8,"citation":"카카오 위치정보 이용약관 제8조(위치정보 수집·이용·제공사실 확인자료의 보관)"}],"key_facts":["위치정보의 보호 및 이용 등에 관한 법률 제16조 제2항에 근거합니다.","해당 자료는 6개월간 보관합니다."],"note":"통합 약관 제16조에는 같은 주제가 '6개월 이상'으로 다르게 적혀 있음 — 오답 문서로 낚일 위험이 있는 문서 간 충돌 케이스"},{"id":"T06","question":"회사는 카카오계정 가입 신청을 받으면 예외 없이 무조건 승낙하나요?","ptype":"trap","gold_articles":[{"doc":"카카오계정 약관","article":6,"citation":"카카오계정 약관 제6조(카카오계정 이용의 제한)"}],"key_facts":["원칙적으로 가입 신청자에게 카카오계정의 이용을 승낙합니다.","다만 허위 정보 입력, 타인 명의 도용 등 일정한 사유가 있는 경우에는 그 사유가 해소될 때까지 승낙을 유보하거나 승낙하지 않을 수 있습니다."]},{"id":"T07","question":"회사는 카카오계정 서비스를 예외 없이 항상 365일 24시간 중단 없이 제공하나요?","ptype":"trap","gold_articles":[{"doc":"카카오계정 약관","article":8,"citation":"카카오계정 약관 제8조(카카오계정 서비스의 변경 및 종료)"}],"key_facts":["365일, 24시간 쉬지 않고 제공하기 위하여 최선의 노력을 다합니다.","다만 설비 점검, 정전, 계약 종료, 불가항력 등의 사유가 있는 경우 서비스의 전부 또는 일부를 제한하거나 중지할 수 있습니다."]},{"id":"T08","question":"위치정보 이용약관이 변경될 때 회사는 항상 15일 전에만 공지하면 되나요?","ptype":"trap","gold_articles":[{"doc":"카카오 위치정보 이용약관","article":2,"citation":"카카오 위치정보 이용약관 제2조(이용약관의 효력 및 변경)"}],"key_facts":["원칙적으로 변경사항을 적용일자 최소 15일 전에 공지합니다.","다만 이용자 권리의 중대한 변경을 발생시키는 경우 적용일 최소 30일 전에 이메일 등으로 개별적으로 고지합니다."]},{"id":"T09","question":"통합서비스 이용계약이 해지되면 작성한 게시물이 예외 없이 전부 삭제되나요?","ptype":"trap","gold_articles":[{"doc":"카카오 통합서비스약관","article":13,"citation":"카카오 통합서비스약관 제13조(이용계약 해지)"}],"key_facts":["이용계약이 해지되면 여러분의 정보나 작성한 게시물 등 모든 데이터는 삭제됩니다.","다만 제3자에 의해 스크랩되거나 다른 이용자의 게시물에 댓글을 추가하는 등의 경우에는 해당 게시물이 삭제되지 않을 수 있습니다."]},{"id":"T10","question":"카카오 통합서비스약관에 따르면 회사는 손해가 발생하면 예외 없이 항상 배상하나요?","ptype":"trap_doc_disambiguation","gold_articles":[{"doc":"카카오 통합서비스약관","article":15,"citation":"카카오 통합서비스약관 제15조(손해배상 등)"}],"key_facts":["회사의 과실로 인하여 손해를 입게 될 경우 배상하지만, 회사의 과실 없이 발생된 손해에 대해서는 책임을 부담하지 않습니다."],"note":"손해배상 조항이 계정약관 제16조, 통합약관 제18조에도 거의 동일하게 존재 — 질문이 '통합서비스약관'을 명시했을 때 정확히 그 문서로 가는지 테스트"}]}''')["questions"]


def _trap_norm(s):
    return _tre.sub(r"\s+", "", _tuni.normalize("NFC", str(s)))


def _trap_coverage(answer_text, key_facts):
    ans_norm = _trap_norm(answer_text)
    if not key_facts:
        return None
    hits = 0
    for kf in key_facts:
        kf_norm = _trap_norm(kf)
        if len(kf_norm) < 3:
            continue
        grams = [kf_norm[i:i + 3] for i in range(len(kf_norm) - 2)]
        if not grams:
            continue
        found = sum(1 for g in grams if g in ans_norm)
        if found / len(grams) >= 0.5:
            hits += 1
    return hits / len(key_facts)


def run_trap_stress_test():
    rows = []
    for q in _TRAP_QUESTIONS:
        gold_keys = {(g["doc"], g["article"]) for g in q["gold_articles"]}
        t0 = _ttime.perf_counter()
        out = answer_question(q["question"])
        latency = _ttime.perf_counter() - t0

        retrieved = out.get("retrieved", [])
        rank = None
        for r, pair in enumerate(retrieved, 1):
            if (pair[0], pair[1]) in gold_keys:
                rank = r
                break

        coverage = _trap_coverage(out.get("answer", ""), q["key_facts"])
        cov_str = f"{coverage:.2f}" if coverage is not None else "N/A"

        print(f"\n[{q['id']}] {q['question']}")
        print(f"   검색 순위: {rank}  |  커버리지(추정): {cov_str}  |  지연: {latency:.1f}s")
        print(f"   답변: {out.get('answer', '')[:300]}")
        note = q.get("note")
        if note:
            print(f"   (주의: {note})")

        rows.append({
            "qid": q["id"], "rank": rank, "coverage_est": coverage,
            "latency_s": round(latency, 2), "answer": out.get("answer", ""),
            "retrieved": retrieved,
        })

    with open("trap_stress_test_results.json", "w", encoding="utf-8") as f:
        _tj.dump(rows, f, ensure_ascii=False, indent=2)
    print("\n결과 저장: trap_stress_test_results.json")
    print("각 답변을 직접 읽고 key_facts와 비교해 실제로 맞았는지 눈으로 확인하세요 —")
    print("coverage_est는 글자 겹침 추정치일 뿐, 논리적으로 반대 결론을 냈는지는 못 잡아냅니다")
    print("(예: P08처럼 근거를 맞게 인용하고 결론만 뒤집는 경우).")
    return rows


print("[함정 테스트 준비] run_trap_stress_test() 를 호출하세요.")
