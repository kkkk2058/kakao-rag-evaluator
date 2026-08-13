# 6번 셀 — (검증용) 커버리지 재시도가 '채점용 F1'에 실제로 도움이 되는지  ※ 제출물 아님
# ─────────────────────────────────────────────────────────────────────────────
# 왜 이 셀이 필요한가:
#   커버리지 재시도의 채택 게이트는 두 가지만 본다.
#     (a) _cov_grounded        — 근거에 없는 말을 지어냈는가
#     (b) _cov_covered_count 증가 — 빠졌던 문장을 더 담았는가   ← recall 계열
#   둘 다 "얼마나 길어졌는가"를 보지 않는다. 그런데 실제 채점의 keyfact F1 은
#   precision 을 깎으므로, '커버리지는 올랐는데 F1 은 떨어진' 채택이 생길 수 있고
#   지금 하네스로는 그게 보이지 않는다(cov 답변만 찍히고 base 가 없어서).
#
#   이 셀은 answer_question 과 똑같은 순서로 돌되 base(재시도 전)와 cov(재시도 후)를
#   따로 붙잡아 두 답변의 토큰 F1 을 나란히 찍는다. 코드는 하나도 바꾸지 않는다.
#
#   토크나이저: 공백 어절. 실제 채점 결과(practice_result)와 대조했을 때 문자 bigram
#   (평균오차 0.042)보다 어절(0.036)이 공식 채점기에 더 가까웠다.
# ─────────────────────────────────────────────────────────────────────────────
import json as _c6_json
import re as _c6_re
import time as _c6_time
import unicodedata as _c6_ud
from collections import Counter as _C6Counter

_C6_GOLD = _c6_json.loads(r'''[["P01", "공개", "사업자/단체 카카오계정은 계정 정보에 등록된 담당자 몇 명이 이용할 수 있으며, 다른 사람과 공유하는 것은 허용되나요?", ["사업자/단체 카카오계정은 계정 정보에 등록된 담당자 1인만 이용할 수 있습니다.", "이를 다른 사람에게 공유하는 것은 금지됩니다."]], ["P02", "공개", "회사가 예측하거나 통제할 수 없는 사유로 서비스가 중단된 경우, 복구가 몇 시간 이상 지연되면 회사는 공지사항에 게시하여 알리나요?", ["2시간 이상 복구가 지연되는 경우 카카오 고객센터 공지사항 등에 게시하여 알려 드립니다.", "회사는 상황을 파악하는 즉시 최대한 빠른 시일 내에 서비스를 복구하도록 노력합니다."]], ["P03", "공개", "카카오계정 약관에서 회사가 개별 서비스와 연동하여 카카오계정에서 제공한다고 열거한 '카카오계정 서비스'의 내용 5가지는 각각 무엇인가요?", ["카카오계정 서비스의 내용은 통합로그인, SSO(Single Sign On), 카카오계정 정보 통합 관리, 사업자/단체 카카오계정, 기타 회사가 제공하는 서비스입니다.", "통합로그인은 카카오계정이 적용된 개별 서비스에서 하나의 카카오계정과 비밀번호로 로그인할 수 있는 통합 회원 인증 서비스입니다.", "SSO(Single Sign On)는 웹브라우저나 특정 모바일 기기에서 카카오계정 1회 로그인으로 이용 중인 개별 서비스간 추가 로그인 없이 자동 접속하는 서비스입니다.", "카카오계정 정보 통합 관리는 개별 서비스 이용을 위해 카카오계정 정보를 통합 관리하며, 개별 서비스의 유형에 따라 전문기관을 통한 실명확인 및 본인인증을 요청하고 이를 카카오계정 정보로 저장합니다.", "사업자/단체 카카오계정은 사업자/단체 명의로 카카오 서비스를 이용하기 위해 만들어진 카카오계정으로서 해당 사업자/단체의 책임 하에 권한을 위임받은 담당자가 이용, 관리할 수 있는 계정 서비스입니다."]], ["P04", "공개", "회사가 위치기반서비스의 이용을 제한하거나 중지한 때에는 이용자에게 무엇을 어떤 방법으로 알리나요?", ["회사가 서비스 이용을 제한하거나 중지한 때에는 그 사유 및 제한기간 등을 알립니다.", "그 사유 및 제한기간 등은 회사 홈페이지 등을 통해 사전 공지하거나 이용자에게 통지합니다."]], ["P05", "공개", "회사가 위치정보 수집·이용·제공사실 확인자료를 기록·보존하는 근거는 위치정보의 보호 및 이용 등에 관한 법률 제 몇 조 제 몇 항이며, 그 자료는 어디에 기록되어 몇 개월간 보관되나요?", ["회사는 위치정보의 보호 및 이용 등에 관한 법률 제16조 제2항에 근거합니다.", "위치정보 수집·이용·제공사실 확인자료를 위치정보시스템에 자동으로 기록·보존합니다.", "해당 자료는 6개월간 보관합니다."]], ["P06", "공개", "카카오계정이 없는 사람이 통합서비스에 가입하려면 무엇을 먼저 해야 하며, 통합서비스 이용계약은 동의·확인·승낙의 어떤 순서로 체결되나요?", ["통합서비스에 가입하기 위해서는 카카오계정이 필요하며, 카카오계정이 없는 경우 카카오계정을 먼저 생성하여야 합니다.", "통합서비스 이용계약은 여러분이 본 약관의 내용에 동의한 후 회사가 여러분의 카카오계정 정보 등을 확인한 후 승낙함으로써 체결됩니다."]], ["P07", "공개", "서비스 명칭에 '카카오'가 사용되더라도 카카오 통합서비스약관의 '통합서비스'에 포함되지 않는 서비스는 누가 제공하는 서비스이며, 약관은 그 예로 무엇을 들고 있나요?", ["서비스 명칭에 '카카오'가 사용되더라도 회사가 아닌 카카오 계열사에서 제공하는 서비스는 본 약관의 통합서비스에 포함되지 않습니다.", "그 예는 ㈜카카오모빌리티가 제공하는 카카오 T 택시 서비스입니다.", "여러분은 회사가 아닌 계열사를 포함한 제3자가 제공하는 서비스에 가입되지는 않습니다."]], ["P08", "공개", "카카오 통합 약관과 세부지침(회사가 정한 서비스의 개별 이용약관·운영정책·규칙 등)의 내용이 충돌하는 경우, 본 약관이 세부지침보다 우선하여 적용되나요?", ["아니오. 본 약관과 세부지침의 내용이 충돌할 경우 세부지침에 따릅니다.", "본 약관에 규정되지 않은 사항에 대해서는 관련법령 또는 회사가 정한 서비스의 개별 이용약관, 운영정책 및 규칙 등(세부지침)의 규정에 따릅니다."]], ["P09", "공개", "이용자가 서비스 사용을 중단하거나 카카오계정 및 Daum 아이디를 탈퇴한 이후, 게시물에 관하여 회사에 부여한 라이선스의 효력은 어떻게 되나요?", ["본 라이선스는 여러분이 서비스의 사용을 중단하거나 카카오계정 및/또는 Daum 아이디를 탈퇴한 후에도 존속하게 됩니다.", "여러분이 회사에게 제공하는 라이선스는 전 세계적이고 영구적인 라이선스입니다."]], ["P10", "공개", "8세 이하의 아동 등의 생명 또는 신체 보호를 위해 보호의무자가 개인위치정보의 이용 또는 제공에 동의하려면 어떤 서류에 무엇을 첨부하여 어디에 제출해야 하며, 그 동의는 어떤 효력을 갖나요?", ["8세 이하의 아동 등의 생명 또는 신체의 보호를 위하여 개인위치정보의 이용 또는 제공에 동의를 하고자 하는 보호의무자는 서면동의서에 보호의무자임을 증명하는 서면을 첨부하여 회사에 제출하여야 합니다.", "보호의무자가 개인위치정보의 이용 또는 제공에 동의하는 경우에는 본인의 동의가 있는 것으로 봅니다.", "보호의무자는 이 경우 이용자의 권리를 모두 가집니다."]], ["T01", "함정", "회사의 과실로 손해가 발생하면 회사가 예외 없이 항상 배상해주나요?", ["회사의 과실로 손해를 입게 될 경우 배상하지만, 회사의 과실 없이 발생된 손해에 대해서는 책임을 부담하지 않습니다.", "간접 손해, 특별 손해, 결과적 손해 등에 대해서도 책임을 부담하지 않습니다."]], ["T02", "함정", "위치기반서비스 이용 목적을 달성하면 개인위치정보는 예외 없이 즉시 파기되나요?", ["원칙적으로 목적을 달성한 때에는 지체없이 개인위치정보를 파기합니다.", "다만, 게시물 또는 콘텐츠와 함께 위치정보가 저장되는 서비스의 경우 해당 게시물 또는 콘텐츠의 보관기간 동안 개인위치정보가 보관됩니다."]], ["T03", "함정", "회사는 이용자가 게시한 모든 콘텐츠를 반드시 검토할 의무가 있나요?", ["아니오. 회사가 모든 콘텐츠를 검토할 의무가 있는 것은 아닙니다."]], ["T04", "함정", "회원은 카카오계정 정보를 전부 자유롭게 수정할 수 있나요?", ["카카오계정 웹사이트 또는 개별 서비스 내 설정 화면을 통해 정보를 열람하고 수정할 수 있습니다.", "다만 카카오계정, 전화번호, 단말기 식별번호, 기타 본인확인정보 등 일부 정보는 수정이 불가능할 수 있습니다."]], ["T05", "함정", "카카오 위치정보 이용약관에 따르면 위치정보 수집·이용·제공사실 확인자료는 정확히 몇 개월 보관되나요?", ["위치정보의 보호 및 이용 등에 관한 법률 제16조 제2항에 근거합니다.", "해당 자료는 6개월간 보관합니다."]], ["T06", "함정", "회사는 카카오계정 가입 신청을 받으면 예외 없이 무조건 승낙하나요?", ["원칙적으로 가입 신청자에게 카카오계정의 이용을 승낙합니다.", "다만 허위 정보 입력, 타인 명의 도용 등 일정한 사유가 있는 경우에는 그 사유가 해소될 때까지 승낙을 유보하거나 승낙하지 않을 수 있습니다."]], ["T07", "함정", "회사는 카카오계정 서비스를 예외 없이 항상 365일 24시간 중단 없이 제공하나요?", ["365일, 24시간 쉬지 않고 제공하기 위하여 최선의 노력을 다합니다.", "다만 설비 점검, 정전, 계약 종료, 불가항력 등의 사유가 있는 경우 서비스의 전부 또는 일부를 제한하거나 중지할 수 있습니다."]], ["T08", "함정", "위치정보 이용약관이 변경될 때 회사는 항상 15일 전에만 공지하면 되나요?", ["원칙적으로 변경사항을 적용일자 최소 15일 전에 공지합니다.", "다만 이용자 권리의 중대한 변경을 발생시키는 경우 적용일 최소 30일 전에 이메일 등으로 개별적으로 고지합니다."]], ["T09", "함정", "통합서비스 이용계약이 해지되면 작성한 게시물이 예외 없이 전부 삭제되나요?", ["이용계약이 해지되면 여러분의 정보나 작성한 게시물 등 모든 데이터는 삭제됩니다.", "다만 제3자에 의해 스크랩되거나 다른 이용자의 게시물에 댓글을 추가하는 등의 경우에는 해당 게시물이 삭제되지 않을 수 있습니다."]], ["T10", "함정", "카카오 통합서비스약관에 따르면 회사는 손해가 발생하면 예외 없이 항상 배상하나요?", ["회사의 과실로 인하여 손해를 입게 될 경우 배상하지만, 회사의 과실 없이 발생된 손해에 대해서는 책임을 부담하지 않습니다."]]]''')


def _c6_tok(s):
    return _c6_re.sub(r"[^\w가-힣0-9]+", " ", _c6_ud.normalize("NFC", s)).split()


def _c6_prf(pred, gold):
    p, g = _C6Counter(_c6_tok(pred)), _C6Counter(_c6_tok(gold))
    if not p or not g:
        return 0.0, 0.0, 0.0
    ov = sum((p & g).values())
    if ov == 0:
        return 0.0, 0.0, 0.0
    pr, rc = ov / sum(p.values()), ov / sum(g.values())
    return 2 * pr * rc / (pr + rc), pr, rc


def _c6_run(question):
    """answer_question() 과 동일한 흐름. base / cov / 게이트 판정을 전부 반환한다."""
    t0 = _c6_time.time()
    ctxs = _retrieve(question)
    gen_ctxs = _prune_contexts_by_coverage(question, ctxs) or ctxs[:1]
    prepared_ctx, evidence_plain = _build_evidence_blocks(question, gen_ctxs)
    binary = _infer_binary_conclusion(question, evidence_plain)

    ans = _finalize_answer(question, _generate(
        question, gen_ctxs, binary_conclusion=binary, prepared_ctx=prepared_ctx))

    issues = _validate_answer_structure(question, ans, binary_conclusion=binary)
    if issues and _c6_time.time() - t0 < RETRY_DEADLINE_S:
        ans = _finalize_answer(question, _generate(
            question, gen_ctxs, previous_answer=ans, retry_issues=issues,
            binary_conclusion=binary, prepared_ctx=prepared_ctx))

    base = ans                       # ← 커버리지 재시도 '직전' 답변
    out = dict(base=base, final=base, ev_len=len(evidence_plain),
               missing=[], cov=None, grounded=None, cov_cnt=None,
               base_cnt=None, adopted=False, retrieved=len(ctxs), used=len(gen_ctxs))

    if _c6_time.time() - t0 >= RETRY_DEADLINE_S:
        out["missing"] = ["(시간초과로 커버리지 재시도 생략)"]
        return out

    missing = _cov_find_missing(question, evidence_plain, base)
    out["missing"] = missing
    if not missing:
        return out

    cov_issues = [
        "다음 근거 내용이 답변에 빠졌습니다. 근거의 표현을 그대로 사용해 보완하고, "
        "근거에 없는 내용은 만들지 마세요: \u300c" + u + "\u300d"
        for u in missing
    ]
    cov = _finalize_answer(question, _generate(
        question, gen_ctxs, previous_answer=base, retry_issues=cov_issues,
        binary_conclusion=binary, prepared_ctx=prepared_ctx))

    grounded = _cov_grounded(cov, evidence_plain)
    b_cnt = _cov_covered_count(base, missing)
    c_cnt = _cov_covered_count(cov, missing)
    adopted = grounded and c_cnt > b_cnt
    out.update(cov=cov, grounded=grounded, base_cnt=b_cnt, cov_cnt=c_cnt,
               adopted=adopted, final=cov if adopted else base)
    return out


print("=" * 112)
print("커버리지 재시도 검증 — base(재시도 전) vs 최종 답변의 채점용 토큰 F1")
print("=" * 112)
print(f"{'ID':<6}{'유형':<5}{'근거자':>6}{'flag':>5}{'baseF1':>8}{'covF1':>8}"
      f"{'Δ':>8}{'baseP':>7}{'covP':>7}{'길이비':>7}  판정")
print("-" * 112)

_c6_rows = []
for qid, kind, question, kfs in _C6_GOLD:
    r = _c6_run(question)
    gold = " ".join(kfs)
    bf, bp, _ = _c6_prf(r["base"], gold)
    ff, fp, _ = _c6_prf(r["final"], gold)
    cf = _c6_prf(r["cov"], gold)[0] if r["cov"] else None
    delta = ff - bf
    ratio = len(r["final"]) / max(1, len(gold))

    if not r["missing"]:
        verdict = "미발동"
    elif not r["adopted"]:
        verdict = f"거부({'ungrounded' if r['grounded'] is False else '커버리지 미증가'})"
    elif delta > 0.005:
        verdict = "✅ 개선"
    elif delta < -0.005:
        verdict = "★ 악화 — 커버리지는 올랐는데 F1 하락"
    else:
        verdict = "무변화"

    _c6_rows.append((qid, kind, bf, ff, delta, r))
    _cfs = f"{cf:.3f}" if cf is not None else "—"
    print(f"{qid:<6}{kind:<5}{r['ev_len']:>6}{len(r['missing']):>5}"
          f"{bf:>8.3f}{_cfs:>8}{delta:>+8.3f}"
          f"{bp:>7.2f}{fp:>7.2f}{ratio:>6.2f}x  {verdict}")

print("-" * 112)
_n = len(_c6_rows)
_b = sum(r[2] for r in _c6_rows) / _n
_f = sum(r[3] for r in _c6_rows) / _n
_up = sum(1 for r in _c6_rows if r[4] > 0.005)
_dn = sum(1 for r in _c6_rows if r[4] < -0.005)
_ad = sum(1 for r in _c6_rows if r[5]["adopted"])
print(f"{'평균':<6}{'':<5}{'':>6}{'':>5}{_b:>8.3f}{_f:>8.3f}{_f - _b:>+8.3f}")
print()
print(f"채택 {_ad}/{_n}  ·  개선 {_up}  ·  악화 {_dn}  ·  "
      f"F1 순변화 {_f - _b:+.4f}  (30점 환산 {(_f - _b) * 30:+.2f}점)")
print()
if _dn:
    print("★ 악화 문항 — 게이트는 통과했는데 채점 F1 은 떨어진 경우 (precision 감시 부재의 증거)")
    for qid, kind, bf, ff, d, r in _c6_rows:
        if d < -0.005:
            print(f"\n  [{qid}] {bf:.3f} → {ff:.3f} ({d:+.3f})  "
                  f"길이 {len(r['base'])}자 → {len(r['final'])}자")
            for u in r["missing"]:
                print(f"     지목: {u[:80]}")
else:
    print("악화 0건 — 현재 게이트로 충분하다는 근거(이 20문항 범위에서).")
print()
print("=" * 112)
print("커버리지가 발동한 문항 상세")
print("=" * 112)
for qid, kind, bf, ff, d, r in _c6_rows:
    if not r["missing"] or r["missing"][0].startswith("("):
        continue
    print(f"\n── {qid} ({kind})  flag={len(r['missing'])}  "
          f"grounded={r['grounded']}  covered {r['base_cnt']}→{r['cov_cnt']}  "
          f"채택={r['adopted']}  F1 {bf:.3f}→{ff:.3f}")
    for u in r["missing"]:
        print(f"   지목: {u[:90]}")
    print(f"   base: {r['base'][:150]}")
    if r["cov"]:
        print(f"   cov : {r['cov'][:150]}")
