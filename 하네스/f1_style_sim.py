# -*- coding: utf-8 -*-
"""핵심내용 F1 근사 시뮬레이터 — 답변 스타일별 F1을 비교한다.
공식 채점의 정확한 토크나이저는 비공개이므로 두 가지 근사로 교차 확인한다:
  (1) 어절에서 흔한 조사/어미를 벗겨낸 content-token F1
  (2) 문자 bigram F1 (조사 변형에 둔감)
"""
import re, unicodedata
from collections import Counter

_JOSA = r"(은|는|이|가|을|를|에|에서|에게|의|와|과|으로|로|도|만|까지|부터|라|이라|께|한테|보다|처럼|마다|조차|밖에|든지|이나|나)$"
_EOMI = r"(합니다|입니다|됩니다|습니다|하다|한다|하는|하여|해야|하고|하며|였습니다|드립니다|드리겠습니다|겠습니다|됩니다|있습니다|없습니다)$"
_STOP = {"그","이","저","등","및","또는","경우","대한","대해","위한","위하여","수","것","때","본","해당","관한","각","제"}

def content_tokens(s):
    s = unicodedata.normalize("NFC", s)
    s = re.sub(r"[^\w가-힣0-9]+", " ", s)
    out = []
    for w in s.split():
        w = re.sub(_EOMI, "", w)
        w = re.sub(_JOSA, "", w)
        if len(w) >= 2 and w not in _STOP:
            out.append(w)
        elif w.isdigit():
            out.append(w)
    return out

def bigrams(s):
    s = re.sub(r"\s+", "", unicodedata.normalize("NFC", s))
    return [s[i:i+2] for i in range(len(s)-1)]

def f1(pred, gold, tok):
    p, g = Counter(tok(pred)), Counter(tok(gold))
    if not p or not g: return 0.0
    ov = sum((p & g).values())
    if ov == 0: return 0.0
    prec, rec = ov/sum(p.values()), ov/sum(g.values())
    return 2*prec*rec/(prec+rec), prec, rec

def report(name, pred, gold_facts):
    gold = " ".join(gold_facts)
    a = f1(pred, gold, content_tokens)
    b = f1(pred, gold, bigrams)
    print(f"  {name:<28} 어절F1={a[0]:.3f} (P={a[1]:.2f} R={a[2]:.2f})   bigramF1={b[0]:.3f} (P={b[1]:.2f} R={b[2]:.2f})  len={len(pred)}")


# -------------------------------------------------------------------------------------
# 답변 스타일 비교 — 프롬프트 설계 근거(설계노트 §10-2)를 재현한다.
# 공식 채점 토크나이저가 비공개이므로 두 근사의 순위가 일치하는지로 결론의 견고성을 본다.
# -------------------------------------------------------------------------------------
if __name__ == "__main__":
    KF_P02 = [
        "2시간 이상 복구가 지연되는 경우 카카오 고객센터 공지사항 등에 게시하여 알려 드립니다.",
        "회사는 상황을 파악하는 즉시 최대한 빠른 시일 내에 서비스를 복구하도록 노력합니다.",
    ]
    SENT = ("회사가 상황을 파악하는 즉시 최대한 빠른 시일 내에 서비스를 복구하도록 노력하되, "
            "2시간 이상 복구가 지연될 시 카카오 서비스 공지사항, 카카오 고객센터 공지사항 등에 "
            "게시하여 알려 드립니다.")

    print("[P02] 답변 스타일별 F1 — 봉우리가 좁다(덜 써도 더 써도 급락)")
    report("결론만", "2시간 이상입니다.", KF_P02)
    report("질문 복원 + 결론", "복구가 2시간 이상 지연되면 회사는 공지사항에 게시하여 알려 드립니다.", KF_P02)
    report("정답 문장 통째 복사(권장)", SENT, KF_P02)
    report("코드가 근거 표기를 앞에 붙인 경우",
           "카카오계정 약관 제8조(카카오계정 서비스의 변경 및 종료)에 따르면, " + SENT, KF_P02)

    KF_P08 = [
        "아니오. 본 약관과 세부지침의 내용이 충돌할 경우 세부지침에 따릅니다.",
        "본 약관에 규정되지 않은 사항에 대해서는 관련법령 또는 회사가 정한 서비스의 개별 이용약관, "
        "운영정책 및 규칙 등(세부지침)의 규정에 따릅니다.",
    ]
    EV = ("본 약관에 규정되지 않은 사항에 대해서는 관련법령 또는 회사가 정한 서비스의 개별 이용약관, "
          "운영정책 및 규칙 등의 규정에 따르며, 본 약관과 세부지침의 내용이 충돌할 경우에는 "
          "세부지침에 따릅니다.")

    print("\n[P08] 판정어 위치는 F1 중립 — bag-of-tokens라 순서 비용이 0이다")
    report("판정 먼저(제출 (7) 방식)", "아니오, " + EV, KF_P08)
    report("근거 먼저 + 끝에 판정(현행)", EV + " 아니오.", KF_P08)
    report("(7) 실제 오답: 첫 토큰부터 뒤집힘",
           "네, 본 약관과 세부지침의 내용이 충돌할 경우 본 약관이 세부지침보다 우선하여 적용됩니다.",
           KF_P08)
