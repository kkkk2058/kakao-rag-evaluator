# -*- coding: utf-8 -*-
"""
리트리버를 기초부터 단계적으로 발전시키며 각 단계의 한계를 실제 사례로 보여준다.
공개 10문항은 모든 방법이 만점(MRR=1.0)이라 차이가 안 보이므로,
각 단계가 약한 지점을 드러내는 변형 질문(패러프레이즈)을 추가로 넣어 테스트한다.
"""
import json
import math
import re
import unicodedata

D = "/Users/sehoonkim/kakao-rag-evaluator/약관원문_확정"
ARTICLES = json.load(open(D + "/articles.json", encoding="utf-8"))
CORPUS_TEXT = [f"{a['title']} {a['text']}" for a in ARTICLES]
GOLD = json.load(open(
    "/Users/sehoonkim/kakao-rag-evaluator/대회 정보/gold_questions_public10.json",
    encoding="utf-8",
))["questions"]

# 원래 공개 문항에, "말을 바꿔 쓴" 버전을 몇 개 추가한다 (조사 변형 / 유사어 치환 / 의미는 같지만 단어가 다른 질문)
ADVERSARIAL = [
    {
        "id": "ADV-P01",
        "question": "회사 담당자 하나만 사업자 계정을 쓸 수 있는 거 맞나요?",  # P01의 핵심어 순서·조사를 크게 바꿈
        "gold": [("카카오계정 약관", 10)],
    },
    {
        "id": "ADV-P08",
        "question": "본 약관 내용이 세부지침이랑 다르면 뭐가 먼저 적용돼?",  # 원문 어휘를 구어체 유사어로 치환
        "gold": [("카카오 통합 약관", 3)],
    },
    {
        "id": "ADV-P05",
        "question": "위치정보 관련 기록은 어느 법 조항 근거로 남기고 보관 기간은 며칠이나 되나요?",  # "몇 개월"->"며칠", 문장 구조 재배열
        "gold": [("카카오 위치정보 이용약관", 8)],
    },
]

ALL_CASES = GOLD + ADVERSARIAL


def normalize_gold(q):
    if "gold_articles" in q:
        return {(g["doc"], g["article"]) for g in q["gold_articles"]}
    return set(q["gold"])


# =====================================================================================
# 1단계 — 가장 기초: 단어가 겹치는 개수만 세기 (TF-IDF도, BM25도 아님)
# =====================================================================================
def naive_word_overlap_score(question, doc_text):
    q_words = set(re.findall(r"[가-힣]+|[A-Za-z0-9]+", question))
    d_words = re.findall(r"[가-힣]+|[A-Za-z0-9]+", doc_text)
    d_words_set = set(d_words)
    return len(q_words & d_words_set)


def retrieve_step1(question, topn=10):
    scores = [naive_word_overlap_score(question, t) for t in CORPUS_TEXT]
    ranked = sorted(range(len(ARTICLES)), key=lambda i: scores[i], reverse=True)
    return ranked[:topn]


# =====================================================================================
# 2단계 — BM25 (단어 단위): 단순 겹침 개수 대신 "희귀 단어일수록 가중치를 높이고,
#          문서가 길다고 유리해지지 않도록" 정규화한다.
# =====================================================================================
from rank_bm25 import BM25Okapi


def word_tokens(text):
    return re.findall(r"[가-힣]+|[A-Za-z0-9]+", text)


_BM25_WORD = BM25Okapi([word_tokens(t) for t in CORPUS_TEXT])


def retrieve_step2(question, topn=10):
    scores = _BM25_WORD.get_scores(word_tokens(question))
    ranked = sorted(range(len(ARTICLES)), key=lambda i: scores[i], reverse=True)
    return ranked[:topn]


# =====================================================================================
# 3단계 — BM25 (문자 bigram): 한국어 조사 변형에 안 흔들리게 토크나이즈 방식만 교체.
# =====================================================================================
def bigram_tokens(text):
    t = re.sub(r"\s+", "", unicodedata.normalize("NFC", text))
    if len(t) < 2:
        return [t] if t else []
    return [t[i:i + 2] for i in range(len(t) - 1)]


_BM25_BIGRAM = BM25Okapi([bigram_tokens(t) for t in CORPUS_TEXT])


def retrieve_step3(question, topn=10):
    scores = _BM25_BIGRAM.get_scores(bigram_tokens(question))
    ranked = sorted(range(len(ARTICLES)), key=lambda i: scores[i], reverse=True)
    return ranked[:topn]


# =====================================================================================
# 4단계 — BM25(bigram) 상위 30개 + 리랭커 재정렬: 어휘가 아예 달라도(패러프레이즈) 의미로 매칭.
#          (여기서는 GPU/모델 다운로드가 없는 환경이라 실제 리랭커 대신, 있다면 이렇게 붙는다는
#          자리만 표시하고, 실측은 Colab에서 harness_full_pipeline.py로 확인한다.)
# =====================================================================================
def retrieve_step4_note():
    return (
        "이 환경은 GPU가 없어 리랭커를 직접 돌리지 못한다. "
        "실제 4단계 실측치는 Colab에서 harness_full_pipeline.py 로 확인했고, "
        "공개 10문항 MRR=1.0 (앞서 팀 8 결과로도 재확인됨)."
    )


# =====================================================================================
# 평가
# =====================================================================================
def mrr_report(name, retrieve_fn, cases):
    total_rr = 0.0
    fails = []
    for q in cases:
        gold_keys = normalize_gold(q)
        ranked = retrieve_fn(q["question"])
        rank = None
        for r, i in enumerate(ranked, 1):
            key = (ARTICLES[i]["doc"], ARTICLES[i]["article_no"])
            if key in gold_keys:
                rank = r
                break
        rr = 1.0 / rank if rank else 0.0
        total_rr += rr
        if rr < 1.0:
            top1 = ARTICLES[ranked[0]]
            fails.append((q["id"], rank, f"{top1['doc']} 제{top1['article_no']}조"))
    mrr = total_rr / len(cases)
    print(f"\n[{name}] MRR = {mrr:.3f}  ({len(cases)}문항)")
    if fails:
        for qid, rank, wrong_top1 in fails:
            print(f"   ✗ {qid}: 정답 순위={rank}  1위로 잘못 뽑은 것={wrong_top1}")
    else:
        print("   전부 1위 적중")
    return mrr


def main():
    print("=" * 70)
    print("공개 10문항만 (전부 쉬운 문제 — 방법 간 차이가 잘 안 드러남)")
    mrr_report("1단계: 단순 단어 겹침 개수", retrieve_step1, GOLD)
    mrr_report("2단계: BM25(단어 단위)", retrieve_step2, GOLD)
    mrr_report("3단계: BM25(문자 bigram)", retrieve_step3, GOLD)

    print("\n" + "=" * 70)
    print("패러프레이즈 3문항 추가 (조사/어휘를 바꿔 쓴 버전 — 여기서 차이가 드러남)")
    mrr_report("1단계: 단순 단어 겹침 개수", retrieve_step1, ADVERSARIAL)
    mrr_report("2단계: BM25(단어 단위)", retrieve_step2, ADVERSARIAL)
    mrr_report("3단계: BM25(문자 bigram)", retrieve_step3, ADVERSARIAL)

    print("\n" + "=" * 70)
    print("전체(공개10 + 패러프레이즈3) 종합 비교")
    mrr_report("1단계: 단순 단어 겹침 개수", retrieve_step1, ALL_CASES)
    mrr_report("2단계: BM25(단어 단위)", retrieve_step2, ALL_CASES)
    mrr_report("3단계: BM25(문자 bigram)", retrieve_step3, ALL_CASES)
    print(f"\n[4단계: +리랭커] {retrieve_step4_note()}")


if __name__ == "__main__":
    main()
