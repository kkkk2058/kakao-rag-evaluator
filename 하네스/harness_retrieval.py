# -*- coding: utf-8 -*-
"""
검색(retrieval) 전용 오프라인 하네스 — GPU 없이 로컬에서 바로 실행 가능.
BM25 토크나이즈 방식, 리랭커 사용 여부 등을 바꿔가며 공개 10문항 MRR을 비교한다.
(생성 모델 답변 품질까지 보려면 harness_full_pipeline.py를 Colab에서 사용)

사용법:
    python harness_retrieval.py

결과는 harness_results.csv 에 한 줄씩 누적 저장된다.
"""
import csv
import json
import os
import re
import time
import unicodedata

BASE = os.path.dirname(os.path.abspath(__file__))
ARTICLES_PATH = os.path.join(BASE, "..", "약관원문_확정", "articles.json")
GOLD_PATH = os.path.join(BASE, "..", "대회 정보", "gold_questions_public10.json")
CSV_PATH = os.path.join(BASE, "harness_results.csv")

ARTICLES = json.load(open(ARTICLES_PATH, encoding="utf-8"))
CORPUS_TEXT = [f"{a['title']} {a['text']}" for a in ARTICLES]
GOLD = json.load(open(GOLD_PATH, encoding="utf-8"))["questions"]


def bigram_tokens(text):
    t = re.sub(r"\s+", "", unicodedata.normalize("NFC", text))
    if len(t) < 2:
        return [t] if t else []
    return [t[i:i + 2] for i in range(len(t) - 1)]


def word_tokens(text):
    return re.findall(r"[가-힣]+|[A-Za-z0-9]+", text)


def build_bm25(tokenizer):
    from rank_bm25 import BM25Okapi
    return BM25Okapi([tokenizer(t) for t in CORPUS_TEXT])


def evaluate_config(name, tokenizer, reranker=None, bm25_pool=30, topn_for_mrr=10):
    """gold 조항이 상위 topn_for_mrr 안에서 처음 등장하는 순위로 MRR을 계산한다."""
    bm25 = build_bm25(tokenizer)
    total_rr = 0.0
    total_latency = 0.0
    hit1 = 0
    rows = []

    for q in GOLD:
        gold_keys = {(g["doc"], g["article"]) for g in q["gold_articles"]}
        t0 = time.perf_counter()

        scores = bm25.get_scores(tokenizer(q["question"]))
        pool = sorted(range(len(ARTICLES)), key=lambda i: scores[i], reverse=True)[:bm25_pool]

        if reranker is not None:
            pairs = [(q["question"], CORPUS_TEXT[i][:1000]) for i in pool]
            raw = reranker.predict(pairs)
            ranked = [i for i, _ in sorted(zip(pool, raw), key=lambda x: x[1], reverse=True)]
        else:
            ranked = pool

        total_latency += time.perf_counter() - t0

        rank = None
        for r, i in enumerate(ranked[:topn_for_mrr], 1):
            key = (ARTICLES[i]["doc"], ARTICLES[i]["article_no"])
            if key in gold_keys:
                rank = r
                break
        rr = 1.0 / rank if rank else 0.0
        total_rr += rr
        if rank == 1:
            hit1 += 1
        rows.append((q["id"], rank, round(rr, 3)))

    n = len(GOLD)
    result = {
        "config": name,
        "mrr": round(total_rr / n, 4),
        "hit@1": f"{hit1}/{n}",
        "avg_latency_ms": round(total_latency / n * 1000, 1),
    }
    return result, rows


def log_result(result):
    header = ["timestamp", "config", "mrr", "hit@1", "avg_latency_ms"]
    exists = os.path.exists(CSV_PATH)
    with open(CSV_PATH, "a", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        if not exists:
            w.writerow(header)
        w.writerow([
            time.strftime("%Y-%m-%d %H:%M:%S"), result["config"],
            result["mrr"], result["hit@1"], result["avg_latency_ms"],
        ])


def main():
    configs = [
        ("bm25_bigram", bigram_tokens, None),
        ("bm25_word", word_tokens, None),
    ]
    for name, tok, reranker in configs:
        result, rows = evaluate_config(name, tok, reranker=reranker)
        log_result(result)
        print(f"[{name}] MRR={result['mrr']}  hit@1={result['hit@1']}  avg_latency={result['avg_latency_ms']}ms")
        for qid, rank, rr in rows:
            if rank != 1:
                print(f"   -> {qid}: rank={rank} rr={rr}  (1위 아님, 확인 필요)")

    print(f"\n결과 누적 파일: {CSV_PATH}")
    print(
        "리랭커까지 비교하려면 Colab에서:\n"
        "  from sentence_transformers import CrossEncoder\n"
        "  reranker = CrossEncoder('BAAI/bge-reranker-v2-m3', device='cuda')\n"
        "  result, rows = evaluate_config('bm25_bigram+reranker', bigram_tokens, reranker=reranker)\n"
        "  log_result(result)"
    )


if __name__ == "__main__":
    main()
