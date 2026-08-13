# -*- coding: utf-8 -*-
# =====================================================================================
#  6번 셀 — (검증용) 근거 소실 지점 추적  ※ 제출물 아님
# =====================================================================================
#  왜 필요한가:
#    4번 셀은 "gold 조항이 retrieved 에 있는가"만 보고, 없으면 굶김 / 있으면 "생성 문제"로
#    단정한다. 그런데 _retrieve() 와 실제 프롬프트 사이에는 좁히는 단계가 두 개 더 있다.
#
#      _retrieve()                    → retrieved (채점에 보고되는 값)
#      _prune_contexts_by_coverage()  → 조항을 통째로 버릴 수 있다 (retrieved 에는 그대로 남음)
#      _build_evidence_blocks()       → 1200자 초과 조항은 항 단위로 잘린다
#      ────────────────────────────────
#      prepared_ctx                   ← 모델이 실제로 본 텍스트
#
#    즉 "검색됨"은 "모델이 봤다"가 아니다. 이 셀은 그 차이를 직접 잰다.
#
#  무엇을 재는가 (key_fact 마다):
#    prepared_ctx 안에 그 사실이 문자 bigram 으로 남아 있는가
#      · 남아 있다  → 모델은 봤는데 답에 안 썼다      → 생성/프롬프트 트랙
#      · 사라졌다   → 파이프라인이 잃었다. 어느 단계인지까지 같이 찍는다
#                     (미검색 / pruning 탈락 / 구조축약)
#
#  비용: 생성을 돌리지 않으므로 10문항 수 초. 1번 셀 실행 후 아무 때나.
#  전제: 1번 셀과 4번 셀(_CV_GOLD 정의)이 먼저 실행돼 있어야 한다.
# =====================================================================================
import re as _et_re
import unicodedata as _et_ud

_ET_SEEN_THRESHOLD = 0.60   # 이 이상이면 "모델이 봤다"로 본다
_ET_CTX_LIMIT = 1200        # 1번 셀 FULL_ARTICLE_CHAR_LIMIT 과 같은 값


def _et_bigrams(text):
    t = _et_re.sub(r"\s+", "", _et_ud.normalize("NFC", str(text)))
    return {t[i:i + 2] for i in range(len(t) - 1)} if len(t) >= 2 else set()


def _et_recall(fact, haystack):
    """fact 의 문자 bigram 중 몇 %가 haystack 에 남아 있는가."""
    f = _et_bigrams(fact)
    if not f:
        return 0.0
    return len(f & _et_bigrams(haystack)) / len(f)


def _et_key(article):
    return (article["doc"], article["article_no"])


def _et_label(key):
    return f"{key[0]}#{key[1]}"


def _et_locate_fact(fact, gold_keys):
    """key_fact 가 실제로 어느 조항(들)에 들어 있는지 원문에서 직접 찾는다.

    4번 셀은 gold_articles 중 하나로 귀속시키는데, 4개 약관에 거의 동일한
    보일러플레이트 조항이 있어서(설계노트 5-1) 같은 사실이 여러 조항에 동시에
    존재한다. 그 경우 '특정 조항 미검색'은 굶김의 근거가 되지 못한다.
    """
    holders = []
    for a in ARTICLES:
        if _et_recall(fact, a["text"]) >= _ET_SEEN_THRESHOLD:
            holders.append(_et_key(a))
    return holders or list(gold_keys)


def run_evidence_trace(gold=None, verbose=True):
    gold = gold if gold is not None else _CV_GOLD
    rows = []

    seen_but_unused = 0
    lost_no_retrieval = 0
    lost_in_pruning = 0
    lost_in_structural = 0

    for qid, question, key_facts, gold_articles in gold:
        retrieved = _retrieve(question)
        retrieved_keys = [_et_key(a) for a in retrieved]

        generation_contexts = _prune_contexts_by_coverage(question, retrieved)
        if not generation_contexts:
            generation_contexts = retrieved[:1]
        kept_keys = [_et_key(a) for a in generation_contexts]

        prepared_ctx, _ = _build_evidence_blocks(question, generation_contexts)

        dropped = [k for k in retrieved_keys if k not in kept_keys]
        truncated = [
            _et_key(a) for a in generation_contexts
            if len(a["text"]) > _ET_CTX_LIMIT
        ]

        fact_rows = []
        for fact in key_facts:
            ctx_recall = _et_recall(fact, prepared_ctx)
            holders = _et_locate_fact(fact, {tuple(g) for g in gold_articles})

            if ctx_recall >= _ET_SEEN_THRESHOLD:
                stage, note = "봄", ""
                seen_but_unused += 1
            elif not any(h in retrieved_keys for h in holders):
                stage = "미검색"
                note = "출처 어느 조항도 검색 안 됨"
                lost_no_retrieval += 1
            elif any(h in dropped for h in holders):
                stage = "pruning"
                note = "검색됐지만 생성 context 에서 탈락"
                lost_in_pruning += 1
            elif any(h in truncated for h in holders):
                stage = "구조축약"
                note = "조항은 남았지만 해당 항이 잘림"
                lost_in_structural += 1
            else:
                stage = "기타"
                note = "조항은 프롬프트에 있는데 문장이 안 보임"

            fact_rows.append((fact, ctx_recall, stage, note, holders))

        rows.append((qid, question, retrieved_keys, kept_keys, dropped, truncated, fact_rows))

    if verbose:
        for qid, question, retrieved_keys, kept_keys, dropped, truncated, fact_rows in rows:
            print(f"── {qid}  검색 {len(retrieved_keys)}개 → 생성 {len(kept_keys)}개")
            print(f"   retrieved : {', '.join(_et_label(k) for k in retrieved_keys)}")
            if dropped:
                print(f"   ✂ pruning 탈락: {', '.join(_et_label(k) for k in dropped)}")
            if truncated:
                print(f"   ✂ 구조축약 적용: {', '.join(_et_label(k) for k in truncated)}")
            for fact, ctx_recall, stage, note, holders in fact_rows:
                mark = "  " if stage == "봄" else "!!"
                print(f"   {mark} ctx_recall={ctx_recall:.2f} [{stage}] {fact[:52]}…")
                if note:
                    print(f"        {note} (출처: {', '.join(_et_label(h) for h in holders)})")
            print()

        total = seen_but_unused + lost_no_retrieval + lost_in_pruning + lost_in_structural
        print("=" * 78)
        print(f"key_fact 총 {total}건")
        print(f"  모델이 봤음 (이후 누락은 생성 트랙) : {seen_but_unused}")
        print(f"  검색 단계에서 소실               : {lost_no_retrieval}")
        print(f"  pruning 단계에서 소실            : {lost_in_pruning}   ← _prune_contexts_by_coverage")
        print(f"  구조축약 단계에서 소실           : {lost_in_structural} ← _select_structural_article_evidence")
        print("=" * 78)
        print("읽는 법: '모델이 봤음'이 대부분이면 검색·pruning 은 건드릴 필요 없고 프롬프트/모델만 손보면 된다.")
        print("         pruning·구조축약 소실이 있으면 그건 검색 지표(MRR)에 안 잡히는 손실이다.")

    return rows


_ = run_evidence_trace()
