# dense 검색 계층 실험 — 현재 BM25(조전체+항최고점 RRF) → 리랭커 파이프라인에
# dense(bge-m3) 채널을 추가하면 무엇이 달라지는지 측정한다.
#
#   1단계: 72개 조항 순위 방식 8종 비교 (BM25 / dense / 하이브리드)
#   2단계: 각 1단계 pool을 리랭커에 넣어 최종 순위 비교
#   MMR:   dense 임베딩으로 최종 1~4개 조항을 다양성 재선택
#
# 실행:  ~/kakao-rag-evaluator2/.venv/bin/python dense_ablation.py [--full] [--no-rerank]
#   --full       170문항 합성셋까지 (기본은 공개10+함정10)
#   --no-rerank  1단계만 (리랭커 생략, 수 초 내 완료)
import json, math, os, re, sys, time, unicodedata, pickle
from collections import Counter

ROOT = '/Users/sehoonkim/kakao-rag-evaluator'
CACHE = os.path.join(os.path.dirname(os.path.abspath(__file__)), '.emb_cache')
os.environ.setdefault('HF_HUB_OFFLINE', '1')
os.environ.setdefault('TOKENIZERS_PARALLELISM', 'false')

STAGE2_FULL = '--stage2-full' in sys.argv       # 190문항 × 리랭커 (약 40분)
FULL = '--full' in sys.argv or STAGE2_FULL
NO_RERANK = '--no-rerank' in sys.argv

# =====================================================================================
# 1. 노트북의 BM25 계층 재현 (bm25_ablation.py와 동일 — rank_bm25.BM25Okapi 공식 그대로)
# =====================================================================================
class BM25Okapi:
    def __init__(self, corpus, k1=1.5, b=0.75, epsilon=0.25):
        self.k1, self.b, self.epsilon = k1, b, epsilon
        self.corpus_size = len(corpus)
        self.doc_len = [len(d) for d in corpus]
        self.avgdl = sum(self.doc_len) / self.corpus_size
        self.doc_freqs = [Counter(d) for d in corpus]
        nd = Counter()
        for d in corpus:
            for w in set(d): nd[w] += 1
        self.idf = {}
        idf_sum = 0.0; neg = []
        for w, f in nd.items():
            v = math.log(self.corpus_size - f + 0.5) - math.log(f + 0.5)
            self.idf[w] = v; idf_sum += v
            if v < 0: neg.append(w)
        eps = self.epsilon * (idf_sum / len(self.idf))
        for w in neg: self.idf[w] = eps

    def get_scores(self, query):
        out = [0.0] * self.corpus_size
        for q in query:
            idf = self.idf.get(q, 0.0)
            if not idf: continue
            for i, df in enumerate(self.doc_freqs):
                f = df.get(q, 0)
                if not f: continue
                out[i] += idf * (f * (self.k1 + 1)) / (
                    f + self.k1 * (1 - self.b + self.b * self.doc_len[i] / self.avgdl))
        return out


def bigram_tokens(text):
    t = re.sub(r"\s+", "", unicodedata.normalize("NFC", text))
    if len(t) < 2: return [t] if t else []
    return [t[i:i + 2] for i in range(len(t) - 1)]


def split_subchunks(text):
    parts = re.split(r"(?=[①②③④⑤⑥⑦⑧⑨⑩])|(?=(?<!\d)\d+\.[^\d])", text)
    parts = [p.strip() for p in parts if p.strip()]
    return parts or [text]


ART = json.load(open(f'{ROOT}/약관원문_확정/articles.json'))
N = len(ART)
CORPUS = [f"{a['doc']} {a['title']} {a['text']}" for a in ART]
SUB = []                                    # [(article_index, subchunk_text), ...]
for i, a in enumerate(ART):
    for sc in split_subchunks(a['text']):
        SUB.append((i, f"{a['doc']} {a['title']} {sc}"))

BM_WHOLE = BM25Okapi([bigram_tokens(t) for t in CORPUS])
BM_SUB = BM25Okapi([bigram_tokens(t) for _, t in SUB])


def _rank_of(scores_by_idx):
    """{idx: score} → {idx: 1-base 순위}"""
    order = sorted(scores_by_idx, key=lambda i: scores_by_idx[i], reverse=True)
    return {i: r for r, i in enumerate(order, 1)}


def _rrf(*rank_maps, k=60):
    out = {}
    for i in range(N):
        out[i] = sum(1.0 / (k + rm.get(i, N + k)) for rm in rank_maps)
    return out


def _best_per_article(pair_scores):
    """서브청크 점수 → 조항별 최고점"""
    best = {}
    for (ai, _), v in zip(SUB, pair_scores):
        if ai not in best or v > best[ai]: best[ai] = v
    return {i: best.get(i, -1e9) for i in range(N)}


def bm25_whole_scores(q):
    s = BM_WHOLE.get_scores(bigram_tokens(q))
    return {i: s[i] for i in range(N)}


def bm25_sub_scores(q):
    return _best_per_article(BM_SUB.get_scores(bigram_tokens(q)))


# =====================================================================================
# 2. dense 계층 — bge-m3 (리랭커 bge-reranker-v2-m3와 같은 계열, 이미 로컬 캐시됨)
# =====================================================================================
EMB_MODEL = os.environ.get('EMB_MODEL', 'BAAI/bge-m3')
_ST = None


def _encode(texts, tag, is_query=False):
    """디스크 캐시 붙인 임베딩. 코퍼스는 한 번만 인코딩된다."""
    os.makedirs(CACHE, exist_ok=True)
    key = os.path.join(CACHE, f"{EMB_MODEL.replace('/', '_')}__{tag}.pkl")
    if os.path.exists(key):
        with open(key, 'rb') as f:
            cached = pickle.load(f)
        if cached['texts'] == texts:
            return cached['emb']
    global _ST
    if _ST is None:
        from sentence_transformers import SentenceTransformer
        t0 = time.time()
        _ST = SentenceTransformer(EMB_MODEL, device='mps')
        print(f"[로딩] {EMB_MODEL} ({time.time()-t0:.1f}s)", file=sys.stderr)
    t0 = time.time()
    emb = _ST.encode(texts, batch_size=8, normalize_embeddings=True,
                     show_progress_bar=False, convert_to_numpy=True)
    print(f"[인코딩] {tag}: {len(texts)}건 {time.time()-t0:.1f}s", file=sys.stderr)
    with open(key, 'wb') as f:
        pickle.dump({'texts': texts, 'emb': emb}, f)
    return emb


E_WHOLE = None; E_SUB = None; E_Q = {}


def _init_dense(questions):
    global E_WHOLE, E_SUB, E_Q
    E_WHOLE = _encode(CORPUS, 'whole')
    E_SUB = _encode([t for _, t in SUB], 'sub')
    qs = [q for _, _, q, _ in questions]
    tag = 'q_full' if FULL else 'q20'
    qe = _encode(qs, tag, is_query=True)
    E_Q = {q: qe[i] for i, q in enumerate(qs)}


def dense_whole_scores(q):
    s = E_WHOLE @ E_Q[q]
    return {i: float(s[i]) for i in range(N)}


def dense_sub_scores(q):
    return _best_per_article(E_SUB @ E_Q[q])


# =====================================================================================
# 3. 1단계 순위 방식 (전부 72개 조항 전체 순위를 반환)
# =====================================================================================
def rank_from(scores):
    return sorted(range(N), key=lambda i: scores[i], reverse=True)


def R_bm25_whole(q):  return rank_from(bm25_whole_scores(q))
def R_bm25_sub(q):    return rank_from(bm25_sub_scores(q))
def R_bm25_rrf(q):    # ← 현재 노트북 1단계
    return rank_from(_rrf(_rank_of(bm25_whole_scores(q)), _rank_of(bm25_sub_scores(q))))
def R_dense_whole(q): return rank_from(dense_whole_scores(q))
def R_dense_sub(q):   return rank_from(dense_sub_scores(q))
def R_dense_rrf(q):
    return rank_from(_rrf(_rank_of(dense_whole_scores(q)), _rank_of(dense_sub_scores(q))))
def R_hybrid2(q):     # BM25 RRF + dense whole
    return rank_from(_rrf(_rank_of(bm25_whole_scores(q)), _rank_of(bm25_sub_scores(q)),
                          _rank_of(dense_whole_scores(q))))
def R_hybrid4(q):     # BM25 2채널 + dense 2채널
    return rank_from(_rrf(_rank_of(bm25_whole_scores(q)), _rank_of(bm25_sub_scores(q)),
                          _rank_of(dense_whole_scores(q)), _rank_of(dense_sub_scores(q))))


STAGE1 = [
    ('① BM25 조전체',                 R_bm25_whole),
    ('② BM25 항최고점',               R_bm25_sub),
    ('③ BM25 ①+② RRF  ← 현재',       R_bm25_rrf),
    ('④ dense 조전체',                R_dense_whole),
    ('⑤ dense 항최고점',              R_dense_sub),
    ('⑥ dense ④+⑤ RRF',              R_dense_rrf),
    ('⑦ 하이브리드 ③+④ (3채널)',      R_hybrid2),
    ('⑧ 하이브리드 ③+⑥ (4채널)',      R_hybrid4),
]

# =====================================================================================
# 4. 2단계 리랭커 — 노트북 _retrieve와 동일한 2채널 RRF(rrf_k=10)
# =====================================================================================
_RR = None


def _reranker():
    global _RR
    if _RR is None:
        from sentence_transformers import CrossEncoder
        t0 = time.time()
        _RR = CrossEncoder('BAAI/bge-reranker-v2-m3', max_length=512, device='mps')
        print(f"[로딩] 리랭커 ({time.time()-t0:.1f}s)", file=sys.stderr)
    return _RR


def _best_subchunk_text(q, ai):
    """노트북 _best_subchunk_for_question 재현 — 질문과 BM25 최고점인 항 하나"""
    chunks = split_subchunks(ART[ai]['text'])
    if len(chunks) == 1: return chunks[0]
    bm = BM25Okapi([bigram_tokens(c) for c in chunks])
    s = bm.get_scores(bigram_tokens(q))
    return chunks[max(range(len(chunks)), key=lambda j: s[j])]


def rerank(q, pool, rrf_k=10):
    """노트북과 동일: 조전체 채널 + 최적 항 채널을 리랭커로 각각 스코어 → RRF"""
    rr = _reranker()
    whole = rr.predict([(q, CORPUS[i]) for i in pool], batch_size=8, show_progress_bar=False)
    sub = rr.predict(
        [(q, f"{ART[i]['doc']} {ART[i]['title']} {_best_subchunk_text(q, i)}") for i in pool],
        batch_size=8, show_progress_bar=False)
    wp = {pool[j]: r for r, j in enumerate(
        sorted(range(len(pool)), key=lambda j: whole[j], reverse=True), 1)}
    sp = {pool[j]: r for r, j in enumerate(
        sorted(range(len(pool)), key=lambda j: sub[j], reverse=True), 1)}
    fused = {i: 1.0 / (rrf_k + wp[i]) + 1.0 / (rrf_k + sp[i]) for i in pool}
    return sorted(pool, key=lambda i: fused[i], reverse=True), fused


# =====================================================================================
# 5. MMR — dense 임베딩으로 최종 후보를 다양성 재선택
#    boilerplate 중복 조항(§5-1: '약관 외 준칙'이 4개 문서에 거의 동일)에 대한 대응 가설
# =====================================================================================
def mmr_reorder(q, ranked, lam=0.7, top=4, cand=10):
    """상위 cand개 안에서 MMR로 top개를 고른 뒤, 나머지는 원래 순서로 뒤에 붙인다."""
    pool = ranked[:cand]
    qv = E_Q[q]
    sel = []
    while pool and len(sel) < top:
        best, best_v = None, -1e9
        for i in pool:
            rel = float(E_WHOLE[i] @ qv)
            div = max((float(E_WHOLE[i] @ E_WHOLE[j]) for j in sel), default=0.0)
            v = lam * rel - (1 - lam) * div
            if v > best_v: best, best_v = i, v
        sel.append(best); pool.remove(best)
    return sel + [i for i in ranked if i not in sel]


# =====================================================================================
# 6. 평가
# =====================================================================================
def norm(s): return re.sub(r"\s+", "", unicodedata.normalize("NFC", s))


def load_questions():
    qs = []
    g = json.load(open(f'{ROOT}/gold_questions_public10.json'))
    for q in g['questions']:
        qs.append(('공개', q['id'], q['question'], q['gold_articles']))
    t = json.load(open(f'{ROOT}/하네스/trap_questions.json'))
    items = t if isinstance(t, list) else next(v for v in t.values() if isinstance(v, list))
    for q in items:
        qs.append(('함정', q['id'], q['question'], q['gold_articles']))
    if FULL:
        s = json.load(open(f'{ROOT}/하네스/오프라인실험/evalset_synth.json'))
        items = s if isinstance(s, list) else s['questions']
        for q in items:
            qs.append(('합성', q['id'], q['question'], q['gold_articles']))
    return qs


def gold_key(gold):
    return {(norm(g['doc']), int(g['article'])) for g in gold}


def evaluate(ranker, qs, pool_k=30):
    """MRR + Recall@1/@3 + pool_k 안에 정답이 있는 비율(= 리랭커가 볼 수 있는 상한)"""
    tot = r1 = r3 = rp = 0.0
    fails = []
    for kind, qid, question, gold in qs:
        gk = gold_key(gold)
        rank = ranker(question)
        pos = next((p for p, ai in enumerate(rank, 1)
                    if (norm(ART[ai]['doc']), ART[ai]['article_no']) in gk), None)
        if pos:
            tot += 1.0 / pos
            r1 += pos == 1
            r3 += pos <= 3
            rp += pos <= pool_k
            if pos > 1: fails.append(f"{qid}@{pos}")
        else:
            fails.append(f"{qid}@miss")
    n = len(qs)
    return dict(mrr=tot / n, r1=r1 / n, r3=r3 / n, rpool=rp / n, fails=fails)


def select_final(order, fused, max_k=4, min_k=1, rerank_margin=0.01):
    """노트북 _retrieve의 컷오프를 그대로 재현 — 실제로 생성기에 넘어가는 조항 집합."""
    top = fused[order[0]]
    sel = [order[0]]
    for i in order[1:max_k]:
        if fused[i] >= top - rerank_margin: sel.append(i)
        else: break
    if len(sel) < min_k: sel = order[:min_k]
    return sel[:max_k]


def evaluate_selection(select_fn, qs):
    """최종 선택 집합 기준 — gold 조항을 몇 % 회수했나 + 무관 조항이 몇 개 섞였나.
    MRR은 1등만 보지만 key_fact F1은 회수한 조항 '전부'의 원문에서 나온다."""
    got = tot = 0; n_sel = 0; noise = 0
    misses = []
    for kind, qid, question, gold in qs:
        gk = gold_key(gold)
        sel = select_fn(question)
        hit = {(norm(ART[i]['doc']), ART[i]['article_no']) for i in sel} & gk
        got += len(hit); tot += len(gk)
        n_sel += len(sel); noise += len(sel) - len(hit)
        if len(hit) < len(gk): misses.append(f"{qid}({len(hit)}/{len(gk)})")
    return dict(cov=got / tot, avg_k=n_sel / len(qs), noise=noise / len(qs), misses=misses)


def report(title, rows, qs, pool_k=30):
    pub = [q for q in qs if q[0] == '공개']
    trap = [q for q in qs if q[0] == '함정']
    syn = [q for q in qs if q[0] == '합성']
    print(f"\n{'='*104}\n{title}\n{'='*104}")
    hdr = f"{'방식':30s} {'공개10':>7s} {'함정10':>7s}"
    if syn: hdr += f" {'합성170':>8s}"
    hdr += f" {'종합MRR':>8s} {'R@1':>6s} {'R@3':>6s} {'R@'+str(pool_k):>6s}   1위 실패"
    print(hdr)
    print('-' * 104)
    for name, fn in rows:
        a = evaluate(fn, pub, pool_k)['mrr']
        b = evaluate(fn, trap, pool_k)['mrr']
        c = evaluate(fn, qs, pool_k)
        line = f"{name:30s} {a:7.4f} {b:7.4f}"
        if syn:
            line += f" {evaluate(fn, syn, pool_k)['mrr']:8.4f}"
        f20 = [x for x in c['fails'] if not x.startswith('S')]
        shown = ', '.join(f20[:6]) + ('…' if len(f20) > 6 else '')
        line += (f" {c['mrr']:8.4f} {c['r1']:6.3f} {c['r3']:6.3f} {c['rpool']:6.3f}"
                 f"   {shown or '없음'}")
        print(line)


if __name__ == '__main__':
    qs = load_questions()
    pub = sum(1 for q in qs if q[0] == '공개'); trap = sum(1 for q in qs if q[0] == '함정')
    syn = sum(1 for q in qs if q[0] == '합성')
    print(f"코퍼스 {N}개 조항 / 서브청크 {len(SUB)}개")
    print(f"문항: 공개 {pub} + 함정 {trap}" + (f" + 합성 {syn}" if syn else "") + f" = {len(qs)}")
    print(f"임베딩: {EMB_MODEL}")

    _init_dense(qs)

    report('[1단계] 리랭커 없이 — 검색기 자체 성능 (R@30 = 리랭커에 정답이 전달될 확률)',
           STAGE1, qs)

    if NO_RERANK:
        sys.exit(0)

    # ---- 2단계: 1단계 pool → 리랭커 -------------------------------------------------
    # 리랭커 호출은 비싸므로 (질문, pool) 단위로 캐시한다.
    _rr_cache = {}

    def make_stage2(stage1_fn, pool_k):
        def f(q):
            pool = stage1_fn(q)[:pool_k]
            key = (q, tuple(sorted(pool)))
            if key not in _rr_cache:
                _rr_cache[key] = rerank(q, pool)
            order, _ = _rr_cache[key]
            return order + [i for i in stage1_fn(q) if i not in set(pool)]
        return f

    def make_mmr(stage1_fn, pool_k, lam):
        base = make_stage2(stage1_fn, pool_k)
        return lambda q: mmr_reorder(q, base(q), lam=lam)

    def make_select(stage1_fn, pool_k):
        """1단계 → 리랭커 → 노트북 컷오프까지 태운 최종 선택 집합"""
        def f(q):
            pool = stage1_fn(q)[:pool_k]
            key = (q, tuple(sorted(pool)))
            if key not in _rr_cache:
                _rr_cache[key] = rerank(q, pool)
            order, fused = _rr_cache[key]
            return select_final(order, fused)
        return f

    t0 = time.time()

    if STAGE2_FULL:
        # 190문항 전체를 리랭커까지 태운다. 공개10+함정10은 이미 포화(MRR 1.0)라
        # 방식 간 차이가 안 보이므로, 합성 170문항까지 넣어야 판별이 된다.
        arms = [
            ('③ BM25 RRF pool=30 ← 현재',    R_bm25_rrf, 30),
            ('⑦ 하이브리드(+dense) pool=30',   R_hybrid2, 30),
            ('· BM25 RRF pool=72 (풀링없음)', R_bm25_rrf, N),
        ]
        report('[2단계·190문항] 1단계 pool → 리랭커',
               [(nm, make_stage2(fn, k)) for nm, fn, k in arms], qs)

        print(f"\n{'='*104}\n[최종 선택 집합] 리랭커+컷오프까지 태운 뒤 실제로 생성기에 넘어가는 조항\n{'='*104}")
        print(f"{'방식':30s} {'gold회수율':>9s} {'평균조항수':>9s} {'무관조항/문항':>11s}   미회수 문항")
        print('-' * 104)
        for nm, fn, k in arms:
            r = evaluate_selection(make_select(fn, k), qs)
            m = ', '.join(r['misses'][:6]) + ('…' if len(r['misses']) > 6 else '')
            print(f"{nm:30s} {r['cov']:9.4f} {r['avg_k']:9.2f} {r['noise']:11.2f}   {m or '없음'}")
        print(f"\n(리랭커 총 {time.time()-t0:.0f}s)")
        sys.exit(0)

    stage2 = [
        ('③ BM25 RRF → 리랭커 ← 현재',   make_stage2(R_bm25_rrf, 30)),
        ('④ dense 조전체 → 리랭커',      make_stage2(R_dense_whole, 30)),
        ('⑦ 하이브리드3 → 리랭커',       make_stage2(R_hybrid2, 30)),
        ('⑧ 하이브리드4 → 리랭커',       make_stage2(R_hybrid4, 30)),
        ('· 풀링 없음(72 전체) → 리랭커', make_stage2(lambda q: list(range(N)), N)),
    ]
    report('[2단계] 1단계 pool=30 → 리랭커 2채널 RRF (노트북 _retrieve와 동일)', stage2, qs)

    report('[MMR] 리랭커 결과를 dense 임베딩 MMR로 상위 4개 다양성 재선택',
           [('현재(③→리랭커, MMR 없음)', make_stage2(R_bm25_rrf, 30)),
            ('MMR λ=0.9 (다양성 약)',   make_mmr(R_bm25_rrf, 30, 0.9)),
            ('MMR λ=0.7',              make_mmr(R_bm25_rrf, 30, 0.7)),
            ('MMR λ=0.5 (다양성 강)',   make_mmr(R_bm25_rrf, 30, 0.5))], qs)

    print(f"\n(리랭커 총 {time.time()-t0:.0f}s)")
