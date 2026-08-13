# 결과기_최종.ipynb 의 BM25 계층을 순수 파이썬으로 재현 (rank_bm25.BM25Okapi 공식 그대로)
import json, math, re, unicodedata
from collections import Counter

class BM25Okapi:
    def __init__(self, corpus, k1=1.5, b=0.75, epsilon=0.25):
        self.k1, self.b, self.epsilon = k1, b, epsilon
        self.corpus_size = len(corpus)
        self.doc_len = [len(d) for d in corpus]
        self.avgdl = sum(self.doc_len)/self.corpus_size
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
        eps = self.epsilon * (idf_sum/len(self.idf))
        for w in neg: self.idf[w] = eps
    def get_scores(self, query):
        out = [0.0]*self.corpus_size
        for q in query:
            idf = self.idf.get(q, 0.0)
            if not idf: continue
            for i, df in enumerate(self.doc_freqs):
                f = df.get(q, 0)
                if not f: continue
                out[i] += idf * (f*(self.k1+1)) / (f + self.k1*(1 - self.b + self.b*self.doc_len[i]/self.avgdl))
        return out

def bigram_tokens(text):
    t = re.sub(r"\s+", "", unicodedata.normalize("NFC", text))
    if len(t) < 2: return [t] if t else []
    return [t[i:i+2] for i in range(len(t)-1)]

def word_tokens(text):                      # 비교용: 어절 토크나이즈
    return re.findall(r"\w+", unicodedata.normalize("NFC", text))

def split_subchunks(text):
    parts = re.split(r"(?=[①②③④⑤⑥⑦⑧⑨⑩])|(?=(?<!\d)\d+\.[^\d])", text)
    parts = [p.strip() for p in parts if p.strip()]
    return parts or [text]

ART = json.load(open('/Users/sehoonkim/kakao-rag-evaluator/약관원문_확정/articles.json'))
CORPUS = [f"{a['doc']} {a['title']} {a['text']}" for a in ART]
CORPUS_NODOC = [f"{a['title']} {a['text']}" for a in ART]       # 문서명 제거 ablation

SUB = []
for i, a in enumerate(ART):
    for sc in split_subchunks(a['text']):
        SUB.append((i, f"{a['doc']} {a['title']} {sc}"))

def build(corpus, tk): return BM25Okapi([tk(t) for t in corpus])

IDX = {
 'whole_bi'  : build(CORPUS, bigram_tokens),
 'whole_wd'  : build(CORPUS, word_tokens),
 'whole_nodoc': build(CORPUS_NODOC, bigram_tokens),
 'sub_bi'    : build([t for _, t in SUB], bigram_tokens),
}

def whole_rank(q, idx='whole_bi'):
    s = IDX[idx].get_scores(bigram_tokens(q) if idx!='whole_wd' else word_tokens(q))
    return sorted(range(len(ART)), key=lambda i: s[i], reverse=True)

def sub_rank(q):
    s = IDX['sub_bi'].get_scores(bigram_tokens(q))
    best = {}
    for (ai, _), v in zip(SUB, s):
        if ai not in best or v > best[ai]: best[ai] = v
    return sorted(best, key=lambda i: best[i], reverse=True)

def rrf_rank(q, k=60):
    wp = {i: r for r, i in enumerate(whole_rank(q), 1)}
    sp = {i: r for r, i in enumerate(sub_rank(q), 1)}
    fused = {i: 1.0/(k+wp[i]) + 1.0/(k+sp.get(i, len(ART)+k)) for i in wp}
    return sorted(fused, key=lambda i: fused[i], reverse=True)

def load_questions():
    qs = []
    g = json.load(open('/Users/sehoonkim/kakao-rag-evaluator/gold_questions_public10.json'))
    for q in g['questions']:
        qs.append(('공개', q['id'], q['question'], q['gold_articles']))
    t = json.load(open('/Users/sehoonkim/kakao-rag-evaluator/하네스/trap_questions.json'))
    items = t if isinstance(t, list) else next(v for v in t.values() if isinstance(v, list))
    for q in items:
        qs.append(('함정', q['id'], q['question'], q['gold_articles']))
    return qs

def norm(s): return re.sub(r"\s+", "", unicodedata.normalize("NFC", s))

def mrr(ranker, qs):
    tot = 0.0; fails = []
    for kind, qid, question, gold in qs:
        gk = {(norm(g['doc']), int(g['article'])) for g in gold}
        rank = ranker(question)
        rr = 0.0
        for pos, ai in enumerate(rank, 1):
            if (norm(ART[ai]['doc']), ART[ai]['article_no']) in gk:
                rr = 1.0/pos
                if pos > 1: fails.append(f"{qid}@{pos}")
                break
        tot += rr
    return tot/len(qs), fails

if __name__ == '__main__':
    qs = load_questions()
    pub = [q for q in qs if q[0]=='공개']; trap = [q for q in qs if q[0]=='함정']
    print(f"문항: 공개 {len(pub)} + 함정 {len(trap)} = {len(qs)}\n")
    variants = [
        ('① 조 전체 BM25 (bigram)',        lambda q: whole_rank(q)),
        ('② 항 최고점 BM25 (bigram)',      sub_rank),
        ('③ ①+② RRF 결합  ← 최종 코드',   rrf_rank),
        ('· 조 전체, 어절 토크나이즈',      lambda q: whole_rank(q,'whole_wd')),
        ('· 조 전체, 문서명 제거',          lambda q: whole_rank(q,'whole_nodoc')),
    ]
    print(f"{'방식':32s} {'공개10':>8s} {'함정10':>8s} {'종합20':>8s}   1위 실패")
    for name, fn in variants:
        a,_ = mrr(fn, pub); b,_ = mrr(fn, trap); c,f = mrr(fn, qs)
        print(f"{name:32s} {a:8.4f} {b:8.4f} {c:8.4f}   {', '.join(f) or '없음'}")
