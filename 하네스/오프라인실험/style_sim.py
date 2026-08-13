# 답변 스타일별 keyfact F1 시뮬레이터 — 생성 모델 없이 "답변 형태"만 바꿔가며 측정
import json, re, unicodedata

ROOT='/Users/sehoonkim/kakao-rag-evaluator/'
ART=json.load(open(ROOT+'약관원문_확정/articles.json'))
BY={(a['doc'].replace(' ',''),a['article_no']):a for a in ART}
GOLD=json.load(open(ROOT+'gold_questions_public10.json'))['questions']

def tok(t): return re.findall(r"\w+", unicodedata.normalize("NFKC", t))
def f1(ans, facts):
    A,B=set(tok(ans)),set(tok(' '.join(facts)))
    i=len(A&B)
    if not i: return 0.0
    p,r=i/len(A),i/len(B)
    return 2*p*r/(p+r)
def bgset(t):
    t=re.sub(r"\s+","",unicodedata.normalize("NFC",t))
    return {t[i:i+2] for i in range(len(t)-1)}
def sim(a,b):
    A,B=bgset(a),bgset(b)
    return len(A&B)/max(1,len(A|B))

def split_subchunks(text):
    p=[x.strip() for x in re.split(r"(?=[①②③④⑤⑥⑦⑧⑨⑩])|(?=(?<!\d)\d+\.[^\d])",text) if x.strip()]
    return p or [text]
def sentences(text):
    out=[]
    for line in re.split(r"\n+", text):
        for s in re.split(r"(?<=[.!?])\s+", line):
            s=s.strip()
            if s: out.append(s)
    return out
CLAUSE=re.compile(r"(?<=하되|하고|하며|하나|으며|지만)[,]\s+")
def clauses(text):
    out=[]
    for s in sentences(text):
        parts=[p.strip() for p in CLAUSE.split(s) if p.strip()]
        out.extend(parts if len(parts)>1 else [s])
    return out

def best_units(units, facts):
    """각 key_fact에 가장 잘 맞는 단위를 골라 원문 순서로 반환 (oracle)"""
    chosen=set()
    for kf in facts:
        best=max(range(len(units)), key=lambda i: sim(units[i], kf))
        chosen.add(best)
    return [units[i] for i in sorted(chosen)]

def greedy_units(units, facts):
    """F1을 직접 최대화하는 greedy 선택 (상한)"""
    sel=[]; cur=0.0
    while True:
        gain=None
        for i in range(len(units)):
            if i in sel: continue
            s=f1(' '.join(units[j] for j in sorted(sel+[i])), facts)
            if s>cur+1e-9 and (gain is None or s>gain[1]): gain=(i,s)
        if gain is None: break
        sel.append(gain[0]); cur=gain[1]
    return [units[i] for i in sorted(sel)]

def styles(question, art, facts):
    text=art['text']
    subs=split_subchunks(text); sents=sentences(text); cls=clauses(text)
    kf_sent=best_units(sents,facts)
    kf_sub =best_units(subs,facts)
    return {
      '결론만(최상위 절 1개)': best_units(cls,facts[:1])[0] if cls else text,
      '질문 되풀이+결론'    : question+' '+(best_units(cls,facts[:1])[0] if cls else ''),
      '정답 문장 통째 복사'  : ' '.join(kf_sent),
      '정답 항(項) 전체 복사': ' '.join(kf_sub),
      '조 전체 복사'        : text,
      '절 greedy oracle(상한)': ' '.join(greedy_units(cls,facts)),
    }

if __name__=='__main__':
    import statistics as st
    rows={}
    for q in GOLD:
        g=q['gold_articles'][0]; art=BY[(g['doc'].replace(' ',''),int(g['article']))]
        for name,ans in styles(q['question'],art,q['key_facts']).items():
            rows.setdefault(name,[]).append(f1(ans,q['key_facts']))
    print(f"{'답변 스타일':24s} {'평균F1':>7s} {'중앙':>7s} {'최소':>7s} {'최대':>7s}")
    for name,v in rows.items():
        print(f"{name:24s} {st.mean(v):7.3f} {st.median(v):7.3f} {min(v):7.3f} {max(v):7.3f}")
    # 실제 제출 답변
    sub={x['qid']:x['answer'] for x in json.load(open(ROOT+'answers_public_8.json'))['answers']}
    act=[f1(sub[q['id']],q['key_facts']) for q in GOLD]
    print(f"{'★ 실제 제출 답변':24s} {st.mean(act):7.3f} {st.median(act):7.3f} {min(act):7.3f} {max(act):7.3f}")
