import json, re, statistics as st
import style_sim as S

def blocks(text):
    raw=(text or '').strip()
    c=[p.strip() for p in re.split(r"(?=[①②③④⑤⑥⑦⑧⑨⑩])",raw) if p.strip()]
    if len(c)>=2 and any(re.match(r"^[①②③④⑤⑥⑦⑧⑨⑩]",p) for p in c): return c
    n=[p.strip() for p in re.split(r"(?=(?<!\d)\d+\.[^\d])",raw) if p.strip()]
    if len(n)>=2 and any(re.match(r"^\d+\.",p) for p in n): return n
    return [raw]

def lexcov(q,t):
    Q=S.bgset(q)
    return len(Q & S.bgset(t))/max(1,len(Q)) if Q else 0.0

def decompose(q):
    m=re.sub(r"\s+(그리고|또한|그렇다면|그러면)\s+"," ||| ",q)
    m=re.sub(r"(하며|이며|되고|되며|이고|하고)\s*,\s*",r"\1 ||| ",m)
    m=re.sub(r"[;；]+"," ||| ",m); m=re.sub(r"[?？]+\s+(?=\S)"," ||| ",m)
    p=[x.strip() for x in m.split("|||") if len(x.strip())>=6]
    return p if len(p)>=2 else [q]

def structural(question, art, max_blocks=4, soft=2200, ratio=0.55, gap=0.12):
    bl=blocks(art['text'])
    if len(bl)<=1: return art['text'].strip()
    queries=[question]+[p for p in decompose(question) if p!=question]
    rows=[[lexcov(q,b) for b in bl] for q in queries]
    anchors={max(range(len(bl)),key=lambda i:r[i]) for r in rows}
    strength=[max(r[i] for r in rows) for i in range(len(bl))]
    gbest=max(strength)
    sel=set(anchors); total=sum(len(bl[i]) for i in sel)
    cand={i+d for i in anchors for d in (-1,1) if 0<=i+d<len(bl)}-sel
    for i in sorted(cand,key=lambda i:strength[i],reverse=True):
        if len(sel)>=max_blocks: break
        if not (gbest<=0 or strength[i]>=gbest*ratio or (gbest-strength[i])<=gap): continue
        if total+len(bl[i])>soft: continue
        sel.add(i); total+=len(bl[i])
    return "\n".join(bl[i] for i in sorted(sel)).strip()

def art_of(q): 
    g=q['gold_articles'][0]; return S.BY[(g['doc'].replace(' ',''),int(g['article']))]

def evidence(q, lim):
    a=art_of(q)
    return a['text'] if len(a['text'])<=lim else structural(q['question'],a)

if __name__=='__main__':
    print("=== 근거를 그대로 답변으로 썼을 때의 F1 (실제 gold 10문항) ===")
    print(f"{'qid':5s} {'조길이':>6s} {'조전체':>7s} {'구조선택':>8s} {'문장oracle':>10s}  구조길이")
    A=[];B=[];C=[]
    for q in S.GOLD:
        art=art_of(q); full=art['text']; stru=structural(q['question'],art)
        orc=' '.join(S.best_units(S.sentences(full),q['key_facts']))
        a,b,c=S.f1(full,q['key_facts']),S.f1(stru,q['key_facts']),S.f1(orc,q['key_facts'])
        A.append(a);B.append(b);C.append(c)
        print(f"{q['id']:5s} {len(full):6d} {a:7.3f} {b:8.3f} {c:10.3f}  {len(stru):5d}자")
    print(f"{'평균':5s} {'':6s} {st.mean(A):7.3f} {st.mean(B):8.3f} {st.mean(C):10.3f}")

    print("\n=== FULL_ARTICLE_CHAR_LIMIT 스윕 (이하=전문 / 초과=구조선택) ===")
    print(f"{'임계값':>8s} {'평균F1':>8s} {'전문 적용':>9s}")
    res={}
    for lim in (0,150,200,300,400,600,800,1200,2000,99999):
        v=[];nf=0
        for q in S.GOLD:
            if len(art_of(q)['text'])<=lim: nf+=1
            v.append(S.f1(evidence(q,lim),q['key_facts']))
        res[lim]=st.mean(v)
        print(f"{lim:8d} {res[lim]:8.3f} {nf:7d}/10{'  ← 현재 코드' if lim==1200 else ''}")
    best=max(res,key=res.get)
    print(f"\n최적 {best} (F1 {res[best]:.3f}) · 현재 1200 (F1 {res[1200]:.3f}) · 차 {res[best]-res[1200]:+.3f} → 30점 환산 {(res[best]-res[1200])*30:+.2f}점")
