# 공개 10문항의 제작 방식을 모방한 합성 평가셋 생성기
# 방식: 원문 문장에서 핵심 요소(수치/열거/절차)를 마스킹해 의문사로 치환한다.
import json, re, unicodedata, random

ROOT='/Users/sehoonkim/kakao-rag-evaluator/'
ART=json.load(open(ROOT+'약관원문_확정/articles.json'))
DOCS=("카카오계정 약관","카카오 위치정보 이용약관","카카오 통합서비스약관","카카오 통합 약관")
random.seed(20260812)

def sentences(text):
    out=[]
    for line in re.split(r"\n+",text):
        for s in re.split(r"(?<=[.!?])\s+",line):
            s=s.strip()
            if len(s)>=20: out.append(s)
    return out
def subchunks(text):
    p=[x.strip() for x in re.split(r"(?=[①②③④⑤⑥⑦⑧⑨⑩])|(?=(?<!\d)\d+\.[^\d])",text) if x.strip()]
    return p or [text]
def clean(s):
    s=re.sub(r"^[①-⑮]\s*","",s); s=re.sub(r"^\d+[.)]\s*","",s)
    return s.strip()

UNIT={'시간':'몇 시간','개월':'몇 개월','일':'며칠','년':'몇 년','명':'몇 명',
      '세':'몇 세','인':'몇 명','가지':'몇 가지','개':'몇 개','회':'몇 회'}

def q_numeric(art, sents):
    """수치를 '몇 ~'로 마스킹한 질문 (P01·P02·P05형)"""
    for s in sents:
        m=re.search(r"(\d+)\s*(시간|개월|년|명|세|인|가지|회)(?!\w)", s)
        if not m: continue
        core=clean(s)
        masked=core[:m.start()+ (len(core)-len(s))] if False else re.sub(
            r"(\d+)\s*("+m.group(2)+r")(?!\w)", UNIT[m.group(2)], clean(s), count=1)
        stem=masked[:150].rstrip(' .')
        stem=re.sub(r"(합니다|됩니다|드립니다|입니다|하겠습니다|드리겠습니다)$","",stem).rstrip(' ,')
        return f"{stem}나요?", [s]
    return None

def q_enumerate(art, subs):
    """N개 항목 열거 질문 (P03형)"""
    items=[c for c in subs if re.match(r"^(?:[①-⑮]|\d+[.)])", c)]
    if len(items)<3: return None
    n=len(items)
    return (f"{art['title']}에서 정하고 있는 {n}가지 항목은 각각 무엇인가요?",
            [clean(i) for i in items[:min(n,6)]])

def q_procedure(art, sents):
    """절차·효력을 묻는 질문 (P04·P06·P09형)"""
    for s in sents:
        if not re.search(r"(하여야|해야|합니다|됩니다|따릅니다|존속|통지|공지|제출)", s): continue
        core=clean(s)
        head=core[:40]
        return f"{art['title']}과 관련하여, {head}… 이후에는 어떻게 되나요?", [s]
    return None

def q_trap(art, sents):
    """전제 뒤집기 함정 질문 (T계열형) — '예외 없이 항상' 단정을 붙인다"""
    for s in sents:
        if not re.search(r"(다만|단,|제외|아니|않습니다|경우에는)", s): continue
        core=clean(s); head=core[:45]
        return f"{head}… 관련하여, 예외 없이 항상 그렇게 적용되나요?", [s]
    return None

def build():
    out=[]; qid=0
    for a in ART:
        sents=sentences(a['text']); subs=subchunks(a['text'])
        if not sents: continue
        for kind,fn,arg in (('numeric',q_numeric,sents),('enumerate',q_enumerate,subs),
                            ('procedure',q_procedure,sents),('trap',q_trap,sents)):
            r=fn(a,arg)
            if not r: continue
            q,kf=r
            if len(q)<25 or len(q)>140: continue
            # 문서명 명시 30% (공개셋 실측)
            if random.random()<0.30: q=f"{a['doc']}에 따르면 {q}"
            qid+=1
            out.append({"id":f"S{qid:03d}","ptype":kind,"question":q,
                        "gold_articles":[{"doc":a['doc'],"article":a['article_no'],
                                          "citation":f"{a['doc']} 제{a['article_no']}조"}],
                        "key_facts":kf})
    return out

if __name__=='__main__':
    import statistics as st
    qs=build()
    json.dump({"questions":qs}, open('/private/tmp/claude-501/-Users-sehoonkim-kakao-rag-evaluator/e3c7ddb1-acc0-4319-93ef-b00cffa999c8/scratchpad/evalset_synth.json','w'), ensure_ascii=False, indent=1)
    def bg(t):
        t=re.sub(r"\s+","",unicodedata.normalize("NFC",t)); return {t[i:i+2] for i in range(len(t)-1)}
    BY={(x['doc'].replace(' ',''),x['article_no']):x for x in ART}
    cov=[]; named=0
    for q in qs:
        g=q['gold_articles'][0]; art=BY[(g['doc'].replace(' ',''),g['article'])]
        cov.append(len(bg(q['question'])&bg(art['text']))/max(1,len(bg(q['question']))))
        named+= any(d in q['question'] for d in DOCS)
    from collections import Counter
    print("생성 문항:",len(qs),"| 유형",dict(Counter(q['ptype'] for q in qs)))
    print(f"커버 대상 조항 {len({(q['gold_articles'][0]['doc'],q['gold_articles'][0]['article']) for q in qs})}/72")
    print(f"bigram커버 평균 {st.mean(cov):.3f} (공개셋 0.594) | 문서명 명시 {named/len(qs):.0%} (공개셋 30%)")
    print(f"질문 길이 평균 {st.mean([len(q['question']) for q in qs]):.0f}자 (공개셋 81자)")
    print("\n샘플:")
    for q in random.sample(qs,6): print(f"  [{q['ptype']:9s}] {q['question'][:96]}")
