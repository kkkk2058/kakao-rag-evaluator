# -*- coding: utf-8 -*-
import json, re, os

BASE = "/Users/sehoonkim/kakao-rag-evaluator/약관원문_확정"

CHAP_RE = re.compile(r'^제\s*(\d+)\s*장\s*(.*)$')
ART_RE = re.compile(r'^제\s*(\d+)\s*조\s*(?:\(([^)]*)\))?\s*$')

FOOTER_MARKERS = (
    "공고일자", "시행일자", "변경 전", "서비스 이용정보", "이용약관", "위치정보 이용약관",
    "개인정보 처리방침", "운영정책", "청소년보호정책", "브랜드보호정책", "권리침해신고안내",
    "공지사항", "새로운 알림이 있습니다", "Copyright", "맨위로", "다른 약관 보기",
    "카카오 서비스 약관", "카카오 통합서비스약관", "카카오계정 약관",
)

def is_footer(line):
    return any(line.startswith(m) for m in FOOTER_MARKERS)

def parse_html_derived(path, doc_name):
    lines = [l.rstrip('\n') for l in open(path, encoding='utf-8')]

    # locate body start = second occurrence of article "제 1 조" header (TOC lists it once, body repeats it)
    first_idx = second_idx = None
    for i, l in enumerate(lines):
        m = ART_RE.match(l)
        if m and m.group(1) == '1':
            if first_idx is None:
                first_idx = i
            else:
                second_idx = i
                break
    body_start = second_idx if second_idx is not None else first_idx
    assert body_start is not None, f"could not locate body start for {doc_name}"

    articles = []
    current = None
    for l in lines[body_start:]:
        am = ART_RE.match(l)
        cm = CHAP_RE.match(l)
        if am:
            if current:
                articles.append(current)
            current = {"doc": doc_name, "article_no": int(am.group(1)),
                       "title": (am.group(2) or "").strip(), "text_lines": []}
            continue
        if cm:
            continue  # chapter heading, not content
        if current is None:
            continue
        if is_footer(l):
            # once footer/nav content starts, stop collecting further lines entirely
            break
        current["text_lines"].append(l)
    if current:
        articles.append(current)

    for a in articles:
        a["text"] = "\n".join(a.pop("text_lines")).strip()
    return articles


# ---- PDF-derived 카카오 통합 약관 (2022-08-25) ----
INTEGRATED_2022_TITLES = {
    1: "목적", 2: "약관의 명시, 효력 및 변경", 3: "약관 외 준칙",
    4: "카카오계정 또는 Daum 아이디 생성 및 연결",
    5: "카카오계정 또는 Daum 아이디 생성 거절 및 유보",
    6: "카카오계정 또는 Daum 아이디 등의 관리",
    7: "다양한 서비스 제공 및 변경 등", 8: "서비스 이용 방법 및 주의점",
    9: "게시물의 관리", 10: "권리의 귀속 및 저작물의 이용",
    11: "유료 서비스의 이용", 12: "게시판 이용 상거래",
    13: "서비스의 이용, 변경 및 종료", 14: "이용계약 해지",
    15: "개인정보의 보호", 16: "위치기반서비스 제공", 17: "인증서비스",
    18: "손해배상 등", 19: "청소년보호", 20: "통지 및 공지", 21: "분쟁의 해결",
}

def parse_integrated_2022(path, doc_name, max_article=21):
    text = open(path, encoding='utf-8').read()
    text = text.replace('\x00', ' ')  # PDF extraction artifact, not real whitespace to re.sub(r'\s+')
    text = re.sub(r'\s+', ' ', text)  # collapse all whitespace/newlines

    marker_re = re.compile(r'제\s*(\d+)\s*조')
    expected = 1
    boundaries = []  # (article_no, match_start, match_end)
    for m in marker_re.finditer(text):
        if expected > max_article:
            break
        if int(m.group(1)) == expected:
            boundaries.append((expected, m.start(), m.end()))
            expected += 1

    assert len(boundaries) == max_article, f"expected {max_article} articles, found {len(boundaries)}: {boundaries}"

    # chapter headings always sit at the tail of a segment (right before the next article begins)
    chap_re = re.compile(r'제\s*\d+\s*장.*$')

    articles = []
    for idx, (art_no, start, end) in enumerate(boundaries):
        content_start = end
        content_end = boundaries[idx + 1][1] if idx + 1 < len(boundaries) else len(text)
        raw = text[content_start:content_end].strip()
        # strip the leading title text (already captured separately) from the content
        title = INTEGRATED_2022_TITLES[art_no]
        title_norm = re.sub(r'\s+', '', title)
        # find where title ends in raw (whitespace-insensitive) and cut it off
        stripped = re.sub(r'\s+', '', raw)
        if stripped.startswith(title_norm):
            # walk raw forward char-by-char counting non-space chars consumed
            consumed = 0
            i = 0
            while i < len(raw) and consumed < len(title_norm):
                if not raw[i].isspace():
                    consumed += 1
                i += 1
            raw = raw[i:].strip()
        # drop any chapter heading that bled into this article's tail (chapter markers sit between articles)
        raw = chap_re.sub(' ', raw).strip()
        # trim trailing footer (공고일자/시행일자/문의사항) from the very last article
        raw = re.split(r'공고일자\s*:', raw)[0].strip()
        raw = re.sub(r'\s+', ' ', raw).strip()
        raw = re.sub(r'[•\s]+$', '', raw).strip()
        articles.append({
            "doc": doc_name, "article_no": art_no,
            "title": title, "text": raw,
        })
    return articles


def main():
    all_articles = []
    all_articles += parse_html_derived(
        os.path.join(BASE, "카카오계정약관_20260529.txt"), "카카오계정 약관")
    all_articles += parse_html_derived(
        os.path.join(BASE, "카카오위치정보이용약관_20260716.txt"), "카카오 위치정보 이용약관")
    all_articles += parse_html_derived(
        os.path.join(BASE, "카카오통합서비스약관_20260529.txt"), "카카오 통합서비스약관")
    all_articles += parse_integrated_2022(
        os.path.join(BASE, "카카오통합약관_20220825_아카이브.txt"), "카카오 통합 약관")

    out_path = os.path.join(BASE, "articles.json")
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(all_articles, f, ensure_ascii=False, indent=2)

    from collections import Counter
    c = Counter(a["doc"] for a in all_articles)
    print(f"TOTAL articles: {len(all_articles)}")
    for doc, n in c.items():
        print(f"  {doc}: {n}")
    print(f"saved -> {out_path}")
    return all_articles

if __name__ == "__main__":
    main()
