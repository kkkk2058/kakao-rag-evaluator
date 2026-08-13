# 오프라인 실험 (GPU 불필요)

`결과기_최종.ipynb`의 검색·생성 형태를 GPU 없이 로컬에서 재현·검증하는 스크립트 모음.
2026-08-12 세션에서 제작. 모든 경로는 프로젝트 루트 절대경로라 어디서 실행해도 된다.

## 파일

| 파일 | 무엇 | 표본 |
|---|---|---|
| `make_evalset.py` | 72개 조항 문장을 4규칙으로 질문화 → `evalset_synth.json` 생성 | 170문항 출력 |
| `evalset_synth.json` | 생성된 합성 평가셋 (검색 실험 전용) | 170문항 / 68조 커버 |
| `bm25_ablation.py` | BM25 검색 계층을 순수 파이썬으로 재구현, 5방식 MRR 비교 | 공개10+함정10, 170 |
| `style_sim.py` | 답변 형태별 keyfact F1 (봉우리 실험) | 공개 10문항 |
| `structural_sweep.py` | `FULL_ARTICLE_CHAR_LIMIT` 임계값 스윕 | 공개 10문항 |

## 실행

```bash
python3 make_evalset.py       # 170문항 재생성 (seed 고정, 결정론적)
python3 bm25_ablation.py      # 검색 방식 비교 (공개10+함정10)
python3 style_sim.py          # 답변 스타일별 F1 (공개10)
python3 structural_sweep.py   # 길이 임계값 스윕 (공개10)
```

## 핵심 한계 (반드시 인지)

1. **합성 170문항은 검색 실험에만 유효.** 정답 key_fact를 원문 문장 그대로 떼어 만들어서,
   F1 실험에 쓰면 "정답 문장 복사"가 자동 만점이 된다. F1 계열은 공개 10문항으로만 한다.
2. **합성셋은 실제보다 쉽다.** 최장 연속복사 26자(공개셋 15.6자). 절대 MRR이 아니라
   **방식 간 상대 비교**로만 읽는다.
3. **리랭커·생성 모델은 재현 안 됨(GPU 필요).** bm25_ablation은 BM25 계층까지만,
   style/structural은 어휘 폴백 경로로 근사한다.

## 의존

- `articles.json` (약관원문_확정/) — 72개 조항 본문
- `gold_questions_public10.json` (루트) — 공개 정답셋
- `하네스/trap_questions.json` — 자체 함정 10문항
- `answers_public_8.json` (루트) — 실제 제출 답변 (style_sim에서 대조용)
- 순수 표준 라이브러리 + 자체 BM25 구현 (rank_bm25 불필요)
