# Stage 1: 인코더-디코더 트랜스포머 직접 구현

번역용 인코더-디코더 트랜스포머를 PyTorch로 **바닥부터** 구현하고, 토이 데이터 overfit으로
학습 파이프라인이 올바른지 검증한 단계의 기록.

## 무엇을 했는가

*Attention is All You Need*(2017)의 인코더-디코더(seq2seq) 구조를 외부 트랜스포머 모듈
(`nn.Transformer` 등) 없이 직접 구현했다. 멀티헤드 어텐션·크로스어텐션·마스킹·위치 인코딩까지
모두 손으로 작성. 구성 요소:

| 파일 | 내용 |
|---|---|
| `src/tokenizer.py` | 한·영 공용 ByteLevel BPE (특수토큰 `<pad><bos><eos><unk>`) |
| `src/model.py` | `MultiHeadAttention`, `FeedForward`, `Encoder/DecoderLayer`, `Seq2SeqTransformer` |
| `src/data.py` | 병렬 코퍼스 로딩, teacher forcing 배치, 패딩 |
| `src/train.py` | CE 손실 + AdamW + 워밍업 스케줄 + bf16 autocast |
| `src/translate.py` | greedy 디코딩 |
| `tests/test_model.py`, `tests/check_overfit.py` | 단위 테스트, overfit 검증 |

## 설계 결정과 이유

| 결정 | 이유 |
|---|---|
| **인코더-디코더** (vs 디코더-온리) | 번역이 1순위. 입력 전체를 인코더가 이해 후 디코더가 생성하는 구조가 소형 모델 번역에 유리. |
| **Pre-LayerNorm** (잔차 경로 앞 정규화) | from-scratch 학습 안정성. Post-LN보다 워밍업 의존도가 낮고 그래디언트가 잘 흐름. |
| **사인/코사인 위치 인코딩** | 학습 파라미터가 없어 임의 길이에 일반화. 구현이 단순. |
| **ByteLevel BPE (공용 vocab)** | 한국어+영어를 한 토크나이저로. 바이트 단위라 OOV가 없고, 단일 vocab이라 임베딩 공유가 자연스러움. |
| **Weight tying** (입력 임베딩 = 출력 투영) | 단일 vocab 전제에서 파라미터 절감 + 일반화 도움. |
| **임베딩 √d_model 스케일** | 원 논문 관례. 임베딩과 위치 인코딩의 크기 균형. |

마스킹은 어텐션 점수에 더하는 **가산 마스크**(차단 위치 `-inf`)로 통일했다:
- 인코더/크로스어텐션: 소스 패딩 마스크
- 디코더 셀프어텐션: 상삼각 causal 마스크 + 타깃 패딩 마스크

## 검증 결과 (정량적)

### 1) 단위 테스트 (`tests/test_model.py`, CPU)

| 항목 | 결과 |
|---|---|
| forward 로짓 shape `(B, L_tgt, vocab)` | 통과 |
| **causal mask 누출** — 미래 토큰 변경 시 과거 위치 로짓 변화 | **0.00e+00** (완전 차단) |
| 패딩 입력 forward | NaN/Inf 없음 |
| weight tying (동일 텐서 공유) | 확인 |

### 2) 토이 overfit (GPU, RX 7800 XT)

- 데이터: 한-영 문장 쌍 60개 (`data/toy_parallel.tsv`)
- 모델: d_model=256, enc/dec 3층, heads=4, d_ff=1024 → **5.67M 파라미터**
- 설정: dropout=0, 400 epoch, AdamW, bf16

| epoch | loss | perplexity |
|---|---|---|
| 1 | 6.5220 | 679.94 |
| 50 | 0.0122 | 1.01 |
| 100 | 0.0014 | 1.00 |
| 400 | **0.0000** | **1.00** |

- **학습셋 exact-match: 60/60 = 100.0%** (greedy 번역이 정답과 문자열 완전 일치)
- 예: `안녕하세요 → Hello`, `김치는 맵지만 맛있어요 → Kimchi is spicy but delicious`

> overfit이 깔끔하게 도달(loss→0, 100% 재현)했다는 것은 **순전파·역전파·마스킹·손실 계산이
> 모두 정확**하다는 강한 신호다. 일반화 성능은 다른 문제이며 Stage 2(실코퍼스)에서 다룬다.

## 한계와 다음 단계 (Stage 2)

- 현재는 **ko→en 단방향**, 토이 데이터 암기 수준. 일반화 평가 없음(목적이 구조 검증이라 의도된 범위).
- Stage 2 계획: 한-영 **병렬 코퍼스** 학습, **양방향(한↔영)** — 소스에 방향 태그 추가,
  대화 데이터, 샘플링(top-k/p·temperature), 긴 문서 **청킹 파이프라인**, 검증셋 BLEU 등 지표.
