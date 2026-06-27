# Stage 2: 실코퍼스 번역 학습 (한↔영, 논문 도메인)

Stage 1에서 검증한 인코더-디코더 모델을 실제 병렬 코퍼스로 학습해 **쓸 수 있는 번역기**로 키운 단계.
번역이 1순위(특히 **ko→en 논문 도메인**)이고 간단한 대화도 목표였다.

## 데이터

| 코퍼스 | 용도 | 규모(정제 후) | 출처/성격 |
|---|---|---|---|
| OPUS-100 (en-ko) | 기반(일반/대화) | 704,745쌍 (원본 1M, 70.5% 유지) | 자막·웹, 노이즈 있음 |
| AI Hub 기술과학 한영 | 논문 파인튜닝 | 1,195,187쌍 (99.99% 유지) | 한국학술정보 학술논문, 문어체 |

- 정제 규칙: 빈 문장·미번역(양쪽 동일)·언어 불일치·길이비 이상·괄호 효과음 제거 + 중복 제거.
- OPUS는 노이즈가 많아 30%가 걸러졌고, AI Hub는 큐레이션 데이터라 거의 그대로 유지됨.

## 모델·학습 구성

- **토크나이저**: 한·영 공용 ByteLevel BPE, **vocab 32,000** (학술 용어도 OOV 없이 인코딩).
- **모델**: Transformer-base 인코더-디코더, **60.5M 파라미터** (d_model=512, 6+6층, 8 head, d_ff=2048).
- **양방향**: 소스 선두에 방향 태그(`<2en>`/`<2ko>`)를 붙여 한 모델로 ko→en, en→ko 모두 처리.
- **학습**: teacher forcing + 패딩 마스크 CE(label smoothing 0.1), AdamW + 워밍업·역제곱근 LR, **bf16 autocast**, 검증 손실·ko→en BLEU 모니터링, 검증손실 기준 best 체크포인트.
- **resume**: 옵티마이저·스케줄러·step까지 저장해 중단/이어하기 지원(`--resume`), 파인튜닝 시작점 지정(`--init-from`).

## 기반 모델 학습 (OPUS, 양방향)

5 epoch, 배치 96, GPU(RX 7800 XT).

| 지표 | 결과 |
|---|---|
| best val_loss | 3.860 (ppl 47.4) |
| ko→en BLEU | 0.45 → **7.36** (단조 상승) |
| 처리량 | ~1,150 ex/s |

정성: **일상 대화는 양방향 모두 자연스럽게 번역**(예: "How are you doing today?"→"오늘 어떻게 지내?"). 학술 문장은 약함.

### 트러블슈팅: 학습 중 OOM
- **증상**: 배치 128로 학습 중 step ~13,900에서 `HIP out of memory`(1.94GB 할당 시도, 596MB만 남음). 약 3.4GB가 "예약됐지만 미사용"(단편화).
- **원인**: 배치 128이 긴 문장 배치 + 메모리 단편화에서 16GB를 초과.
- **시도**: `PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True` → **ROCm/HIP는 미지원**(경고 후 무시됨).
- **해결**: 배치 96으로 낮춰 ~3GB 마진 확보 → 완주. (`python | tee` 파이프라인이 종료코드를 가리는 문제는 `set -o pipefail`로 수정해 실패를 정확히 감지하도록 함.)

## 논문 도메인 파인튜닝 (AI Hub 기술과학)

`--init-from`으로 기반 모델에서 출발, 학술 데이터로 양방향 파인튜닝(LR 3e-4, 배치 96).

| step | val_loss | ppl | ko→en BLEU |
|---|---|---|---|
| 2,000 | 1.88 | 6.6 | 30.4 |
| 14,000 | 1.27 | 3.6 | 41.8 |
| 28,000 (중단) | 1.15 | 3.2 | **44.10** |

- BLEU **7.36 → 44.10** (학술 도메인). val_loss 3.86 → 1.15.
- 8% 학습만에 BLEU 30 돌파 → 도메인 적응이 매우 빠름. epoch 1 후반부터 ~43~44에서 평탄해져 step 28,000에서 중단.

### 핵심 발견: catastrophic forgetting (재앙적 망각)
학술 데이터만으로 파인튜닝하니 **논문 번역은 급상승했지만 일상 대화 능력을 잃음**.

| 입력(ko) | 기반 모델 | 파인튜닝 |
|---|---|---|
| 이 실험 결과는 기존 방법보다 우수한 성능을 보였다. | The results are seen in the nature of the nature... ❌ | This experimental result showed better performance than the existing method. ✅ |
| 안녕하세요, 만나서 반갑습니다. | Hello, nice to meet you. ✅ | The baby, the child, and the child are in the way. ❌ |

- **원인**: 학술 코퍼스에 일상 표현이 거의 없어, 모델이 논문 도메인으로 과특화되며 OPUS에서 배운 대화 분포를 덮어씀.
- **결정**: 이 번역기는 별도 기능(논문 번역 전용)으로 사용하므로 **논문 특화 모델로 유지**. 대화는 기반 모델(`runs/base/best.pt`)이 담당하는 **2-모델 구성**. (한 모델로 둘 다 원하면 학술+OPUS 혼합 재학습이 표준 해법이며, 향후 과제로 남김.)

## 논문 번역 검증 (분량 있는 텍스트, 청킹)

문장 단위 모델이므로 긴 글은 **문장 분할 → 각각 번역 → 재조립**으로 처리. 논문 초록(6문장) 번역 예:

> 본 연구에서는 한국어와 영어 간 번역을 위한 새로운 인코더-디코더 모델을 제안한다.
> → *This study proposes a new encoder-decoder model for translation between Korean and English.*

> 실험 결과, 제안한 모델은 기존의 통계 기반 방법보다 우수한 번역 품질을 보였다.
> → *As a result of the experiment, the proposed model showed better translational quality than the existing statistical-based method.*

- 6문장 중 5문장이 거의 출판체에 근접. 긴 복문도 절단 없이 일관 번역.
- **약점**: 드문 전문어 오역 — `말뭉치`→"distal drought", `벤치마크`→"ventmark", `추론(inference)`→"reasoning". 더 다양한 학술 데이터/큰 모델로 개선 여지.

## 산출물

| 경로 | 내용 |
|---|---|
| `runs/base/best.pt` | 기반 모델(일반·대화), ko→en BLEU 7.36 |
| `runs/finetune/best.pt` | 논문 특화 모델, ko→en 학술 BLEU 44.10 |
| `runs/*/tokenizer.json` | 공용 BPE(32k) |

## 결정과 트레이드오프

- **양방향 + 방향 태그**: 한 모델로 두 방향 → 파라미터 효율적, 우선 방향(ko→en) 학술 노출 충분.
- **배치 96**: 128은 OOM, 64는 안전하지만 VRAM 저활용 → 96이 절충(완주 + VRAM 활용).
- **2-모델 구성**: forgetting을 혼합학습으로 풀 수도 있으나, 용도가 분리(논문 전용)되어 특화 유지가 단순·효과적.
- **평가 지표**: 우선 방향인 ko→en BLEU를 핵심 지표로 모니터링(en→ko는 한국어 토큰화 이슈로 BLEU 신뢰도 낮음).
