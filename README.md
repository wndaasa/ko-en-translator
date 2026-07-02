# 한 <-> 영 번역 Transformer

PyTorch로 Transformer를 **직접 구현**하는 프로젝트입니다.
주 목적은 **한국어 <-> 영어 번역 전용 소형 모델**이며, 부차적으로 간단한 대화도 가능하도록 합니다.

> 이 프로젝트의 모델은 "작고 빠른 번역기"에 초점을 둡니다.

## 목표와 범위

- **1순위: 한↔영 번역.** 작은 크기에서 번역 품질을 우선.
- **2순위: 간단한 대화.** 짧은 멀티턴 수준.
- 비목표: RAG, 범용 챗봇.

긴 문서(예: 50페이지 논문) 번역은 초기(Stage 2)에는 **문장 단위 청킹**(분할→번역→재조립)으로
처리했으나, 문장별 번역은 용어·대명사·문체가 문장마다 흔들리는 한계가 있었습니다. 현재는
**여러 문장을 문맥 윈도우로 묶어 통째로 번역하는 문서 모드**(Stage 6)가 기본입니다 — 어텐션의
$O(n^2)$ 비용 대신 선형 순환(minRNN, $O(n)$)을 채택해 소형 모델로도 긴 컨텍스트를 감당합니다.

## 아키텍처 결정: 인코더-디코더 seq2seq

번역이 1순위이므로 *Attention is All You Need*(2017)의 **인코더-디코더(seq2seq)** 구조를 채택합니다.

| 구분 | 인코더-디코더 (채택) | 디코더-온리 (GPT) |
|---|---|---|
| 번역 적합성 | 높음 — 입력 전체를 인코더가 이해 후 디코더가 생성 | 보통 — 조건부 생성(프롬프트)으로 학습 |
| 소형 모델 번역 품질 | 유리 (파라미터 효율적) | 상대적으로 불리 |
| 학습 효과 | 인코더/디코더 셀프어텐션 + **크로스어텐션** 전부 구현 | 단일 스택 |
| 대화 처리 | seq2seq(입력→응답) | 더 자연스러움 |

대화는 seq2seq 형태(입력→응답, T5/mT5 방식)로 처리합니다.

> **현재 주력은 minRNN**(Stage 4~)입니다. 인코더-디코더 구조는 유지하되 어텐션을 **minGRU 선형
> 순환**으로 대체 — 학습은 병렬 스캔으로 병렬화되고, 디코딩은 스텝당 $O(1)$·메모리 $O(L)$입니다.
> 동일 파라미터에서 트랜스포머와 동급 이상 품질을 확인했고(상세: [docs/stage4-minrnn.md](docs/stage4-minrnn.md)),
> 긴 문맥일수록 격차가 벌어져 문서 모드(Stage 6)의 토대가 됐습니다. positional embedding이 없어
> (순환이 위치를 담당) max_len을 바꿔도 가중치를 그대로 이어 쓸 수 있습니다.

## 개발 환경

| 항목 | 내용 |
|---|---|
| GPU | NVIDIA RTX 4090 (24GB) — RunPod 클라우드 |
| 백엔드 | Linux 컨테이너 + CUDA 12.8 |
| 프레임워크 | PyTorch 2.8.0 (cu128), Python 3.12 |

> Stage 0~5는 로컬 AMD RX 7800 XT(16GB, WSL2 + ROCm)에서 진행했고, 대용량 학습이 필요한
> Stage 6부터 클라우드로 이전했습니다. 구 로컬 환경의 구축 과정과 트러블슈팅(ROCDXG,
> `HSA_ENABLE_DXG_DETECTION` 등)은 [docs/environment-setup.md](docs/environment-setup.md)에
> 기록으로 남겨 두었습니다 — 현재 환경에서는 표준 CUDA 휠이면 충분하고 별도 설정이 없습니다.

## 진행 상태

- [x] **Stage 0 — 환경 셋업** (완료): WSL2 + ROCm + ROCDXG, GPU 학습 동작 검증
- [x] **Stage 1 — 트랜스포머 직접 구현** (완료): 한·영 공용 BPE → 인코더-디코더 모델 → 학습 루프 → 토이 overfit 검증 (loss 6.52→0.00, 학습셋 exact-match 100%). 상세: [docs/stage1-implementation.md](docs/stage1-implementation.md)
- [x] **Stage 2 — 실코퍼스 번역 학습** (완료): OPUS-100 양방향 기반 학습 + AI Hub 기술과학 논문 파인튜닝. ko→en BLEU 7.36(일반) → **44.10(학술)**. 긴 문단은 문장 청킹으로 번역. 상세: [docs/stage2-translation.md](docs/stage2-translation.md)
  - 산출물: `runs/base/best.pt`(일반·대화), `runs/finetune/best.pt`(논문 특화). 학술 파인튜닝은 catastrophic forgetting으로 대화 능력을 잃어 **2-모델 구성**으로 사용.
- [x] **Stage 3 — MCE (형태소 합성 인코더)** (완료, **음성 결과**): 플랫 BPE 대신 문자(음절)→단어 CharCNN 합성 인코더를 직접 구현해 베이스라인과 동일 조건으로 비교. 동일 step에서 MCE가 val_loss·BLEU 모두 일관되게 뒤지고 학습은 ~5배 느림 → 가설 미지지(BPE가 강한 베이스라인). "가설→공정한 실험→분석"의 정직한 음성 결과. 상세: [docs/roadmap-stage3-mce.md](docs/roadmap-stage3-mce.md)
- [x] **Stage 4 — minRNN (선형 순환 seq2seq)** (완료, **양성 결과**): 어텐션을 minGRU 선형 순환으로 대체(학습 병렬·추론 O(1)). 동일 파라미터(60.5M)에서 **품질은 동일 step 기준 트랜스포머를 앞서고**, 디코딩 길이 L=4096에서 **메모리 ~2.5배 적고 속도 ~2.2배 빠름**(O(L) vs O(L²)). 상세: [docs/stage4-minrnn.md](docs/stage4-minrnn.md)
- [x] **Stage 5 — 도메인 혼합 학습** (완료, **양성 결과**): 일상(OPUS)+격식(AI Hub) 균형 혼합 + 도메인 태그(`<casual>`/`<formal>`)로 minRNN 학습, 도메인별 BLEU 측정. 격식 추가 후에도 일상체가 단일도메인보다 오히려 높음(positive transfer) → **작은 모델이 두 레지스터를 망각 없이 공존**. 상세: [docs/stage5-domain-mix.md](docs/stage5-domain-mix.md)
- [x] **Stage 6 — 문서/문맥 단위 번역** (완료, **양성 결과**): 앞뒤 문장을 `<eos>`로 이어붙인 문맥 결합(context-concatenation, k-to-k)으로 문서 단위 번역. minRNN 268M 2단계 학습 — 문장 사전학습(OPUS 8M+, val_loss 2.537) → 문맥 파인튜닝. 파이프라인을 TED2020으로 검증 후 **본 타깃인 논문 도메인(AI Hub 기술과학 60,483편, 179K 윈도우)** 학습: val_loss **0.906**/ppl 2.5, 문서모드 ko→en BLEU **35.46**, 의학·공학 전문용어 정확 번역(실사용 수준). 긴 컨텍스트에서 minRNN이 트랜스포머 대비 우위(디코딩 L=4096 메모리 2.65배·속도 2.9배, 학습 메모리 10.6GB vs 16.3GB) → **소형 GPU 한 장으로 문서 번역 학습 가능**. 상세: [docs/stage6-context.md](docs/stage6-context.md)

> **데이터·가중치는 저장소에 포함하지 않습니다.** 코퍼스는 대용량·라이선스(AI Hub 재배포 금지) 이슈로 `prepare_data.py`/`prepare_aihub.py`로 재생성하며, 학습 가중치(`runs/`)는 추후 Hugging Face로 별도 배포 예정입니다. 데이터 준비 명령은 [CLAUDE.md](CLAUDE.md) 참고.
>
> **출처 명시**: 이 프로젝트는 [AI허브(aihub.or.kr)](https://www.aihub.or.kr)의 「한국어-영어 번역 말뭉치(기술과학)」를 활용하며, 해당 AI데이터는 **한국지능정보사회진흥원(NIA)의 사업결과**입니다. 이 데이터로 학습된 모델(2차적 저작물)에도 동일하게 이를 밝힙니다.

## 환경 재현

CUDA GPU가 있는 Linux면 표준 PyTorch 설치로 충분합니다:

```bash
pip install torch tokenizers sacrebleu
```

> 구 로컬 환경(WSL2 + ROCm, AMD GPU)을 재현하려면 `scripts/00~02` 셋업 스크립트와
> [docs/environment-setup.md](docs/environment-setup.md)를 참고하세요. 이 경우 GPU 실행 시
> `HSA_ENABLE_DXG_DETECTION=1`이 필요합니다.

## 사용법

```bash
# 학습 (토이 병렬셋 overfit — Stage 1)
python -m src.train --dropout 0.0 --epochs 400

# 번역
python -m src.translate --text "안녕하세요"

# 단위 테스트 / overfit 검증
python -m tests.test_model
python -m tests.check_overfit
```

## 프로젝트 구조

```
Transformer/
├─ README.md
├─ docs/
│  ├─ environment-setup.md      # 환경 구축 + 트러블슈팅 기록
│  ├─ stage1-implementation.md  # Stage 1 구현/결정/결과
│  ├─ stage2-translation.md     # Stage 2 실코퍼스 학습/파인튜닝/결과
│  ├─ roadmap-stage3-mce.md     # Stage 3 MCE 설계/실험/음성 결과
│  ├─ stage4-minrnn.md          # Stage 4 minRNN 구현/비교 결과
│  ├─ stage5-domain-mix.md      # Stage 5 도메인 혼합 학습/평가
│  └─ stage6-context.md         # Stage 6 문서/문맥 단위 번역
├─ scripts/                     # 환경 셋업 스크립트 (00~02)
├─ src/
│  ├─ tokenizer.py              # 한·영 공용 ByteLevel BPE (방향 태그 포함)
│  ├─ model.py                  # 인코더-디코더 트랜스포머 (from scratch)
│  ├─ data.py                   # 병렬 코퍼스 로딩/배치 (양방향 MTDataset, 문서모드 지원)
│  ├─ prepare_data.py           # OPUS-100 다운로드/정제
│  ├─ prepare_aihub.py          # AI Hub 기술과학 CSV 정제
│  ├─ prepare_mixed.py          # (Stage 5) 도메인 혼합 코퍼스 생성
│  ├─ prepare_large.py          # (Stage 6) OPUS moses 대용량 코퍼스 다운로드/정제/중복제거
│  ├─ prepare_context.py        # (Stage 6) 문서 경계 보존 문맥 윈도우 생성 (moses/CSV)
│  ├─ char_tokenizer.py         # (Stage 3) 문자(음절) 토크나이저
│  ├─ char_encoder.py           # (Stage 3) CharCNN 합성기 + MCE 트랜스포머
│  ├─ minrnn.py                 # (Stage 4) minGRU 선형순환 seq2seq
│  ├─ train.py                  # 학습 루프 (resume·init-from·--arch bpe/mce/minrnn·--context)
│  └─ translate.py              # greedy 디코딩 번역 (bpe/mce/minrnn, 문서모드)
├─ tests/
│  ├─ test_model.py             # shape/causality 단위 테스트
│  ├─ test_mce.py               # MCE 구조 단위 테스트
│  ├─ test_minrnn.py            # minRNN 구조 단위 테스트
│  ├─ benchmark_efficiency.py   # 디코딩 메모리/속도 벤치 (Transformer vs minRNN)
│  └─ check_overfit.py          # 토이 overfit exact-match 검증
├─ data/
│  ├─ toy_parallel.tsv          # 한-영 토이 병렬 코퍼스 (Stage 1)
│  ├─ processed/                # OPUS·AI Hub 정제 TSV (생성물)
│  └─ raw_opus/                 # OPUS moses 원본 (Stage 6, 생성물)
├─ artifacts/                   # Stage 1 토이 산출물
└─ runs/                        # 학습 산출물 — base/finetune(Stage 2), big/ctx/ctx_paper(Stage 6)
```
