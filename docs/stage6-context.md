# Stage 6: 문서/문맥 단위 번역 — 앞뒤 문맥으로 일관성 살리기 (양성)

> 질문: 문장을 하나씩 끊어 번역하면 용어·대명사·문체가 문장마다 흔들린다. 앞뒤 문장을 **함께**
> 넣어 문맥을 주면 일관성이 살아날까? 그리고 이 긴 컨텍스트를 **소형 모델로** 감당할 수 있을까?
> **결과: 문맥 결합(context-concatenation)으로 문서 단위 번역이 동작하고**, 긴 컨텍스트에서
> minRNN이 트랜스포머 대비 메모리 우위라 24GB 한 장으로 학습이 가능했다. 본 타깃인 **논문 도메인
> (AI Hub 기술과학 60,483편)** 문맥 파인튜닝으로 ko→en 문서모드 BLEU **35.46**, ppl 2.5 달성.

2단계로 진행했다. Phase 1(문장 사전학습)로 토대를 만들고, Phase 2(문맥 파인튜닝)로 문서 단위에 적응.
Phase 2는 파이프라인을 TED2020(강연)으로 먼저 검증한 뒤, 본 타깃인 논문 코퍼스로 학습했다.

## Phase 1 — 문장 사전학습 (토대)

- **데이터**: OPUS moses(NLLB·CCMatrix·OpenSubtitles·ParaCrawl) 직접 다운로드·정제·전역 중복제거,
  train 16.99M쌍 / val 5K. `src/prepare_large.py`.
- **모델**: minRNN 268M(d_model=1024, enc/dec 8층, d_ff=4096, heads=16, vocab=32000), 양방향,
  `--compile`(로그스캔 fusion으로 학습 가속·메모리 절감), batch 64, max_len 128, warmup 8000.
- **경과**: 1차 학습이 BLEU 정체 감시로 step 130k에서 조기 종료됐으나, val_loss는 계속 하강 중이었다
  (BLEU는 300샘플 노이즈로 출렁였을 뿐). `resume`로 이어 학습해 **val_loss 2.652 → 2.537
  (ppl 14.2 → 12.6)**, ko→en BLEU ~26에서 수렴. 이 `best.pt`가 Phase 2의 시작점.

> 교훈: 조기종료 기준은 노이즈가 큰 BLEU가 아니라 val_loss로 잡아야 했다. lr 스케줄(inverse-sqrt)을
> 이어붙일 때 warmup 값이 어긋나면 lr 궤적이 틀어지므로 재개 시 동일 warmup을 명시해야 한다.

## Phase 2 — 문맥 파인튜닝 (진짜 목표)

### 데이터: 문서 경계 살리기 (다운로드 없음)

OPUS moses 파일은 **빈 줄로 문서 경계를 표시**한다(TED2020 = 3,753개 강연). 이 경계를 살려
문서 내 연속 문장을 **슬라이딩 윈도우**로 이어붙였다. `src/prepare_context.py`:

- window=45 문장, stride=22(≈50% 겹침) → 문서당 여러 문맥 청크. train **13,588** / val 213 윈도우.
- 문장 결합 구분자는 **`<eos>` 재사용**(vocab 무변경 → Phase 1 임베딩을 그대로 `init-from`).
  TSV에는 제어문자 마커로 저장하고, 학습 시 `data.py`가 문장 사이에 `<eos>`를 삽입한다.
- 형식: 소스 `= <2en> + enc(s1) <eos> enc(s2) <eos> … enc(sk) <eos>`, 타깃도 동일(k-to-k).
- 강연=격식 독백체라 논문/공문서 문체에 가장 근접. ko→en 단방향에 집중.

토큰 길이 분포(window 45): ko p50=786 / p95=990, en p50=870 / p95=1054 → **max_len 1024**에 정합.

### 학습

- `init-from` Phase 1 `best.pt`, `--context --no-bidi`, max_len 1024, `--compile`.
- **메모리**: max_len 1024는 크로스어텐션 비용이 커서(O(L·S)) batch 8은 OOM. batch **4** +
  `expandable_segments`로 15GB에 안착. lr 2e-4, warmup 500.
- **결과**: val_loss **2.140 / ppl 8.5**에서 plateau(step ~17k, ep6). best.pt 상시 저장.

> minRNN은 positional embedding이 없다(선형 순환이 위치를 담당). 그래서 max_len을 128→1024로
> 바꿔도 파라미터가 동일해 `init-from`이 그대로 로드된다 — 문장→문서 전이가 매끄러웠던 이유.

### 평가: 문서 단위 디코딩

teacher-forcing val_loss는 좋은데(2.14) 기존 BLEU가 0.03으로 무너졌다. 원인은 모델이 아니라 평가:
문장용 greedy는 **첫 `<eos>`에서 멈춰** 문서의 첫 문장만 뽑는다. 입력 문장 수(k)만큼 `<eos>`를
생성하면 멈추는 **문서 모드 디코딩**(`translate.greedy_translate_context`)으로 다시 재니:

- **문서 모드 ko→en BLEU 21.14** (val 짧은 문서 15개, k-to-k 디코딩)
- 정성: 24문장 문맥을 한 번에 받아 "visualize data"를 일관 번역하는 등 **문서 단위로 흐름 유지**.
  짧은 문서("(박수)"→"(Applause)")도 정확. 일부 오역·희귀 토큰 혼입은 남는 한계.

### 본 학습: 논문 도메인 (AI Hub 기술과학)

TED로 파이프라인을 검증한 뒤, 본 타깃인 논문 코퍼스로 학습했다. **CSV는 논문별로 섞여 있고
`file_name`=논문·`sn`=문장 순서**라, `file_name`으로 묶고 `sn`으로 정렬해 원문 흐름을 복원했다
(`prepare_context.py --csv`). 특허·의학 문장이 논리적 순서로 이어짐을 확인.

- **데이터**: 60,483편(의학·ICT·기계·전기·전자), 정제 후 113만 문장쌍 → window 12/stride 6으로
  **179,136 문맥 윈도우**(TED의 13배). 논문 문장이 짧아(~350토큰) max_len 640이면 충분, 그만큼
  batch를 12까지 키워(19GB) 학습이 3배 빠름(~99 ex/s).
- **학습**: Phase 1 문장 `best.pt`에서 `init-from`, ko→en, batch 12, lr 2e-4.
  **val_loss 0.906 / ppl 2.5**(step ~16k, ep2)에서 plateau. 논문은 정형 문어체라 TED(2.14)보다
  훨씬 낮게 수렴.
- **평가**: 문서모드 k-to-k 디코딩 **ko→en BLEU 35.46**(val 짧은 문서 20개). 정성 샘플:

  | 한국어 | 모델 출력 |
  |---|---|
  | 한 명의 환자가 부분 반응을 보였다 | one patient showed a partial response |
  | 이 화학 재활용 기술의 과정은 다음과 같다 | The process of this chemical recycling technology is as follows |
  | 꾸준한 구강 건강 관리 습관 형성이 중요하다 | It is important to form a steady oral health management habit |

  전문 용어(partial response, chemical recycling, oral health)를 정확히 번역 — 실사용 수준.
  산출물: `runs/ctx_paper/best.pt`.

## 효율 벤치: 왜 minRNN인가 (핵심 증거)

`tests/benchmark_efficiency.py`로 268M 실모델에서 디코더 forward를 길이별 측정(batch 1):

| L | Transformer mem / time | minRNN mem / time |
|---|---|---|
| 128 | 1054 MB / 2.7 ms | 1038 MB / 2.4 ms |
| 1024 | 1212 MB / 12.1 ms | 1083 MB / 9.5 ms |
| 2048 | 1640 MB / 35.0 ms | 1135 MB / 18.7 ms |
| 4096 | 3288 MB / 114 ms | 1239 MB / **39 ms** |

L=4096에서 **메모리 2.65배·속도 2.9배** 우위(O(L²) vs O(L)). L=128(minRNN 최악 조건)에선 차이가
거의 없다가 길이가 길어질수록 벌어진다 — 문서 단위 번역이 정확히 이 구간이다.

실학습 조건(max_len 1024, batch 4, forward+backward+optimizer 1스텝)에서도:

| 아키텍처 | seq | 학습 peak 메모리 |
|---|---|---|
| minRNN | 835 | **10.60 GB** |
| Transformer | 963 | 16.26 GB |

→ 같은 조건에서 트랜스포머가 1.5배 이상 쓰고, max_len을 더 키우면 먼저 OOM에 도달한다.
**긴 문맥 학습을 24GB 한 장으로 해낼 수 있던 건 minRNN 덕분.**

## 한계 (정직하게)

- 논문 정성평가(BLEU 35.46)는 짧은 문서 20개 표본이라 참고치. 긴 문서 greedy 디코딩이 O(L²)라
  느려 전수 평가는 미실행. 공문서 도메인은 별도 코퍼스가 필요하다(현재는 논문에 한정).
  OpenSubtitles(44,660 자막문서)는 문서 경계가 align xml에만 있어 파싱이 추가로 필요하고
  구어체라 후순위로 뒀다.
- 문서 모드 BLEU(21.14)는 짧은 문서 15개 표본이라 참고치. 긴 문서 greedy 디코딩은 O(L²)라 느려
  전수 평가는 미실행. 정지 조건(입력 문장 수 기준)도 더 다듬을 여지가 있다.
- window/stride, max_len은 실측으로 정했으나(1024) 더 긴 문맥(2048)은 현재 데이터 길이(~1000토큰)로는
  이점이 안 드러난다. 더 긴 문서 코퍼스가 있어야 minRNN의 장점을 끝까지 밀어붙일 수 있다.
- 산출물: `src/prepare_context.py`, `src/data.py`(문서모드 `sep_char`/`CONTEXT_SEP`),
  `src/train.py`(`--context`/`--no-bidi`), `src/translate.py`(`greedy_translate_context`),
  `tests/benchmark_efficiency.py`(config 플래그화). 가중치 `runs/ctx/best.pt`.
