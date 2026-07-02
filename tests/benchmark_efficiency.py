"""디코딩 효율 벤치마크: 트랜스포머(셀프어텐션 O(L²)) vs minRNN(선형순환 O(L)).

시퀀스 길이 L을 키우며 디코더 forward의 peak 메모리·시간을 측정한다. 가중치는 무관(구조만
측정)하므로 랜덤 초기화 모델을 쓴다. 추론 시 minRNN은 순환 step으로 O(1) 메모리까지 가능하나,
여기서는 학습/병렬 forward 기준(트랜스포머 O(L²) vs minRNN O(L))의 스케일링을 본다.

기본값은 소형 스모크테스트(빠른 확인용)이고, 실제 268M 모델 스케일은 플래그로 지정한다:

  # 소형 스모크
  python -m tests.benchmark_efficiency
  # 268M 실모델 스케일(포폴용): L=128/512/1024/2048 대비
  python -m tests.benchmark_efficiency --d-model 1024 --layers 8 --heads 16 \
      --d-ff 4096 --lengths 128 512 1024 2048 --batch 1
"""
from __future__ import annotations

import argparse
import time

import torch

from src.minrnn import MinRNNConfig, MinRNNSeq2Seq
from src.model import ModelConfig, Seq2SeqTransformer

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
VOCAB, SRC = 32000, 50


def bench(model, name: str, lengths: list[int], batch: int, warmup: bool):
    model.eval()
    print(f"\n=== {name} ===")
    print(f"{'L':>6} | {'peak_mem(MB)':>12} | {'time(ms)':>9}")
    src = torch.randint(4, VOCAB, (batch, SRC), device=DEVICE)
    with torch.no_grad():
        memory = model.encode(src)
        for L in lengths:
            tgt = torch.randint(4, VOCAB, (batch, L), device=DEVICE)
            try:
                if warmup:  # 첫 커널 컴파일/캐시 편향 제거용 워밍업 1회
                    _ = model.decode(tgt, memory, src)
                if DEVICE == "cuda":
                    torch.cuda.synchronize(); torch.cuda.reset_peak_memory_stats()
                t0 = time.time()
                _ = model.decode(tgt, memory, src)
                if DEVICE == "cuda":
                    torch.cuda.synchronize()
                dt = (time.time() - t0) * 1000
                peak = torch.cuda.max_memory_allocated() / 1024**2 if DEVICE == "cuda" else 0
                print(f"{L:>6} | {peak:>12.1f} | {dt:>9.1f}")
            except RuntimeError as e:
                print(f"{L:>6} | {'OOM/err':>12} | {str(e)[:40]}")
                torch.cuda.empty_cache() if DEVICE == "cuda" else None
                break


def main():
    ap = argparse.ArgumentParser(description="디코딩 효율 벤치(TF vs minRNN)")
    ap.add_argument("--d-model", type=int, default=512)
    ap.add_argument("--layers", type=int, default=6)
    ap.add_argument("--heads", type=int, default=8)
    ap.add_argument("--d-ff", type=int, default=2048)
    ap.add_argument("--batch", type=int, default=1)
    ap.add_argument("--lengths", type=int, nargs="+",
                    default=[128, 256, 512, 1024, 2048, 4096])
    ap.add_argument("--warmup", action="store_true",
                    help="측정 전 워밍업 1회(커널 캐시 편향 제거)")
    args = ap.parse_args()

    print(f"device={DEVICE} | d_model={args.d_model} layers={args.layers} "
          f"heads={args.heads} d_ff={args.d_ff} | batch={args.batch} src_len={SRC}")
    common = dict(vocab_size=VOCAB, d_model=args.d_model, n_heads=args.heads,
                  n_enc_layers=args.layers, n_dec_layers=args.layers, d_ff=args.d_ff,
                  max_len=max(args.lengths) + 8)
    tf = Seq2SeqTransformer(ModelConfig(**common)).to(DEVICE)
    print(f"Transformer params={tf.num_params():,}")
    bench(tf, "Transformer (self-attn O(L^2))", args.lengths, args.batch, args.warmup)
    del tf
    torch.cuda.empty_cache() if DEVICE == "cuda" else None

    mr = MinRNNSeq2Seq(MinRNNConfig(**common)).to(DEVICE)
    print(f"minRNN params={mr.num_params():,}")
    bench(mr, "minRNN (linear recurrence O(L))", args.lengths, args.batch, args.warmup)


if __name__ == "__main__":
    main()
