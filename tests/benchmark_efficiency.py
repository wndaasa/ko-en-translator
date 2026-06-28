"""디코딩 효율 벤치마크: 트랜스포머(셀프어텐션 O(L²)) vs minRNN(선형순환 O(L)).

시퀀스 길이 L을 키우며 디코더 forward의 peak 메모리·시간을 측정한다. 가중치는 무관(구조만
측정)하므로 랜덤 초기화 모델을 쓴다. 추론 시 minRNN은 순환 step으로 O(1) 메모리까지 가능하나,
여기서는 학습/병렬 forward 기준(트랜스포머 O(L²) vs minRNN O(L))의 스케일링을 본다.

  HSA_ENABLE_DXG_DETECTION=1 python -m tests.benchmark_efficiency
"""
from __future__ import annotations

import time

import torch

from src.minrnn import MinRNNConfig, MinRNNSeq2Seq
from src.model import ModelConfig, Seq2SeqTransformer

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
VOCAB, D, LAYERS, HEADS, DFF, SRC = 32000, 512, 6, 8, 2048, 50
LENGTHS = [128, 256, 512, 1024, 2048, 4096]


def bench(model, name: str):
    model.eval()
    print(f"\n=== {name} ===")
    print(f"{'L':>6} | {'peak_mem(MB)':>12} | {'time(ms)':>9}")
    src = torch.randint(4, VOCAB, (1, SRC), device=DEVICE)
    with torch.no_grad():
        memory = model.encode(src)
        for L in LENGTHS:
            tgt = torch.randint(4, VOCAB, (1, L), device=DEVICE)
            if DEVICE == "cuda":
                torch.cuda.synchronize(); torch.cuda.reset_peak_memory_stats()
            t0 = time.time()
            try:
                _ = model.decode(tgt, memory, src)
                if DEVICE == "cuda":
                    torch.cuda.synchronize()
                dt = (time.time() - t0) * 1000
                peak = torch.cuda.max_memory_allocated() / 1024**2 if DEVICE == "cuda" else 0
                print(f"{L:>6} | {peak:>12.1f} | {dt:>9.1f}")
            except RuntimeError as e:
                print(f"{L:>6} | {'OOM/err':>12} | {str(e)[:40]}")
                break


def main():
    print(f"device={DEVICE} | d_model={D} layers={LAYERS} | src_len={SRC}")
    tf = Seq2SeqTransformer(ModelConfig(
        vocab_size=VOCAB, d_model=D, n_heads=HEADS, n_enc_layers=LAYERS,
        n_dec_layers=LAYERS, d_ff=DFF, max_len=max(LENGTHS) + 8)).to(DEVICE)
    mr = MinRNNSeq2Seq(MinRNNConfig(
        vocab_size=VOCAB, d_model=D, n_heads=HEADS, n_enc_layers=LAYERS,
        n_dec_layers=LAYERS, d_ff=DFF, max_len=max(LENGTHS) + 8)).to(DEVICE)
    bench(tf, "Transformer (self-attn O(L^2))")
    bench(mr, "minRNN (linear recurrence O(L))")


if __name__ == "__main__":
    main()
