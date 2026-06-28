"""minRNN 단위 테스트 (pytest 불필요).

  python -m tests.test_minrnn

검증: 병렬 스캔==순차 / MinGRU forward==step / BiMinGRU 패딩 불변 /
      디코더 causal / forward shape·weight tying
"""
from __future__ import annotations

import torch

from src.minrnn import (BiMinGRU, MinGRU, MinRNNConfig, MinRNNSeq2Seq,
                        parallel_scan_log)
from src.tokenizer import PAD_ID


def test_scan_matches_sequential():
    torch.manual_seed(0)
    b, t, d = 2, 30, 8
    log_a = -torch.rand(b, t, d)              # log(0,1) → 음수
    log_b = torch.randn(b, t, d) * 0.5
    h_par = parallel_scan_log(log_a, log_b)
    a, bb = log_a.exp(), log_b.exp()
    h = torch.zeros(b, d)
    seq = []
    for i in range(t):
        h = a[:, i] * h + bb[:, i]
        seq.append(h)
    h_seq = torch.stack(seq, dim=1)
    diff = (h_par - h_seq).abs().max().item()
    assert diff < 1e-4, f"스캔 불일치 {diff}"
    print(f"[1] 병렬 스캔 == 순차 (최대오차 {diff:.2e})")


def test_mingru_forward_eq_step():
    torch.manual_seed(0)
    cell = MinGRU(8).eval()
    x = torch.randn(2, 20, 8)
    out_par = cell(x)
    h = torch.zeros(2, 8)
    seq = [(_ := None)] * 0
    for i in range(x.size(1)):
        h = cell.step(x[:, i], h)
        seq.append(h)
    out_seq = torch.stack(seq, dim=1)
    diff = (out_par - out_seq).abs().max().item()
    assert diff < 1e-4, f"forward!=step {diff}"
    print(f"[2] MinGRU forward == step (최대오차 {diff:.2e})")


def test_bidir_padding_invariant():
    torch.manual_seed(0)
    bi = BiMinGRU(8).eval()
    x = torch.randn(1, 6, 8)
    keep_full = torch.ones(1, 6, dtype=torch.bool)
    out1 = bi(x, keep_full)
    # 뒤에 패딩 2개 추가(keep=False) → 앞 6개 출력 불변이어야
    xp = torch.cat([x, torch.randn(1, 2, 8)], dim=1)
    keep_pad = torch.tensor([[True] * 6 + [False] * 2])
    out2 = bi(xp, keep_pad)[:, :6]
    diff = (out1 - out2).abs().max().item()
    assert diff < 1e-4, f"패딩이 실토큰 출력 바꿈 {diff}"
    print(f"[3] BiMinGRU 패딩 불변 (최대오차 {diff:.2e})")


def build():
    cfg = MinRNNConfig(vocab_size=50, d_model=32, n_enc_layers=2, n_dec_layers=2,
                       n_heads=4, d_ff=64, max_len=32)
    return cfg, MinRNNSeq2Seq(cfg).eval()


def test_decoder_causal():
    cfg, model = build()
    src = torch.randint(4, cfg.vocab_size, (1, 6))
    tgt = torch.randint(4, cfg.vocab_size, (1, 8))
    with torch.no_grad():
        out1 = model(src, tgt)
        tgt2 = tgt.clone()
        t = 3
        tgt2[0, t + 1:] = torch.randint(4, cfg.vocab_size, (tgt.size(1) - (t + 1),))
        out2 = model(src, tgt2)
    diff = (out1[0, : t + 1] - out2[0, : t + 1]).abs().max().item()
    assert diff < 1e-4, f"디코더 causal 누출 {diff}"
    print(f"[4] 디코더 causal OK (위치<= {t} 변화 {diff:.2e})")


def test_forward_shape_tying():
    cfg, model = build()
    src = torch.tensor([[7, 8, 9, PAD_ID, PAD_ID]])
    tgt = torch.tensor([[1, 10, 11, 2, PAD_ID]])
    with torch.no_grad():
        logits = model(src, tgt)
    assert logits.shape == (1, 5, cfg.vocab_size), logits.shape
    assert torch.isfinite(logits).all()
    assert model.lm_head.weight is model.embed.weight
    print(f"[5] forward shape {tuple(logits.shape)}, weight tying OK | 파라미터 {model.num_params():,}")


if __name__ == "__main__":
    test_scan_matches_sequential()
    test_mingru_forward_eq_step()
    test_bidir_padding_invariant()
    test_decoder_causal()
    test_forward_shape_tying()
    print("ALL minRNN TESTS PASSED")
