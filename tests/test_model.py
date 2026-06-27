"""모델 구조 단위 테스트 (pytest 불필요, 직접 실행).

  python -m tests.test_model

검증 항목:
  1) forward 로짓 shape
  2) 디코더 causal mask 가 미래 토큰 누출을 막는지
  3) 인코더가 src 패딩에 NaN 없이 동작하는지
  4) weight tying 이 실제로 같은 텐서를 공유하는지
"""
from __future__ import annotations

import torch

from src.model import ModelConfig, Seq2SeqTransformer
from src.tokenizer import BOS_ID, EOS_ID, PAD_ID


def build():
    cfg = ModelConfig(vocab_size=50, d_model=32, n_heads=4, n_enc_layers=2, n_dec_layers=2, d_ff=64, max_len=32)
    model = Seq2SeqTransformer(cfg).eval()  # dropout 끄기
    return cfg, model


def test_forward_shape():
    cfg, model = build()
    b, ls, lt = 3, 7, 5
    src = torch.randint(4, cfg.vocab_size, (b, ls))
    tgt = torch.randint(4, cfg.vocab_size, (b, lt))
    logits = model(src, tgt)
    assert logits.shape == (b, lt, cfg.vocab_size), logits.shape
    assert torch.isfinite(logits).all()
    print("[1] forward shape OK:", tuple(logits.shape))


def test_causality():
    """미래 위치의 tgt 토큰을 바꿔도 그 이전 위치의 로짓은 변하지 않아야 한다."""
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
    assert diff < 1e-5, f"causal mask 누출! 위치<= {t} 로짓 변화 {diff}"
    print(f"[2] causality OK (위치<= {t} 최대 변화 {diff:.2e})")


def test_padding_runs():
    cfg, model = build()
    src = torch.tensor([[5, 6, 7, PAD_ID, PAD_ID]])
    tgt = torch.tensor([[BOS_ID, 8, 9, EOS_ID, PAD_ID]])
    with torch.no_grad():
        logits = model(src, tgt)
    assert torch.isfinite(logits).all(), "패딩 입력에서 NaN/Inf 발생"
    print("[3] padding forward OK (NaN 없음)")


def test_weight_tying():
    cfg, model = build()
    assert model.lm_head.weight is model.embed.weight, "weight tying 안 됨"
    print(f"[4] weight tying OK | 고유 파라미터 수: {model.num_params():,}")


if __name__ == "__main__":
    test_forward_shape()
    test_causality()
    test_padding_runs()
    test_weight_tying()
    print("ALL TESTS PASSED")
