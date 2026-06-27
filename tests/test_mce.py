"""MCE 구조 단위 테스트 (pytest 불필요).

  python -m tests.test_mce

검증: forward shape / 디코더 causal mask / 패딩 단어 NaN 없음 / weight tying
"""
from __future__ import annotations

import torch

from src.char_encoder import MCEConfig, MCETransformer
from src.char_tokenizer import PAD_CHAR_ID
from src.tokenizer import PAD_ID


def build():
    cfg = MCEConfig(
        char_vocab_size=30, tgt_vocab_size=50, d_model=32, n_heads=4,
        n_enc_layers=2, n_dec_layers=2, d_ff=64, max_len=32, max_chars=8,
        d_char=16, char_channels=16, char_widths=(1, 2, 3),
    )
    return cfg, MCETransformer(cfg).eval()


def rand_words(b, w, c, vocab, n_pad_words=0):
    """(B,W,C) 문자 ids. 각 단어는 앞부분 실제문자 + 0 패딩. 마지막 n_pad_words개는 전부 패딩."""
    x = torch.zeros(b, w, c, dtype=torch.long)
    for bi in range(b):
        for wi in range(w - n_pad_words):
            length = torch.randint(1, c + 1, (1,)).item()
            x[bi, wi, :length] = torch.randint(2, vocab, (length,))
    return x


def test_forward_shape():
    cfg, model = build()
    b, w, t = 3, 6, 5
    src = rand_words(b, w, cfg.max_chars, cfg.char_vocab_size)
    tag = torch.randint(0, 2, (b,))
    tgt = torch.randint(4, cfg.tgt_vocab_size, (b, t))
    logits = model(src, tag, tgt)
    assert logits.shape == (b, t, cfg.tgt_vocab_size), logits.shape
    assert torch.isfinite(logits).all()
    print("[1] forward shape OK:", tuple(logits.shape))


def test_causality():
    cfg, model = build()
    src = rand_words(1, 5, cfg.max_chars, cfg.char_vocab_size)
    tag = torch.zeros(1, dtype=torch.long)
    tgt = torch.randint(4, cfg.tgt_vocab_size, (1, 8))
    with torch.no_grad():
        out1 = model(src, tag, tgt)
        tgt2 = tgt.clone()
        t = 3
        tgt2[0, t + 1:] = torch.randint(4, cfg.tgt_vocab_size, (tgt.size(1) - (t + 1),))
        out2 = model(src, tag, tgt2)
    diff = (out1[0, : t + 1] - out2[0, : t + 1]).abs().max().item()
    assert diff < 1e-5, f"causal 누출 {diff}"
    print(f"[2] causality OK (위치<= {t} 최대 변화 {diff:.2e})")


def test_padding_words():
    cfg, model = build()
    # 마지막 2개 단어가 패딩(전부 PAD_CHAR_ID)인 배치
    src = rand_words(2, 6, cfg.max_chars, cfg.char_vocab_size, n_pad_words=2)
    tag = torch.tensor([0, 1])
    tgt = torch.tensor([[1, 8, 9, 2, PAD_ID], [1, 7, 2, PAD_ID, PAD_ID]])
    with torch.no_grad():
        logits = model(src, tag, tgt)
    assert torch.isfinite(logits).all(), "패딩 단어/타깃에서 NaN/Inf"
    assert src[0, -1].eq(PAD_CHAR_ID).all(), "테스트 설정: 마지막 단어는 패딩이어야"
    print("[3] padding(단어/타깃) forward OK (NaN 없음)")


def test_weight_tying():
    cfg, model = build()
    assert model.lm_head.weight is model.tgt_embed.weight
    print(f"[4] weight tying OK | 고유 파라미터 수: {model.num_params():,}")


if __name__ == "__main__":
    test_forward_shape()
    test_causality()
    test_padding_words()
    test_weight_tying()
    print("ALL MCE TESTS PASSED")
