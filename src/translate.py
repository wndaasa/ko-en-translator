"""학습된 모델로 번역(greedy 디코딩)."""
from __future__ import annotations

import argparse
from pathlib import Path

import torch

from .model import ModelConfig, Seq2SeqTransformer
from .tokenizer import BOS_ID, EOS_ID, Tokenizer, decode, encode, load_tokenizer, tag_id


def load_model(ckpt_path: str | Path, device: str) -> tuple[Seq2SeqTransformer, str]:
    ckpt = torch.load(ckpt_path, map_location=device)
    cfg = ModelConfig(**ckpt["config"])
    model = Seq2SeqTransformer(cfg).to(device)
    model.load_state_dict(ckpt["model"])
    model.eval()
    return model, ckpt.get("direction", "ko2en")


@torch.no_grad()
def greedy_translate(
    model: Seq2SeqTransformer,
    tokenizer: Tokenizer,
    text: str,
    device: str,
    target_lang: str | None = None,
    max_new: int = 64,
) -> str:
    """target_lang('en'|'ko')이 주어지면 소스 선두에 방향 태그를 부착한다(양방향 모델용).
    None이면 태그 없이 인코딩(단방향 stage1 모델 호환)."""
    ids = encode(tokenizer, text, add_eos=True)
    if target_lang is not None:
        ids = [tag_id(target_lang)] + ids
    src = torch.tensor([ids], dtype=torch.long, device=device)
    memory = model.encode(src)
    ys = torch.tensor([[BOS_ID]], dtype=torch.long, device=device)
    for _ in range(max_new):
        h = model.decode(ys, memory, src)
        next_id = int(model.lm_head(h)[:, -1].argmax(-1).item())
        ys = torch.cat([ys, torch.tensor([[next_id]], device=device)], dim=1)
        if next_id == EOS_ID:
            break
    return decode(tokenizer, ys[0].tolist())


@torch.no_grad()
def greedy_translate_mce(model, char_tok, bpe_tok: Tokenizer, text: str, device: str,
                         target_lang: str, max_new: int = 128) -> str:
    """MCE 모델용 greedy 번역. 소스는 문자(음절)로 인코딩, 방향 태그 사용."""
    from .char_tokenizer import UNK_CHAR_ID, PAD_CHAR_ID
    cfg = model.cfg
    words = char_tok.encode_sentence(text)
    words = [w[: cfg.max_chars] for w in words][: cfg.max_len - 1]
    if not words:
        words = [[UNK_CHAR_ID]]
    src = torch.full((1, len(words), cfg.max_chars), PAD_CHAR_ID, dtype=torch.long, device=device)
    for wi, w in enumerate(words):
        length = min(len(w), cfg.max_chars)
        src[0, wi, :length] = torch.tensor(w[:length], dtype=torch.long, device=device)
    tag = torch.tensor([0 if target_lang == "en" else 1], dtype=torch.long, device=device)
    memory, keep = model.encode(src, tag)
    ys = torch.tensor([[BOS_ID]], dtype=torch.long, device=device)
    for _ in range(max_new):
        h = model.decode(ys, memory, keep)
        nxt = int(model.lm_head(h)[:, -1].argmax(-1).item())
        ys = torch.cat([ys, torch.tensor([[nxt]], device=device)], dim=1)
        if nxt == EOS_ID:
            break
    return decode(bpe_tok, ys[0].tolist())


def _main() -> None:
    ap = argparse.ArgumentParser(description="번역 (greedy)")
    ap.add_argument("--artifacts", default="artifacts")
    ap.add_argument("--text", help="번역할 문장. 생략 시 대화형 모드.")
    ap.add_argument("--to", choices=["en", "ko"], default=None,
                    help="타깃 언어(양방향 모델). 생략 시 태그 없음(stage1 단방향).")
    ap.add_argument("--max-new", type=int, default=64)
    args = ap.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    art = Path(args.artifacts)
    tokenizer = load_tokenizer(art / "tokenizer.json")
    model, direction = load_model(art / "model.pt", device)
    print(f"device={device} | direction={direction}")

    if args.text is not None:
        print(greedy_translate(model, tokenizer, args.text, device, args.to, args.max_new))
        return
    print("문장을 입력하세요 (빈 줄 종료):")
    while True:
        try:
            line = input("> ").strip()
        except EOFError:
            break
        if not line:
            break
        print(greedy_translate(model, tokenizer, line, device, args.to, args.max_new))


if __name__ == "__main__":
    _main()
