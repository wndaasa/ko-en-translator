"""번역 모델 학습기.

stage1(토이) / stage2(OPUS 양방향) 공용. teacher forcing + 패딩 마스크 CE +
AdamW(워밍업·역제곱근) + bf16 autocast. 검증 손실과 ko→en BLEU로 모니터링하고
검증 손실 기준 best 체크포인트를 저장한다.
"""
from __future__ import annotations

import argparse
import functools
import math
import time
from pathlib import Path

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

from .char_encoder import MCEConfig, MCETransformer
from .char_tokenizer import CharTokenizer, build_char_vocab
from .data import MCEDataset, MTDataset, collate, mce_collate, read_pairs
from .minrnn import MinRNNConfig, MinRNNSeq2Seq
from .model import ModelConfig, Seq2SeqTransformer
from .tokenizer import PAD_ID, load_tokenizer
from .translate import greedy_translate, greedy_translate_mce


def lr_lambda(step: int, warmup: int) -> float:
    step = max(step, 1)
    return min(step / warmup, (warmup / step) ** 0.5)


def get_device() -> str:
    return "cuda" if torch.cuda.is_available() else "cpu"


def forward_logits(model, batch, device, arch):
    """아키텍처에 맞게 forward. bpe: (src, dec_in) / mce: (src_char, tag, dec_in)."""
    dec_in = batch["dec_in"].to(device)
    if arch == "mce":
        return model(batch["src"].to(device), batch["tag"].to(device), dec_in)
    return model(batch["src"].to(device), dec_in)


@torch.no_grad()
def evaluate(model, tokenizer, val_pairs, val_loader, device, use_amp, bleu_samples, arch, char_tok):
    """검증 손실(양방향)과 ko→en BLEU(우선 방향)를 계산."""
    model.eval()
    total_loss, total_tok = 0.0, 0
    for batch in val_loader:
        labels = batch["labels"].to(device)
        with torch.autocast(device_type="cuda", dtype=torch.bfloat16, enabled=use_amp):
            logits = forward_logits(model, batch, device, arch)
            loss = F.cross_entropy(
                logits.reshape(-1, logits.size(-1)), labels.reshape(-1),
                ignore_index=PAD_ID, reduction="sum",
            )
        total_loss += loss.item()
        total_tok += (labels != PAD_ID).sum().item()
    val_loss = total_loss / max(total_tok, 1)

    # ko -> en BLEU (영어 타깃, sacrebleu 표준)
    import sacrebleu
    sample = val_pairs[:bleu_samples]
    if arch == "mce":
        hyps = [greedy_translate_mce(model, char_tok, tokenizer, ko, device, "en", 128)
                for ko, _ in sample]
    else:
        hyps = [greedy_translate(model, tokenizer, ko, device, target_lang="en", max_new=128)
                for ko, _ in sample]
    refs = [en for _, en in sample]
    bleu = sacrebleu.corpus_bleu(hyps, [refs]).score
    model.train()
    return val_loss, bleu


def train(args: argparse.Namespace) -> None:
    device = get_device()
    art = Path(args.artifacts)
    art.mkdir(parents=True, exist_ok=True)
    tokenizer = load_tokenizer(art / "tokenizer.json")
    vocab_size = tokenizer.get_vocab_size()

    arch = args.arch
    char_tok = None
    val_pairs = read_pairs(args.val_data)
    if args.max_val:
        val_pairs = val_pairs[: args.max_val]

    if arch == "mce":
        char_vocab_path = Path(args.char_vocab) if args.char_vocab else (art / "char_vocab.json")
        if not char_vocab_path.exists():
            build_char_vocab([args.train_data], char_vocab_path, min_freq=2)
        char_tok = CharTokenizer.from_file(char_vocab_path)
        train_ds = MCEDataset(args.train_data, char_tok, tokenizer, max_len=args.max_len,
                              max_chars=args.max_chars, bidirectional=True)
        val_ds = MCEDataset(args.val_data, char_tok, tokenizer, max_len=args.max_len,
                            max_chars=args.max_chars, bidirectional=True, limit=args.max_val)
        coll = functools.partial(mce_collate, max_chars=args.max_chars)
        cfg = MCEConfig(
            char_vocab_size=char_tok.vocab_size(), tgt_vocab_size=vocab_size,
            d_model=args.d_model, n_heads=args.n_heads, n_enc_layers=args.layers,
            n_dec_layers=args.layers, d_ff=args.d_ff, max_len=args.max_len,
            max_chars=args.max_chars, dropout=args.dropout,
        )
        model = MCETransformer(cfg).to(device)
    else:
        # bpe(트랜스포머)와 minrnn은 데이터 파이프라인(토큰 기반)이 동일.
        train_ds = MTDataset(args.train_data, tokenizer, max_len=args.max_len, bidirectional=True)
        val_ds = MTDataset(args.val_data, tokenizer, max_len=args.max_len, bidirectional=True,
                           limit=args.max_val)
        coll = collate
        if arch == "minrnn":
            cfg = MinRNNConfig(
                vocab_size=vocab_size, d_model=args.d_model, n_heads=args.n_heads,
                n_enc_layers=args.layers, n_dec_layers=args.layers, d_ff=args.d_ff,
                max_len=args.max_len, dropout=args.dropout,
            )
            model = MinRNNSeq2Seq(cfg).to(device)
        else:
            cfg = ModelConfig(
                vocab_size=vocab_size, d_model=args.d_model, n_heads=args.n_heads,
                n_enc_layers=args.layers, n_dec_layers=args.layers, d_ff=args.d_ff,
                max_len=args.max_len, dropout=args.dropout,
            )
            model = Seq2SeqTransformer(cfg).to(device)

    train_loader = DataLoader(
        train_ds, batch_size=args.batch_size, shuffle=True, collate_fn=coll,
        num_workers=args.num_workers, persistent_workers=args.num_workers > 0, drop_last=True,
    )
    val_loader = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False, collate_fn=coll)
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, betas=(0.9, 0.98), weight_decay=0.01)
    sched = torch.optim.lr_scheduler.LambdaLR(opt, lambda s: lr_lambda(s, args.warmup))
    use_amp = device == "cuda"

    # 파인튜닝: 사전학습 가중치로 시작(옵티마이저/스케줄은 새로 시작).
    if args.init_from:
        ck = torch.load(args.init_from, map_location=device)
        model.load_state_dict(ck["model"])
        print(f"init-from: {args.init_from} 가중치 로드 (옵티마이저는 새로 시작)")

    steps_per_epoch = len(train_loader)
    total_steps = steps_per_epoch * args.epochs
    best_val = float("inf")
    step = 0
    start_epoch = 1

    # 이어하기: 모델+옵티마이저+스케줄러+step 복원 (epoch 단위 재개).
    if args.resume:
        ck = torch.load(args.resume, map_location=device)
        model.load_state_dict(ck["model"])
        opt.load_state_dict(ck["opt"])
        sched.load_state_dict(ck["sched"])
        step = ck["step"]
        start_epoch = ck["next_epoch"]
        best_val = ck.get("best_val", float("inf"))
        print(f"resume: {args.resume} | step {step}, epoch {start_epoch}부터 재개")

    print(f"device={device} | params={model.num_params():,} | vocab={vocab_size}")
    print(f"train examples={len(train_ds):,} | steps/epoch={steps_per_epoch:,} | total steps={total_steps:,}")

    def save_resume(next_epoch: int) -> None:
        """이어하기에 필요한 전체 상태 저장. next_epoch=재개 시 시작할 epoch."""
        torch.save({"model": model.state_dict(), "opt": opt.state_dict(),
                    "sched": sched.state_dict(), "step": step, "next_epoch": next_epoch,
                    "best_val": best_val, "config": cfg.__dict__, "arch": arch, "bidirectional": True},
                   art / "resume.pt")

    t0 = time.time()
    running = 0.0
    model.train()
    for epoch in range(start_epoch, args.epochs + 1):
        for batch in train_loader:
            labels = batch["labels"].to(device)
            opt.zero_grad()
            with torch.autocast(device_type="cuda", dtype=torch.bfloat16, enabled=use_amp):
                logits = forward_logits(model, batch, device, arch)
                loss = F.cross_entropy(
                    logits.reshape(-1, logits.size(-1)), labels.reshape(-1),
                    ignore_index=PAD_ID, label_smoothing=args.label_smoothing,
                )
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            sched.step()
            step += 1
            running += loss.item()

            if step % args.log_every == 0:
                dt = time.time() - t0
                ips = args.log_every * args.batch_size / dt
                avg = running / args.log_every
                print(f"step {step:6d}/{total_steps} | ep {epoch} | loss {avg:.3f} "
                      f"| lr {sched.get_last_lr()[0]:.2e} | {ips:.0f} ex/s")
                running = 0.0
                t0 = time.time()

            if step % args.eval_every == 0:
                val_loss, bleu = evaluate(model, tokenizer, val_pairs, val_loader,
                                          device, use_amp, args.bleu_samples, arch, char_tok)
                ppl = math.exp(min(val_loss, 20))
                tag = ""
                if val_loss < best_val:
                    best_val = val_loss
                    torch.save({"model": model.state_dict(), "config": cfg.__dict__,
                                "arch": arch, "bidirectional": True}, art / "best.pt")
                    tag = "  <- best 저장"
                # 크래시/중단 대비 전체 상태 저장(현재 epoch 재시작 기준).
                save_resume(epoch)
                print(f"  [eval] step {step} | val_loss {val_loss:.3f} | ppl {ppl:.1f} "
                      f"| ko->en BLEU {bleu:.2f}{tag}")
                t0 = time.time()

        torch.save({"model": model.state_dict(), "config": cfg.__dict__, "arch": arch,
                    "bidirectional": True}, art / "last.pt")
        save_resume(epoch + 1)  # epoch 완료 → 다음 epoch부터 재개
    print(f"학습 종료. best val_loss={best_val:.3f} | 체크포인트: {art}/best.pt, {art}/last.pt")


def build_argparser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(description="번역 모델 학습 (양방향)")
    ap.add_argument("--train-data", default="data/processed/train.tsv")
    ap.add_argument("--val-data", default="data/processed/val.tsv")
    ap.add_argument("--artifacts", default="runs/base")
    ap.add_argument("--init-from", default=None, help="사전학습 가중치(파인튜닝 시작점). 옵티마이저는 새로 시작.")
    ap.add_argument("--resume", default=None, help="resume.pt 경로. 중단된 학습을 이어서 진행.")
    ap.add_argument("--arch", default="bpe", choices=["bpe", "mce", "minrnn"],
                    help="구조: 베이스라인 트랜스포머(bpe) / MCE 문자합성 / minRNN 선형순환.")
    ap.add_argument("--char-vocab", default=None, help="MCE 문자 vocab 경로(없으면 train-data로 생성).")
    ap.add_argument("--max-chars", type=int, default=20, help="MCE 단어당 최대 문자 수.")
    ap.add_argument("--epochs", type=int, default=4)
    ap.add_argument("--batch-size", type=int, default=64)
    ap.add_argument("--lr", type=float, default=5e-4)
    ap.add_argument("--warmup", type=int, default=4000)
    ap.add_argument("--label-smoothing", type=float, default=0.1)
    ap.add_argument("--max-len", type=int, default=128)
    ap.add_argument("--num-workers", type=int, default=4)
    ap.add_argument("--log-every", type=int, default=100)
    ap.add_argument("--eval-every", type=int, default=2000)
    ap.add_argument("--bleu-samples", type=int, default=300)
    ap.add_argument("--max-val", type=int, default=0, help="검증셋 pair 수 제한(0=전체). 큰 val에서 eval 가속.")
    # 모델 크기 (Transformer-base)
    ap.add_argument("--d-model", type=int, default=512)
    ap.add_argument("--n-heads", type=int, default=8)
    ap.add_argument("--layers", type=int, default=6)
    ap.add_argument("--d-ff", type=int, default=2048)
    ap.add_argument("--dropout", type=float, default=0.1)
    return ap


if __name__ == "__main__":
    train(build_argparser().parse_args())
