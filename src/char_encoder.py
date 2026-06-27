"""MCE: 형태소 합성 인코더 (Morpheme-Compositional Encoder).

플랫 BPE 토큰 임베딩 대신, 단어의 문자(한글=음절)를 CharCNN으로 합성해 단어 벡터를 만든 뒤
단어 단위 트랜스포머 인코더에 넣는다. 디코더·cross-attention·타깃 BPE는 베이스라인과 동일하며,
베이스라인의 EncoderLayer/DecoderLayer/PositionalEncoding을 그대로 재사용한다.

가설: 처음 보는 단어도 구성 문자로부터 의미를 합성 → 저빈도 전문어 일반화 개선.
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field

import torch
import torch.nn as nn
import torch.nn.functional as F

from .char_tokenizer import PAD_CHAR_ID
from .model import DecoderLayer, EncoderLayer, ModelConfig, PositionalEncoding
from .tokenizer import PAD_ID


@dataclass
class MCEConfig:
    char_vocab_size: int
    tgt_vocab_size: int
    d_model: int = 512
    n_heads: int = 8
    n_enc_layers: int = 6
    n_dec_layers: int = 6
    d_ff: int = 2048
    max_len: int = 128            # 최대 단어 수(+태그) / 타깃 토큰 수
    max_chars: int = 20           # 단어당 문자 수
    d_char: int = 128
    char_channels: int = 128      # conv 폭별 채널 수
    char_widths: tuple = (1, 2, 3, 4, 5)
    dropout: float = 0.1
    pad_id: int = PAD_ID          # 타깃 BPE 패딩 id
    n_dir_tags: int = 2           # 방향 태그: 0=en, 1=ko


class CharCNN(nn.Module):
    """단어의 문자 시퀀스 → 단어 벡터 1개. (여러 폭 conv + max-over-time + highway)"""

    def __init__(self, cfg: MCEConfig):
        super().__init__()
        self.embed = nn.Embedding(cfg.char_vocab_size, cfg.d_char, padding_idx=PAD_CHAR_ID)
        self.convs = nn.ModuleList(
            [nn.Conv1d(cfg.d_char, cfg.char_channels, kernel_size=w) for w in cfg.char_widths]
        )
        total = cfg.char_channels * len(cfg.char_widths)
        self.hw_t = nn.Linear(total, total)   # highway transform gate
        self.hw_h = nn.Linear(total, total)   # highway nonlinear
        self.proj = nn.Linear(total, cfg.d_model)
        self.dropout = nn.Dropout(cfg.dropout)

    def forward(self, char_ids: torch.Tensor) -> torch.Tensor:
        # char_ids: (B, W, C) → 단어 임베딩 (B, W, d_model)
        b, w, c = char_ids.shape
        x = self.embed(char_ids.view(b * w, c)).transpose(1, 2)  # (BW, d_char, C)
        feats = [F.relu(conv(x)).max(dim=2).values for conv in self.convs]  # 각 (BW, ch)
        h = torch.cat(feats, dim=1)                              # (BW, total)
        t = torch.sigmoid(self.hw_t(h))
        h = t * F.relu(self.hw_h(h)) + (1 - t) * h               # highway
        out = self.proj(self.dropout(h))                         # (BW, d_model)
        return out.view(b, w, -1)


class MCETransformer(nn.Module):
    def __init__(self, cfg: MCEConfig):
        super().__init__()
        self.cfg = cfg
        self.charcnn = CharCNN(cfg)
        self.tag_embed = nn.Embedding(cfg.n_dir_tags, cfg.d_model)
        self.pos = PositionalEncoding(cfg.d_model, cfg.max_len + 8)
        self.emb_dropout = nn.Dropout(cfg.dropout)

        # 베이스라인 레이어 재사용(같은 시그니처의 설정 객체를 만들어 전달)
        lcfg = ModelConfig(
            vocab_size=cfg.tgt_vocab_size, d_model=cfg.d_model, n_heads=cfg.n_heads,
            n_enc_layers=cfg.n_enc_layers, n_dec_layers=cfg.n_dec_layers, d_ff=cfg.d_ff,
            max_len=cfg.max_len, dropout=cfg.dropout, pad_id=cfg.pad_id,
        )
        self.encoder = nn.ModuleList([EncoderLayer(lcfg) for _ in range(cfg.n_enc_layers)])
        self.decoder = nn.ModuleList([DecoderLayer(lcfg) for _ in range(cfg.n_dec_layers)])
        self.enc_norm = nn.LayerNorm(cfg.d_model)
        self.dec_norm = nn.LayerNorm(cfg.d_model)

        self.tgt_embed = nn.Embedding(cfg.tgt_vocab_size, cfg.d_model, padding_idx=cfg.pad_id)
        self.lm_head = nn.Linear(cfg.d_model, cfg.tgt_vocab_size, bias=False)
        self.lm_head.weight = self.tgt_embed.weight  # weight tying (디코더측)
        self.apply(self._init_weights)

    @staticmethod
    def _init_weights(m: nn.Module) -> None:
        if isinstance(m, nn.Linear):
            nn.init.xavier_uniform_(m.weight)
            if m.bias is not None:
                nn.init.zeros_(m.bias)
        elif isinstance(m, nn.Embedding):
            nn.init.normal_(m.weight, mean=0.0, std=0.02)
            if m.padding_idx is not None:
                with torch.no_grad():
                    m.weight[m.padding_idx].zero_()

    # ---- 마스크 ----
    def _word_keep(self, char_ids: torch.Tensor) -> torch.Tensor:
        # (B,W,C) → (B,W) bool: 실제 단어=True (문자가 하나라도 pad가 아니면)
        return (char_ids != PAD_CHAR_ID).any(dim=-1)

    def _additive(self, keep: torch.Tensor) -> torch.Tensor:
        # keep (B,S) bool → (B,1,1,S) 가산 마스크
        return torch.where(keep[:, None, None, :], 0.0, float("-inf"))

    def _causal(self, length: int, device: torch.device) -> torch.Tensor:
        return torch.triu(torch.full((length, length), float("-inf"), device=device), diagonal=1)[None, None]

    def _tgt_pad(self, ids: torch.Tensor) -> torch.Tensor:
        return self._additive(ids != self.cfg.pad_id)

    # ---- 인코딩/디코딩 ----
    def encode(self, src_char_ids: torch.Tensor, tag_idx: torch.Tensor):
        b = src_char_ids.size(0)
        word_emb = self.charcnn(src_char_ids)                       # (B,W,d)
        tag = self.tag_embed(tag_idx).unsqueeze(1)                  # (B,1,d)
        x = torch.cat([tag, word_emb], dim=1)                       # (B,1+W,d)
        keep = torch.cat(
            [torch.ones(b, 1, dtype=torch.bool, device=x.device), self._word_keep(src_char_ids)],
            dim=1,
        )                                                           # (B,1+W)
        x = self.emb_dropout(self.pos(x))
        smask = self._additive(keep)
        for layer in self.encoder:
            x = layer(x, smask)
        return self.enc_norm(x), keep

    def decode(self, tgt: torch.Tensor, memory: torch.Tensor, src_keep: torch.Tensor) -> torch.Tensor:
        tgt_mask = self._causal(tgt.size(1), tgt.device) + self._tgt_pad(tgt)
        mem_mask = self._additive(src_keep)
        x = self.tgt_embed(tgt) * math.sqrt(self.cfg.d_model)
        x = self.emb_dropout(self.pos(x))
        for layer in self.decoder:
            x = layer(x, memory, tgt_mask, mem_mask)
        return self.dec_norm(x)

    def forward(self, src_char_ids: torch.Tensor, tag_idx: torch.Tensor, tgt: torch.Tensor) -> torch.Tensor:
        memory, keep = self.encode(src_char_ids, tag_idx)
        return self.lm_head(self.decode(tgt, memory, keep))

    def num_params(self) -> int:
        seen, total = set(), 0
        for p in self.parameters():
            if id(p) in seen:
                continue
            seen.add(id(p))
            total += p.numel()
        return total
