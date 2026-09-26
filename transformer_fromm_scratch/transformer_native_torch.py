"""
Minimal encoder-decoder Transformer ("Attention is All You Need") from scratch.
PyTorch, with per-function tests and a simple EN->RU translation dataset.
Run: python transformer_from_scratch.py
"""

from __future__ import annotations

import argparse
import math
import random
import time
from typing import Optional

import datasets
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
from torch.nn import Transformer as TransformerBase


# -----------------------------------------------------------------------------
# Config
# -----------------------------------------------------------------------------
PAD_IDX = 0
SOS_IDX = 1
EOS_IDX = 2
UNK_IDX = 3

D_MODEL = 128
N_HEADS = 4
N_LAYERS = 2
D_FF = 512
DROPOUT = 0.1
MAX_LEN = 64


# -----------------------------------------------------------------------------
# 1. Positional encoding: PE(pos, 2i) = sin(pos/10000^(2i/d)); PE(pos, 2i+1) = cos(...)
# -----------------------------------------------------------------------------
def positional_encoding(
    seq_len: int, d_model: int, device: torch.device
) -> torch.Tensor:
    """Create sinusoidal positional encodings. Shape: (1, seq_len, d_model)."""
    pe = torch.zeros(1, seq_len, d_model, device=device)
    # your code here: compute position (0 .. seq_len-1) and dimension indices
    position = torch.arange(seq_len, device=device, dtype=torch.float).unsqueeze(1)
    div_term = torch.exp(
        torch.arange(0, d_model, 2, device=device).float()
        * (-math.log(10000.0) / d_model)
    )
    pe[0, :, 0::2] = torch.sin(position * div_term)
    pe[0, :, 1::2] = torch.cos(position * div_term)
    return pe


def test_positional_encoding():
    pe = positional_encoding(10, D_MODEL, torch.device("cpu"))
    assert pe.shape == (1, 10, D_MODEL), pe.shape
    assert not torch.isnan(pe).any()
    print("  [OK] positional_encoding")


# -----------------------------------------------------------------------------
# 2. Scaled dot-product attention: Attention(Q,K,V) = softmax(QK^T / sqrt(d_k)) V
# -----------------------------------------------------------------------------
def scaled_dot_product_attention(
    q: torch.Tensor,  # (batch, n_heads, seq_len, d_k)
    k: torch.Tensor,
    v: torch.Tensor,
    mask: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    """Single-head scaled dot-product attention. Returns (batch, n_heads, seq_len, d_v)."""
    d_k = q.size(-1)
    # your code here: scores = q @ k.transpose(-2,-1) / sqrt(d_k), then softmax + @ v
    scores = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(d_k)
    if mask is not None:
        scores = scores.masked_fill(mask == 0, float("-inf"))
    attn = F.softmax(scores, dim=-1)
    out = torch.matmul(attn, v)
    return out


def test_scaled_dot_product_attention():
    B, H, T, D = 2, 4, 5, 32
    q = k = v = torch.randn(B, H, T, D)
    out = scaled_dot_product_attention(q, k, v)
    assert out.shape == (B, H, T, D), out.shape
    print("  [OK] scaled_dot_product_attention")


# -----------------------------------------------------------------------------
# 3. Multi-head attention: concat heads and project
# -----------------------------------------------------------------------------
class MultiHeadAttention(nn.Module):
    """Multi-head attention. Paper: MultiHead(Q,K,V) = Concat(head_1,...,head_h)W^O."""

    def __init__(self, d_model: int, n_heads: int, dropout: float = 0.1):
        super().__init__()
        assert d_model % n_heads == 0
        self.d_k = d_model // n_heads
        self.n_heads = n_heads
        self.d_model = d_model
        self.W_q = nn.Linear(d_model, d_model)
        self.W_k = nn.Linear(d_model, d_model)
        self.W_v = nn.Linear(d_model, d_model)
        self.W_o = nn.Linear(d_model, d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(
        self,
        q: torch.Tensor,  # (batch, seq_len, d_model)
        k: torch.Tensor,
        v: torch.Tensor,
        mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        batch = q.size(0)
        # your code here: linear project q,k,v; split into heads; scaled_dot_product_attention; concat; W_o
        q = self.W_q(q).view(batch, -1, self.n_heads, self.d_k).transpose(1, 2)
        k = self.W_k(k).view(batch, -1, self.n_heads, self.d_k).transpose(1, 2)
        v = self.W_v(v).view(batch, -1, self.n_heads, self.d_k).transpose(1, 2)
        attn_out = scaled_dot_product_attention(q, k, v, mask)
        attn_out = attn_out.transpose(1, 2).contiguous().view(batch, -1, self.d_model)
        return self.dropout(self.W_o(attn_out))


def test_multi_head_attention():
    mha = MultiHeadAttention(D_MODEL, N_HEADS, DROPOUT)
    B, T, D = 2, 7, D_MODEL
    x = torch.randn(B, T, D)
    out = mha(x, x, x)
    assert out.shape == (B, T, D), out.shape
    print("  [OK] MultiHeadAttention")


# -----------------------------------------------------------------------------
# 4. Position-wise feed-forward: FFN(x) = max(0, x W_1 + b_1) W_2 + b_2
# -----------------------------------------------------------------------------
class FeedForward(nn.Module):
    def __init__(self, d_model: int, d_ff: int, dropout: float = 0.1):
        super().__init__()
        # your code here: two linear layers (d_model -> d_ff -> d_model), ReLU in between
        self.linear1 = nn.Linear(d_model, d_ff)
        self.linear2 = nn.Linear(d_ff, d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.dropout(self.linear2(F.relu(self.linear1(x))))


def test_feed_forward():
    ff = FeedForward(D_MODEL, D_FF, DROPOUT)
    x = torch.randn(2, 5, D_MODEL)
    assert ff(x).shape == x.shape
    print("  [OK] FeedForward")


# -----------------------------------------------------------------------------
# 5. Encoder layer: self-attention + add&norm + FFN + add&norm
# -----------------------------------------------------------------------------
class EncoderLayer(nn.Module):
    def __init__(self, d_model: int, n_heads: int, d_ff: int, dropout: float):
        super().__init__()
        self.self_attn = MultiHeadAttention(d_model, n_heads, dropout)
        self.ff = FeedForward(d_model, d_ff, dropout)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(
        self,
        x: torch.Tensor,
        src_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        # your code here: x = norm(x + dropout(self_attn(x,x,x,mask))); same for ff
        x = self.norm1(x + self.dropout(self.self_attn(x, x, x, src_mask)))
        x = self.norm2(x + self.dropout(self.ff(x)))
        return x


def test_encoder_layer():
    layer = EncoderLayer(D_MODEL, N_HEADS, D_FF, DROPOUT)
    x = torch.randn(2, 6, D_MODEL)
    assert layer(x).shape == x.shape
    print("  [OK] EncoderLayer")


# -----------------------------------------------------------------------------
# 6. Decoder layer: masked self-attn + cross-attn (to encoder) + FFN, each with add&norm
# -----------------------------------------------------------------------------
class DecoderLayer(nn.Module):
    def __init__(self, d_model: int, n_heads: int, d_ff: int, dropout: float):
        super().__init__()
        self.self_attn = MultiHeadAttention(d_model, n_heads, dropout)
        self.cross_attn = MultiHeadAttention(d_model, n_heads, dropout)
        self.ff = FeedForward(d_model, d_ff, dropout)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.norm3 = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(
        self,
        x: torch.Tensor,
        enc_out: torch.Tensor,
        tgt_mask: Optional[torch.Tensor] = None,
        src_tgt_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        # your code here: masked self-attn; then cross-attn (q from x, k,v from enc_out); then ff
        x = self.norm1(x + self.dropout(self.self_attn(x, x, x, tgt_mask)))
        x = self.norm2(
            x + self.dropout(self.cross_attn(x, enc_out, enc_out, src_tgt_mask))
        )
        x = self.norm3(x + self.dropout(self.ff(x)))
        return x


def test_decoder_layer():
    layer = DecoderLayer(D_MODEL, N_HEADS, D_FF, DROPOUT)
    x = torch.randn(2, 5, D_MODEL)
    enc = torch.randn(2, 7, D_MODEL)
    assert layer(x, enc).shape == x.shape
    print("  [OK] DecoderLayer")


# -----------------------------------------------------------------------------
# 7. Full transformer: embeddings + PE + encoder stack + decoder stack + output projection
# -----------------------------------------------------------------------------
def make_causal_mask(seq_len: int, device: torch.device) -> torch.Tensor:
    """Lower-triangular mask so position i cannot attend to j > i. Shape (1, 1, seq_len, seq_len)."""
    return torch.tril(torch.ones(seq_len, seq_len, device=device)).view(
        1, 1, seq_len, seq_len
    )


def make_pad_mask(padding_mask: torch.Tensor) -> torch.Tensor:
    """padding_mask: (batch, seq_len), 1 = valid, 0 = pad. Returns (batch, 1, 1, seq_len) for attention."""
    return padding_mask.unsqueeze(1).unsqueeze(2)  # (B, 1, 1, T)


class Transformer(nn.Module):
    def __init__(
        self,
        src_vocab_size: int,
        tgt_vocab_size: int,
        d_model: int = D_MODEL,
        n_heads: int = N_HEADS,
        n_layers: int = N_LAYERS,
        d_ff: int = D_FF,
        dropout: float = DROPOUT,
        max_len: int = MAX_LEN,
    ):
        super().__init__()
        self.d_model = d_model
        self.max_len = max_len
        self.src_embed = nn.Embedding(src_vocab_size, d_model, padding_idx=PAD_IDX)
        self.tgt_embed = nn.Embedding(tgt_vocab_size, d_model, padding_idx=PAD_IDX)
        self.pe = positional_encoding(
            max_len, d_model, torch.device("cpu")
        )  # moved to device in forward
        self.encoder_layers = nn.ModuleList(
            [EncoderLayer(d_model, n_heads, d_ff, dropout) for _ in range(n_layers)]
        )
        self.decoder_layers = nn.ModuleList(
            [DecoderLayer(d_model, n_heads, d_ff, dropout) for _ in range(n_layers)]
        )
        self.fc_out = nn.Linear(d_model, tgt_vocab_size)

    def forward(
        self,
        src: torch.Tensor,  # (batch, src_len)
        tgt: torch.Tensor,  # (batch, tgt_len)
        src_pad_mask: Optional[torch.Tensor] = None,
        tgt_pad_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        device = src.device
        self.pe = self.pe.to(device)
        src_len, tgt_len = src.size(1), tgt.size(1)

        # Encoder
        src_emb = self.src_embed(src) * math.sqrt(self.d_model) + self.pe[:, :src_len]
        src_attn_mask = (
            make_pad_mask(src_pad_mask) if src_pad_mask is not None else None
        )
        enc_out = src_emb
        for layer in self.encoder_layers:
            enc_out = layer(enc_out, src_attn_mask)

        # Decoder
        tgt_emb = self.tgt_embed(tgt) * math.sqrt(self.d_model) + self.pe[:, :tgt_len]
        causal = make_causal_mask(tgt_len, device)
        tgt_attn_mask = causal
        if tgt_pad_mask is not None:
            pm = make_pad_mask(tgt_pad_mask).expand(-1, -1, tgt_len, -1)
            tgt_attn_mask = tgt_attn_mask * pm
        src_tgt_mask = make_pad_mask(src_pad_mask) if src_pad_mask is not None else None
        dec_out = tgt_emb
        for layer in self.decoder_layers:
            dec_out = layer(dec_out, enc_out, tgt_attn_mask, src_tgt_mask)

        return self.fc_out(dec_out)


def test_transformer():
    src_vocab, tgt_vocab = 100, 100
    model = Transformer(src_vocab, tgt_vocab)
    B, src_len, tgt_len = 2, 5, 4
    src = torch.randint(1, src_vocab, (B, src_len))
    tgt = torch.randint(1, tgt_vocab, (B, tgt_len))
    logits = model(src, tgt)
    assert logits.shape == (B, tgt_len, tgt_vocab), logits.shape
    print("  [OK] Transformer")


# -----------------------------------------------------------------------------
# 7b. Wrapper around nn.Transformer: embeddings + PE + output projection (same interface as Transformer)
# -----------------------------------------------------------------------------
class TransformerWithEmbeddings(nn.Module):
    """Wraps nn.Transformer with embeddings and output projection so it accepts token indices like Transformer."""

    def __init__(
        self,
        src_vocab_size: int,
        tgt_vocab_size: int,
        d_model: int = D_MODEL,
        nhead: int = N_HEADS,
        num_encoder_layers: int = N_LAYERS,
        num_decoder_layers: int = N_LAYERS,
        dim_feedforward: int = D_FF,
        dropout: float = DROPOUT,
        max_len: int = MAX_LEN,
    ):
        super().__init__()
        self.d_model = d_model
        self.max_len = max_len
        self.src_embed = nn.Embedding(src_vocab_size, d_model, padding_idx=PAD_IDX)
        self.tgt_embed = nn.Embedding(tgt_vocab_size, d_model, padding_idx=PAD_IDX)
        self.pe = positional_encoding(max_len, d_model, torch.device("cpu"))
        self.transformer = TransformerBase(
            d_model=d_model,
            nhead=nhead,
            num_encoder_layers=num_encoder_layers,
            num_decoder_layers=num_decoder_layers,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            activation="relu",
            batch_first=False,
            norm_first=False,
        )
        self.fc_out = nn.Linear(d_model, tgt_vocab_size)

    def forward(
        self,
        src: torch.Tensor,  # (batch, src_len)
        tgt: torch.Tensor,  # (batch, tgt_len)
        src_pad_mask: Optional[torch.Tensor] = None,
        tgt_pad_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        device = src.device
        self.pe = self.pe.to(device)
        src_len, tgt_len = src.size(1), tgt.size(1)

        src_emb = self.src_embed(src) * math.sqrt(self.d_model) + self.pe[:, :src_len]
        tgt_emb = self.tgt_embed(tgt) * math.sqrt(self.d_model) + self.pe[:, :tgt_len]

        # nn.Transformer expects (S, N, E) when batch_first=False
        src_emb = src_emb.transpose(0, 1)  # (S, N, E)
        tgt_emb = tgt_emb.transpose(0, 1)  # (T, N, E)

        # key_padding_mask: (N, S) or (N, T), True = ignore (masked)
        src_key_padding_mask = (src_pad_mask == 0) if src_pad_mask is not None else None
        tgt_key_padding_mask = (tgt_pad_mask == 0) if tgt_pad_mask is not None else None

        tgt_mask = TransformerBase.generate_square_subsequent_mask(
            tgt_len, device=device
        )

        out = self.transformer(
            src_emb,
            tgt_emb,
            tgt_mask=tgt_mask,
            src_key_padding_mask=src_key_padding_mask,
            tgt_key_padding_mask=tgt_key_padding_mask,
        )  # (T, N, E)
        out = out.transpose(0, 1)  # (N, T, E)
        return self.fc_out(out)


# -----------------------------------------------------------------------------
# 8. Simple EN-RU dataset (tiny in-memory for learning)
# -----------------------------------------------------------------------------
def get_simple_en_ru_dataset():
    """Return a tiny list of (english, russian) pairs. No external deps; extend with HF later."""
    # Small parallel sentences (you can replace with datasets.load_dataset("opus100", "en-ru") and take 1k)
    pairs = [
        ("Hello world", "Привет мир"),
        ("I like cats", "Я люблю кошек"),
        ("The cat sat on the mat", "Кот сидел на коврике"),
        ("She is reading a book", "Она читает книгу"),
        ("We need more data", "Нам нужно больше данных"),
        ("This is a test", "Это тест"),
        ("Machine learning is useful", "Машинное обучение полезно"),
        ("Attention is all you need", "Внимание — это всё что нужно"),
        ("Good morning", "Доброе утро"),
        ("How are you", "Как дела"),
    ]
    return pairs


def test_dataset():
    pairs = get_simple_en_ru_dataset()
    assert len(pairs) >= 5
    assert isinstance(pairs[0], (list, tuple)) and len(pairs[0]) == 2
    print("  [OK] get_simple_en_ru_dataset")


# -----------------------------------------------------------------------------
# 9. Tokenizer stub (character or simple whitespace; replace with subword for real MT)
# -----------------------------------------------------------------------------
class SimpleTokenizer:
    """Minimal tokenizer: space split + special tokens. Good for learning; use HuggingFace tokenizer for real MT."""

    def __init__(self, sentences: list[str], is_ru: bool = False):
        self.is_ru = is_ru
        self.special = {
            "<pad>": PAD_IDX,
            "<sos>": SOS_IDX,
            "<eos>": EOS_IDX,
            "<unk>": UNK_IDX,
        }
        vocab = set(self.special.keys())
        for s in sentences:
            for t in s.split():
                vocab.add(t.lower())
        self.id2tok = sorted(vocab)
        self.tok2id = {t: i for i, t in enumerate(self.id2tok)}

    def encode(self, text: str, add_special: bool = True) -> list[int]:
        ids = [self.tok2id.get("<sos>", SOS_IDX)] if add_special else []
        for t in text.split():
            ids.append(self.tok2id.get(t.lower(), self.tok2id.get("<unk>", UNK_IDX)))
        if add_special:
            ids.append(self.tok2id.get("<eos>", EOS_IDX))
        return ids

    def decode(self, ids: list[int]) -> str:
        return " ".join(
            self.id2tok[i] if i < len(self.id2tok) else "<unk>"
            for i in ids
            if i not in (PAD_IDX, SOS_IDX, EOS_IDX)
        )

    @property
    def vocab_size(self) -> int:
        return len(self.id2tok)


def test_tokenizer():
    pairs = get_simple_en_ru_dataset()
    en_sentences = [p[0] for p in pairs]
    tok = SimpleTokenizer(en_sentences, is_ru=False)
    enc = tok.encode("Hello world")
    assert enc[0] == tok.tok2id.get("<sos>", SOS_IDX)
    assert tok.decode(enc)  # no crash
    print("  [OK] SimpleTokenizer")


# -----------------------------------------------------------------------------
# 10. Training step (one batch)
# -----------------------------------------------------------------------------
def train_step(
    model: Transformer,
    src: torch.Tensor,
    tgt: torch.Tensor,
    optimizer: torch.optim.Optimizer,
    criterion: nn.Module,
    src_pad_mask: Optional[torch.Tensor] = None,
    tgt_pad_mask: Optional[torch.Tensor] = None,
) -> float:
    """One forward + backward; returns loss scalar."""
    # tgt input: tgt[:, :-1]; target: tgt[:, 1:]
    tgt_in = tgt[:, :-1]
    tgt_out = tgt[:, 1:]
    tgt_in_pad = tgt_pad_mask[:, 1:] if tgt_pad_mask is not None else None
    logits = model(src, tgt_in, src_pad_mask, tgt_in_pad)
    loss = criterion(logits.reshape(-1, logits.size(-1)), tgt_out.reshape(-1))
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    return loss.item()


def evaluate_step(
    model: Transformer,
    src: torch.Tensor,
    tgt: torch.Tensor,
    criterion: nn.Module,
    src_pad_mask: Optional[torch.Tensor] = None,
    tgt_pad_mask: Optional[torch.Tensor] = None,
) -> float:
    # Keep evaluation consistent with train_step: teacher forcing with shift.
    tgt_in = tgt[:, :-1]
    tgt_out = tgt[:, 1:]
    tgt_in_pad = tgt_pad_mask[:, 1:] if tgt_pad_mask is not None else None
    logits = model(src, tgt_in, src_pad_mask, tgt_in_pad)
    loss = criterion(logits.reshape(-1, logits.size(-1)), tgt_out.reshape(-1))
    return loss.item()


def translate_and_show_gt(
    model: Transformer,
    src: str,
    tgt: str,
    tok_src: SimpleTokenizer,
    tok_tgt: SimpleTokenizer,
    src_pad_mask: Optional[torch.Tensor] = None,
    tgt_pad_mask: Optional[torch.Tensor] = None,
) -> str:
    model.eval()
    with torch.no_grad():
        src_ids = tok_src.encode(src)
        tgt_ids = tok_tgt.encode(tgt)
        src = torch.tensor([src_ids], dtype=torch.long)
        tgt = torch.tensor([tgt_ids], dtype=torch.long)
        src_mask_t = src_mask.unsqueeze(0)
        tgt_mask_t = tgt_mask.unsqueeze(0)
        logits = model(src, tgt, src_mask_t, tgt_mask_t)
        loss = criterion(logits.reshape(-1, logits.size(-1)), tgt.reshape(-1))
        return loss.item()


def pad_seq(ids: list[int], length: int) -> tuple[list[int], torch.Tensor]:
    pad = length - len(ids)
    mask = [1] * len(ids) + [0] * max(0, pad)
    ids = ids + [PAD_IDX] * max(0, pad)
    return ids[:length], torch.tensor(mask[:length], dtype=torch.float32)


def pad_seq_tensor(ids: torch.Tensor, length: int) -> tuple[torch.Tensor, torch.Tensor]:
    pad = length - ids.size(1)
    mask = [1] * ids.size(1) + [0] * max(0, pad)
    ids = torch.cat(
        [ids, torch.full((ids.size(0), pad), PAD_IDX, dtype=torch.long)], dim=1
    )
    return ids, torch.tensor(mask, dtype=torch.float32)


class TranslationDataset(Dataset):
    """Stores already-tokenized (src_ids, tgt_ids) pairs for batching."""

    def __init__(
        self,
        pairs: list[tuple[str, str]],
        tok_src: SimpleTokenizer,
        tok_tgt: SimpleTokenizer,
        max_len: int,
    ):
        super().__init__()
        self.samples: list[tuple[list[int], list[int]]] = []
        for en, ru in pairs:
            src_ids = tok_src.encode(en)[:max_len]
            tgt_ids = tok_tgt.encode(ru)[:max_len]
            self.samples.append((src_ids, tgt_ids))

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> tuple[list[int], list[int]]:
        return self.samples[idx]


def make_collate_fn(max_len: int):
    def collate_fn(
        batch: list[tuple[list[int], list[int]]],
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        src_seqs = [b[0] for b in batch]
        tgt_seqs = [b[1] for b in batch]

        src_len = min(max(len(s) for s in src_seqs), max_len)
        tgt_len = min(max(len(t) for t in tgt_seqs), max_len)

        bsz = len(batch)
        src = torch.full((bsz, src_len), PAD_IDX, dtype=torch.long)
        tgt = torch.full((bsz, tgt_len), PAD_IDX, dtype=torch.long)
        src_mask = torch.zeros((bsz, src_len), dtype=torch.float32)  # 1=valid, 0=pad
        tgt_mask = torch.zeros((bsz, tgt_len), dtype=torch.float32)  # 1=valid, 0=pad

        for i, s in enumerate(src_seqs):
            s = s[:src_len]
            src[i, : len(s)] = torch.tensor(s, dtype=torch.long)
            src_mask[i, : len(s)] = 1.0

        for i, t in enumerate(tgt_seqs):
            t = t[:tgt_len]
            tgt[i, : len(t)] = torch.tensor(t, dtype=torch.long)
            tgt_mask[i, : len(t)] = 1.0

        return src, src_mask, tgt, tgt_mask

    return collate_fn


def progress_bar(
    iteration: int, total: int, prefix: str = "", bar_len: int = 30
) -> None:
    """Lightweight terminal progress bar (no external deps like tqdm)."""
    if total <= 0:
        return
    iteration = min(max(iteration, 0), total)
    filled = int(round(bar_len * iteration / float(total)))
    filled = min(filled, bar_len)
    bar = "#" * filled + "-" * (bar_len - filled)
    print(f"\r{prefix}[{bar}] {iteration}/{total}", end="", flush=True)


@torch.no_grad()
def greedy_decode(
    model: Transformer,
    src_ids: list[int],
    tok_tgt: SimpleTokenizer,
    max_decode_len: int,
    device: torch.device,
    min_length: int = 2,
) -> str:
    """Greedy autoregressive decoding for EN->RU translation."""
    model.eval()
    src = torch.tensor([src_ids], dtype=torch.long, device=device)

    generated: list[int] = [SOS_IDX]
    for step in range(max_decode_len - 1):  # already have SOS
        tgt = torch.tensor([generated], dtype=torch.long, device=device)
        logits = model(src, tgt)  # (1, tgt_len, vocab)
        next_logits = logits[0, -1, :].clone()

        # Suppress EOS until we have at least min_length content tokens (avoids empty output)
        if step < min_length:
            next_logits[EOS_IDX] = float("-inf")

        next_id = int(torch.argmax(next_logits).item())
        generated.append(next_id)
        if next_id == EOS_IDX:
            break

    return tok_tgt.decode(generated)


def test_train_step():
    pairs = get_simple_en_ru_dataset()
    en_sents = [p[0] for p in pairs]
    ru_sents = [p[1] for p in pairs]
    tok_src = SimpleTokenizer(en_sents, False)
    tok_tgt = SimpleTokenizer(ru_sents, True)
    model = Transformer(tok_src.vocab_size, tok_tgt.vocab_size)
    opt = torch.optim.Adam(model.parameters(), lr=1e-4)
    criterion = nn.CrossEntropyLoss(ignore_index=PAD_IDX)
    src = torch.tensor(
        [tok_src.encode(pairs[0][0])[:8] + [PAD_IDX] * 2]
    )  # pad to same len
    tgt = torch.tensor([tok_tgt.encode(pairs[0][1])[:8] + [PAD_IDX] * 2])
    loss = train_step(model, src, tgt, opt, criterion)
    assert isinstance(loss, float)
    print("  [OK] train_step")


# -----------------------------------------------------------------------------
# Run all tests
# -----------------------------------------------------------------------------
def run_all_tests():
    print("Running tests...")
    test_positional_encoding()
    test_scaled_dot_product_attention()
    test_multi_head_attention()
    test_feed_forward()
    test_encoder_layer()
    test_decoder_layer()
    test_transformer()
    test_dataset()
    test_tokenizer()
    test_train_step()
    print("All tests passed.")


# -----------------------------------------------------------------------------
# Minimal training loop on tiny EN-RU data (run after tests)
# -----------------------------------------------------------------------------
def train_tiny_example(epochs: int = 20):
    """Train on the built-in tiny EN-RU pairs. For real MT use HuggingFace datasets."""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    pairs = get_simple_en_ru_dataset()
    en_sents = [p[0] for p in pairs]
    ru_sents = [p[1] for p in pairs]
    tok_src = SimpleTokenizer(en_sents, False)
    tok_tgt = SimpleTokenizer(ru_sents, True)
    model = Transformer(tok_src.vocab_size, tok_tgt.vocab_size).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)
    criterion = nn.CrossEntropyLoss(ignore_index=PAD_IDX)

    max_src = max(len(tok_src.encode(s)) for s in en_sents)
    max_tgt = max(len(tok_tgt.encode(s)) for s in ru_sents)
    max_src = min(max_src + 2, MAX_LEN)
    max_tgt = min(max_tgt + 2, MAX_LEN)

    def pad_seq(ids: list[int], length: int) -> tuple[list[int], torch.Tensor]:
        pad = length - len(ids)
        mask = [1] * len(ids) + [0] * max(0, pad)
        ids = ids + [PAD_IDX] * max(0, pad)
        return ids[:length], torch.tensor(mask[:length], dtype=torch.float32)

    for epoch in range(epochs):
        total_loss = 0.0
        for en, ru in pairs:
            src_ids, src_mask = pad_seq(tok_src.encode(en), max_src)
            tgt_ids, tgt_mask = pad_seq(tok_tgt.encode(ru), max_tgt)
            src = torch.tensor([src_ids], dtype=torch.long, device=device)
            tgt = torch.tensor([tgt_ids], dtype=torch.long, device=device)
            src_mask_t = src_mask.unsqueeze(0).to(device)
            tgt_mask_t = tgt_mask.unsqueeze(0).to(device)
            loss = train_step(
                model, src, tgt, optimizer, criterion, src_mask_t, tgt_mask_t
            )
            total_loss += loss
        if (epoch + 1) % 5 == 0:
            print(f"  Epoch {epoch + 1}/{epochs} loss = {total_loss / len(pairs):.4f}")
    print(
        "Done. For real EN-RU MT, use: datasets.load_dataset('Helsinki-NLP/opus-100', 'en-ru')."
    )


class _SummaryWrapper(nn.Module):
    def __init__(self, tr: Transformer):
        super().__init__()
        self.tr = tr

    def forward(self, src: torch.Tensor) -> torch.Tensor:
        # torch.nn.Transformer (native) expects *float* (S, N, E) when batch_first=False.
        # torchsummary uses batch_size=2, so it feeds (2, S, N, E) — strip to (S, N, E).
        x = src.float()
        while x.dim() > 3:
            x = x[0]
        tgt = x
        return self.tr(x, tgt)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--lr", type=float, default=1e-3)  # 1e-4
    parser.add_argument("--train-samples", type=int, default=15000)
    parser.add_argument("--val-samples", type=int, default=200)
    parser.add_argument("--test-samples", type=int, default=200)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    random.seed(args.seed)
    torch.manual_seed(args.seed)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    en_ru_dataset = datasets.load_dataset("Helsinki-NLP/opus-100", "en-ru")

    splits = set(en_ru_dataset.keys())
    val_split = "validation" if "validation" in splits else "train"
    test_split = "test" if "test" in splits else "train"

    train_pairs_raw = en_ru_dataset["train"]["translation"][: args.train_samples]
    val_pairs_raw = en_ru_dataset[val_split]["translation"][: args.val_samples]
    test_pairs_raw = en_ru_dataset[test_split]["translation"][: args.test_samples]

    en_train = [p["en"] for p in train_pairs_raw]
    ru_train = [p["ru"] for p in train_pairs_raw]

    # Build tokenizers on train split only.
    tok_src = SimpleTokenizer(en_train, False)
    tok_tgt = SimpleTokenizer(ru_train, True)

    train_pairs = [(p["en"], p["ru"]) for p in train_pairs_raw]
    val_pairs = [(p["en"], p["ru"]) for p in val_pairs_raw]
    test_pairs = [(p["en"], p["ru"]) for p in test_pairs_raw]

    print(f"Vocab size: {tok_src.vocab_size}, {tok_tgt.vocab_size}")
    # model = Transformer(tok_src.vocab_size, tok_tgt.vocab_size).to(device)
    model = TransformerWithEmbeddings(
        tok_src.vocab_size,
        tok_tgt.vocab_size,
        d_model=D_MODEL,
        nhead=N_HEADS,
        num_encoder_layers=N_LAYERS,
        num_decoder_layers=N_LAYERS,
        dim_feedforward=D_FF,
        dropout=DROPOUT,
        max_len=MAX_LEN,
    ).to(device)

    n_params = sum(p.numel() for p in model.parameters())
    print(f"Model parameters: {n_params:,}")

    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    criterion = nn.CrossEntropyLoss(ignore_index=PAD_IDX)

    train_ds = TranslationDataset(train_pairs, tok_src, tok_tgt, max_len=MAX_LEN)
    val_ds = TranslationDataset(val_pairs, tok_src, tok_tgt, max_len=MAX_LEN)
    test_ds = TranslationDataset(test_pairs, tok_src, tok_tgt, max_len=MAX_LEN)

    collate_fn = make_collate_fn(max_len=MAX_LEN)

    train_loader = DataLoader(
        train_ds,
        batch_size=args.batch_size,
        shuffle=True,
        drop_last=False,
        collate_fn=collate_fn,
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=args.batch_size,
        shuffle=False,
        drop_last=False,
        collate_fn=collate_fn,
    )
    test_loader = DataLoader(
        test_ds,
        batch_size=args.batch_size,
        shuffle=False,
        drop_last=False,
        collate_fn=collate_fn,
    )

    # N=3 translations from eval (validation) set each epoch.
    eval_print_pairs = train_pairs[:3]  # val_pairs[:3]

    for epoch in range(args.epochs):
        model.train()
        train_loss_sum = 0.0
        start = time.time()

        for batch_idx, (src, src_mask, tgt, tgt_mask) in enumerate(
            train_loader, start=1
        ):
            src = src.to(device)
            src_mask = src_mask.to(device)
            tgt = tgt.to(device)
            tgt_mask = tgt_mask.to(device)

            loss = train_step(
                model,
                src,
                tgt,
                optimizer,
                criterion,
                src_pad_mask=src_mask,
                tgt_pad_mask=tgt_mask,
            )
            train_loss_sum += loss
            progress_bar(
                batch_idx,
                len(train_loader),
                prefix=f"Epoch {epoch + 1}/{args.epochs} Train ",
            )

        print()  # finish progress bar line
        avg_train_loss = train_loss_sum / max(1, len(train_loader))

        # Validation loss
        model.eval()
        val_loss_sum = 0.0
        with torch.no_grad():
            for src, src_mask, tgt, tgt_mask in val_loader:
                src = src.to(device)
                src_mask = src_mask.to(device)
                tgt = tgt.to(device)
                tgt_mask = tgt_mask.to(device)
                val_loss_sum += evaluate_step(
                    model,
                    src,
                    tgt,
                    criterion,
                    src_pad_mask=src_mask,
                    tgt_pad_mask=tgt_mask,
                )

        avg_val_loss = val_loss_sum / max(1, len(val_loader))
        elapsed = time.time() - start
        print(
            f"Epoch {epoch + 1}/{args.epochs} done. "
            f"train_loss={avg_train_loss:.4f} val_loss={avg_val_loss:.4f} time={elapsed:.1f}s"
        )

        # Print translations.
        # print("Eval translations:")
        # for i, (en_text, ru_gt) in enumerate(eval_print_pairs, start=1):
        #     src_ids = tok_src.encode(en_text)[:MAX_LEN]
        #     ru_pred = greedy_decode(
        #         model=model,
        #         src_ids=src_ids,
        #         tok_tgt=tok_tgt,
        #         max_decode_len=MAX_LEN,
        #         device=device,
        #     )
        #     print(f"  {i}) EN: {en_text}")
        #     print(f"     RU_pred: {ru_pred}")
        #     print(f"     RU_gt:   {ru_gt}")

        # print()

    # Final test loss
    model.eval()
    test_loss_sum = 0.0
    with torch.no_grad():
        for src, src_mask, tgt, tgt_mask in test_loader:
            src = src.to(device)
            src_mask = src_mask.to(device)
            tgt = tgt.to(device)
            tgt_mask = tgt_mask.to(device)
            test_loss_sum += evaluate_step(
                model,
                src,
                tgt,
                criterion,
                src_pad_mask=src_mask,
                tgt_pad_mask=tgt_mask,
            )

    avg_test_loss = test_loss_sum / max(1, len(test_loader))
    print(f"Test_loss={avg_test_loss:.4f}")

    # Print translations.
    print("Eval translations:")
    for i, (en_text, ru_gt) in enumerate(eval_print_pairs, start=1):
        src_ids = tok_src.encode(en_text)[:MAX_LEN]
        ru_pred = greedy_decode(
            model=model,
            src_ids=src_ids,
            tok_tgt=tok_tgt,
            max_decode_len=MAX_LEN,
            device=device,
            min_length=2,
        )
        print(f"  {i}) EN: {en_text}")
        print(f"     RU_pred: {ru_pred}")
        print(f"     RU_gt:   {ru_gt}")
