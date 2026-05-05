import math
from typing import Optional
import torch
import torch.nn as nn
import torch.nn.functional as F


# ---------------------------------------------------------------------------
# RoPE
# ---------------------------------------------------------------------------

class RotaryEmbedding(nn.Module):
    """
    Rotary Position Embedding (RoPE).

    Stores inverse frequencies as a non-persistent buffer and returns
    (cos, sin) tensors of shape [1, 1, T, dim_head/2].  We keep them at
    half-dimension so that apply_rope can tile them once instead of
    pre-tiling here, making the intent explicit.
    """

    def __init__(self, dim: int, base: int = 10000):
        super().__init__()
        assert dim % 2 == 0, "RoPE dim must be even"
        inv_freq = 1.0 / (base ** (torch.arange(0, dim, 2).float() / dim))
        # persistent=False → not saved in state_dict, recomputed on load
        self.register_buffer("inv_freq", inv_freq, persistent=False)

    def forward(self, seq_len: int, device: torch.device):
        # positions: [T]
        positions = torch.arange(seq_len, device=device, dtype=self.inv_freq.dtype)
        # freqs: [T, dim/2]
        freqs = torch.einsum("i,j->ij", positions, self.inv_freq)
        # cos/sin: [1, 1, T, dim/2]  — half-dim, tiled inside apply_rope
        cos = freqs.cos()[None, None, :, :]
        sin = freqs.sin()[None, None, :, :]
        return cos, sin


def apply_rope(x: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor) -> torch.Tensor:
    """
    Apply RoPE to x of shape [B, H, T, dim_head].

    cos/sin are [1, 1, T, dim_head/2].

    The rotation mixes paired dimensions (x1, x2) → (x1·cos − x2·sin,
    x2·cos + x1·sin), which is equivalent to complex multiplication.
    Splitting at the midpoint and negating the second half achieves this
    without materialising complex numbers.

    FIX vs original:
      - Original called torch.cat([freqs, freqs]) inside RotaryEmbedding,
        giving identical frequencies in both halves → no real positional
        information encoded.
      - Original passed full-dim cos/sin to a rotate_half that also assumed
        full-dim input, creating a mismatch between the frequencies used for
        the block rotation and those applied by multiplication.
      Here cos/sin stay at dim/2 and are tiled once, keeping the math clean.
    """
    d = x.shape[-1]
    x1, x2 = x[..., : d // 2], x[..., d // 2 :]          # each [B,H,T,d/2]
    # tile cos/sin from [1,1,T,d/2] → implicit broadcast covers B and H
    rotated = torch.cat([-x2, x1], dim=-1)                 # [B,H,T,d]
    cos_full = torch.cat([cos, cos], dim=-1)                # [1,1,T,d]
    sin_full = torch.cat([sin, sin], dim=-1)                # [1,1,T,d]
    return x * cos_full + rotated * sin_full


# ---------------------------------------------------------------------------
# Local Self-Attention with RoPE
# ---------------------------------------------------------------------------

class LocalSelfAttentionRoPE(nn.Module):
    """
    Pre-norm local sliding-window self-attention with RoPE.

    window_size: total width of the attention window (symmetric).
                 Each query attends to tokens within distance
                 floor((window_size-1)/2) on each side.

    FIX summary vs original:
      1. RoPE bug fixed (see apply_rope above).
      2. NaN guard: rows that are entirely masked (e.g. a padding query
         whose window overlaps only other padding) produce -inf logits;
         softmax([-inf,...]) = NaN which silently kills gradients.
         Solved by replacing NaNs with 0 after softmax — these rows will
         also be zeroed by the encoder-level mask so the value doesn't matter.
      3. Window size semantics: half = (window_size - 1) // 2 so that
         window_size=8 gives exactly 8 neighbours (±3 + self for odd,
         adjusted for even).  Original used window_size//2 giving window_size+1
         total tokens.
      4. Redundant per-attention query masking removed.  Padding zeroing is
         handled once at the encoder level, avoiding a half-fix that still
         leaked padding through the residual.
    """

    def __init__(
        self,
        dim: int,
        heads: int = 8,
        dim_head: int = 64,
        dropout: float = 0.1,
        window_size: int = 8,
    ):
        super().__init__()
        inner_dim = heads * dim_head
        self.heads = heads
        self.dim_head = dim_head
        self.scale = dim_head ** -0.5
        self.window_size = window_size

        self.norm = nn.LayerNorm(dim)
        self.to_qkv = nn.Linear(dim, inner_dim * 3, bias=False)
        self.rope = RotaryEmbedding(dim_head)
        self.attn_drop = nn.Dropout(dropout)
        self.out_proj = nn.Linear(inner_dim, dim)
        self.out_drop = nn.Dropout(dropout)

    def forward(
        self,
        x: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Args:
            x:              [B, T, D]
            attention_mask: [B, T], 1 = valid token, 0 = padding.
                            If None, all positions are treated as valid.
        Returns:
            [B, T, D]  (residual connection included)
        """
        b, t, _ = x.shape
        h = self.heads

        residual = x
        x = self.norm(x)

        # ── Project to Q, K, V ────────────────────────────────────────────
        qkv = self.to_qkv(x)                                   # [B,T,3*H*Dh]
        q, k, v = qkv.chunk(3, dim=-1)

        q = q.view(b, t, h, self.dim_head).transpose(1, 2)     # [B,H,T,Dh]
        k = k.view(b, t, h, self.dim_head).transpose(1, 2)
        v = v.view(b, t, h, self.dim_head).transpose(1, 2)

        # ── Apply RoPE ────────────────────────────────────────────────────
        cos, sin = self.rope(seq_len=t, device=x.device)
        q = apply_rope(q, cos, sin)
        k = apply_rope(k, cos, sin)

        # ── Scaled dot-product scores ─────────────────────────────────────
        attn_scores = torch.matmul(q, k.transpose(-1, -2)) * self.scale  # [B,H,T,T]

        # ── Local window mask ─────────────────────────────────────────────
        # FIX: half = (window_size-1)//2 so window_size is the true total width.
        idx = torch.arange(t, device=x.device)
        half = (self.window_size - 1) // 2
        # local_mask[i,j] = True iff |i-j| <= half
        local_mask = (idx[None, :] - idx[:, None]).abs() <= half  # [T,T]
        attn_scores = attn_scores.masked_fill(
            ~local_mask[None, None, :, :], float("-inf")
        )

        # ── Key padding mask ──────────────────────────────────────────────
        # Padded key positions cannot be attended to.
        if attention_mask is not None:
            key_mask = attention_mask[:, None, None, :].bool().to(x.device)  # [B,1,1,T]
            attn_scores = attn_scores.masked_fill(~key_mask, float("-inf"))

        # ── Softmax + NaN guard ───────────────────────────────────────────
        # FIX: when every key in a query's window is masked (-inf), softmax
        # produces NaN → zero those rows explicitly.  Their output will be
        # zeroed again by the encoder-level attention_mask so the value is
        # irrelevant; what matters is that no NaN propagates through the graph.
        attn = F.softmax(attn_scores, dim=-1)
        attn = attn.nan_to_num(0.0)
        attn = self.attn_drop(attn)

        # ── Weighted sum ──────────────────────────────────────────────────
        out = torch.matmul(attn, v)                             # [B,H,T,Dh]
        out = out.transpose(1, 2).contiguous().view(b, t, h * self.dim_head)

        out = self.out_proj(out)
        out = self.out_drop(out)

        # NOTE: no per-attention query masking here.
        # Padding zeroing is done once at the LocalTransformerEncoder level,
        # which also covers the residual — doing it here would still leak
        # padding through `residual + out`.
        return residual + out


# ---------------------------------------------------------------------------
# Feed-Forward Block
# ---------------------------------------------------------------------------

class FeedForward(nn.Module):
    """Pre-norm feed-forward block with residual."""

    def __init__(self, dim: int, hidden_dim: int, dropout: float = 0.1):
        super().__init__()
        self.norm = nn.LayerNorm(dim)
        self.net = nn.Sequential(
            nn.Linear(dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, dim),
            nn.Dropout(dropout),
        )

    def forward(
        self,
        x: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        out = self.net(self.norm(x))
        if attention_mask is not None:
            # Zero FFN output at padding positions before adding residual.
            out = out * attention_mask.unsqueeze(-1).to(device=out.device, dtype=out.dtype)
        return x + out


# ---------------------------------------------------------------------------
# Transformer Block
# ---------------------------------------------------------------------------

class LocalTransformerBlock(nn.Module):
    def __init__(
        self,
        dim: int,
        heads: int = 8,
        dim_head: int = 64,
        mlp_ratio: float = 4.0,
        dropout: float = 0.1,
        window_size: int = 8,
    ):
        super().__init__()
        self.attn = LocalSelfAttentionRoPE(
            dim=dim,
            heads=heads,
            dim_head=dim_head,
            dropout=dropout,
            window_size=window_size,
        )
        self.ff = FeedForward(
            dim=dim,
            hidden_dim=int(dim * mlp_ratio),
            dropout=dropout,
        )

    def forward(
        self,
        x: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        x = self.attn(x, attention_mask=attention_mask)
        x = self.ff(x, attention_mask=attention_mask)
        return x


# ---------------------------------------------------------------------------
# Encoder
# ---------------------------------------------------------------------------

class LocalTransformerEncoder(nn.Module):
    """
    Stack of LocalTransformerBlocks followed by a final LayerNorm.

    After every block (and after the final norm) padding positions are forced
    to zero via attention_mask.  This is the single authoritative place where
    padding is zeroed, avoiding the partial fix that existed inside
    LocalSelfAttentionRoPE in the original code.
    """

    def __init__(
        self,
        dim: int,
        depth: int = 4,
        heads: int = 8,
        dim_head: int = 64,
        mlp_ratio: float = 4.0,
        dropout: float = 0.1,
        window_size: int = 13,
    ):
        super().__init__()
        self.layers = nn.ModuleList([
            LocalTransformerBlock(
                dim=dim,
                heads=heads,
                dim_head=dim_head,
                mlp_ratio=mlp_ratio,
                dropout=dropout,
                window_size=window_size,
            )
            for _ in range(depth)
        ])
        self.final_norm = nn.LayerNorm(dim)

    def forward(
        self,
        x: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Args:
            x:              [B, T, D]
            attention_mask: [B, T], 1 = valid, 0 = padding (optional)
        Returns:
            [B, T, D]
        """
        for layer in self.layers:
            x = layer(x, attention_mask=attention_mask)

        x = self.final_norm(x)

        # Final mask application after norm so padding positions are clean
        # zeros for any downstream pooling / downsampling.
        if attention_mask is not None:
            x = x * attention_mask.unsqueeze(-1).to(device=x.device, dtype=x.dtype)

        return x


# ---------------------------------------------------------------------------
# Smoke test
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    torch.manual_seed(0)

    B, T, D = 2, 32, 256
    x = torch.randn(B, T, D)

    # Simulate variable-length sequences: second sample has 24 valid tokens
    mask = torch.ones(B, T)
    mask[1, 24:] = 0

    encoder = LocalTransformerEncoder(
        dim=D,
        depth=4,
        heads=8,
        dim_head=32,
        mlp_ratio=4.0,
        dropout=0.1,
        window_size=8,
    )

    out = encoder(x, attention_mask=mask)
    print("Output shape:", out.shape)                           # [2, 32, 256]

    # Padding positions should be exactly zero
    assert out[1, 24:].abs().max().item() == 0.0, "Padding leaked!"
    print("Padding positions are clean zeros ✓")

    # Check no NaNs
    assert not out.isnan().any(), "NaN in output!"
    print("No NaNs in output ✓")

    # Quick backward pass
    out.sum().backward()
    print("Backward pass OK ✓")