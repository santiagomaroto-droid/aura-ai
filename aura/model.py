"""
AURA Core Architecture (v2.1)
=============================
Adaptive Unitary Resonant Architecture with Associative Complex Matrix States.

Mathematical Specification:
1. Outer-Product Associative Update:
   Delta S_t = beta_t * (k_t (x) v_t)
   where beta_t in R^{H x K} is a per-dimension selective write gate.

2. Unitary Complex State Rotation (Value Axis):
   S_t = (lambda_t * e^{i * theta_t}) * S_{t-1} + Delta S_t
   where rotation is on the value axis to keep key queries orthogonal and invariant.

3. Constant O(1) Memory Inference:
   State memory is strictly fixed at H x K x V complex numbers, completely independent
   of sequence length (zero KV-Cache).
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Tuple, List


class RMSNorm(nn.Module):
    """Root Mean Square Layer Normalization."""
    def __init__(self, dim: int, eps: float = 1e-6):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.eps) * self.weight


class AURAMatrixCell(nn.Module):
    """
    Resonant Associative Matrix Core Cell.
    State shape: [Batch, Heads, d_k, d_v] in Complex Space.
    """
    def __init__(self, d_model: int, n_heads: int = 4, d_k: int = 24, d_v: int = 24):
        super().__init__()
        self.d_model = d_model
        self.n_heads = n_heads
        self.d_k = d_k
        self.d_v = d_v

        # Projections
        self.proj_q = nn.Linear(d_model, n_heads * d_k, bias=False)
        self.proj_k = nn.Linear(d_model, n_heads * d_k, bias=False)
        self.proj_v = nn.Linear(d_model, n_heads * d_v, bias=False)

        # Decay and Phase modulation on VALUE axis
        self.proj_decay = nn.Linear(d_model, n_heads * d_v, bias=True)
        self.proj_theta = nn.Linear(d_model, n_heads * d_v, bias=False)

        # Base geometric frequencies
        freqs = torch.exp(torch.linspace(math.log(0.001), math.log(0.5), n_heads * d_v))
        self.register_buffer("base_freqs", freqs)

        # Per-dimension selective write gate on KEY axis
        self.proj_beta = nn.Linear(d_model, n_heads * d_k, bias=True)

        # Readout projections
        self.readout_norm = RMSNorm(n_heads * d_v)
        self.proj_out = nn.Linear(n_heads * d_v, d_model, bias=False)
        self.proj_gate = nn.Linear(d_model, d_model, bias=False)

        self._init_weights()

    def _init_weights(self):
        # Long memory decay bias by default: sigmoid(4) ~ 0.982
        nn.init.constant_(self.proj_decay.bias, 4.0)
        nn.init.normal_(self.proj_theta.weight, std=0.005)
        nn.init.constant_(self.proj_beta.bias, -1.0)
        for proj in [self.proj_q, self.proj_k, self.proj_v]:
            nn.init.normal_(proj.weight, std=0.02)
        nn.init.normal_(self.proj_out.weight, std=0.005)
        nn.init.normal_(self.proj_gate.weight, std=0.005)

    def forward(
        self, x: torch.Tensor, prev_state: Optional[torch.Tensor] = None
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        B, T, _ = x.shape
        device = x.device
        H, K, V = self.n_heads, self.d_k, self.d_v

        q = F.normalize(self.proj_q(x).view(B, T, H, K), dim=-1)
        k = F.normalize(self.proj_k(x).view(B, T, H, K), dim=-1)
        v = self.proj_v(x).view(B, T, H, V)

        decay = torch.sigmoid(self.proj_decay(x)).view(B, T, H, V)
        delta_theta = torch.tanh(self.proj_theta(x)).view(B, T, H, V) * 0.5
        theta = self.base_freqs.view(1, 1, H, V) + delta_theta

        beta = torch.sigmoid(self.proj_beta(x)).view(B, T, H, K)

        a_real = decay * torch.cos(theta)
        a_imag = decay * torch.sin(theta)

        if prev_state is None:
            S_real = torch.zeros(B, H, K, V, device=device)
            S_imag = torch.zeros(B, H, K, V, device=device)
        else:
            S_real = prev_state.real
            S_imag = prev_state.imag

        outputs = []
        for t in range(T):
            ar = a_real[:, t, :, :].unsqueeze(2)  # [B, H, 1, V]
            ai = a_imag[:, t, :, :].unsqueeze(2)

            S_new_real = S_real * ar - S_imag * ai
            S_new_imag = S_real * ai + S_imag * ar

            kt = k[:, t, :, :]
            vt = v[:, t, :, :]
            bt = beta[:, t, :, :]

            gated_k = kt * bt
            delta_S = torch.einsum("bhk,bhv->bhkv", gated_k, vt)

            S_real = S_new_real + delta_S
            S_imag = S_new_imag

            qt = q[:, t, :, :]
            y = torch.einsum("bhk,bhkv->bhv", qt, S_real)
            outputs.append(y.reshape(B, -1).unsqueeze(1))

        all_y = torch.cat(outputs, dim=1)
        readout = self.proj_out(self.readout_norm(all_y))
        gate = F.silu(self.proj_gate(x))
        return readout * gate, torch.complex(S_real, S_imag)

    def step(
        self, x_t: torch.Tensor, prev_state: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """O(1) Step-by-step autoregressive inference."""
        B = x_t.shape[0]
        H, K, V = self.n_heads, self.d_k, self.d_v

        q = F.normalize(self.proj_q(x_t).view(B, H, K), dim=-1)
        k = F.normalize(self.proj_k(x_t).view(B, H, K), dim=-1)
        v = self.proj_v(x_t).view(B, H, V)

        decay = torch.sigmoid(self.proj_decay(x_t)).view(B, H, V)
        delta_theta = torch.tanh(self.proj_theta(x_t)).view(B, H, V) * 0.5
        theta = self.base_freqs.view(1, H, V) + delta_theta

        beta = torch.sigmoid(self.proj_beta(x_t)).view(B, H, K)

        ar = (decay * torch.cos(theta)).unsqueeze(2)  # [B, H, 1, V]
        ai = (decay * torch.sin(theta)).unsqueeze(2)


        S_real = prev_state.real
        S_imag = prev_state.imag

        S_new_real = S_real * ar - S_imag * ai
        S_new_imag = S_real * ai + S_imag * ar

        gated_k = k * beta
        delta_S = torch.einsum("bhk,bhv->bhkv", gated_k, v)

        S_real = S_new_real + delta_S
        S_imag = S_new_imag

        y = torch.einsum("bhk,bhkv->bhv", q, S_real)
        readout = self.proj_out(self.readout_norm(y.reshape(B, -1)))
        gate = F.silu(self.proj_gate(x_t))
        out = readout * gate

        return out, torch.complex(S_real, S_imag)


class AURAMatrixBlock(nn.Module):
    def __init__(self, d_model: int, n_heads: int = 4, d_k: int = 24, d_v: int = 24):
        super().__init__()
        d_ffn = int(d_model * 2.67)
        self.norm1 = RMSNorm(d_model)
        self.cell = AURAMatrixCell(d_model, n_heads, d_k, d_v)
        self.norm2 = RMSNorm(d_model)
        self.mlp = nn.Sequential(
            nn.Linear(d_model, d_ffn, bias=False),
            nn.SiLU(),
            nn.Linear(d_ffn, d_model, bias=False),
        )

    def forward(self, x: torch.Tensor, prev_state: Optional[torch.Tensor] = None):
        res, state = self.cell(self.norm1(x), prev_state)
        x = x + res
        x = x + self.mlp(self.norm2(x))
        return x, state

    def step(self, x_t: torch.Tensor, prev_state: torch.Tensor):
        res, state = self.cell.step(self.norm1(x_t), prev_state)
        x_t = x_t + res
        x_t = x_t + self.mlp(self.norm2(x_t))
        return x_t, state


class AURAMatrixModel(nn.Module):
    """
    Complete Generative Language Model based on AURA-Matrix.
    """
    def __init__(
        self,
        vocab_size: int,
        d_model: int = 96,
        n_layers: int = 3,
        n_heads: int = 4,
        d_k: int = 24,
        d_v: int = 24,
    ):
        super().__init__()
        self.vocab_size = vocab_size
        self.d_model = d_model
        self.n_layers = n_layers
        self.n_heads = n_heads
        self.d_k = d_k
        self.d_v = d_v

        self.embed = nn.Embedding(vocab_size, d_model)
        self.blocks = nn.ModuleList([
            AURAMatrixBlock(d_model, n_heads, d_k, d_v) for _ in range(n_layers)
        ])
        self.norm = RMSNorm(d_model)
        self.head = nn.Linear(d_model, vocab_size, bias=False)
        self.head.weight = self.embed.weight

    def forward(self, x: torch.Tensor, targets: Optional[torch.Tensor] = None):
        h = self.embed(x)
        for block in self.blocks:
            h, _ = block(h)
        h = self.norm(h)
        logits = self.head(h)
        loss = None
        if targets is not None:
            loss = F.cross_entropy(logits.view(-1, self.vocab_size), targets.view(-1))
        return logits, loss

    @torch.no_grad()
    def generate(
        self,
        prompt_ids: torch.Tensor,
        max_new_tokens: int = 60,
        temperature: float = 0.75,
        top_k: Optional[int] = 25,
    ) -> torch.Tensor:
        """Autoregressive generation with strictly O(1) state memory."""
        self.eval()
        B = prompt_ids.shape[0]
        device = prompt_ids.device

        # Warm up
        h = self.embed(prompt_ids)
        states = []
        for block in self.blocks:
            h, s = block(h)
            states.append(s)
        h = self.norm(h)

        logits = self.head(h[:, -1, :]) / max(temperature, 1e-4)
        if top_k is not None:
            v, _ = torch.topk(logits, min(top_k, logits.size(-1)))
            logits[logits < v[:, [-1]]] = -float("inf")
        probs = F.softmax(logits, dim=-1)
        tok = torch.multinomial(probs, 1)
        generated = [tok]

        # O(1) step stepping
        curr = tok
        for _ in range(max_new_tokens - 1):
            h_t = self.embed(curr).squeeze(1)
            new_states = []
            for i, block in enumerate(self.blocks):
                h_t, s_new = block.step(h_t, states[i])
                new_states.append(s_new)
            states = new_states

            h_t = self.norm(h_t)
            logits = self.head(h_t) / max(temperature, 1e-4)
            if top_k is not None:
                v, _ = torch.topk(logits, min(top_k, logits.size(-1)))
                logits[logits < v[:, [-1]]] = -float("inf")
            probs = F.softmax(logits, dim=-1)
            curr = torch.multinomial(probs, 1)
            generated.append(curr)

        return torch.cat([prompt_ids, torch.cat(generated, dim=1)], dim=1)
