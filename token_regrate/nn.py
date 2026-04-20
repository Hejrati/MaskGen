
import math

import torch
import torch.nn as nn
import torch.nn.functional as F


class TokenRegretCritic(nn.Module):
    def __init__(
        self,
        hidden_dim,
        text_dim,
        timestep_dim=32,
        mlp_dim=512,
        logits_topk=8,
        prompt_gap_topk=0,
        use_hidden=True,
    ):
        """Construct a per-token critic that predicts scalar counterfactual regret."""
        super().__init__()
        self.use_hidden = bool(use_hidden)
        self.logits_topk = int(logits_topk)
        self.prompt_gap_topk = max(0, int(prompt_gap_topk))
        self.timestep_dim = int(timestep_dim)
        in_dim = self.timestep_dim + self.logits_topk + self.prompt_gap_topk + text_dim
        if self.use_hidden:
            in_dim += hidden_dim
        self.net = nn.Sequential(
            nn.Linear(in_dim, mlp_dim),
            nn.SiLU(),
            nn.Linear(mlp_dim, mlp_dim),
            nn.SiLU(),
            nn.Linear(mlp_dim, 1),
        )

    def _timestep_embedding(self, timesteps):
        """Create sinusoidal embeddings for normalized diffusion timesteps."""
        half = self.timestep_dim // 2
        freqs = torch.exp(-math.log(10000) * torch.arange(half, device=timesteps.device, dtype=torch.float32) / max(half, 1))
        args = timesteps.float().unsqueeze(1) * freqs.unsqueeze(0)
        emb = torch.cat([torch.sin(args), torch.cos(args)], dim=-1)
        if self.timestep_dim % 2 == 1:
            emb = F.pad(emb, (0, 1))
        return emb

    def _logit_features(self, logits):
        """Build compact generator-state features from token logits."""
        k = min(self.logits_topk, logits.shape[-1])
        topk = torch.topk(logits, k=k, dim=-1).values
        if k < self.logits_topk:
            topk = F.pad(topk, (0, self.logits_topk - k))
        return topk

    def _prompt_gap_features(self, prompt_logits_delta, logits):
        """Build compact cond-minus-uncond logit-gap features."""
        if self.prompt_gap_topk <= 0:
            return None
        if prompt_logits_delta is None:
            return logits.new_zeros((*logits.shape[:-1], self.prompt_gap_topk))
        k = min(self.prompt_gap_topk, prompt_logits_delta.shape[-1])
        topk = torch.topk(prompt_logits_delta, k=k, dim=-1).values
        if k < self.prompt_gap_topk:
            topk = F.pad(topk, (0, self.prompt_gap_topk - k))
        return topk

    def forward(self, hidden_states, logits, timesteps, text_features, prompt_logits_delta=None):
        """Predict scalar regret per token position."""
        bsz, seq_len, _ = logits.shape
        t_feat = self._timestep_embedding(timesteps).unsqueeze(1).expand(bsz, seq_len, -1)
        logit_feat = self._logit_features(logits)
        prompt_gap_feat = self._prompt_gap_features(prompt_logits_delta, logits)
        text_feat = text_features.unsqueeze(1).expand(bsz, seq_len, -1)
        chunks = [t_feat, logit_feat, text_feat]
        if self.use_hidden:
            chunks.insert(1, hidden_states)
        if prompt_gap_feat is not None:
            insert_at = 2 if self.use_hidden else 1
            chunks.insert(insert_at, prompt_gap_feat)
        x = torch.cat(chunks, dim=-1)
        return self.net(x).squeeze(-1)
