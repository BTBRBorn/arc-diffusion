import torch
from torch import nn
import torch.nn.functional as F


class MLP(nn.Module):
    def __init__(self, *args, **kwargs):
        super().__init__()
        config = kwargs["config"]
        self.e_proj = nn.Linear(config.emb_dim, 4 * config.emb_dim, bias=False)
        self.gelu = nn.GELU()

        self.c_proj = nn.Linear(config.emb_dim * 4, config.emb_dim, bias=False)
        assert not hasattr(self.c_proj, "RESIDUAL_INIT")
        self.c_proj.RESIDUAL_INIT = 1

    def forward(self, x: torch.Tensor):
        return self.c_proj(self.gelu(self.e_proj(x)))


class SelfAttention(nn.Module):
    def __init__(self, *args, **kwargs):
        super().__init__()
        config = kwargs["config"]
        self.config = config
        self.device = kwargs["device"]
        self.QKV = nn.Linear(config.emb_dim, 3 * config.emb_dim, bias=False)

        self.proj = nn.Linear(config.emb_dim, config.emb_dim, bias=False)
        assert not hasattr(self.proj, "RESIDUAL_INIT")
        self.proj.RESIDUAL_INIT = 1

    def forward(self, x: torch.Tensor, attention_mode):
        B, T, C = x.size()
        Q, K, V = self.QKV(x).split(
            self.config.emb_dim, dim=2
        )  # (B, T, 3C) -> ((B, T, C), (B, T, C), (B, T, C))

        Q = Q.view(B, T, self.config.n_head, self.config.head_size).transpose(
            1, 2
        )  # (B, T, C) -> (B, n_head, T, head_size)
        K = K.view(B, T, self.config.n_head, self.config.head_size).transpose(
            1, 2
        )  # (B, T, C) -> (B, n_head, T, head_size)
        V = V.view(B, T, self.config.n_head, self.config.head_size).transpose(
            1, 2
        )  # (B, T, C) -> (B, n_head, T, head_size)
        if attention_mode == "flash_attention":
            with nn.attention.sdpa_kernel(nn.attention.SDPBackend.FLASH_ATTENTION):
                y = F.scaled_dot_product_attention(Q, K, V, is_causal=False)

        elif attention_mode == "efficient_memory":
            with nn.attention.sdpa_kernel(nn.attention.SDPBackend.EFFICIENT_ATTENTION):
                y = F.scaled_dot_product_attention(Q, K, V, is_causal=False)

        elif attention_mode == "standard":
            att = (Q @ K.transpose(-2, -1)) * (
                self.config.head_size**-0.5
            )  # (B, n_head, T, head_size) -> (B, n_head, T, T)
            att = F.softmax(att, dim=-1)

            y = (
                att @ V
            )  # (B, n_head, T, T) @ (B, n_head, T, head_size) -> (B, n_head, T, head_size)

        else:
            raise ValueError(
                "attention_mode has to be 'flash_attention', 'efficient_memory' or 'standard'."
            )

        y = y.transpose(2, 1).reshape(B, T, C)

        return self.proj(y)


class Block(nn.Module):
    def __init__(self, *args, **kwargs):
        super().__init__()
        config = kwargs["config"]
        self.device = kwargs["device"]
        self.ln1 = nn.LayerNorm(config.emb_dim, bias=False)
        self.msa = SelfAttention(config=config, device=self.device)
        self.ln2 = nn.LayerNorm(config.emb_dim, bias=False)
        self.mlp = MLP(config=config)

    def forward(self, x: torch.Tensor, attention_mode):
        x = x + self.msa(self.ln1(x), attention_mode)
        x = x + self.mlp(self.ln2(x))
        return x


class GPT(nn.Module):
    def __init__(self, *args, **kwargs):
        super().__init__()
        config = kwargs["config"]
        self.device = kwargs['device']
        self.config = config
        self.wte = nn.Embedding(config.vocab_size, config.emb_dim)
        self.pte = nn.Embedding(config.block_size, config.emb_dim)
        self.blocks = nn.ModuleList(
            [Block(config=config, device=self.device) for _ in range(config.n_layer)]
        )
        self.ln = nn.LayerNorm(config.emb_dim, bias=False)
        self.f_head = nn.Linear(config.emb_dim, config.vocab_size, bias=False)

        self.register_buffer(
            "pos_inx", torch.arange(config.block_size, device=self.device)
        )

        self.f_head.weight = self.wte.weight
        assert id(self.f_head.weight) == id(self.wte.weight)

        self.apply(self._weight_init)

    def _weight_init(self, module):
        if isinstance(module, nn.Linear):
            std = module.in_features ** (-0.5)
            if hasattr(module, "RESIDUAL_INIT"):
                std *= self.config.n_layer ** (-0.5)
            nn.init.trunc_normal_(module.weight, mean=0, std=std)
            if module.bias is not None:
                nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            std = self.config.emb_dim ** (-0.5)
            nn.init.trunc_normal_(module.weight, mean=0, std=std)

    def configure_optimizer(self):
        config = self.config

        optim_groups = [
            {
                "params": [p for p in self.parameters() if p.dim() >= 2],
                "weight_decay": config.weight_decay,
            },
            {
                "params": [p for p in self.parameters() if p.dim() < 2],
                "weight_decay": 0.0,
            },
        ]

        optimizer = torch.optim.AdamW(
            optim_groups,
            lr=config.learning_rate,
            betas=(0.9, 0.95),
            fused=True,
        )

        return optimizer

    def forward(self, x: torch.Tensor, attention_mode="flash_attention"):
        B, T = x.size()
        x = self.wte(x) + self.pte(self.pos_inx[:T])  # (B, T, C) + (T, C) -> (B, T, C)
        for block in self.blocks:
            x = block(x, attention_mode)  # (B, T, C) -> (B, T, C)
        return self.f_head(self.ln(x))  # (B, T, vocab_size)