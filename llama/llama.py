from collections.abc import Iterable
from itertools import islice
from functools import wraps
from typing import Any, Optional, Union, Tuple
from dataclasses import dataclass, field
from safetensors import safe_open
import math

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.distributed as dist
from linear import RowParallelLinear, ColumnParallelLinear, AvgGrad
torch.manual_seed(42)


# @dataclass
# class LlamaConfig:
#     hidden_size: int = 2048  # Llama 3-1b
#     vocab_size: int = 128256
#     num_hidden_layers: int = 16
#     num_attention_heads: int = 32
#     num_key_value_heads: int = 8
#     intermediate_size: int = 8192
#     pad_token_id: int = None
#     max_position_embeddings: int = 131072
#     rope_theta: float = 500000.0
#     rope_scaling: dict = field(
#         default_factory=lambda: {
#             "factor": 32.0,
#             "high_freq_factor": 4.0,
#             "low_freq_factor": 1.0,
#             "original_max_position_embeddings": 8192,
#             "rope_type": "llama3"
#         }
#     )
#     # partial_rotary_factor: float = 1.0
#     rms_norm_eps: float = 1e-05


@dataclass
class LlamaConfig:
    hidden_size: int = 4096  # Llama 3-8b
    vocab_size: int = 128256
    num_hidden_layers: int = 16
    num_attention_heads: int = 32
    num_key_value_heads: int = 8
    intermediate_size: int = 14336
    pad_token_id: int = None
    max_position_embeddings: int = 131072
    rope_theta: float = 500000.0
    rope_scaling: dict = field(
        default_factory=lambda: {
            "factor": 32.0,
            "high_freq_factor": 4.0,
            "low_freq_factor": 1.0,
            "original_max_position_embeddings": 8192,
            "rope_type": "llama3"
        }
    )
    # partial_rotary_factor: float = 1.0
    rms_norm_eps: float = 1e-05


class LlamaRotaryEmbedding(nn.Module):
    inv_freq: torch.Tensor  # fix linting for `register_buffer`

    def __init__(self, config: LlamaConfig, device=None):
        super().__init__()
        self.config = config
        self.max_seq_len_cached = config.max_position_embeddings
        self.original_max_seq_len = config.max_position_embeddings
        
        inv_freq = self._compute_llama3_parameters(device)
        self.register_buffer("inv_freq", inv_freq, persistent=False)
        self.original_inv_freq = self.inv_freq

    def _compute_default_rope_parameters(
        self,
        device: Optional["torch.device"] = None,
        # seq_len: Optional[int] = None,
    ) -> tuple["torch.Tensor", float]:
        config = self.config
        theta = config.rope_theta
        # partial_rotary_factor = config.partial_rotary_factor
        head_dim = config.hidden_size // config.num_attention_heads
        # dim = int(head_dim * partial_rotary_factor)
    
        attention_factor = 1.0  # Unused in this type of RoPE
        inv_freq = 1.0 / (theta ** (torch.arange(0, head_dim, 2, dtype=torch.int64).to(device=device, dtype=torch.float) / head_dim))
        return inv_freq  #, attention_factor
    
    def _compute_llama3_parameters(
        self,
        device: "torch.device",
        seq_len: Optional[int] = None
    ) -> tuple["torch.Tensor", float]:
        config = self.config
        inv_freq = self._compute_default_rope_parameters(device)
    
        factor = config.rope_scaling["factor"]  # `8` in the original implementation
        low_freq_factor = config.rope_scaling["low_freq_factor"]  # `1` in the original implementation
        high_freq_factor = config.rope_scaling["high_freq_factor"]  # `4` in the original implementation
        old_context_len = config.rope_scaling["original_max_position_embeddings"]
    
        low_freq_wavelen = old_context_len / low_freq_factor
        high_freq_wavelen = old_context_len / high_freq_factor
    
        wavelen = 2 * math.pi / inv_freq
        # wavelen < high_freq_wavelen: do nothing
        # wavelen > low_freq_wavelen: divide by factor
        inv_freq_llama = torch.where(wavelen > low_freq_wavelen, inv_freq / factor, inv_freq)
        # otherwise: interpolate between the two, using a smooth factor
        smooth_factor = (old_context_len / wavelen - low_freq_factor) / (high_freq_factor - low_freq_factor)
        smoothed_inv_freq = (1 - smooth_factor) * inv_freq_llama / factor + smooth_factor * inv_freq_llama
        is_medium_freq = ~(wavelen < high_freq_wavelen) * ~(wavelen > low_freq_wavelen)
        inv_freq_llama = torch.where(is_medium_freq, smoothed_inv_freq, inv_freq_llama)
    
        return inv_freq_llama  #, attention_factor
    
    @torch.no_grad()
    def forward(self, x, position_ids):
        inv_freq_expanded = self.inv_freq[None, :, None].float()
        inv_freq_expanded = inv_freq_expanded.expand(position_ids.shape[0], -1, 1).to(x.device)
        position_ids_expanded = position_ids[:, None, :].float()

        device_type = x.device.type
        with torch.autocast(device_type=device_type, enabled=False):  # Force float32
            freqs = (inv_freq_expanded.float() @ position_ids_expanded.float()).transpose(1, 2)
            emb = torch.cat((freqs, freqs), dim=-1)
            cos = emb.cos()  # * self.attention_scaling
            sin = emb.sin()  # * self.attention_scaling

        return cos.to(dtype=x.dtype), sin.to(dtype=x.dtype)


class LlamaAttention(nn.Module):
    def __init__(
        self,
        layer_idx: int,
        hidden_size: int,
        num_query_heads: int,
        num_kv_heads: int,
        flash_attn: bool = True,
        tp_rank: int = -1,
        tp_size: int = 1,
    ):
        super().__init__()
        assert hidden_size % num_query_heads == 0, "hidden_size must be divisible by num_query_heads"
        assert num_query_heads % num_kv_heads == 0, "num_query_heads must be divisible by num_kv_heads"
        
        self.hidden_size = hidden_size
        self.num_query_heads = num_query_heads
        self.num_kv_heads = num_kv_heads
        self.num_queries_per_kv = num_query_heads // num_kv_heads
        self.head_dim = hidden_size // num_query_heads
        self.kv_dim = num_kv_heads * self.head_dim
        self.flash_attn = flash_attn
        self.tp_rank = tp_rank
        self.tp_size = tp_size
        
        if tp_size > 1:
            self.q_proj = ColumnParallelLinear(hidden_size, hidden_size, tp_rank=tp_rank, tp_size=tp_size)
            self.k_proj = ColumnParallelLinear(hidden_size, self.kv_dim, tp_rank=tp_rank, tp_size=tp_size)
            self.v_proj = ColumnParallelLinear(hidden_size, self.kv_dim, tp_rank=tp_rank, tp_size=tp_size)
            self.o_proj = RowParallelLinear(hidden_size, hidden_size, tp_rank=tp_rank, tp_size=tp_size, split_input=False)
        else:
            self.q_proj = nn.Linear(hidden_size, hidden_size, bias=False)
            self.k_proj = nn.Linear(hidden_size, self.kv_dim, bias=False)
            self.v_proj = nn.Linear(hidden_size, self.kv_dim, bias=False)
            self.o_proj = nn.Linear(hidden_size, hidden_size, bias=False)

    @staticmethod
    def rotate_half(x):
        x1 = x[..., : x.shape[-1] // 2]
        x2 = x[..., x.shape[-1] // 2 :]
        return torch.cat((-x2, x1), dim=-1)

    @staticmethod
    def apply_rotary_emb(q, k, cos, sin, unsqueeze_dim=1):    
        cos = cos.unsqueeze(unsqueeze_dim)
        sin = sin.unsqueeze(unsqueeze_dim)
        q_embed = (q * cos) + (LlamaAttention.rotate_half(q) * sin)
        k_embed = (k * cos) + (LlamaAttention.rotate_half(k) * sin)
        return q_embed, k_embed

    def load(self, q_proj_w, k_proj_w, v_proj_w, o_proj_w):
        if self.tp_size > 1:
            start = self.hidden_size * self.tp_rank // self.tp_size
            end = self.hidden_size * (self.tp_rank + 1) // self.tp_size
            kv_start = self.kv_dim * self.tp_rank // self.tp_size
            kv_end = self.kv_dim * (self.tp_rank + 1) // self.tp_size

            self.q_proj.weight = nn.Parameter(q_proj_w[start:end, :])
            self.k_proj.weight = nn.Parameter(k_proj_w[kv_start:kv_end, :])
            self.v_proj.weight = nn.Parameter(v_proj_w[kv_start:kv_end, :])
            self.o_proj.weight = nn.Parameter(o_proj_w[:, start:end])
        else:
            self.q_proj.weight = nn.Parameter(q_proj_w)
            self.k_proj.weight = nn.Parameter(k_proj_w)
            self.v_proj.weight = nn.Parameter(v_proj_w)
            self.o_proj.weight = nn.Parameter(o_proj_w)

    def forward(self, x, pos_emb, mask=None) -> torch.Tensor:
        B, T, C = x.shape
        assert C == self.hidden_size
        # print(f'tp_rank {self.tp_rank} x before proj:', x.shape)
        
        # Q Shape: (batch_size, num_query_heads, seq_len, head_dim)        
        # K/V Shape: (batch_size, num_kv_heads, seq_len, head_dim)
        q = self.q_proj(x).view(B, T, -1, self.head_dim).transpose(1, 2)
        k = self.k_proj(x).view(B, T, -1, self.head_dim).transpose(1, 2)
        v = self.v_proj(x).view(B, T, -1, self.head_dim).transpose(1, 2)
        
        cos, sin = pos_emb
        q, k = LlamaAttention.apply_rotary_emb(q, k, cos, sin)
        
        if self.flash_attn:
            attn_out = torch.nn.functional.scaled_dot_product_attention(
                q, k, v, is_causal=True, enable_gqa=True
            )
        else:
            # Shape: (batch_size, num_query_heads, seq_len, head_dim)
            k = k.repeat_interleave(self.num_queries_per_kv, dim=1)
            v = v.repeat_interleave(self.num_queries_per_kv, dim=1)
            
            attn_weights = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(self.head_dim)
            mask = torch.ones(T, T, dtype=torch.bool).tril(diagonal=0)
            attn_weights.masked_fill_(mask.logical_not(), float("-inf"))
            # if mask is not None:
            #     scores = scores.masked_fill(mask == 0, float('-inf'))
            attn_weights = torch.softmax(attn_weights, dim=-1)
            attn_out = torch.matmul(attn_weights, v)
        
        attn_out = attn_out.transpose(1, 2).contiguous().view(B, T, -1)
        return self.o_proj(attn_out)


class LlamaMLP(nn.Module):
    def __init__(
        self,
        hidden_size: int,
        intermediate_size: int,
        bias: bool = False,
        tp_rank: int = -1,
        tp_size: int = 1,
    ) -> None:
        super().__init__()
        self.hidden_size = hidden_size
        self.intermediate_size = intermediate_size
        self.bias = bias
        self.tp_rank = tp_rank
        self.tp_size = tp_size

        if tp_size > 1:
            self.gate_up_proj = ColumnParallelLinear(
                idim=hidden_size,
                odim=intermediate_size*2,
                add_bias=bias,
                tp_rank=tp_rank,
                tp_size=tp_size,
            )
            self.down_proj = RowParallelLinear(
                idim=intermediate_size,
                odim=hidden_size,
                add_bias=bias,
                tp_rank=tp_rank,
                tp_size=tp_size,
                split_input=False,
            )
        else:
            self.gate_up_proj = nn.Linear(
                in_features=hidden_size,
                out_features=intermediate_size*2,
                bias=bias,
            )
            self.down_proj = nn.Linear(
                in_features=intermediate_size,
                out_features=hidden_size,
                bias=bias,
            )
            # self.act_fn = SiluAndMul()

    def load(self, gate_proj_w, up_proj_w, down_proj_w):
        if self.tp_size > 1:
            start = self.intermediate_size * self.tp_rank // self.tp_size
            end = self.intermediate_size * (self.tp_rank + 1) // self.tp_size
            gate_up_w = torch.cat([gate_proj_w[start:end, :], up_proj_w[start:end, :]], dim=0)
            self.gate_up_proj.weight = nn.Parameter(gate_up_w)
            self.down_proj.weight = nn.Parameter(down_proj_w[:, start:end])
        else:
            self.gate_up_proj.weight = nn.Parameter(torch.cat([gate_proj_w, up_proj_w], dim=0))
            self.down_proj.weight = nn.Parameter(down_proj_w)

    def forward(self, x):
        gate, x = self.gate_up_proj(x).chunk(2, dim=-1)
        x = F.silu(gate) * x
        x = self.down_proj(x)
        return x


class LlamaRMSNorm(nn.Module):
    def __init__(self, hidden_size, eps=1e-6):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(hidden_size))
        self.variance_epsilon = eps

    def forward(self, x):
        input_dtype = x.dtype
        x = x.to(torch.float32)
        variance = x.pow(2).mean(-1, keepdim=True)
        x = x * torch.rsqrt(variance + self.variance_epsilon)
        x = self.weight * x.to(input_dtype)
        AvgGrad.apply(self.weight)
        return x

    def extra_repr(self):
        return f"{tuple(self.weight.shape)}, eps={self.variance_epsilon}"


class LlamaDecoderLayer(nn.Module):
    '''Extending from GradientCheckpointingLayer in Huggingface's implemention'''
    
    def __init__(self, config: LlamaConfig, layer_idx: int, tp_rank: int, tp_size: int):
        super().__init__()
        self.hidden_size = config.hidden_size
        self.tp_rank = tp_rank
        self.tp_size = tp_size

        self.attn = LlamaAttention(
            layer_idx=layer_idx,
            hidden_size=config.hidden_size,
            num_query_heads=config.num_attention_heads,
            num_kv_heads=config.num_key_value_heads,
            tp_rank=tp_rank,
            tp_size=tp_size,
        )

        self.mlp = LlamaMLP(
            hidden_size=config.hidden_size,
            intermediate_size=config.intermediate_size,
            tp_rank=tp_rank,
            tp_size=tp_size,
        )
        self.pre_attn_layernorm = LlamaRMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.pre_mlp_layernorm = LlamaRMSNorm(config.hidden_size, eps=config.rms_norm_eps)

    def load(self, **kwargs):
        self.attn.load(
            kwargs['q_proj_w'],
            kwargs['k_proj_w'],
            kwargs['v_proj_w'],
            kwargs['o_proj_w'],
        )
        self.mlp.load(
            kwargs['gate_w'],
            kwargs['up_w'],
            kwargs['down_w'],
        )

    def forward(
        self, x: torch.Tensor,
        pos_emb: Optional[tuple[torch.Tensor, torch.Tensor]],
        mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        residual = x
        x = self.pre_attn_layernorm(x)
        x = self.attn(x, pos_emb)
        x = residual + x

        # Fully Connected
        residual = x
        x = self.pre_mlp_layernorm(x)
        x = self.mlp(x)
        x = residual + x
        return x


class LlamaModel(nn.Module):
    def __init__(self, config: LlamaConfig, device='meta', tp_rank: int = -1, tp_size: int = 1):
        super().__init__()
        self.config = config
        self.padding_idx = config.pad_token_id
        self.vocab_size = config.vocab_size

        with torch.device(device):
            self.embed_tokens = nn.Embedding(config.vocab_size, config.hidden_size, self.padding_idx)
            self.layers = nn.ModuleList(
                [LlamaDecoderLayer(config, layer_idx, tp_rank, tp_size) 
                 for layer_idx in range(config.num_hidden_layers)]
            )
            self.norm = LlamaRMSNorm(config.hidden_size, eps=config.rms_norm_eps)
            self.lm_head = nn.Linear(config.hidden_size, config.vocab_size, bias=False)

        self.rotary_emb = LlamaRotaryEmbedding(config=config)

    def forward(self, x, mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        x = self.embed_tokens(x)
        B, T, _ = x.shape
        pos_ids = torch.arange(T, dtype=torch.long).unsqueeze(0).expand(B, -1).to(x.device)
        pos_emb = self.rotary_emb(x, pos_ids)

        for layer in self.layers:
            x = layer(x, pos_emb)

        x = self.norm(x)
        x = self.lm_head(x)
        return x


def tp_attn(x, pos_emb, queue, qw, kw, vw, ow, gate_w, up_w, down_w, hidden_size, num_query_heads, num_kv_heads, tp_rank, tp_size):
    import os
    os.environ['MASTER_ADDR'] = "127.0.0.1"
    os.environ['MASTER_PORT'] = "23333"
    dist.init_process_group(backend='gloo', init_method='env://', world_size=tp_size, rank=tp_rank)
    print(f"TP Rank {tp_rank} initialized.")

    config = LlamaConfig()
    tp_dim = hidden_size // tp_size
    tp_int_dim = config.intermediate_size // tp_size

    layer = LlamaDecoderLayer(
        config=config,
        layer_idx=0,
        tp_rank=tp_rank,
        tp_size=tp_size,
    )
    layer.load(
        q_proj_w=qw,
        k_proj_w=kw,
        v_proj_w=vw,
        o_proj_w=ow,
        gate_w=gate_w,
        up_w=up_w,
        down_w=down_w,
    )

    y_attn = layer.attn(x, pos_emb)
    y_0 = layer(x, pos_emb)
    y = y_0.sum()
    y.backward()
    
    y_attn_m = queue.get()
    y_0_m = queue.get()
    start, end = tp_rank * tp_dim, (tp_rank + 1) * tp_dim
    q_grad_m = queue.get()[start:end, :]  # q_grad torch.Size([16, 16]), down_grad torch.Size([16, 32])
    start, end = tp_rank * tp_int_dim, (tp_rank + 1) * tp_int_dim
    down_grad_m = queue.get()[:, start:end]

    # print(f'Result@{tp_rank}: q_grad {q_grad_m.shape}, down_grad {down_grad_m.shape}')
    print(f'Difference between attn output@{tp_rank}:', (y_attn - y_attn_m).abs().mean())
    print(f'Difference between output@{tp_rank}:', (y_0 - y_0_m).abs().mean())
    print(f'Difference between Q grad@{tp_rank}:', (layer.attn.q_proj.weight.grad - q_grad_m).abs().mean())
    print(f'Difference between Down grad@{tp_rank}:', (layer.mlp.down_proj.weight.grad - down_grad_m).abs().mean())
    print(f'tp_rank {tp_rank}: {y_0}')


if __name__ == "__main__":
    import torch.multiprocessing as mp

    # config = LlamaConfig(
    #     hidden_size = 16,
    #     num_attention_heads = 8,
    #     num_key_value_heads = 2,
    #     intermediate_size = 16,
    # )

    config = LlamaConfig()
    hidden_size=config.hidden_size
    num_query_heads=config.num_attention_heads
    num_kv_heads=config.num_key_value_heads
    head_dim = hidden_size // num_query_heads

    layer = LlamaDecoderLayer(
        config=config,
        layer_idx=0,
        tp_rank=0,
        tp_size=1
    )

    qw = torch.randn(hidden_size, hidden_size, requires_grad=True)
    kw = torch.randn(head_dim * num_kv_heads, hidden_size, requires_grad=True)
    vw = torch.randn(head_dim * num_kv_heads, hidden_size, requires_grad=True)
    ow = torch.randn(hidden_size, hidden_size, requires_grad=True)
    gate_w = torch.randn(config.intermediate_size, hidden_size, requires_grad=True)
    up_w = torch.randn(config.intermediate_size, hidden_size, requires_grad=True)
    down_w = torch.randn(hidden_size, config.intermediate_size, requires_grad=True)

    layer.load(
        q_proj_w=qw,
        k_proj_w=kw,
        v_proj_w=vw,
        o_proj_w=ow,
        gate_w=gate_w,
        up_w=up_w,
        down_w=down_w,
    )

    T, B = 6, 1
    x = torch.randn(B, T, hidden_size)
    rotary_emb = LlamaRotaryEmbedding(config=config)
    pos_ids = torch.arange(T, dtype=torch.long).unsqueeze(0).expand(B, -1)
    pos_emb = rotary_emb(x, pos_ids)

    y_attn = layer.attn(x, pos_emb)
    y_0 = layer(x, pos_emb)
    print('Normal forward:', y_0)
    y = y_0.sum()
    y.backward()

    tp_size = 2
    queues = [mp.Queue() for _ in range(tp_size)]
    workers = [mp.Process(target=tp_attn, args=(x, pos_emb, queues[tp_rank],
                                                qw, kw, vw, ow,
                                                gate_w, up_w, down_w,
                                                config.hidden_size,
                                                config.num_attention_heads,
                                                config.num_key_value_heads,
                                                tp_rank, tp_size)) for tp_rank in range(tp_size)]
    for w in workers:
        w.start()

    y_attn = y_attn.detach().share_memory_()
    y_0 = y_0.detach().share_memory_()
    q_grad = layer.attn.q_proj.weight.grad.detach().share_memory_()
    down_grad = layer.mlp.down_proj.weight.grad.detach().share_memory_()

    for q in queues:
        q.put(y_attn)
        q.put(y_0)
        q.put(q_grad)
        q.put(down_grad)

    for w in workers:
        w.join()
