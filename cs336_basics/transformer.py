
from doctest import OutputChecker
from turtle import forward
from numpy import ones_like
import torch
from  einops import einsum,rearrange
import einx
from torch import Tensor, nn
import math
from jaxtyping import Bool, Float, Int

from tests.conftest import mask


class Linear(torch.nn.Module):
    def __init__(self, in_features, out_features, device=None, dtype=None):
        """初始化线性变换模块
        参数:
            in_features (int): 输入的最终维度
            out_features (int): 输出的最终维度  
            device (torch.device | None): 参数存储设备，默认为None
            dtype (torch.dtype | None): 参数数据类型，默认为None
        """
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.device = device
        self.dtype = dtype
        self.weight = nn.Parameter(torch.empty((out_features, in_features), device=device, dtype=dtype))
        _sigma = (2 / (in_features + out_features))**0.5
        nn.init.trunc_normal_(self.weight, mean=0.0, std=_sigma, a = -3 * _sigma, b = 3 * _sigma)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """对输入应用线性变换"""
        return einsum(self.weight, x, "out_features in_features, ... in_features -> ... out_features")

class Embedding(torch.nn.Module):
    def __init__(self, num_embeddings, embedding_dim, device=None, dtype=None):
        """初始化嵌入模块
        参数:
            num_embeddings (int): 词表大小（词汇量）
            embedding_dim (int): 嵌入向量的维度（即d_model）
            device (torch.device | None): 参数存储设备，默认为None
            dtype (torch.dtype | None): 参数数据类型，默认为None
        """
        super().__init__()
        self.Embedding = nn.Parameter(torch.empty((num_embeddings, embedding_dim), device=device, dtype=dtype))
        _sigma = 1 
        nn.init.trunc_normal_(self.Embedding, mean=0.0, std=_sigma, a = -3 * _sigma, b = 3 * _sigma)

    def forward(self, token_ids: torch.Tensor) -> torch.Tensor:
        """根据输入的 token ID 查找对应的嵌入向量"""
        return self.Embedding[token_ids]

class RMSnorm(torch.nn.Module):
    def __init__(self, d_model: int, eps: float = 1e-5, device=None, dtype=None):
        """初始化RMSNorm模块
        参数:
            d_model (int): 模型的隐藏层维度
            eps (float): 数值稳定项，默认为1e-5
            device (torch.device | None): 参数存储设备，默认为None
            dtype (torch.dtype | None): 参数数据类型，默认为None
        """
        super().__init__()
        self.eps = eps
        self.g = nn.Parameter(torch.ones(d_model, device=device, dtype=dtype))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """处理输入张量（形状为(batch_size, sequence_length, d_model)）
        并返回相同形状的张量
        """
        rms = torch.sqrt(torch.mean(x**2, dim=-1, keepdim=True) + self.eps)
        return x  * self.g / rms



class FFN(torch.nn.Module):
    def __init__(self, d_model, d_ff=None, device=None, dtype=None):
        """初始化前馈神经网络模块。
        参数:
            d_model (int): 模型的隐藏层维度
            d_ff (int | None): 内部前馈层维度，默认约为 8/3 * d_model 并调整为64的倍数
            device (torch.device | None): 参数存储设备，默认为None
            dtype (torch.dtype | None): 参数数据类型，默认为None
        """
        super().__init__()
        if d_ff is None:
            d_ff = int(math.ceil(d_model * (8/3) / 64) * 64)
        self.d_ff = d_ff
        self.w1 = nn.Parameter(torch.empty((self.d_ff, d_model), device=device, dtype=dtype))
        self.w2 = nn.Parameter(torch.empty((d_model, self.d_ff), device=device, dtype=dtype))
        self.w3 = nn.Parameter(torch.empty((self.d_ff, d_model), device=device, dtype=dtype))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        对输入张量进行前馈神经网络变换
        参数:
            x (torch.Tensor): 输入张量，形状为(batch_size, sequence_length, d_model)
        """
        GLU = lambda x : einsum(x, torch.sigmoid(x),"... d_ff, ... d_ff -> ... d_ff")
        W1x = einsum(x, self.w1, "... d_model, d_ff d_model -> ... d_ff")
        W3x = einsum(x, self.w3, "... d_model, d_ff d_model -> ... d_ff")
        FFN3 = GLU(W1x) * W3x


        return  einsum(FFN3, self.w2, "... d_ff, d_model d_ff -> ... d_model")


class RotaryEmbedding(nn.Module):
    def __init__(self, context_length: int, dim: int, theta: float = 10000.0):
        super().__init__()
        self.register_buffer(
            "_freq_cis_cache",
            RotaryEmbedding._init_cache(context_length, dim, theta), persistent=False
        )
    
    @staticmethod
    def _init_cache(context_length: int, dim: int, theta: float) -> Float[Tensor, " 2 context_length half_dim"]:
        assert dim % 2 == 0

        d = torch.arange(0, dim, 2) / dim
        freqs = theta ** -d
        t = torch.arange(context_length)

        freqs = einsum(t, freqs, "t, f -> t f")

        cos, sin = torch.cos(freqs), torch.sin(freqs)
        return torch.stack((cos, sin))

    def forward(self, x: Float[Tensor, " ... seq d"], pos_ids: Int[Tensor, " ... seq"]) -> Float[Tensor, " ... seq d"]:
        x1, x2 = rearrange(x, '... (half_d xy) -> xy ... half_d', xy=2)

        # Standard
        # cos, sin = self._freq_cis_cache[:, pos_ids, :]

        # einx
        cos, sin = einx.get_at('cos_sin [pos] half_dim, ... -> cos_sin ... half_dim', self._freq_cis_cache, pos_ids)

        # 2D rotation matrix applied to pairs in x
        x1_rot = cos * x1 - sin * x2
        x2_rot = sin * x1 + cos * x2
        result = einx.rearrange('... x_half, ... x_half -> ... (x_half (1 + 1))', x1_rot, x2_rot).contiguous()
        return result
    
    def extra_repr(self):
        return f"context_length={self._freq_cis_cache.shape[0]}, dim/2={self._freq_cis_cache.shape[1]}"




def softmax_stable(in_tensor: Tensor, dim: int):
    max_val, _ = in_tensor.max(dim=dim, keepdim=True)
    in_new = in_tensor - max_val
    return torch.softmax(in_new, dim=dim)



class multihead_self_attention(torch.nn.Module):
    def __init__(self, d_model: int, num_heads:int,context_length:int=1024,device=None, dtype=None):
        """
        """
        super().__init__()
        self.d_model = d_model
        self.num_heads = num_heads
        self.d_k = d_model //  num_heads
        self.d_v = self.d_k
        self.context_length = context_length
        self.wq = nn.Parameter(torch.empty((num_heads*self.d_k,d_model), device=device,dtype=dtype))
        self.wk = nn.Parameter(torch.empty((num_heads*self.d_k,d_model), device=device,dtype=dtype))
        self.wv = nn.Parameter(torch.empty((num_heads*self.d_v,d_model), device=device,dtype=dtype))
        self.wo = nn.Parameter(torch.empty((d_model,num_heads*self.d_v), device=device,dtype=dtype))

    def run_scaled_dot_product_attention( self,
        Q: Float[Tensor, " ... queries d_k"],
        K: Float[Tensor, " ... keys d_k"],
        V: Float[Tensor, " ... values d_v"],
        mask: Bool[Tensor, " ... queries keys"] | None = None,
    ) -> Float[Tensor, " ... queries d_v"]:
        """
        Given key (K), query (Q), and value (V) tensors, return
        the output of your scaled dot product attention implementation.

        Args:
            Q (Float[Tensor, " ... queries d_k"]): Query tensor
            K (Float[Tensor, " ... keys d_k"]): Key tensor
            V (Float[Tensor, " ... values d_v"]): Values tensor
            mask (Bool[Tensor, " ... queries keys"] | None): Mask tensor
        Returns:
            Float[Tensor, " ... queries d_v"]: Output of SDPA
        """
        # raise NotImplementedError
        d_k = Q.shape[-1]
        scores = einsum(Q, K,"... queries d_k, ... keys d_k -> ... queries keys") / d_k**0.5
        masked_scores = scores.masked_fill(mask==False,float("-inf"))
        softmax_out =  softmax_stable(masked_scores, dim=-1)
        attention = einsum(softmax_out, V, "... queries keys, ... keys d_v -> ... queries d_v")
        return attention
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass with RoPE and causal attention."""
        batch, seq, _ = x.shape

        # QKV projections: (batch, seq, d_model) -> (batch, seq, num_heads * d_k)
        q = einsum(x, self.wq, "... d_model, d_kq d_model -> ... d_kq")
        k = einsum(x, self.wk, "... d_model, d_kk d_model -> ... d_kk")
        v = einsum(x, self.wv, "... d_model, d_kv d_model -> ... d_kv")

        # Reshape into heads: (batch, seq, num_heads*d_k) -> (batch, num_heads, seq, d_k)
        q = rearrange(q, "batch seq (num_heads d_k) -> batch num_heads seq d_k", num_heads=self.num_heads)
        k = rearrange(k, "batch seq (num_heads d_k) -> batch num_heads seq d_k", num_heads=self.num_heads)
        v = rearrange(v, "batch seq (num_heads d_v) -> batch num_heads seq d_v", num_heads=self.num_heads)

        # Apply RoPE
        rope = RotaryEmbedding(context_length=self.context_length, dim=self.d_k)
        pos = torch.arange(seq, device=x.device)
        pos = rearrange(pos, 's -> 1 1 s').expand(batch, self.num_heads, seq)
        q = rope(q, pos)
        k = rope(k, pos)

        # Causal mask: upper triangle (excluding diagonal) = masked
        causal_mask = torch.triu(torch.ones(seq, seq, device=x.device, dtype=torch.bool), diagonal=1)

        # Scaled dot product attention
        attn = self.run_scaled_dot_product_attention(q, k, v, causal_mask)

        # Reshape heads back: (batch, num_heads, seq, d_v) -> (batch, seq, num_heads*d_v)
        attn = rearrange(attn, "batch num_heads seq d_v -> batch seq (num_heads d_v)")

        # Output projection
        out = einsum(attn, self.wo, "batch seq d_model, d_model d_model -> batch seq d_model")

        return out


        
class transformer_block(torch.nn.Module):
    def __init__(self, d_model:int, num_heads:int,d_ff:int):
        super().__init__()
        self.d_model =d_model
        self.num_heads = num_heads
        self.d_ff =d_ff
    def forward(self, x):
        MHA = multihead_self_attention(d_model=self.d_model, num_heads=self.num_heads)
        norm = RMSnorm(d_model=self.d_model)
        y1 = x + MHA(norm(x))
        ffn = FFN(d_model=self.d_model,d_ff=self.d_ff)
        y2 = y1 + ffn(norm(x))
        return y2


