
import torch
from  einops import einsum,rearrange
import einx
from torch import Tensor, nn
import math
from jaxtyping import Bool, Float, Int


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


class RotaryPositionalEmbedding(nn.Module):
    def __init__(self, theta: float, d_k: int, max_seq_len: int, device=None):
        """初始化RoPE模块并创建缓冲区(如需要)
        参数:
            theta (float): RoPE的Θ参数值
            d_k (int): 查询向量和键向量的维度
            max_seq_len (int): 输入的最大序列长度
            device (torch.device | None): 缓冲区存储设备，默认为None
        """

    def forward(self, x: torch.Tensor, token_positions: torch.Tensor) -> torch.Tensor:
        """处理输入张量(形状为(..., seq_len, d_k))并返回相同形状的张量
        参数:
            x: 任意批次维度的输入张量
            token_positions: 形状为(..., seq_len)的位置张量，指定x在序列维度的位置
            使用token_positions参数对预计算的cos/sin张量进行切片

        """

def softmax_stable(in_tensor: Tensor, dim: int):
    max_val, _ = in_tensor.max(dim=dim, keepdim=True)
    in_new = in_tensor - max_val
    return torch.softmax(in_new, dim=dim)



class multihead_self_attention(torch.nn.Module):
    def __init__(self, d_model: int, num_heads:int ,device=None, dtype=None):
        """
        """
        super().__init__()
        self.d_model = d_model
        self.num_heads = num_heads
        self.d_k = d_model //  num_heads
        self.d_v = self.d_k
        self.wq = nn.Parameter(torch.empty((num_heads*self.d_k,d_model), device=device,dtype=dtype))
        self.wk = nn.Parameter(torch.empty((num_heads*self.d_k,d_model), device=device,dtype=dtype))
        self.wv = nn.Parameter(torch.empty((num_heads*self.d_v,d_model), device=device,dtype=dtype))
        self.wo = nn.Parameter(torch.empty((d_model,num_heads*self.d_v), device=device,dtype=dtype))
    def run_scaled_dot_product_attention(
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
        """
        """
        q = einsum(x, self.wq, "... d_model, d_kq d_model -> ... d_kq")
        k = einsum(x, self.wk, "... d_model, d_kk d_model -> ... d_kk")
        v = einsum(x, self.wv, "... d_model, d_kv d_model -> ... d_kv")
        
        q = rearrange(q,"batch, seq, num_heads * d_k -> ")

        







