
import torch
from  einops import einsum 
from torch import nn
import math

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