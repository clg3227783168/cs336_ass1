from __future__ import annotations
import torch.nn as nn
from jaxtyping import Float, Bool, Int
import torch
from torch import Tensor
import math
from einops import rearrange, einsum
from collections.abc import Iterable
import os
import json
import einx
import logging

def softmax(x, dim=-1):
    rescaled_input = x - torch.max(x, dim=dim, keepdim=True)[0]
    exponentiated_rescaled_input = torch.exp(rescaled_input)
    return exponentiated_rescaled_input / torch.sum(exponentiated_rescaled_input, dim=dim, keepdim=True)

# def linear(
#     weights: Float[Tensor, " d_out d_in"],
#     in_features: Float[Tensor, " ... d_in"],
# ) -> Float[Tensor, " ... d_out"]:
#     # return einsum(in_features, weights, "... d_in, d_out d_in -> ... d_out") 都可以
#     return torch.matmul(in_features, weights.t())

# class Linear(nn.Module):
#     def __init__(weights: Float[Tensor, " d_out d_in"]):
#         super().__init__()
#         self.weights = weights
#     def forward(self,
#         in_features: Float[Tensor, " ... d_in"]
#     ) -> Float[Tensor, " ... d_out"]:
#         return einsum(in_features, weights, "... d_in, d_out d_in -> ... d_out")
class Linear(nn.Module):
    def __init__(self, d_in: int, d_out: int):
        """A linear layer initialized with truncated normal fan-in fan-out.

        Args:
            d_in: int
                The number of input features.
            d_out: int
                The number of output features.
        """
        
        super().__init__()
        std = math.sqrt(2 / (d_in + d_out))
        self.weight: Float[Tensor, " d_out d_in"] = nn.Parameter(
            nn.init.trunc_normal_(torch.empty(d_out, d_in), std=std, a=-3*std, b=3*std),
            requires_grad=True
        )

    def forward(self, x: Float[Tensor, " ... d_in"]) -> Float[Tensor, " ... d_out"]:
        return einsum(x, self.weight, "... d_in, d_out d_in -> ... d_out")
    
    def extra_repr(self):
        return f"d_out={self.weight.shape[0]}, d_in={self.weight.shape[1]}"

def silu(x: torch.Tensor):
    return x * torch.sigmoid(x)

class SiLU(nn.Module):
    def __init__(self, d_model: int):
        super().__init__()
        d_ff = 4 * d_model
        self.w1 = Linear(d_model, d_ff)
        self.w2 = Linear(d_ff, d_model)

    def forward(self, x):
        return self.w2(silu(self.w1(x)))

class SwiGLU(nn.Module):
    def __init__(self, d_model: int, d_ff: int):
        super().__init__()
        self.w1 = Linear(d_model, d_ff)
        self.w2 = Linear(d_ff, d_model)
        self.w3 = Linear(d_model, d_ff)

    def forward(self, x):
        return self.w2(silu(self.w1(x)) * self.w3(x))

class RMSNorm(nn.Module):
    """
    This module implements root mean square layer normalization, as
    described in Eq. 4 of https://arxiv.org/abs/1910.07467

    Args:
        hidden_size: int
            Dimensionality of the input to normalize.
        eps: float, default is 1e-5
            A value added to the denominator for numerical stability.

    Returns:
        FloatTensor of same shape as input.
    """

    def __init__(
        self,
        hidden_size: int,
        eps: float = 1e-5,
        device=None,
    ):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(hidden_size, device=device))
        self.eps = eps

    def forward(self, x):
        """
        Args:
            x: FloatTensor of shape `(batch_size, *)`.
                The input to apply root mean square layer normalization on.

        Returns:
            FloatTensor of same shape as input
        """
        # NOTE: in practice, many implementations will
        # manually upcast the input to fp32 here to prevent overflow when you
        # square the input.
        # https://github.com/pytorch/pytorch/issues/66707
        in_dtype = x.dtype

        x = x.to(torch.float32)
        rms = torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.eps)
        x = x * rms

        return (self.weight * x).to(in_dtype)
    
    def extra_repr(self):
        return f"hidden_size={self.weight.shape[0]}, eps={self.eps}"