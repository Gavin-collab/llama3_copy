# Copyright (c) Meta Platforms, Inc. and affiliates.
# This software may be used and distributed in accordance with the terms of the Llama 3 Community License Agreement.

import math
from dataclasses import dataclass
from typing import Optional, Tuple

import fairscale.nn.model_parallel.initialize as fs_init
import torch
import torch.nn.functional as F
from fairscale.nn.model_parallel.layers import (
    ColumnParallelLinear,
    RowParallelLinear,
    VocabParallelEmbedding,
)
from torch import nn


@dataclass
class ModelArgs:
    dim: int = 4096
    n_layers: int = 32
    n_heads: int = 32
    n_kv_heads: Optional[int] = None
    vocab_size: int = -1
    multiple_of: int = 256  # make SwiGLU hidden layer size multiple of large power of 2
    ffn_dim_multiplier: Optional[float] = None
    norm_eps: float = 1e-5
    rope_theta: float = 500000

    max_batch_size: int = 32
    max_seq_len: int = 2048


# a normalization layer
class RMSNorm(torch.nn.Module):
    def __init__(self, dim: int, eps: float = 1e-6):
        super().__init__()
        self.eps = eps
        # set the [1,1..] size: given dim, this is to be learnt
        self.weight = nn.Parameter(torch.ones(dim))

    def _norm(self, x):
        return x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.eps)

    def forward(self, x):
        output = self._norm(x.float()).type_as(x)
        # multiply output element-wise by the weights
        return output * self.weight


def precompute_freqs_cis(dim: int, end: int, theta: float = 10000.0):
    freqs = 1.0 / (theta ** (torch.arange(0, dim, 2)[: (dim // 2)].float() / dim))
    t = torch.arange(end, device=freqs.device, dtype=torch.float32)
    freqs = torch.outer(t, freqs)
    freqs_cis = torch.polar(torch.ones_like(freqs), freqs)  # complex64
    return freqs_cis


def reshape_for_broadcast(freqs_cis: torch.Tensor, x: torch.Tensor):
    ndim = x.ndim
    assert 0 <= 1 < ndim
    assert freqs_cis.shape == (x.shape[1], x.shape[-1])
    shape = [d if i == 1 or i == ndim - 1 else 1 for i, d in enumerate(x.shape)]
    return freqs_cis.view(*shape)


def apply_rotary_emb(
    xq: torch.Tensor,
    xk: torch.Tensor,
    freqs_cis: torch.Tensor,
) -> Tuple[torch.Tensor, torch.Tensor]:
    xq_ = torch.view_as_complex(xq.float().reshape(*xq.shape[:-1], -1, 2))
    xk_ = torch.view_as_complex(xk.float().reshape(*xk.shape[:-1], -1, 2))
    freqs_cis = reshape_for_broadcast(freqs_cis, xq_)
    xq_out = torch.view_as_real(xq_ * freqs_cis).flatten(3)
    xk_out = torch.view_as_real(xk_ * freqs_cis).flatten(3)
    return xq_out.type_as(xq), xk_out.type_as(xk)


def repeat_kv(x: torch.Tensor, n_rep: int) -> torch.Tensor:
    """torch.repeat_interleave(x, dim=2, repeats=n_rep)"""
    bs, slen, n_kv_heads, head_dim = x.shape
    if n_rep == 1:
        return x
    return (
        x[:, :, :, None, :]
        .expand(bs, slen, n_kv_heads, n_rep, head_dim)
        .reshape(bs, slen, n_kv_heads * n_rep, head_dim)
    )


class Attention(nn.Module):
    def __init__(self, args: ModelArgs):
        super().__init__()
        # number of kv_heads = number of heads is kv_heads is not available
        self.n_kv_heads = args.n_heads if args.n_kv_heads is None else args.n_kv_heads
        # some parallel size
        model_parallel_size = fs_init.get_model_parallel_world_size()
        # number of local heads = head floored by parallel size
        self.n_local_heads = args.n_heads // model_parallel_size
        # local kv heads  =  kv heads floored by parallel size
        self.n_local_kv_heads = self.n_kv_heads // model_parallel_size
        # some number = local-heads floored by local kv heads
        self.n_rep = self.n_local_heads // self.n_local_kv_heads
        # each head dimension = dimensions divided by number of heads
        self.head_dim = args.dim // args.n_heads

        # dense processing 
        # query weight: transform embeddings into queries
        self.wq = ColumnParallelLinear(
            # dimensions as input dimension
            args.dim,
            args.n_heads * self.head_dim,
            bias=False,
            gather_output=False,
            init_method=lambda x: x,
        )
        # key weight: embeddings to keys 
        self.wk = ColumnParallelLinear(
            args.dim,
            self.n_kv_heads * self.head_dim,
            bias=False,
            gather_output=False,
            init_method=lambda x: x,
        )
        # value weight: embeddings to values
        self.wv = ColumnParallelLinear(
            args.dim,
            self.n_kv_heads * self.head_dim,
            bias=False,
            gather_output=False,
            init_method=lambda x: x,
        )
        self.wo = RowParallelLinear(
            args.n_heads * self.head_dim,
            args.dim,
            bias=False,
            input_is_parallel=True,
            init_method=lambda x: x,
        )

        # caching keys
        # (max batch size, max sequence length, max local key and value heads, each head dimension)
        self.cache_k = torch.zeros(
            (
                args.max_batch_size,
                args.max_seq_len,
                self.n_local_kv_heads,
                self.head_dim,
            )
        ).cuda()
        # cache value
        self.cache_v = torch.zeros(
            (
                args.max_batch_size,
                args.max_seq_len,
                self.n_local_kv_heads,
                self.head_dim,
            )
        ).cuda()

    def forward(
        self,
        # input
        x: torch.Tensor,
        # start position
        start_pos: int,
        # frequencies
        freqs_cis: torch.Tensor,
        # mask to apply
        mask: Optional[torch.Tensor],
    ):
        # get batch size and sequece length
        bsz, seqlen, _ = x.shape
        # get queries, keys and values
        xq, xk, xv = self.wq(x), self.wk(x), self.wv(x)

        # reshape everything into ( batch size, sequence length, heads num, each head dimension )
        xq = xq.view(bsz, seqlen, self.n_local_heads, self.head_dim)
        xk = xk.view(bsz, seqlen, self.n_local_kv_heads, self.head_dim)
        xv = xv.view(bsz, seqlen, self.n_local_kv_heads, self.head_dim)

        # queries and keys but combined with their positional information
        xq, xk = apply_rotary_emb(xq, xk, freqs_cis=freqs_cis)

        # key and value cache pdated
        self.cache_k = self.cache_k.to(xq)
        self.cache_v = self.cache_v.to(xq)

        self.cache_k[:bsz, start_pos : start_pos + seqlen] = xk
        self.cache_v[:bsz, start_pos : start_pos + seqlen] = xv

        keys = self.cache_k[:bsz, : start_pos + seqlen]
        values = self.cache_v[:bsz, : start_pos + seqlen]

        # repeat k/v heads if n_kv_heads < n_heads
        keys = repeat_kv(
            keys, self.n_rep
        )  # (bs, cache_len + seqlen, n_local_heads, head_dim)
        values = repeat_kv(
            values, self.n_rep
        )  # (bs, cache_len + seqlen, n_local_heads, head_dim)

        # transpose queries, keys and keys
        xq = xq.transpose(1, 2)  # (bs, n_local_heads, seqlen, head_dim)
        keys = keys.transpose(1, 2)  # (bs, n_local_heads, cache_len + seqlen, head_dim)
        values = values.transpose(
            1, 2
        )  # (bs, n_local_heads, cache_len + seqlen, head_dim)
        # do some matmul 
        scores = torch.matmul(xq, keys.transpose(2, 3)) / math.sqrt(self.head_dim)
        if mask is not None:
            scores = scores + mask  # (bs, n_local_heads, seqlen, cache_len + seqlen)
        
        scores = F.softmax(scores.float(), dim=-1).type_as(xq)
        # apply values with the softmax scores 
        output = torch.matmul(scores, values)  # (bs, n_local_heads, seqlen, head_dim)
        output = output.transpose(1, 2).contiguous().view(bsz, seqlen, -1)
        return self.wo(output)


class FeedForward(nn.Module):
    def __init__(
        self, 
        dim: int, # dimension
        hidden_dim: int, # hidden dimension
        multiple_of: int, # some multiplier
        ffn_dim_multiplier: Optional[float], # another multipler
    ):
        super().__init__()
        # hidden dim = 2 hidden dims / 3
        hidden_dim = int(2 * hidden_dim / 3) 
        # custom dim factor multiplier
        if ffn_dim_multiplier is not None:
            hidden_dim = int(ffn_dim_multiplier * hidden_dim)

        hidden_dim = multiple_of * ((hidden_dim + multiple_of - 1) // multiple_of)

        self.w1 = ColumnParallelLinear(
            dim, hidden_dim, bias=False, gather_output=False, init_method=lambda x: x
        )
        self.w2 = RowParallelLinear(
            hidden_dim, dim, bias=False, input_is_parallel=True, init_method=lambda x: x
        )
        self.w3 = ColumnParallelLinear(
            dim, hidden_dim, bias=False, gather_output=False, init_method=lambda x: x
        )

    def forward(self, x):
        # dense input
        # dense input 2
        # multiply them 
        # activation
        # go through another dense
        return self.w2(F.silu(self.w1(x)) * self.w3(x))


class TransformerBlock(nn.Module):
    def __init__(self, layer_id: int, args: ModelArgs):
        super().__init__()
        # specify heads
        self.n_heads = args.n_heads
        # get dimensions
        self.dim = args.dim
        # dimension for each head = dimensions / heads but floored
        self.head_dim = args.dim // args.n_heads
        # create attention layer
        self.attention = Attention(args)
        # feed-forward module
        self.feed_forward = FeedForward(
            # has such dim
            dim=args.dim,
            # 4 times of hidden dimension
            hidden_dim=4 * args.dim,
            # some tensor oor number from parameters
            multiple_of=args.multiple_of,
            # some tensor or number from parameters
            ffn_dim_multiplier=args.ffn_dim_multiplier,
        )
        # the layer id
        self.layer_id = layer_id
        # create needed normalization layers
        self.attention_norm = RMSNorm(args.dim, eps=args.norm_eps)
        self.ffn_norm = RMSNorm(args.dim, eps=args.norm_eps)

    def forward(
        self,
        # the tensor to be processed
        x: torch.Tensor,
        # start position
        start_pos: int,
        # sinusoidal frequencies
        freqs_cis: torch.Tensor,
        # some mask
        mask: Optional[torch.Tensor],
    ):
        # normalize the input tensor
        # give the attention module the normalized input, start position, frequencies, and mask
        h = x + self.attention(self.attention_norm(x), start_pos, freqs_cis, mask)
        # normalize the attentions
        # let attentions go through a feed-forward module
        # add attentions to the input tensor
        out = h + self.feed_forward(self.ffn_norm(h))
        return out


class Transformer(nn.Module):
    def __init__(self, params: ModelArgs):
        super().__init__()

        self.params = params
        self.vocab_size = params.vocab_size
        self.n_layers = params.n_layers

        self.tok_embeddings = VocabParallelEmbedding(
            params.vocab_size, params.dim, init_method=lambda x: x
        )

        self.layers = torch.nn.ModuleList()
        for layer_id in range(params.n_layers):
            self.layers.append(TransformerBlock(layer_id, params))

        self.norm = RMSNorm(params.dim, eps=params.norm_eps)
        self.output = ColumnParallelLinear(
            params.dim, params.vocab_size, bias=False, init_method=lambda x: x
        )

        self.freqs_cis = precompute_freqs_cis(
            params.dim // params.n_heads,
            params.max_seq_len * 2,
            params.rope_theta,
        )

    @torch.inference_mode()
    def forward(self, tokens: torch.Tensor, start_pos: int):
        # tokens might be (10, 1000), batch_size and sequence length
        _bsz, seqlen = tokens.shape
        # get the embeddings
        h = self.tok_embeddings(tokens)
        # 
        self.freqs_cis = self.freqs_cis.to(h.device)
        # get the frequencies
        freqs_cis = self.freqs_cis[start_pos : start_pos + seqlen]
        mask = None
        if seqlen > 1:
            # create a mask full of some numbers
            mask = torch.full((seqlen, seqlen), float("-inf"), device=tokens.device)
            # why is mask created this way 
            mask = torch.triu(mask, diagonal=1)

            # When performing key-value caching, we compute the attention scores
            # only for the new sequence. Thus, the matrix of scores is of size
            # (seqlen, cache_len + seqlen), and the only masked entries are (i, j) for
            # j > cache_len + i, since row i corresponds to token cache_len + i.

            # concatenate zeros of shape ( [[0,0,0], [0,0,0]] ) 
            # same length as the content already generated
            mask = torch.hstack(
                [torch.zeros((seqlen, start_pos), device=tokens.device), mask]
            ).type_as(h)
        
        for layer in self.layers:
            # give each transformer embeddings, start_position, frequencies, and the mask
            h = layer(h, start_pos, freqs_cis, mask)
        # normalize it
        h = self.norm(h)
        # do some output processing
        output = self.output(h).float()
        return output
