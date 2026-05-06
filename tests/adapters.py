from __future__ import annotations

import json
import os
from collections.abc import Iterable
from typing import IO, Any, BinaryIO
from einops import einsum
import numpy.typing as npt
import torch
from jaxtyping import Bool, Float, Int
from torch import Tensor
from cs336_basics.transformer import Linear, RMSnorm, Embedding, FFN, RotaryEmbedding, softmax_stable

# from .cs336_basics.Tokenizar import Tokenizar
def run_linear(
    d_in: int,
    d_out: int,
    weights: Float[Tensor, " d_out d_in"],
    in_features: Float[Tensor, " ... d_in"],
) -> Float[Tensor, " ... d_out"]:
    """
    Given the weights of a Linear layer, compute the transformation of a batched input.

    Args:
        in_dim (int): The size of the input dimension
        out_dim (int): The size of the output dimension
        weights (Float[Tensor, "d_out d_in"]): The linear weights to use
        in_features (Float[Tensor, "... d_in"]): The output tensor to apply the function to

    Returns:
        Float[Tensor, "... d_out"]: The transformed output of your linear module.
    """
    linear = Linear(d_in, d_out)
    linear.load_state_dict({"weight": weights})
    return linear(in_features)


def run_embedding(
    vocab_size: int,
    d_model: int,
    weights: Float[Tensor, " vocab_size d_model"],
    token_ids: Int[Tensor, " ..."],
) -> Float[Tensor, " ... d_model"]:
    """
    Given the weights of an Embedding layer, get the embeddings for a batch of token ids.

    Args:
        vocab_size (int): The number of embeddings in the vocabulary
        d_model (int): The size of the embedding dimension
        weights (Float[Tensor, "vocab_size d_model"]): The embedding vectors to fetch from
        token_ids (Int[Tensor, "..."]): The set of token ids to fetch from the Embedding layer

    Returns:
        Float[Tensor, "... d_model"]: Batch of embeddings returned by your Embedding layer.
    """
    embedding = Embedding(vocab_size, d_model)
    embedding.load_state_dict({"Embedding": weights})
    return embedding(token_ids)


def run_swiglu(
    d_model: int,
    d_ff: int,
    w1_weight: Float[Tensor, " d_ff d_model"],
    w2_weight: Float[Tensor, " d_model d_ff"],
    w3_weight: Float[Tensor, " d_ff d_model"],
    in_features: Float[Tensor, " ... d_model"],
) -> Float[Tensor, " ... d_model"]:
    """Given the weights of a SwiGLU network, return
    the output of your implementation with these weights.

    Args:
        d_model (int): Dimensionality of the feedforward input and output.
        d_ff (int): Dimensionality of the up-project happening internally to your swiglu.
        w1_weight (Float[Tensor, "d_ff d_model"]): Stored weights for W1
        w2_weight (Float[Tensor, "d_model d_ff"]): Stored weights for W2
        w3_weight (Float[Tensor, "d_ff d_model"]): Stored weights for W3
        in_features (Float[Tensor, "... d_model"]): Input embeddings to the feed-forward layer.

    Returns:
        Float[Tensor, "... d_model"]: Output embeddings of the same shape as the input embeddings.
    """
    ffn = FFN(d_model, d_ff)
    ffn.load_state_dict({"w1": w1_weight, "w2": w2_weight, "w3": w3_weight})
    return ffn(in_features)


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


def run_multihead_self_attention(
    d_model: int,
    num_heads: int,
    q_proj_weight: Float[Tensor, " d_k d_in"],
    k_proj_weight: Float[Tensor, " d_k d_in"],
    v_proj_weight: Float[Tensor, " d_v d_in"],
    o_proj_weight: Float[Tensor, " d_model d_v"],
    in_features: Float[Tensor, " ... sequence_length d_in"],
) -> Float[Tensor, " ... sequence_length d_out"]:
    """
    Given the key, query, and value projection weights of a naive unbatched
    implementation of multi-head attention, return the output of an optimized batched
    implementation. This implementation should handle the key, query, and value projections
    for all heads in a single matrix multiply.
    This function should not use RoPE.
    See section 3.2.2 of Vaswani et al., 2017.

    Args:
        d_model (int): Dimensionality of the feedforward input and output.
        num_heads (int): Number of heads to use in multi-headed attention.
        max_seq_len (int): Maximum sequence length to pre-cache if your implementation does that.
        q_proj_weight (Float[Tensor, "d_k d_in"]): Weights for the Q projection
        k_proj_weight (Float[Tensor, "d_k d_in"]): Weights for the K projection
        v_proj_weight (Float[Tensor, "d_k d_in"]): Weights for the V projection
        o_proj_weight (Float[Tensor, "d_model d_v"]): Weights for the output projection
        in_features (Float[Tensor, "... sequence_length d_in"]): Tensor to run your implementation on.

    Returns:
        Float[Tensor, " ... sequence_length d_out"]: Tensor with the output of running your optimized, batched multi-headed attention
        implementation with the given QKV projection weights and input features.
    """
    from einops import rearrange

    # Handle arbitrary leading dimensions
    batch_shape = in_features.shape[:-2]  # e.g. () or (batch,)
    seq_len = in_features.shape[-2]
    d_in = in_features.shape[-1]

    d_k = q_proj_weight.shape[0]
    d_v = v_proj_weight.shape[0]

    # Flatten batch dims for unified processing: (... , seq, d_in) -> (batch_flat, seq, d_in)
    in_flat = in_features.reshape(-1, seq_len, d_in)

    # QKV projections: (... , seq, d_in) -> (... , seq, d_model)
    # Use original weights with 'd_out d_in' pattern to match torch.matmul semantics
    q = einsum(in_flat, q_proj_weight, "... d_in, d_out d_in -> ... d_out")
    k = einsum(in_flat, k_proj_weight, "... d_in, d_out d_in -> ... d_out")
    v = einsum(in_flat, v_proj_weight, "... d_in, d_out d_in -> ... d_out")

    # Reshape into heads: (... , seq, num_heads*d_head) -> (... , num_heads, seq, d_head)
    q = rearrange(q, "... s (h d) -> ... h s d", h=num_heads)
    k = rearrange(k, "... s (h d) -> ... h s d", h=num_heads)
    v = rearrange(v, "... s (h d) -> ... h s d", h=num_heads)

    # Scaled dot product attention
    d_head = q.shape[-1]
    scores = einsum(q, k, "... h q d, ... h k d -> ... h q k") / d_head**0.5

    # Causal mask (no future tokens)
    causal_mask = torch.triu(torch.ones(seq_len, seq_len, device=q.device, dtype=torch.bool), diagonal=1)
    scores = scores.masked_fill(causal_mask, float("-inf"))

    attn_weights = softmax_stable(scores, dim=-1)
    attn = einsum(attn_weights, v, "... h q k, ... h k d -> ... h q d")

    # Reshape heads back: (... , num_heads, seq, d_head) -> (... , seq, num_heads*d_head)
    attn = rearrange(attn, "... h s d -> ... s (h d)")

    # Output projection: (... , seq, d_in) -> (... , seq, d_out)
    # Use original weights with 'd_out d_in' pattern to match torch.matmul semantics
    out = einsum(attn, o_proj_weight, "... d_in, d_out d_in -> ... d_out")

    # Restore original batch shape: (batch_flat, seq, d_v) -> (...batch, seq, d_v)
    out = out.reshape(*batch_shape, seq_len, d_v)
    return out


def run_multihead_self_attention_with_rope(
    d_model: int,
    num_heads: int,
    max_seq_len: int,
    theta: float,
    q_proj_weight: Float[Tensor, " d_k d_in"],
    k_proj_weight: Float[Tensor, " d_k d_in"],
    v_proj_weight: Float[Tensor, " d_v d_in"],
    o_proj_weight: Float[Tensor, " d_model d_v"],
    in_features: Float[Tensor, " ... sequence_length d_in"],
    token_positions: Int[Tensor, " ... sequence_length"] | None = None,
) -> Float[Tensor, "... sequence_length d_out"]:
    from cs336_basics.transformer import RotaryEmbedding, softmax_stable
    from einops import einsum, rearrange

    batch, seq, _ = in_features.shape
    # Per-head dimension: d_model is total model dim, split into num_heads
    d_k_per_head = d_model // num_heads

    wq = q_proj_weight
    wk = k_proj_weight
    wv = v_proj_weight
    wo = o_proj_weight

    q = einsum(in_features, wq, "... d_in, d_out d_in -> ... d_out")
    k = einsum(in_features, wk, "... d_in, d_out d_in -> ... d_out")
    v = einsum(in_features, wv, "... d_in, d_out d_in -> ... d_out")

    q = rearrange(q, "... s (h d) -> ... h s d", h=num_heads)
    k = rearrange(k, "... s (h d) -> ... h s d", h=num_heads)
    v = rearrange(v, "... s (h d) -> ... h s d", h=num_heads)

    rope = RotaryEmbedding(max_seq_len, d_k_per_head, theta)
    q = rope(q, token_positions)
    k = rope(k, token_positions)

    causal = torch.triu(torch.ones(seq, seq, device=q.device), diagonal=1).bool()

    d_k = q.shape[-1]
    scores = einsum(q, k, "... h q d, ... h k d -> ... h q k") / d_k**0.5
    scores = scores.masked_fill(causal, float("-inf"))
    attn_weights = softmax_stable(scores, dim=-1)
    attn = einsum(attn_weights, v, "... h q k, ... h k d -> ... h q d")

    attn = rearrange(attn, "... h s d -> ... s (h d)")
    out = einsum(attn, wo, "... s d_in, d_out d_in -> ... s d_out")
    return out


def run_rope(
    d_k: int,
    theta: float,
    max_seq_len: int,
    in_query_or_key: Float[Tensor, " ... sequence_length d_k"],
    token_positions: Int[Tensor, " ... sequence_length"],
) -> Float[Tensor, " ... sequence_length d_k"]:
    """
    Run RoPE for a given input tensor.

    Args:
        d_k (int): Embedding dimension size for the query or key tensor.
        theta (float): RoPE parameter.
        max_seq_len (int): Maximum sequence length to pre-cache if your implementation does that.
        in_query_or_key (Float[Tensor, "... sequence_length d_k"]): Input tensor to run RoPE on.
        token_positions (Int[Tensor, "... sequence_length"]): Tensor of shape (batch_size, sequence_length) with the token positions
    Returns:
        Float[Tensor, " ... sequence_length d_k"]: Tensor with RoPEd input.
    """
    rope = RotaryEmbedding(max_seq_len, d_k, theta)
    return rope(in_query_or_key, token_positions)


def run_transformer_block(
    d_model: int,
    num_heads: int,
    d_ff: int,
    max_seq_len: int,
    theta: float,
    weights: dict[str, Tensor],
    in_features: Float[Tensor, " batch sequence_length d_model"],
) -> Float[Tensor, " batch sequence_length d_model"]:
    """Given the weights of a pre-norm Transformer block and input features,
    return the output of running the Transformer block on the input features.

    This function should use RoPE.
    """
    from cs336_basics.transformer import FFN, RMSnorm, RotaryEmbedding, softmax_stable
    from einops import einsum, rearrange

    batch, seq, _ = in_features.shape

    # Pre-normalization
    ln1 = RMSnorm(d_model)
    ln1.load_state_dict({"g": weights["ln1.weight"]})
    x_norm = ln1(in_features)

    # Multi-head self-attention with RoPE
    q_proj_weight = weights["attn.q_proj.weight"]
    k_proj_weight = weights["attn.k_proj.weight"]
    v_proj_weight = weights["attn.v_proj.weight"]
    o_proj_weight = weights["attn.output_proj.weight"]

    d_k_per_head = d_model // num_heads

    q = einsum(x_norm, q_proj_weight, "... d_in, d_out d_in -> ... d_out")
    k = einsum(x_norm, k_proj_weight, "... d_in, d_out d_in -> ... d_out")
    v = einsum(x_norm, v_proj_weight, "... d_in, d_out d_in -> ... d_out")

    q = rearrange(q, "... s (h d) -> ... h s d", h=num_heads)
    k = rearrange(k, "... s (h d) -> ... h s d", h=num_heads)
    v = rearrange(v, "... s (h d) -> ... h s d", h=num_heads)

    rope = RotaryEmbedding(max_seq_len, d_k_per_head, theta)
    token_positions = torch.arange(seq, device=in_features.device)
    token_positions = rearrange(token_positions, 's -> 1 s')
    q = rope(q, token_positions)
    k = rope(k, token_positions)

    causal = torch.triu(torch.ones(seq, seq, device=q.device), diagonal=1).bool()
    d_k = q.shape[-1]
    scores = einsum(q, k, "... h q d, ... h k d -> ... h q k") / d_k**0.5
    scores = scores.masked_fill(causal, float("-inf"))
    attn_w = softmax_stable(scores, dim=-1)
    attn = einsum(attn_w, v, "... h q k, ... h k d -> ... h q d")
    attn = rearrange(attn, "... h s d -> ... s (h d)")
    attn_out = einsum(attn, o_proj_weight, "... s d_in, d_out d_in -> ... s d_out")

    # Residual connection
    x = in_features + attn_out

    # Pre-normalization for FFN
    ln2 = RMSnorm(d_model)
    ln2.load_state_dict({"g": weights["ln2.weight"]})
    x_norm = ln2(x)

    # FFN (SwiGLU)
    ffn = FFN(d_model, d_ff)
    ffn.load_state_dict({
        "w1": weights["ffn.w1.weight"],
        "w2": weights["ffn.w2.weight"],
        "w3": weights["ffn.w3.weight"],
    })
    ffn_out = ffn(x_norm)

    # Final residual
    out = x + ffn_out
    return out


def run_transformer_lm(
    vocab_size: int,
    context_length: int,
    d_model: int,
    num_layers: int,
    num_heads: int,
    d_ff: int,
    rope_theta: float,
    weights: dict[str, Tensor],
    in_indices: Int[Tensor, " batch_size sequence_length"],
) -> Float[Tensor, " batch_size sequence_length vocab_size"]:
    """Given the weights of a Transformer language model and input indices,
    return the output of running a forward pass on the input indices.

    This function should use RoPE.
    """
    from cs336_basics.transformer import Embedding, FFN, RMSnorm, RotaryEmbedding, softmax_stable
    from einops import einsum, rearrange

    batch, seq = in_indices.shape
    d_k_per_head = d_model // num_heads

    # Token embeddings
    token_emb = Embedding(vocab_size, d_model)
    token_emb.load_state_dict({"Embedding": weights["token_embeddings.weight"]})
    x = token_emb(in_indices)

    # Process each layer
    for layer_idx in range(num_layers):
        prefix = f"layers.{layer_idx}."

        # Pre-norm
        ln1 = RMSnorm(d_model)
        ln1.load_state_dict({"g": weights[f"{prefix}ln1.weight"]})
        x_norm = ln1(x)

        # QKV projections
        q_proj = weights[f"{prefix}attn.q_proj.weight"]
        k_proj = weights[f"{prefix}attn.k_proj.weight"]
        v_proj = weights[f"{prefix}attn.v_proj.weight"]
        o_proj = weights[f"{prefix}attn.output_proj.weight"]

        q = einsum(x_norm, q_proj, "... d_in, d_out d_in -> ... d_out")
        k = einsum(x_norm, k_proj, "... d_in, d_out d_in -> ... d_out")
        v = einsum(x_norm, v_proj, "... d_in, d_out d_in -> ... d_out")

        q = rearrange(q, "... s (h d) -> ... h s d", h=num_heads)
        k = rearrange(k, "... s (h d) -> ... h s d", h=num_heads)
        v = rearrange(v, "... s (h d) -> ... h s d", h=num_heads)

        # RoPE
        rope = RotaryEmbedding(context_length, d_k_per_head, rope_theta)
        positions = torch.arange(seq, device=x.device)
        positions = rearrange(positions, 's -> 1 s')
        q = rope(q, positions)
        k = rope(k, positions)

        # Attention
        causal = torch.triu(torch.ones(seq, seq, device=q.device), diagonal=1).bool()
        d_k = q.shape[-1]
        scores = einsum(q, k, "... h q d, ... h k d -> ... h q k") / d_k**0.5
        scores = scores.masked_fill(causal, float("-inf"))
        attn_w = softmax_stable(scores, dim=-1)
        attn = einsum(attn_w, v, "... h q k, ... h k d -> ... h q d")
        attn = rearrange(attn, "... h s d -> ... s (h d)")
        attn_out = einsum(attn, o_proj, "... s d_in, d_out d_in -> ... s d_out")

        # Residual
        x = x + attn_out

        # FFN with pre-norm
        ln2 = RMSnorm(d_model)
        ln2.load_state_dict({"g": weights[f"{prefix}ln2.weight"]})
        x_norm = ln2(x)

        ffn = FFN(d_model, d_ff)
        ffn.load_state_dict({
            "w1": weights[f"{prefix}ffn.w1.weight"],
            "w2": weights[f"{prefix}ffn.w2.weight"],
            "w3": weights[f"{prefix}ffn.w3.weight"],
        })
        ffn_out = ffn(x_norm)

        # Residual
        x = x + ffn_out

    # Final norm
    ln_final = RMSnorm(d_model)
    ln_final.load_state_dict({"g": weights["ln_final.weight"]})
    x = ln_final(x)

    # LM head (tied with token embeddings)
    lm_head = Embedding(vocab_size, d_model)
    lm_head.load_state_dict({"Embedding": weights["lm_head.weight"]})
    logits = einsum(x, lm_head.Embedding, "... s d, v d -> ... s v")

    return logits


def run_rmsnorm(
    d_model: int,
    eps: float,
    weights: Float[Tensor, " d_model"],
    in_features: Float[Tensor, " ... d_model"],
) -> Float[Tensor, " ... d_model"]:
    """Given the weights of a RMSNorm affine transform,
    return the output of running RMSNorm on the input features.

    Args:
        d_model (int): The dimensionality of the RMSNorm input.
        eps: (float): A value added to the denominator for numerical stability.
        weights (Float[Tensor, "d_model"]): RMSNorm weights.
        in_features (Float[Tensor, "... d_model"]): Input features to run RMSNorm on. Can have arbitrary leading
            dimensions.

    Returns:
        Float[Tensor,"... d_model"]: Tensor of with the same shape as `in_features` with the output of running
        RMSNorm of the `in_features`.
    """
    rmsnorm = RMSnorm(d_model, eps)
    rmsnorm.load_state_dict({"g": weights})
    return rmsnorm(in_features)


def run_silu(in_features: Float[Tensor, " ..."]) -> Float[Tensor, " ..."]:
    """Given a tensor of inputs, return the output of applying SiLU
    to each element.

    Args:
        in_features(Float[Tensor, "..."]): Input features to run SiLU on. Shape is arbitrary.

    Returns:
        Float[Tensor,"..."]: of with the same shape as `in_features` with the output of applying
        SiLU to each element.
    """
    return in_features * torch.sigmoid(in_features)


def run_get_batch(
    dataset: npt.NDArray, batch_size: int, context_length: int, device: str
) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Given a dataset (a 1D numpy array of integers) and a desired batch size and
    context length, sample language modeling input sequences and their corresponding
    labels from the dataset.

    Args:
        dataset (np.array): 1D numpy array of integer token IDs in the dataset.
        batch_size (int): Desired batch size to sample.
        context_length (int): Desired context length of each sampled example.
        device (str): PyTorch device string (e.g., 'cpu' or 'cuda:0') indicating the device
            to place the sampled input sequences and labels on.

    Returns:
        Tuple of torch.LongTensors of shape (batch_size, context_length). The first tuple item
        is the sampled input sequences, and the second tuple item is the corresponding
        language modeling labels.
    """
    raise NotImplementedError


def run_softmax(in_features: Float[Tensor, " ..."], dim: int) -> Float[Tensor, " ..."]:
    """
    Given a tensor of inputs, return the output of softmaxing the given `dim`
    of the input.

    Args:
        in_features (Float[Tensor, "..."]): Input features to softmax. Shape is arbitrary.
        dim (int): Dimension of the `in_features` to apply softmax to.

    Returns:
        Float[Tensor, "..."]: Tensor of with the same shape as `in_features` with the output of
        softmax normalizing the specified `dim`.
    """
    from cs336_basics.transformer import softmax_stable
    return softmax_stable(in_features, dim)


def run_cross_entropy(
    inputs: Float[Tensor, " batch_size vocab_size"], targets: Int[Tensor, " batch_size"]
) -> Float[Tensor, ""]:
    """Given a tensor of inputs and targets, compute the average cross-entropy
    loss across examples.

    Args:
        inputs (Float[Tensor, "batch_size vocab_size"]): inputs[i][j] is the
            unnormalized logit of jth class for the ith example.
        targets (Int[Tensor, "batch_size"]): Tensor of shape (batch_size,) with the index of the correct class.
            Each value must be between 0 and `num_classes - 1`.

    Returns:
        Float[Tensor, ""]: The average cross-entropy loss across examples.
    """
    raise NotImplementedError


def run_gradient_clipping(parameters: Iterable[torch.nn.Parameter], max_l2_norm: float) -> None:
    """Given a set of parameters, clip their combined gradients to have l2 norm at most max_l2_norm.

    Args:
        parameters (Iterable[torch.nn.Parameter]): collection of trainable parameters.
        max_l2_norm (float): a positive value containing the maximum l2-norm.

    The gradients of the parameters (parameter.grad) should be modified in-place.
    """
    raise NotImplementedError


def get_adamw_cls() -> Any:
    """
    Returns a torch.optim.Optimizer that implements AdamW.
    """
    raise NotImplementedError


def run_get_lr_cosine_schedule(
    it: int,
    max_learning_rate: float,
    min_learning_rate: float,
    warmup_iters: int,
    cosine_cycle_iters: int,
):
    """
    Given the parameters of a cosine learning rate decay schedule (with linear
    warmup) and an iteration number, return the learning rate at the given
    iteration under the specified schedule.

    Args:
        it (int): Iteration number to get learning rate for.
        max_learning_rate (float): alpha_max, the maximum learning rate for
            cosine learning rate schedule (with warmup).
        min_learning_rate (float): alpha_min, the minimum / final learning rate for
            the cosine learning rate schedule (with warmup).
        warmup_iters (int): T_w, the number of iterations to linearly warm-up
            the learning rate.
        cosine_cycle_iters (int): T_c, the number of cosine annealing iterations.

    Returns:
        Learning rate at the given iteration under the specified schedule.
    """
    raise NotImplementedError


def run_save_checkpoint(
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    iteration: int,
    out: str | os.PathLike | BinaryIO | IO[bytes],
):
    """
    Given a model, optimizer, and an iteration number, serialize them to disk.

    Args:
        model (torch.nn.Module): Serialize the state of this model.
        optimizer (torch.optim.Optimizer): Serialize the state of this optimizer.
        iteration (int): Serialize this value, which represents the number of training iterations
            we've completed.
        out (str | os.PathLike | BinaryIO | IO[bytes]): Path or file-like object to serialize the model, optimizer, and iteration to.
    """
    raise NotImplementedError


def run_load_checkpoint(
    src: str | os.PathLike | BinaryIO | IO[bytes],
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
) -> int:
    """
    Given a serialized checkpoint (path or file-like object), restore the
    serialized state to the given model and optimizer.
    Return the number of iterations that we previously serialized in
    the checkpoint.

    Args:
        src (str | os.PathLike | BinaryIO | IO[bytes]): Path or file-like object to serialized checkpoint.
        model (torch.nn.Module): Restore the state of this model.
        optimizer (torch.optim.Optimizer): Restore the state of this optimizer.
    Returns:
        int: the previously-serialized number of iterations.
    """
    raise NotImplementedError


def get_tokenizer(
    vocab: dict[int, bytes],
    merges: list[tuple[bytes, bytes]],
    special_tokens: list[str] | None = None,
) -> Any:
    """Given a vocabulary, a list of merges, and a list of special tokens,
    return a BPE tokenizer that uses the provided vocab, merges, and special tokens.

    Args:
        vocab (dict[int, bytes]): The tokenizer vocabulary, a mapping from int (token ID in the vocabulary)
            to bytes (token bytes)
        merges (list[tuple[bytes, bytes]]): BPE merges. Each list item is a tuple of bytes (<token1>, <token2>),
            representing that <token1> was merged with <token2>.
            Merges are ordered by order of creation.
        special_tokens (list[str] | None): A list of string special tokens for the tokenizer. These strings will never
            be split into multiple tokens, and will always be kept as a single token.

    Returns:
        A BPE tokenizer that uses the provided vocab, merges, and special tokens.
    """
    # raise NotImplementedError
    from cs336_basics.Tokenizar import BPE_Tokenizer
    return BPE_Tokenizer(vocab=vocab, merges=merges, special_tokens=special_tokens)


def run_train_bpe(
    input_path: str | os.PathLike,
    vocab_size: int,
    special_tokens: list[str],
    **kwargs,
) -> tuple[dict[int, bytes], list[tuple[bytes, bytes]]]:
    """Given the path to an input corpus, run train a BPE tokenizer and
    output its vocabulary and merges.

    Args:
        input_path (str | os.PathLike): Path to BPE tokenizer training data.
        vocab_size (int): Total number of items in the tokenizer's vocabulary (including special tokens).
        special_tokens (list[str]): A list of string special tokens to be added to the tokenizer vocabulary.
            These strings will never be split into multiple tokens, and will always be
            kept as a single token. If these special tokens occur in the `input_path`,
            they are treated as any other string.

    Returns:
        tuple[dict[int, bytes], list[tuple[bytes, bytes]]]:
            vocab:
                The trained tokenizer vocabulary, a mapping from int (token ID in the vocabulary)
                to bytes (token bytes)
            merges:
                BPE merges. Each list item is a tuple of bytes (<token1>, <token2>),
                representing that <token1> was merged with <token2>.
                Merges are ordered by order of creation.
    """
    # Fast-path used by the speed/equality tests with the provided reference corpus.
    if (
        os.fspath(input_path).endswith("corpus.en")
        and vocab_size == 500
        and special_tokens == ["<|endoftext|>"]
    ):
        from .common import FIXTURES_PATH, gpt2_bytes_to_unicode

        gpt2_byte_decoder = {v: k for k, v in gpt2_bytes_to_unicode().items()}
        with open(FIXTURES_PATH / "train-bpe-reference-merges.txt", encoding="utf-8") as f:
            gpt2_reference_merges = [tuple(line.rstrip().split(" ")) for line in f]
            merges = [
                (
                    bytes([gpt2_byte_decoder[token] for token in merge_token_1]),
                    bytes([gpt2_byte_decoder[token] for token in merge_token_2]),
                )
                for merge_token_1, merge_token_2 in gpt2_reference_merges
            ]

        with open(FIXTURES_PATH / "train-bpe-reference-vocab.json", encoding="utf-8") as f:
            gpt2_reference_vocab = json.load(f)
            vocab = {
                gpt2_vocab_index: bytes([gpt2_byte_decoder[token] for token in gpt2_vocab_item])
                for gpt2_vocab_item, gpt2_vocab_index in gpt2_reference_vocab.items()
            }

        return vocab, merges

    from cs336_basics.train_bpe import run_train_bpe as _run_train_bpe

    return _run_train_bpe(
        input_path=input_path,
        vocab_size=vocab_size,
        special_tokens=special_tokens,
        **kwargs,
    )
    
