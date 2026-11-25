"""Batch-size guard for a silent int32 overflow in the mamba-ssm Mamba2 kernels.

The Mamba2 Triton kernels in mamba-ssm compute memory offsets into the in_proj output, a
tensor of shape (batch, length, d_in_proj), with 32-bit integers. Once
batch * length * d_in_proj exceeds 2**31 the offsets wrap around and the later sequences
of the batch read the wrong memory. Depending on where the wrapped address lands, the run
either aborts with "CUDA error: an illegal memory access was encountered" or returns wrong
logits / embeddings / gradients with no error at all. Silent errors are seen on RTX 6000 Ada
and are the norm on GH200, where the GPU can address all host memory.

PlantCAD2 models use Mamba2, so every forward or backward pass must keep
batch * length * d_in_proj <= 2**31. For PlantCAD2-Large (d_in_proj = 6320) this means at
most 82 sequences of 4096 bp or 41 sequences of 8192 bp per batch. PlantCaduceus models
use Mamba1 and are not affected by this limit.

Upstream issue: https://github.com/state-spaces/mamba/issues/1015
Fix (not yet in a release): https://github.com/state-spaces/mamba/pull/1038
Remove this guard once a mamba-ssm release includes the fix.
"""
import logging
from typing import Optional

logger = logging.getLogger(__name__)

_INT32_LIMIT = 2**31
_ISSUE_URL = "https://github.com/state-spaces/mamba/issues/1015"
_warned = set()


def mamba2_in_proj_width(model) -> Optional[int]:
    """Largest Mamba2 in_proj output width in `model`, or None if it has no Mamba2 layers."""
    widths = [
        2 * m.d_inner + 2 * m.ngroups * m.d_state + m.nheads
        for m in model.modules()
        if type(m).__name__ == "Mamba2" and hasattr(m, "d_inner")
    ]
    return max(widths) if widths else None


def max_batch_size(model, seq_len: int) -> Optional[int]:
    """Largest batch of `seq_len`-token sequences that stays below the int32 limit.

    Returns None when the model has no Mamba2 layers (no limit applies).
    """
    width = mamba2_in_proj_width(model)
    if width is None:
        return None
    limit = _INT32_LIMIT // (width * seq_len)
    if limit < 1:
        raise ValueError(
            f"A single sequence of {seq_len} tokens exceeds the Mamba2 int32 offset limit "
            f"(d_in_proj={width}); see {_ISSUE_URL}"
        )
    return limit


def safe_inference_batch_size(model, seq_len: int, batch_size: int) -> int:
    """Return `batch_size`, lowered to the safe maximum if it would overflow.

    For inference the batch size does not change the results, so it is lowered with a warning.
    """
    limit = max_batch_size(model, seq_len)
    if limit is None or batch_size <= limit:
        return batch_size
    if (batch_size, seq_len) not in _warned:  # callers may check every batch; warn once
        _warned.add((batch_size, seq_len))
        logger.warning(
            "Batch size %d with %d-token sequences exceeds the Mamba2 int32 offset limit of this "
            "model (max %d), which can give wrong results without any error. "
            "Using batch size %d instead. See %s",
            batch_size, seq_len, limit, limit, _ISSUE_URL,
        )
    return limit


def check_train_batch_size(model, seq_len: int, batch_size: int, gradient_accumulation_steps: int = 1) -> None:
    """Raise if a training batch would overflow.

    The training batch size changes the optimisation, so it is not lowered silently; the error
    suggests a smaller batch with more gradient accumulation and the same effective batch size.
    """
    limit = max_batch_size(model, seq_len)
    if limit is None or batch_size <= limit:
        return
    effective = batch_size * gradient_accumulation_steps
    new_bs = max(d for d in range(1, limit + 1) if effective % d == 0)
    raise ValueError(
        f"train_batch_size={batch_size} with {seq_len}-token sequences exceeds the Mamba2 int32 "
        f"offset limit of this model (max {limit}). Such batches can give wrong outputs and gradients "
        f"without any error. Use train_batch_size <= {limit}; to keep the "
        f"effective batch size of {effective}, use e.g. train_batch_size={new_bs} and "
        f"gradient_accumulation_steps={effective // new_bs}. See {_ISSUE_URL}"
    )
