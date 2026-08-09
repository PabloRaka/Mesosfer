"""
Unified Flash Attention interface with automatic FA3/FA2/SDPA switching.

Exports `flash_attn` module that matches the FA3-style API used by the model,
preferring FA3, then FA2, then PyTorch SDPA.

Usage (drop-in replacement for FA3):
    from mesosfer.model.flash_attention import flash_attn

    # Training (no KV cache)
    y = flash_attn.flash_attn_func(q, k, v, causal=True, window_size=window_size)

    # Inference (with KV cache)
    y = flash_attn.flash_attn_with_kvcache(q, k_cache, v_cache, k=k, v=v, ...)
"""
from types import SimpleNamespace

import os
import torch
import torch.nn.functional as F


# =============================================================================
# Helpers (defined first because backend loaders below depend on them)
# =============================================================================
def _is_rocm():
    return bool(getattr(torch.version, "hip", None))


def _get_compute_dtype():
    from mesosfer.utils.common import COMPUTE_DTYPE
    return COMPUTE_DTYPE


# =============================================================================
# Detection: Try to load Flash Attention backends
# =============================================================================
def _load_flash_attention_3():
    """Try to load Flash Attention 3 (requires Hopper GPU, sm90)."""
    if not torch.cuda.is_available():
        return None
    if _is_rocm():
        return None
    try:
        major, _ = torch.cuda.get_device_capability()
        # FA3 kernels are compiled for Hopper (sm90) only
        # Ada (sm89), Blackwell (sm100) need SDPA fallback until FA3 is recompiled
        if major != 9:
            return None
        import os
        os.environ["HF_HUB_DISABLE_PROGRESS_BARS"] = "1"
        from kernels import get_kernel
        return get_kernel('varunneal/flash-attention-3').flash_attn_interface
    except Exception:
        return None


def _load_flash_attention_2():
    """Try to load Flash Attention 2 from the optional flash-attn package."""
    if not torch.cuda.is_available():
        return None
    if _is_rocm():
        # flash-attn picks its backend at import time from this env var. Unset, its
        # flash_attn_interface does `import flash_attn_2_cuda`, which does not exist in
        # a ROCm build — the ImportError is swallowed below and we silently fall back to
        # SDPA even though flash-attn is installed. setdefault so an explicit
        # FLASH_ATTENTION_TRITON_AMD_ENABLE=FALSE (e.g. to test the CK backend) still wins.
        os.environ.setdefault("FLASH_ATTENTION_TRITON_AMD_ENABLE", "TRUE")
    try:
        try:
            from flash_attn import flash_attn_func, flash_attn_with_kvcache
        except ImportError:
            from flash_attn.flash_attn_interface import flash_attn_func, flash_attn_with_kvcache
        return SimpleNamespace(
            flash_attn_func=flash_attn_func,
            flash_attn_with_kvcache=flash_attn_with_kvcache,
        )
    except Exception:
        try:
            try:
                from flash_attn import flash_attn_func
            except ImportError:
                from flash_attn.flash_attn_interface import flash_attn_func
            return SimpleNamespace(
                flash_attn_func=flash_attn_func,
                flash_attn_with_kvcache=None,
            )
        except Exception:
            return None


_fa3 = _load_flash_attention_3()
HAS_FA3 = _fa3 is not None
_fa2 = _load_flash_attention_2()
HAS_FA2 = _fa2 is not None

# Override for testing/runtime: set MESOSFER_ATTENTION_BACKEND to fa3, fa2, sdpa, or auto.
_env_override = os.environ.get("MESOSFER_ATTENTION_BACKEND") or os.environ.get("mesosfer_ATTENTION_BACKEND")
_override_impl = None if _env_override in (None, "", "auto") else _env_override.lower()
if _override_impl is not None and _override_impl not in {"fa3", "fa2", "sdpa"}:
    raise ValueError("MESOSFER_ATTENTION_BACKEND must be one of: auto, fa3, fa2, sdpa")


def _flash_attention_dtype_supported(backend):
    """Return whether the current training dtype can use the requested FA backend."""
    compute_dtype = _get_compute_dtype()

    if backend == "fa3":
        # FA3 Hopper kernels used here support bf16/fp8; this model trains bf16.
        return not _is_rocm() and compute_dtype == torch.bfloat16

    if backend == "fa2":
        if compute_dtype not in {torch.float16, torch.bfloat16}:
            return False
        if not torch.cuda.is_available():
            return False
        if _is_rocm():
            return True
        if compute_dtype == torch.bfloat16:
            major, _ = torch.cuda.get_device_capability()
            return major >= 8
        return True

    if backend == "sdpa":
        return True

    raise ValueError(f"Unknown attention backend: {backend}")


def _resolve_attention_backend():
    """Resolve backend once: FA3 first, then FA2, then PyTorch SDPA."""
    if _override_impl is not None:
        if _override_impl == "fa3":
            assert HAS_FA3, "Cannot override to FA3: not available on this hardware"
            assert _flash_attention_dtype_supported("fa3"), "Cannot override to FA3: current dtype is unsupported"
            return "fa3"
        if _override_impl == "fa2":
            assert HAS_FA2, "Cannot override to FA2: flash-attn package is not available"
            assert _flash_attention_dtype_supported("fa2"), "Cannot override to FA2: current dtype is unsupported"
            return "fa2"
        if _override_impl == "sdpa":
            return "sdpa"
        raise ValueError(f"Unknown attention implementation override: {_override_impl}")

    if HAS_FA3 and _flash_attention_dtype_supported("fa3"):
        return "fa3"
    if HAS_FA2 and _flash_attention_dtype_supported("fa2"):
        return "fa2"
    return "sdpa"


def _resolve_use_fa3():
    """Backward-compatible boolean for existing callers/tests."""
    return _resolve_attention_backend() == "fa3"


def _resolve_use_fa2():
    """Backward-compatible boolean for code that wants to check FA2 explicitly."""
    return _resolve_attention_backend() == "fa2"


ATTENTION_BACKEND = _resolve_attention_backend()
USE_FA3 = ATTENTION_BACKEND == "fa3"
USE_FA2 = ATTENTION_BACKEND == "fa2"


# =============================================================================
# SDPA helpers
# =============================================================================
def _sdpa_attention(q, k, v, window_size, enable_gqa):
    """
    SDPA attention with sliding window support.
    q, k, v are (B, H, T, D) format.
    """
    Tq = q.size(2)
    Tk = k.size(2)
    window = window_size[0]

    # Full context, same length
    if (window < 0 or window >= Tq) and Tq == Tk:
        return F.scaled_dot_product_attention(
            q.contiguous(), k.contiguous(), v.contiguous(),
            is_causal=True, enable_gqa=enable_gqa
        )

    # Single token generation
    if Tq == 1:
        if window >= 0 and window < Tk:
            # window is "left" tokens we need to include (window + 1) keys total
            start = max(0, Tk - (window + 1))
            k = k[:, :, start:, :]
            v = v[:, :, start:, :]
        return F.scaled_dot_product_attention(
            q.contiguous(), k.contiguous(), v.contiguous(),
            is_causal=False, enable_gqa=enable_gqa
        )

    # Need explicit mask for sliding window/chunk inference
    device = q.device
    # For chunk inference (Tq != Tk), is_causal is not aligned to cache position => build an explicit bool mask
    row_idx = (Tk - Tq) + torch.arange(Tq, device=device).unsqueeze(1)
    col_idx = torch.arange(Tk, device=device).unsqueeze(0)
    mask = col_idx <= row_idx

    # sliding window (left)
    if window >= 0 and window < Tk:
        mask = mask & ((row_idx - col_idx) <= window)

    return F.scaled_dot_product_attention(
        q.contiguous(), k.contiguous(), v.contiguous(),
        attn_mask=mask, enable_gqa=enable_gqa
    )

# =============================================================================
# Public API: Same interface as FA3
# =============================================================================
def flash_attn_func(q, k, v, causal=False, window_size=(-1, -1)):
    """
    Flash Attention for training (no KV cache).

    Args:
        q, k, v: Tensors of shape (B, T, H, D)
        causal: Whether to use causal masking
        window_size: (left, right) sliding window. -1 means unlimited.

    Returns:
        Output tensor of shape (B, T, H, D)
    """
    if ATTENTION_BACKEND == "fa3":
        return _fa3.flash_attn_func(q, k, v, causal=causal, window_size=window_size)
    if ATTENTION_BACKEND == "fa2":
        # ROCm Triton FA2 doesn't support sliding window backward yet; it requires
        # window_size=(-1, -1) to take the full-attention code path. If the caller
        # passes (-1, 0) or (>=seq_len, 0) for "L"-pattern layers, normalize to
        # (-1, -1) so the backend treats it as full attention. Causality is still
        # enforced by causal=True. Genuine sliding window (window>0 and < seq_len)
        # will still hit NotImplementedError on ROCm and should be avoided there.
        if _is_rocm():
            seq_len = q.size(1)
            left, right = window_size
            is_full_left = (left < 0) or (left >= seq_len)
            is_full_right = (right < 0) or (right >= seq_len) or (right == 0 and causal)
            if is_full_left and is_full_right:
                return _fa2.flash_attn_func(q, k, v, causal=causal, window_size=(-1, -1))
        return _fa2.flash_attn_func(q, k, v, causal=causal, window_size=window_size)

    # SDPA fallback: transpose (B, T, H, D) -> (B, H, T, D)
    q = q.transpose(1, 2)
    k = k.transpose(1, 2)
    v = v.transpose(1, 2)
    enable_gqa = q.size(1) != k.size(1)
    y = _sdpa_attention(q, k, v, window_size, enable_gqa)
    return y.transpose(1, 2)  # back to (B, T, H, D)


def flash_attn_with_kvcache(q, k_cache, v_cache, k=None, v=None, cache_seqlens=None,
                            causal=False, window_size=(-1, -1)):
    """
    Flash Attention with KV cache for inference.

    FA3 updates k_cache/v_cache in-place. Our SDPA fallback does the same.

    Args:
        q: Queries, shape (B, T_new, H, D)
        k_cache, v_cache: Pre-allocated cache tensors, shape (B, T_max, H_kv, D)
        k, v: New keys/values to insert, shape (B, T_new, H_kv, D)
        cache_seqlens: Current position in cache, shape (B,) int32
        causal: Whether to use causal masking
        window_size: (left, right) sliding window. -1 means unlimited.

    Returns:
        Output tensor of shape (B, T_new, H, D)
    """
    if ATTENTION_BACKEND == "fa3":
        return _fa3.flash_attn_with_kvcache(
            q, k_cache, v_cache, k=k, v=v, cache_seqlens=cache_seqlens,
            causal=causal, window_size=window_size
        )
    if ATTENTION_BACKEND == "fa2" and getattr(_fa2, "flash_attn_with_kvcache", None) is not None and not _is_rocm():
        return _fa2.flash_attn_with_kvcache(
            q, k_cache, v_cache, k=k, v=v, cache_seqlens=cache_seqlens,
            causal=causal, window_size=window_size
        )

    # SDPA fallback: manually manage KV cache
    B, T_new, H, D = q.shape
    pos = cache_seqlens[0].item()  # assume uniform position across batch

    # Insert new k, v into cache (in-place, matching FA3 behavior)
    if k is not None and v is not None:
        k_cache[:, pos:pos+T_new, :, :] = k
        v_cache[:, pos:pos+T_new, :, :] = v

    # Get full cache up to current position + new tokens
    end_pos = pos + T_new
    k_full = k_cache[:, :end_pos, :, :]
    v_full = v_cache[:, :end_pos, :, :]

    # Transpose to SDPA layout: (B, T, H, D) -> (B, H, T, D)
    q_sdpa = q.transpose(1, 2)
    k_sdpa = k_full.transpose(1, 2)
    v_sdpa = v_full.transpose(1, 2)

    enable_gqa = q_sdpa.size(1) != k_sdpa.size(1)
    y_sdpa = _sdpa_attention(q_sdpa, k_sdpa, v_sdpa, window_size, enable_gqa)

    return y_sdpa.transpose(1, 2)  # back to (B, T, H, D)


# =============================================================================
# Export: flash_attn module interface (drop-in replacement for FA3)
# =============================================================================
if ATTENTION_BACKEND == "fa2" and _is_rocm():
    # TorchInductor cannot reason about the ROCm Triton flash-attn custom op. It emits
    #   assert_size_stride(buf, (2,), (1,), 'torch.ops.flash_attn._flash_attn_forward')
    # and dies at runtime with "TypeError: expected Tensor()". Keep these opaque so
    # dynamo graph-breaks around them; every other block still compiles normally.
    flash_attn_func = torch.compiler.disable(flash_attn_func)
    flash_attn_with_kvcache = torch.compiler.disable(flash_attn_with_kvcache)

flash_attn = SimpleNamespace(
    flash_attn_func=flash_attn_func,
    flash_attn_with_kvcache=flash_attn_with_kvcache,
)
