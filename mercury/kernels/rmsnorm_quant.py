from __future__ import annotations

import torch

try:
    import triton
    import triton.language as tl

    HAS_TRITON = True
except ImportError:  # CPU-only laptops and any machine without a working Triton
    triton = None
    tl = None
    HAS_TRITON = False


def fused_available() -> bool:
    return HAS_TRITON and torch.cuda.is_available()


if HAS_TRITON:

    @triton.jit
    def _add_rmsnorm_quant(
        X,
        R,
        W,
        Y,
        RES_OUT,
        SCALE,
        stride,
        N,
        eps,
        HAS_RES: tl.constexpr,
        QUANT: tl.constexpr,
        BLOCK: tl.constexpr,
    ):
        row = tl.program_id(0)
        cols = tl.arange(0, BLOCK)
        m = cols < N
        x = tl.load(X + row * stride + cols, mask=m, other=0.0).to(tl.float32)
        if HAS_RES:
            x += tl.load(R + row * stride + cols, mask=m, other=0.0).to(tl.float32)
            tl.store(RES_OUT + row * stride + cols, x.to(tl.float16), mask=m)
        rstd = 1.0 / tl.sqrt(tl.sum(x * x, axis=0) / N + eps)
        w = tl.load(W + cols, mask=m, other=0.0).to(tl.float32)
        y = (x * rstd).to(tl.float16).to(tl.float32) * w
        if QUANT:
            s = tl.maximum(tl.max(tl.abs(y), axis=0), 1e-6) / 127.0
            q = y / s
            q = tl.where(q >= 0, q + 0.5, q - 0.5)
            q = tl.minimum(tl.maximum(q, -127.0), 127.0).to(tl.int8)
            tl.store(Y + row * stride + cols, q, mask=m)
            tl.store(SCALE + row, s)
        else:
            tl.store(Y + row * stride + cols, y.to(tl.float16), mask=m)


def add_rmsnorm_quant_triton(
    x: torch.Tensor,
    weight: torch.Tensor,
    eps: float = 1e-6,
    residual: torch.Tensor | None = None,
    quant: bool = False,
):
    if not HAS_TRITON:
        raise RuntimeError("triton is not importable in this environment")
    shape = x.shape
    x2 = x.reshape(-1, shape[-1]).contiguous()
    rows, cols = x2.shape
    block = triton.next_power_of_2(cols)
    r2 = residual.reshape(rows, cols).contiguous() if residual is not None else x2
    res_out = torch.empty_like(x2) if residual is not None else x2
    y = torch.empty((rows, cols), device=x.device, dtype=torch.int8 if quant else x.dtype)
    scale = torch.empty(rows, device=x.device, dtype=torch.float32) if quant else y
    _add_rmsnorm_quant[(rows,)](
        x2,
        r2,
        weight,
        y,
        res_out,
        scale,
        x2.stride(0),
        cols,
        eps,
        HAS_RES=residual is not None,
        QUANT=quant,
        BLOCK=block,
        num_warps=4 if block <= 1024 else 8,
    )
    return (
        y.view(shape),
        scale if quant else None,
        res_out.view(shape) if residual is not None else None,
    )


def add_rmsnorm_quant_torch(
    x: torch.Tensor,
    weight: torch.Tensor,
    eps: float = 1e-6,
    residual: torch.Tensor | None = None,
    quant: bool = False,
):
    """Same arithmetic in the same order as the kernel; the eager competitor and the
    correctness reference. torch.compile of this is the honest competitor."""
    shape = x.shape
    flat = x.reshape(-1, shape[-1]).float()
    res_out = None
    if residual is not None:
        flat = flat + residual.reshape(-1, shape[-1]).float()
        res_out = flat.to(x.dtype).view(shape)
    rstd = torch.rsqrt(flat.pow(2).mean(-1, keepdim=True) + eps)
    y = (flat * rstd).to(torch.float16).float() * weight.float()
    if quant:
        scale = (y.abs().amax(-1, keepdim=True).clamp_min(1e-6) / 127.0).float()
        q = y / scale
        q = torch.where(q >= 0, q + 0.5, q - 0.5).clamp(-127.0, 127.0).to(torch.int8)
        return q.view(shape), scale.squeeze(-1).contiguous(), res_out
    return y.to(x.dtype).view(shape), None, res_out


add_rmsnorm_quant_ref = add_rmsnorm_quant_torch


def add_rmsnorm_quant(
    x: torch.Tensor,
    weight: torch.Tensor,
    eps: float = 1e-6,
    residual: torch.Tensor | None = None,
    quant: bool = False,
    backend: str = "auto",
):
    if backend == "torch" or (backend == "auto" and not (x.is_cuda and HAS_TRITON)):
        return add_rmsnorm_quant_torch(x, weight, eps, residual, quant)
    return add_rmsnorm_quant_triton(x, weight, eps, residual, quant)


def bytes_moved(rows: int, cols: int, has_residual: bool, quant: bool) -> int:
    """Traffic the fused kernel is obliged to do, for the effective-bandwidth column."""
    half = 2
    total = rows * cols * half
    if has_residual:
        total += rows * cols * half
        total += rows * cols * half
    total += cols * half
    total += rows * cols * (1 if quant else half)
    if quant:
        total += rows * 4
    return total
