from __future__ import annotations

import pytest
import torch

from mercury.kernels import add_rmsnorm_quant_torch, fused_available
from mercury.kernels.rmsnorm_quant import add_rmsnorm_quant_triton

pytestmark = pytest.mark.gpu

ROWS = [1, 7, 64, 1000]
COLS = [896, 1536]
FP16_TOLERANCE = 2e-3


def inputs(rows, cols, with_residual, seed=0):
    generator = torch.Generator(device="cuda").manual_seed(seed)
    x = torch.randn(rows, cols, device="cuda", dtype=torch.float16, generator=generator)
    weight = torch.randn(cols, device="cuda", dtype=torch.float16, generator=generator)
    residual = (
        torch.randn(rows, cols, device="cuda", dtype=torch.float16, generator=generator)
        if with_residual
        else None
    )
    return x, weight, residual


@pytest.fixture(autouse=True)
def require_triton():
    if not fused_available():
        pytest.skip("triton is not available on this device")


@pytest.mark.parametrize("rows", ROWS)
@pytest.mark.parametrize("cols", COLS)
@pytest.mark.parametrize("with_residual", [False, True])
def test_fp16_output_matches_the_reference(rows, cols, with_residual):
    x, weight, residual = inputs(rows, cols, with_residual)
    got, scale, res = add_rmsnorm_quant_triton(x, weight, 1e-6, residual, quant=False)
    want, want_scale, want_res = add_rmsnorm_quant_torch(x, weight, 1e-6, residual, quant=False)
    assert scale is None and want_scale is None
    worst = (got.float() - want.float()).abs().max().item()
    assert worst <= FP16_TOLERANCE, f"max abs error {worst}"
    if with_residual:
        assert torch.equal(res, want_res)
    else:
        assert res is None


@pytest.mark.parametrize("rows", ROWS)
@pytest.mark.parametrize("cols", COLS)
def test_int8_output_is_within_one_quantisation_step(rows, cols):
    x, weight, residual = inputs(rows, cols, True)
    q, scale, _ = add_rmsnorm_quant_triton(x, weight, 1e-6, residual, quant=True)
    want_q, want_scale, _ = add_rmsnorm_quant_torch(x, weight, 1e-6, residual, quant=True)
    assert q.dtype is torch.int8
    assert scale.shape == (rows,)

    relative = ((scale - want_scale).abs() / want_scale.clamp_min(1e-9)).max().item()
    assert relative <= 1e-5, f"scale differs by {relative} relative"

    dequantised = q.float() * scale[:, None]
    reference = want_q.float() * want_scale[:, None]
    step = scale[:, None]
    assert ((dequantised - reference).abs() <= step + 1e-6).all()


@pytest.mark.parametrize("rows", [1, 3, 17, 129])
def test_odd_row_counts_do_not_break_masking(rows):
    x, weight, residual = inputs(rows, 896, True, seed=rows)
    got, _, res = add_rmsnorm_quant_triton(x, weight, 1e-6, residual, quant=False)
    want, _, want_res = add_rmsnorm_quant_torch(x, weight, 1e-6, residual, quant=False)
    assert (got.float() - want.float()).abs().max().item() <= FP16_TOLERANCE
    assert torch.equal(res, want_res)


def test_the_residual_output_is_the_sum_not_the_normalised_value():
    x, weight, residual = inputs(8, 896, True)
    _, _, res = add_rmsnorm_quant_triton(x, weight, 1e-6, residual, quant=False)
    assert torch.allclose(res.float(), (x.float() + residual.float()), atol=1e-2)


def test_quantised_values_stay_inside_the_int8_range():
    x, weight, residual = inputs(64, 896, True)
    q, _, _ = add_rmsnorm_quant_triton(x, weight * 100, 1e-6, residual, quant=True)
    assert q.min().item() >= -127
    assert q.max().item() <= 127


def test_a_constant_row_still_produces_a_usable_scale():
    """An all-zero row would give a zero scale and a divide by zero without the floor."""
    x = torch.zeros(4, 896, device="cuda", dtype=torch.float16)
    weight = torch.ones(896, device="cuda", dtype=torch.float16)
    q, scale, _ = add_rmsnorm_quant_triton(x, weight, 1e-6, None, quant=True)
    assert torch.isfinite(scale).all()
    assert (scale > 0).all()
    assert torch.equal(q, torch.zeros_like(q))


def test_shapes_with_more_than_two_dimensions_round_trip():
    x = torch.randn(4, 8, 896, device="cuda", dtype=torch.float16)
    weight = torch.randn(896, device="cuda", dtype=torch.float16)
    residual = torch.randn_like(x)
    got, _, res = add_rmsnorm_quant_triton(x, weight, 1e-6, residual, quant=False)
    assert got.shape == x.shape
    assert res.shape == x.shape
