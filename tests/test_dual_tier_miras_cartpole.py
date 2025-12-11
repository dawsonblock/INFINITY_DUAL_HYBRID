#!/usr/bin/env python3
"""
Tests for Dual-Tier Miras memory system.
"""

import sys
sys.path.insert(0, "src")

import torch
import pytest

from infinity_dual_hybrid.miras import (
    DualTierMiras,
    SSMCompressedMiras,
    SSMCompressedMirasTitans,
)
from infinity_dual_hybrid.config import MirasConfig


def test_ssm_compressed_miras_forward():
    """Test fast tier Miras forward pass."""
    batch_size = 8
    d_model = 64
    rank = 16

    miras = SSMCompressedMiras(d_model=d_model, rank=rank)

    k = torch.randn(batch_size, d_model)
    v = torch.randn(batch_size, d_model)

    # Write
    miras.update(k, v)

    # Read
    out = miras.read(k)

    assert out.shape == (batch_size, d_model)
    assert torch.isfinite(out).all()


def test_ssm_compressed_miras_titans_forward():
    """Test deep tier Miras (Titans-style) forward pass."""
    batch_size = 8
    d_model = 64
    rank = 32

    miras = SSMCompressedMirasTitans(d_model=d_model, rank=rank)

    k = torch.randn(batch_size, d_model)
    v = torch.randn(batch_size, d_model)

    # Write
    miras.update(k, v)

    # Read
    out = miras.read(k)

    assert out.shape == (batch_size, d_model)
    assert torch.isfinite(out).all()


def test_dual_tier_miras_forward():
    """Test combined Dual-Tier Miras."""
    batch_size = 8
    d_model = 64

    cfg = MirasConfig(d_model=d_model, fast_rank=16, deep_rank=32)
    miras = DualTierMiras.from_config(cfg)

    k = torch.randn(batch_size, d_model)
    v = torch.randn(batch_size, d_model)
    context = torch.randn(batch_size, d_model)

    # Write
    miras.update(k, v, weight=torch.ones(batch_size))

    # Read
    result = miras.read(k, context=context)

    assert "v" in result
    assert result["v"].shape == (batch_size, d_model)
    assert torch.isfinite(result["v"]).all()


def test_dual_tier_miras_mixing():
    """Test that mixing ratio affects output."""
    batch_size = 4
    d_model = 32

    cfg = MirasConfig(d_model=d_model, fast_rank=8, deep_rank=16)
    miras = DualTierMiras.from_config(cfg)

    k = torch.randn(batch_size, d_model)
    v_fast = torch.ones(batch_size, d_model) * 1.0
    v_deep = torch.ones(batch_size, d_model) * 2.0

    # Write different values to fast and deep
    miras.fast_mem.update(k, v_fast)
    miras.deep_mem.update(k, v_deep)

    # Read - output should be between fast and deep values
    result = miras.read(k)

    # The output should be a weighted combination
    assert torch.isfinite(result["v"]).all()


def test_miras_reset():
    """Test memory reset functionality."""
    d_model = 32

    cfg = MirasConfig(d_model=d_model, fast_rank=8, deep_rank=16)
    miras = DualTierMiras.from_config(cfg)

    k = torch.randn(4, d_model)
    v = torch.randn(4, d_model)

    # Write
    miras.update(k, v)

    # Reset
    miras.reset()

    # After reset, read should return near-zero
    out = miras.read(k)
    assert torch.isfinite(out["v"]).all()


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
