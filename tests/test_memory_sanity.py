#!/usr/bin/env python3
"""
Sanity tests for memory systems (Miras + LTM).
"""

import sys
sys.path.insert(0, "src")

import torch
import pytest

from infinity_dual_hybrid.miras import DualTierMiras
from infinity_dual_hybrid.ltm import build_ltm, SimpleLTM
from infinity_dual_hybrid.config import MirasConfig, LTMConfig


class TestMirasSanity:
    """Basic sanity checks for Miras."""

    def test_miras_creates(self):
        cfg = MirasConfig(d_model=64)
        miras = DualTierMiras.from_config(cfg)
        assert miras is not None

    def test_miras_read_write(self):
        cfg = MirasConfig(d_model=64)
        miras = DualTierMiras.from_config(cfg)

        k = torch.randn(4, 64)
        v = torch.randn(4, 64)

        miras.update(k, v)
        out = miras.read(k)

        assert "v" in out
        assert out["v"].shape == (4, 64)

    def test_miras_gradients_flow(self):
        cfg = MirasConfig(d_model=32)
        miras = DualTierMiras.from_config(cfg)

        k = torch.randn(2, 32, requires_grad=True)
        v = torch.randn(2, 32)

        out = miras.read(k)
        loss = out["v"].sum()
        loss.backward()

        assert k.grad is not None


class TestLTMSanity:
    """Basic sanity checks for LTM."""

    def test_simple_ltm_creates(self):
        cfg = LTMConfig(d_key=64, d_value=64, use_faiss=False)
        ltm = build_ltm(cfg)
        assert isinstance(ltm, SimpleLTM)

    def test_simple_ltm_store_retrieve(self):
        cfg = LTMConfig(d_key=64, d_value=64, use_faiss=False, max_entries=100)
        ltm = build_ltm(cfg)

        keys = torch.randn(10, 64)
        values = torch.randn(10, 64)

        ltm.store(keys, values)
        retrieved = ltm.retrieve(keys[:2], top_k=3)

        assert retrieved.shape[0] == 2
        assert retrieved.shape[2] == 64

    def test_ltm_empty_retrieve(self):
        cfg = LTMConfig(d_key=64, d_value=64, use_faiss=False)
        ltm = build_ltm(cfg)

        query = torch.randn(2, 64)
        retrieved = ltm.retrieve(query, top_k=3)

        # Should handle empty gracefully
        assert retrieved is not None


class TestIntegration:
    """Integration tests for memory systems."""

    def test_miras_and_ltm_together(self):
        miras_cfg = MirasConfig(d_model=64)
        ltm_cfg = LTMConfig(d_key=64, d_value=64, use_faiss=False)

        miras = DualTierMiras.from_config(miras_cfg)
        ltm = build_ltm(ltm_cfg)

        # Simulate agent forward pass
        obs_encoding = torch.randn(4, 64)

        # Miras read/write
        miras_out = miras.read(obs_encoding)
        miras.update(obs_encoding, miras_out["v"])

        # LTM store/retrieve
        ltm.store(obs_encoding, miras_out["v"])
        ltm_out = ltm.retrieve(obs_encoding, top_k=2)

        assert miras_out["v"].shape == (4, 64)
        assert ltm_out.shape[0] == 4


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
