"""Tests for TaiChi Matrix integration pipeline."""

import numpy as np
import pytest

from taichi_matrix import TaiChiPipeline, PipelineResult


@pytest.fixture
def rng():
    return np.random.default_rng(42)


@pytest.fixture
def sample_input(rng):
    return rng.normal(0, 1, (32, 128))


class TestPipelineInit:
    def test_default_init(self):
        p = TaiChiPipeline()
        assert isinstance(p.status(), dict)
        assert len(p.status()) == 5
        for name in ["router", "mtp", "hex", "quant", "correct"]:
            assert name in p.status()

    def test_status_returns_bools(self):
        p = TaiChiPipeline()
        status = p.status()
        for v in status.values():
            assert isinstance(v, bool)

    def test_enable_subset(self):
        p = TaiChiPipeline(enable=["router", "mtp"])
        assert "router" in p.enabled
        assert "mtp" in p.enabled

    def test_repr(self):
        p = TaiChiPipeline()
        r = repr(p)
        assert "TaiChiPipeline" in r
        assert "enabled" in r


class TestPipelineRun:
    def test_run_returns_pipeline_result(self, sample_input):
        p = TaiChiPipeline()
        result = p.run(sample_input)
        assert isinstance(result, PipelineResult)

    def test_route_mode_is_str(self, sample_input):
        p = TaiChiPipeline()
        result = p.run(sample_input)
        assert result.route_mode in ("steady", "transitional", "turbulent")

    def test_weights_sum_near_one(self, sample_input):
        p = TaiChiPipeline()
        result = p.run(sample_input)
        assert abs(np.sum(result.route_weights) - 1.0) < 0.1

    def test_confidence_in_range(self, sample_input):
        p = TaiChiPipeline()
        result = p.run(sample_input)
        assert 0 <= result.confidence <= 1.0

    def test_timings_has_entries(self, sample_input):
        p = TaiChiPipeline()
        result = p.run(sample_input)
        assert isinstance(result.timings, dict)
        for s in result.modules_available:
            assert s in result.timings

    def test_modules_available_list(self, sample_input):
        p = TaiChiPipeline()
        result = p.run(sample_input)
        assert isinstance(result.modules_available, list)
        assert all(s in ["router", "mtp", "hex", "quant", "correct"]
                   for s in result.modules_available)

    def test_corrected_present_when_correct_available(self, sample_input):
        p = TaiChiPipeline()
        result = p.run(sample_input)
        if "correct" in result.modules_available:
            assert result.corrected is not None

    def test_pipeline_result_is_picklable(self, sample_input):
        import pickle
        p = TaiChiPipeline()
        result = p.run(sample_input)
        data = pickle.dumps(result)
        assert len(data) > 0

    def test_empty_input(self, rng):
        """Empty 6-dim input should work."""
        x = rng.normal(0, 1, (6, 6))
        p = TaiChiPipeline()
        result = p.run(x)
        assert isinstance(result, PipelineResult)

    def test_large_input(self, rng):
        x = rng.normal(0, 1, (256, 512))
        p = TaiChiPipeline()
        result = p.run(x)
        assert isinstance(result, PipelineResult)
        assert 0 <= result.confidence <= 1.0

    def test_enable_none_runs_all_available(self, sample_input):
        p = TaiChiPipeline(enable=None)
        result = p.run(sample_input)
        assert result.modules_available == p.available

    def test_enable_empty_runs_none(self, sample_input):
        p = TaiChiPipeline(enable=[])
        result = p.run(sample_input)
        assert result.modules_available == []

    def test_residue_reduction_in_range(self, sample_input):
        p = TaiChiPipeline()
        result = p.run(sample_input)
        assert result.residue_reduction >= 0
