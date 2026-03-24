"""
Integration tests for PETITE.bsm_shower.BSMShower.
These tests require pre-computed data files in data/ and are skipped otherwise.
"""
import numpy as np
import pytest

from PETITE.particle import Particle
from PETITE.physical_constants import m_electron

from conftest import requires_data


# ---------------------------------------------------------------------------
# BSMShower initialisation
# ---------------------------------------------------------------------------
@requires_data
class TestBSMShowerInit:
    def test_dark_vector_shower_creates(self, dark_vector_shower):
        assert dark_vector_shower is not None

    def test_mcp_shower_creates(self, mcp_shower):
        assert mcp_shower is not None

    def test_dark_vector_active_processes_nonempty(self, dark_vector_shower):
        assert len(dark_vector_shower.active_processes) > 0

    def test_mcp_active_processes_contains_annihilation(self, mcp_shower):
        assert "mcpAnnihilation" in mcp_shower.active_processes

    def test_organizational_parameter_matches_model_key(self, mcp_shower):
        # Organizational parameter should match or be ≤ requested m_chi=0.010
        assert mcp_shower._organizational_parameter <= 0.010 + 1e-6


# ---------------------------------------------------------------------------
# draw_bsm_sample
# ---------------------------------------------------------------------------
@requires_data
class TestDrawBSMSample:
    def test_mcp_annihilation_returns_sample(self, mcp_shower):
        sample = mcp_shower.draw_bsm_sample(1.0, process="mcpAnnihilation")
        assert hasattr(sample, "__len__")
        assert len(sample) > 0

    def test_mcp_sample_nonempty(self, mcp_shower):
        sample = mcp_shower.draw_bsm_sample(1.0, process="mcpAnnihilation")
        assert len(sample) > 0

    def test_invalid_process_raises(self, mcp_shower):
        with pytest.raises(Exception):
            mcp_shower.draw_bsm_sample(1.0, process="NonExistentProcess")


# ---------------------------------------------------------------------------
# get_bsm_weight
# ---------------------------------------------------------------------------
@requires_data
class TestGetBSMWeight:
    def _positron(self, E):
        pz = np.sqrt(E**2 - m_electron**2)
        return Particle([E, 0.0, 0.0, pz], [0.0, 0.0, 0.0],
                        {"PID": -11, "mass": m_electron, "weight": 1.0})

    def test_positron_weight_non_negative(self, mcp_shower):
        p = self._positron(1.0)
        w = mcp_shower.get_bsm_weight(p, "mcpAnnihilation")
        assert w >= 0.0

    def test_incompatible_pid_returns_zero(self, mcp_shower):
        """A muon cannot undergo mcpAnnihilation."""
        p_mu = Particle([1.0, 0.0, 0.0, 0.9], [0.0, 0.0, 0.0],
                        {"PID": 13, "mass": 0.1, "weight": 1.0})
        w = mcp_shower.get_bsm_weight(p_mu, "mcpAnnihilation")
        assert w == 0.0

    def test_list_input_works(self, mcp_shower):
        """get_bsm_weight should also accept a [PID, energy] list."""
        w = mcp_shower.get_bsm_weight([-11, 1.0], "mcpAnnihilation")
        assert w >= 0.0


# ---------------------------------------------------------------------------
# generate_bsm_shower
# ---------------------------------------------------------------------------
@requires_data
class TestGenerateBSMShower:
    def _photon(self, E=1.0):
        return Particle([E, 0.0, 0.0, E], [0.0, 0.0, 0.0],
                        {"PID": 22, "weight": 1.0, "mass": 0.0})

    def test_returns_two_lists(self, mcp_shower):
        p0 = self._photon(1.0)
        sm_shower = mcp_shower.generate_shower(p0)
        result = mcp_shower.generate_bsm_shower(ExDir=list(sm_shower))
        assert isinstance(result, (list, tuple))
        assert len(result) == 2

    def test_sm_shower_nonempty(self, mcp_shower):
        p0 = self._photon(1.0)
        sm_shower = mcp_shower.generate_shower(p0)
        sm_part, bsm_part = mcp_shower.generate_bsm_shower(ExDir=list(sm_shower))
        assert len(sm_part) > 0

    def test_bsm_weights_non_negative(self, mcp_shower):
        p0 = self._photon(1.0)
        sm_shower = mcp_shower.generate_shower(p0)
        _, bsm_part = mcp_shower.generate_bsm_shower(ExDir=list(sm_shower))
        for p in bsm_part:
            assert p.get_ids()["weight"] >= 0.0

    def test_none_input_returns_none(self, mcp_shower):
        result = mcp_shower.generate_bsm_shower(ExDir=None, SParams=None)
        assert result is None
