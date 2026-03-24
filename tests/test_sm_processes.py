"""
Tests for PETITE.sm.sm_processes — differential cross section callables.

All dσ values must be non-negative; spot-check specific kinematic points;
and confirm Bug #1 regression (dsigma_compton_dxdpt variable ordering).
"""
import numpy as np
import pytest

from PETITE.physical_constants import m_electron, alpha_em, target_information
from PETITE.sm.sm_processes import (
    dsigma_brem_dimensionless,
    dsigma_pairprod_dimensionless,
    dsigma_compton_dCT,
    dsigma_annihilation_dCT,
    dsigma_moller_dCT,
    dsigma_muonelectron_dCT,
    dsigma_compton_dxdpt,
    g2_elastic,
    aa, aap,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
GRAPHITE = target_information["graphite"]

def _brem_event_info(T_inc, m_lepton=m_electron):
    return {
        "T_inc": T_inc,
        "m_lepton": m_lepton,
        "Eg_min": 1e-3,
        "Z_T": GRAPHITE["Z_T"],
    }

def _pp_event_info(T_inc):
    return {
        "T_inc": T_inc,
        "Z_T": GRAPHITE["Z_T"],
    }


# ---------------------------------------------------------------------------
# Form factors
# ---------------------------------------------------------------------------
class TestFormFactors:
    def test_aa_positive(self):
        assert aa(6, m_electron) > 0

    def test_aap_positive(self):
        assert aap(6, m_electron) > 0

    def test_g2_elastic_positive(self):
        ei = {"Z_T": 6}
        val = g2_elastic(ei, 1e-6)
        assert val >= 0

    def test_g2_elastic_zero_at_zero_t(self):
        ei = {"Z_T": 6}
        assert g2_elastic(ei, 0.0) == 0.0


# ---------------------------------------------------------------------------
# Bremsstrahlung
# ---------------------------------------------------------------------------
class TestBremCrossSection:
    def _sample(self):
        return np.array([0.5, 0.3, 0.1, 0.5])  # generic interior point

    @pytest.mark.parametrize("T_inc", [0.1, 0.5, 1.0, 5.0])
    def test_non_negative(self, T_inc):
        ei = _brem_event_info(T_inc)
        f = dsigma_brem_dimensionless(ei, 4)
        val = f(self._sample())
        assert float(val) >= 0.0

    def test_batch_mode(self):
        ei = _brem_event_info(1.0)
        f = dsigma_brem_dimensionless(ei, 4, batch_mode=True)
        pts = np.random.uniform(0.1, 0.9, size=(10, 4))
        vals = f(pts)
        assert vals.shape == (10,)
        assert np.all(vals >= 0)

    def test_forbidden_kinematics_zero(self):
        """x1 near 1 → w ≈ ep − m which barely satisfies allowed; x1 > 1 impossible."""
        ei = _brem_event_info(1.0)
        f = dsigma_brem_dimensionless(ei, 4)
        # x2 == x3 → d = 0, dp = 0 → forbidden (d > 0 & dp > 0 fails)
        sample_bad = np.array([0.5, 0.0, 0.0, 0.5])
        assert float(f(sample_bad)) == 0.0


# ---------------------------------------------------------------------------
# Pair Production
# ---------------------------------------------------------------------------
class TestPairProdCrossSection:
    @pytest.mark.parametrize("T_inc", [0.5, 1.0, 5.0])
    def test_non_negative(self, T_inc):
        ei = _pp_event_info(T_inc)
        f = dsigma_pairprod_dimensionless(ei, 4)
        val = f(np.array([0.5, 0.3, 0.1, 0.5]))
        assert float(val) >= 0.0

    def test_batch_mode(self):
        ei = _pp_event_info(1.0)
        f = dsigma_pairprod_dimensionless(ei, 4, batch_mode=True)
        pts = np.random.uniform(0.05, 0.95, size=(10, 4))
        vals = f(pts)
        assert np.all(vals >= 0)


# ---------------------------------------------------------------------------
# Compton dCT
# ---------------------------------------------------------------------------
class TestComptonDCT:
    @pytest.mark.parametrize("ct", [-0.9, 0.0, 0.5, 0.9])
    def test_non_negative(self, ct):
        ei = {"T_inc": 1.0}
        f = dsigma_compton_dCT(ei, 1)
        val = f(np.array([ct]))
        assert float(val) >= 0.0

    def test_batch_mode(self):
        ei = {"T_inc": 1.0}
        f = dsigma_compton_dCT(ei, 1, batch_mode=True)
        cts = np.linspace(-0.9, 0.9, 20).reshape(-1, 1)
        vals = f(cts)
        assert np.all(vals >= 0)

    def test_increases_forward(self):
        """Compton cross section should be larger in the forward direction."""
        ei = {"T_inc": 1.0}
        f = dsigma_compton_dCT(ei, 1)
        val_fwd = float(f(np.array([0.9])))
        val_bwd = float(f(np.array([-0.9])))
        assert val_fwd > val_bwd


# ---------------------------------------------------------------------------
# Annihilation dCT
# ---------------------------------------------------------------------------
class TestAnnihilationDCT:
    @pytest.mark.parametrize("ct", [-0.9, 0.0, 0.5])
    def test_non_negative(self, ct):
        ei = {"T_inc": 1.0}
        f = dsigma_annihilation_dCT(ei, 1)
        val = f(np.array([ct]))
        assert float(val) >= 0.0

    def test_batch_mode(self):
        ei = {"T_inc": 1.0}
        f = dsigma_annihilation_dCT(ei, 1, batch_mode=True)
        cts = np.linspace(-0.9, 0.9, 20).reshape(-1, 1)
        vals = f(cts)
        assert np.all(vals >= 0)


# ---------------------------------------------------------------------------
# Moller dCT
# ---------------------------------------------------------------------------
class TestMollerDCT:
    def test_non_negative_interior(self):
        ei = {"T_inc": 1.0}
        f = dsigma_moller_dCT(ei, 1)
        val = f(np.array([0.0]))
        assert float(val) >= 0.0

    def test_zero_at_ct_equals_one(self):
        """ct=1 → forward scatter → zero (kinematic cutoff)."""
        ei = {"T_inc": 1.0}
        f = dsigma_moller_dCT(ei, 1)
        val = f(np.array([1.0 - 1e-10]))
        assert float(val) == 0.0

    def test_batch_mode(self):
        ei = {"T_inc": 1.0}
        f = dsigma_moller_dCT(ei, 1, batch_mode=True)
        cts = np.linspace(-0.5, 0.5, 10).reshape(-1, 1)
        vals = f(cts)
        assert np.all(vals >= 0)


# ---------------------------------------------------------------------------
# dsigma_compton_dxdpt — Bug #1 regression test
# ---------------------------------------------------------------------------
class TestComptonDxDpt:
    """
    Bug #1: variables `a` and `b` are computed before `x = variables[0]` is
    assigned, causing an UnboundLocalError.  These tests are marked xfail
    until the bug is fixed; they will become passing xfails once corrected.
    """

    @pytest.mark.xfail(strict=True, reason="Bug #1: x used before assignment in dsigma_compton_dxdpt")
    def test_returns_without_name_error(self):
        """After fixing Bug #1 this should not raise UnboundLocalError."""
        ei = {"T_inc": 1.0, "Z_eff": 6, "mV": 0.0}
        result = dsigma_compton_dxdpt(ei, [0.5, 0.1])
        assert np.isfinite(result)

    @pytest.mark.xfail(strict=True, reason="Bug #1: x used before assignment in dsigma_compton_dxdpt")
    def test_returns_non_negative(self):
        ei = {"T_inc": 1.0, "Z_eff": 6, "mV": 0.0}
        result = dsigma_compton_dxdpt(ei, [0.5, 0.1])
        assert result >= 0.0

    @pytest.mark.xfail(strict=True, reason="Bug #1: x used before assignment in dsigma_compton_dxdpt")
    def test_batch_returns_array(self):
        ei = {"T_inc": 1.0, "Z_eff": 6, "mV": 0.0}
        batch = [[0.3, 0.05], [0.5, 0.1], [0.7, 0.2]]
        result = dsigma_compton_dxdpt(ei, batch)
        assert len(result) == 3
        assert all(np.isfinite(v) for v in result)
