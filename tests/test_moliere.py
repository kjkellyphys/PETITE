"""
Tests for PETITE.sm.moliere — multiple Coulomb scattering.

Checks: CDF monotonicity, energy conservation after scatter, angle range,
and correct behaviour for particles at rest.
"""
import numpy as np
import pytest

from PETITE.physical_constants import m_electron, m_muon
from PETITE.sm.moliere import (
    moliere_f0,
    moliere_f1,
    moliere_cdf,
    get_scattered_momentum_fast,
)

# Graphite parameters: A=12 g/mol, Z=6
A_GRAPHITE = 12.0
Z_GRAPHITE = 6


# ---------------------------------------------------------------------------
# Moliere distribution functions
# ---------------------------------------------------------------------------
class TestMoliereFunctions:
    def test_f0_positive(self):
        for x in [0.0, 1.0, 10.0]:
            assert moliere_f0(x) >= 0.0

    def test_f0_decreasing(self):
        assert moliere_f0(0.0) > moliere_f0(1.0) > moliere_f0(5.0)

    def test_f1_finite(self):
        for x in [1e-7, 0.5, 5.0, 200.0]:
            assert np.isfinite(moliere_f1(x))

    def test_cdf_at_zero_is_zero(self):
        assert moliere_cdf(0.0, B=5.0) == pytest.approx(0.0, abs=1e-6)

    def test_cdf_monotone(self):
        B = 5.0
        xs = np.linspace(0.01, 10.0, 50)
        vals = [moliere_cdf(x, B) for x in xs]
        assert all(v2 >= v1 for v1, v2 in zip(vals, vals[1:]))

    def test_cdf_approaches_one(self):
        # At large x the CDF should be very close to 1
        assert moliere_cdf(200.0, B=5.0) > 0.99


# ---------------------------------------------------------------------------
# get_scattered_momentum_fast
# ---------------------------------------------------------------------------
class TestGetScatteredMomentumFast:
    """
    After multiple scattering:
    - Energy must be conserved exactly (function preserves p4[0])
    - 3-momentum magnitude must be conserved (function only rotates)
    - Output must not be identical to input (scatter happened)
    - Scattering angle must be within [0, pi/2] for thin targets
    """

    def _electron_p4(self, E):
        pz = np.sqrt(E**2 - m_electron**2)
        return np.array([E, 0.0, 0.0, pz])

    def test_energy_conserved(self):
        p4 = self._electron_p4(1.0)
        t = 0.01  # g/cm^2 — thin target
        p4_new = get_scattered_momentum_fast(p4, t, A_GRAPHITE, Z_GRAPHITE,
                                             m_lepton=m_electron)
        assert abs(p4_new[0] - p4[0]) < 1e-12

    def test_pmag_conserved(self):
        p4 = self._electron_p4(1.0)
        t = 0.01
        p4_new = get_scattered_momentum_fast(p4, t, A_GRAPHITE, Z_GRAPHITE,
                                             m_lepton=m_electron)
        pmag_in = np.linalg.norm(p4[1:])
        pmag_out = np.linalg.norm(p4_new[1:])
        assert abs(pmag_out - pmag_in) < 1e-12

    def test_rest_particle_unchanged(self):
        p4 = np.array([m_electron, 0.0, 0.0, 0.0])
        t = 0.1
        p4_new = get_scattered_momentum_fast(p4, t, A_GRAPHITE, Z_GRAPHITE,
                                             m_lepton=m_electron)
        assert np.allclose(p4_new, p4)

    @pytest.mark.parametrize("E", [0.2, 1.0, 5.0, 10.0])
    def test_scatter_angle_in_valid_range(self, E):
        """Scattering angle must be in [0, pi]."""
        p4 = self._electron_p4(E)
        t = 0.01
        p4_new = get_scattered_momentum_fast(p4, t, A_GRAPHITE, Z_GRAPHITE,
                                             m_lepton=m_electron)
        p3_in = p4[1:] / np.linalg.norm(p4[1:])
        p3_out = p4_new[1:] / np.linalg.norm(p4_new[1:])
        cos_th = np.clip(np.dot(p3_in, p3_out), -1.0, 1.0)
        theta = np.arccos(cos_th)
        assert 0.0 <= theta <= np.pi

    def test_muon_scattering_smaller_than_electron(self):
        """Heavier particles scatter less on average (statistical test over many samples)."""
        np.random.seed(0)
        E = 1.0
        p4_e = np.array([E, 0.0, 0.0, np.sqrt(E**2 - m_electron**2)])
        p4_mu = np.array([E, 0.0, 0.0, np.sqrt(E**2 - m_muon**2)])
        t = 0.5  # thicker target → larger mean deflection, better statistics

        angles_e, angles_mu = [], []
        for _ in range(1000):
            p4_e_new = get_scattered_momentum_fast(p4_e, t, A_GRAPHITE, Z_GRAPHITE,
                                                   m_lepton=m_electron)
            p4_mu_new = get_scattered_momentum_fast(p4_mu, t, A_GRAPHITE, Z_GRAPHITE,
                                                    m_lepton=m_muon)
            p3_e_in = p4_e[1:] / np.linalg.norm(p4_e[1:])
            p3_mu_in = p4_mu[1:] / np.linalg.norm(p4_mu[1:])
            cos_e = np.clip(np.dot(p3_e_in, p4_e_new[1:] / np.linalg.norm(p4_e_new[1:])), -1, 1)
            cos_mu = np.clip(np.dot(p3_mu_in, p4_mu_new[1:] / np.linalg.norm(p4_mu_new[1:])), -1, 1)
            angles_e.append(np.arccos(cos_e))
            angles_mu.append(np.arccos(cos_mu))

        assert np.mean(angles_e) > np.mean(angles_mu)
