"""
Tests for PETITE.sm.kinematics — four-vector reconstruction functions.

All kinematic functions must conserve 4-momentum (or at least energy for the
small-angle approximations) and produce on-shell particles.
"""
import numpy as np
import pytest

from PETITE.particle import Particle
from PETITE.physical_constants import m_electron, m_muon
from PETITE.sm.kinematics import (
    l_to_lgamma_fourvecs,
    gamma_to_epem_fourvecs,
    compton_fourvecs,
    ee_to_ee_fourvecs,
    mue_to_mue_fourvecs,
    annihilation_fourvecs,
)
from conftest import minkowski_sq


def _electron(E):
    pz = np.sqrt(E**2 - m_electron**2)
    return Particle([E, 0.0, 0.0, pz], [0.0, 0.0, 0.0],
                    {"PID": 11, "mass": m_electron})


def _photon(E):
    return Particle([E, 0.0, 0.0, E], [0.0, 0.0, 0.0],
                    {"PID": 22, "mass": 0.0})


def _positron(E):
    pz = np.sqrt(E**2 - m_electron**2)
    return Particle([E, 0.0, 0.0, pz], [0.0, 0.0, 0.0],
                    {"PID": -11, "mass": m_electron})


def _muon(E):
    pz = np.sqrt(E**2 - m_muon**2)
    return Particle([E, 0.0, 0.0, pz], [0.0, 0.0, 0.0],
                    {"PID": 13, "mass": m_muon})


# ---------------------------------------------------------------------------
# Bremsstrahlung: l -> l + gamma  (small-angle approximation)
# Energy is conserved exactly; 3-momentum is only approximately conserved.
# ---------------------------------------------------------------------------
class TestBremFourvecs:
    @pytest.mark.parametrize("E_inc", [0.5, 1.0, 5.0, 10.0])
    def test_energy_conserved(self, E_inc):
        p0 = _electron(E_inc)
        # Uniform sampling in [0,1]^4
        sample = np.random.uniform(0.01, 0.99, size=4)
        ep4v, g4v = l_to_lgamma_fourvecs(p0, sample)
        assert abs(ep4v[0] + g4v[0] - E_inc) < 1e-9

    def test_photon_massless(self):
        p0 = _electron(1.0)
        sample = np.random.uniform(0.01, 0.99, size=4)
        _, g4v = l_to_lgamma_fourvecs(p0, sample)
        m2 = minkowski_sq(g4v)
        assert abs(m2) < 1e-10

    def test_outgoing_lepton_on_shell(self):
        p0 = _electron(1.0)
        sample = np.random.uniform(0.01, 0.99, size=4)
        ep4v, _ = l_to_lgamma_fourvecs(p0, sample)
        m2 = minkowski_sq(ep4v)
        assert abs(m2 - m_electron**2) < 1e-8


# ---------------------------------------------------------------------------
# Pair production: gamma -> e+ e-  (small-angle approximation)
# ---------------------------------------------------------------------------
class TestPairProdFourvecs:
    @pytest.mark.parametrize("E_gamma", [0.5, 1.0, 5.0])
    def test_energy_conserved(self, E_gamma):
        p0 = _photon(E_gamma)
        sample = np.random.uniform(0.01, 0.99, size=4)
        pp4v, pm4v = gamma_to_epem_fourvecs(p0, sample)
        assert abs(pp4v[0] + pm4v[0] - E_gamma) < 1e-9

    def test_positron_on_shell(self):
        p0 = _photon(1.0)
        sample = np.random.uniform(0.01, 0.99, size=4)
        pp4v, _ = gamma_to_epem_fourvecs(p0, sample)
        assert abs(minkowski_sq(pp4v) - m_electron**2) < 1e-8

    def test_electron_on_shell(self):
        p0 = _photon(1.0)
        sample = np.random.uniform(0.01, 0.99, size=4)
        _, pm4v = gamma_to_epem_fourvecs(p0, sample)
        assert abs(minkowski_sq(pm4v) - m_electron**2) < 1e-8


# ---------------------------------------------------------------------------
# Compton scattering: gamma e- -> gamma e-  (exact CM kinematics)
# ---------------------------------------------------------------------------
class TestComptonFourvecs:
    def test_four_momentum_conserved(self):
        Eg = 1.0
        p0 = _photon(Eg)
        ct = np.random.uniform(-0.99, 0.99)
        pe4v, pV4v = compton_fourvecs(p0, [ct])
        # Total incoming 4-mom = [Eg + m_e, 0, 0, Eg]
        p_in = np.array([Eg + m_electron, 0.0, 0.0, Eg])
        p_out = np.array(pe4v) + np.array(pV4v)
        assert np.allclose(p_in, p_out, atol=1e-9)

    def test_outgoing_photon_massless(self):
        p0 = _photon(1.0)
        ct = 0.5
        _, pV4v = compton_fourvecs(p0, [ct])
        assert abs(minkowski_sq(pV4v)) < 1e-10

    def test_outgoing_electron_on_shell(self):
        p0 = _photon(1.0)
        ct = 0.5
        pe4v, _ = compton_fourvecs(p0, [ct])
        assert abs(minkowski_sq(pe4v) - m_electron**2) < 1e-10

    @pytest.mark.parametrize("ct", [-0.99, 0.0, 0.99])
    def test_various_scattering_angles(self, ct):
        p0 = _photon(1.0)
        pe4v, pV4v = compton_fourvecs(p0, [ct])
        p_in = np.array([1.0 + m_electron, 0.0, 0.0, 1.0])
        p_out = np.array(pe4v) + np.array(pV4v)
        assert np.allclose(p_in, p_out, atol=1e-9)


# ---------------------------------------------------------------------------
# Moller / Bhabha: e e -> e e  (exact CM kinematics)
# ---------------------------------------------------------------------------
class TestEEtoEEFourvecs:
    @pytest.mark.parametrize("E_inc", [0.6, 1.0, 5.0])
    def test_four_momentum_conserved(self, E_inc):
        p0 = _electron(E_inc)
        ct = np.random.uniform(-0.99, 0.99)
        p1, p2 = ee_to_ee_fourvecs(p0, [ct])
        # Incoming: electron + target electron at rest
        p_in = np.array([E_inc + m_electron, 0.0, 0.0,
                         np.sqrt(E_inc**2 - m_electron**2)])
        p_out = np.array(p1) + np.array(p2)
        assert np.allclose(p_in, p_out, atol=1e-9)

    def test_both_electrons_on_shell(self):
        p0 = _electron(1.0)
        p1, p2 = ee_to_ee_fourvecs(p0, [0.5])
        assert abs(minkowski_sq(p1) - m_electron**2) < 1e-10
        assert abs(minkowski_sq(p2) - m_electron**2) < 1e-10


# ---------------------------------------------------------------------------
# Mu-e scattering: mu e -> mu e  (exact CM kinematics)
# ---------------------------------------------------------------------------
class TestMuEFourvecs:
    def test_four_momentum_conserved(self):
        E_mu = 5.0
        p0 = _muon(E_mu)
        ct = 0.3
        pm, pe = mue_to_mue_fourvecs(p0, [ct])
        pz_mu = np.sqrt(E_mu**2 - m_muon**2)
        p_in = np.array([E_mu + m_electron, 0.0, 0.0, pz_mu])
        p_out = np.array(pm) + np.array(pe)
        assert np.allclose(p_in, p_out, atol=1e-9)

    def test_outgoing_muon_on_shell(self):
        p0 = _muon(5.0)
        pm, _ = mue_to_mue_fourvecs(p0, [0.3])
        assert abs(minkowski_sq(pm) - m_muon**2) < 1e-10

    def test_outgoing_electron_on_shell(self):
        p0 = _muon(5.0)
        _, pe = mue_to_mue_fourvecs(p0, [0.3])
        assert abs(minkowski_sq(pe) - m_electron**2) < 1e-10


# ---------------------------------------------------------------------------
# Annihilation: e+ e- -> gamma gamma  (exact CM kinematics)
# ---------------------------------------------------------------------------
class TestAnnihilationFourvecs:
    def test_four_momentum_conserved(self):
        E_pos = 1.0
        p0 = _positron(E_pos)
        ct = 0.5
        pg, pV = annihilation_fourvecs(p0, [ct])
        pz_pos = np.sqrt(E_pos**2 - m_electron**2)
        p_in = np.array([E_pos + m_electron, 0.0, 0.0, pz_pos])
        p_out = np.array(pg) + np.array(pV)
        assert np.allclose(p_in, p_out, atol=1e-9)

    def test_both_photons_massless(self):
        p0 = _positron(1.0)
        pg, pV = annihilation_fourvecs(p0, [0.5])
        assert abs(minkowski_sq(pg)) < 1e-10
        assert abs(minkowski_sq(pV)) < 1e-10

    @pytest.mark.parametrize("ct", [-0.99, 0.0, 0.99])
    def test_various_angles(self, ct):
        p0 = _positron(1.0)
        pg, pV = annihilation_fourvecs(p0, [ct])
        assert abs(minkowski_sq(pg)) < 1e-10
