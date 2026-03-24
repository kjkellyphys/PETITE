"""
Tests for PETITE.particle — Particle class, mass_dict, meson_decay_dict, kinematics helpers.
"""
import numpy as np
import pytest

from PETITE.particle import Particle, mass_dict, meson_decay_dict
from PETITE.physical_constants import m_electron, m_muon, m_pi0, m_pi_pm


# ---------------------------------------------------------------------------
# mass_dict
# ---------------------------------------------------------------------------
class TestMassDict:
    def test_electron_mass(self):
        assert mass_dict[11] == m_electron

    def test_positron_same_as_electron(self):
        assert mass_dict[-11] == mass_dict[11]

    def test_photon_massless(self):
        assert mass_dict[22] == 0.0

    def test_muon_mass(self):
        assert mass_dict[13] == m_muon

    def test_pi0_mass(self):
        assert mass_dict[111] == m_pi0

    def test_pion_charged_mass(self):
        assert mass_dict[211] == m_pi_pm
        assert mass_dict[-211] == m_pi_pm

    # Bug #7: duplicate keys 12, -12, 14, -14 (last definition wins, but values same)
    def test_neutrino_massless(self):
        assert mass_dict[12] == 0.0
        assert mass_dict[-12] == 0.0
        assert mass_dict[14] == 0.0
        assert mass_dict[-14] == 0.0


# ---------------------------------------------------------------------------
# Particle construction
# ---------------------------------------------------------------------------
class TestParticleConstruction:
    def test_electron_from_list(self):
        p = Particle([m_electron, 0.0, 0.0, 0.0], [0.0, 0.0, 0.0],
                     {"PID": 11, "mass": m_electron})
        assert np.allclose(p.get_p0(), [m_electron, 0.0, 0.0, 0.0])

    def test_mass_inferred_from_four_vector(self):
        # Provide no mass; it should be computed from p0
        pz = np.sqrt(1.0**2 - m_electron**2)
        p = Particle([1.0, 0.0, 0.0, pz], [0.0, 0.0, 0.0], {"PID": 11})
        assert abs(p.get_ids()["mass"] - m_electron) < 1e-6

    def test_scalar_energy_initializes_z_momentum(self):
        # If p0 given as scalar it should set pz = sqrt(E^2 - m^2)
        p = Particle(1.0, [0.0, 0.0, 0.0], {"PID": 11, "mass": m_electron})
        E, px, py, pz = p.get_p0()
        assert E == 1.0
        assert px == 0.0
        assert py == 0.0
        assert abs(pz - np.sqrt(1.0**2 - m_electron**2)) < 1e-10

    def test_position_stored(self):
        p = Particle([m_electron, 0.0, 0.0, 0.0], [1.0, 2.0, 3.0],
                     {"PID": 11, "mass": m_electron})
        assert np.allclose(p.get_r0(), [1.0, 2.0, 3.0])

    def test_default_ids_filled_in(self):
        p = Particle([m_electron, 0.0, 0.0, 0.0], [0.0, 0.0, 0.0],
                     {"PID": 11, "mass": m_electron})
        ids = p.get_ids()
        assert "weight" in ids
        assert ids["weight"] == 1.0
        assert "generation_process" in ids

    def test_ended_starts_false(self):
        p = Particle([m_electron, 0.0, 0.0, 0.0], [0.0, 0.0, 0.0],
                     {"PID": 11, "mass": m_electron})
        assert p.get_ended() is False

    def test_set_ended_true(self):
        p = Particle([m_electron, 0.0, 0.0, 0.0], [0.0, 0.0, 0.0],
                     {"PID": 11, "mass": m_electron})
        p.set_ended(True)
        assert p.get_ended() is True

    def test_set_ended_invalid_raises(self):
        p = Particle([m_electron, 0.0, 0.0, 0.0], [0.0, 0.0, 0.0],
                     {"PID": 11, "mass": m_electron})
        with pytest.raises(ValueError):
            p.set_ended("yes")

    def test_copy_is_independent(self):
        p = Particle([1.0, 0.0, 0.0, np.sqrt(1.0**2 - m_electron**2)],
                     [0.0, 0.0, 0.0], {"PID": 11, "mass": m_electron})
        p2 = p.copy()
        assert np.allclose(p2.get_p0(), p.get_p0())
        assert p2 is not p


# ---------------------------------------------------------------------------
# lose_energy
# ---------------------------------------------------------------------------
class TestLoseEnergy:
    def _electron(self, E):
        pz = np.sqrt(E**2 - m_electron**2)
        return Particle([E, 0.0, 0.0, pz], [0.0, 0.0, 0.0],
                        {"PID": 11, "mass": m_electron})

    def test_on_shell_after_energy_loss(self):
        p = self._electron(1.0)
        p.lose_energy(0.1)
        E, px, py, pz = p.get_pf()
        m2 = E**2 - px**2 - py**2 - pz**2
        assert abs(m2 - m_electron**2) < 1e-10

    def test_energy_decreases(self):
        p = self._electron(1.0)
        p.lose_energy(0.1)
        assert p.get_pf()[0] < 1.0

    def test_energy_cannot_go_below_mass(self):
        p = self._electron(0.6)
        p.lose_energy(10.0)
        assert p.get_pf()[0] >= m_electron


# ---------------------------------------------------------------------------
# rotation_matrix
# ---------------------------------------------------------------------------
class TestRotationMatrix:
    def test_z_axis_particle_gives_identity_columns(self):
        """A particle pointing along +z has R that maps z-hat to z-hat."""
        pz = np.sqrt(1.0**2 - m_electron**2)
        p = Particle([1.0, 0.0, 0.0, pz], [0.0, 0.0, 0.0],
                     {"PID": 11, "mass": m_electron})
        R = np.array(p.rotation_matrix())
        # R @ e_z should be e_z
        assert np.allclose(R @ [0, 0, 1], [0, 0, 1], atol=1e-10)

    def test_rotation_matrix_is_orthogonal(self):
        """Rotation matrix must be orthogonal."""
        pz = np.sqrt(1.0**2 - m_electron**2)
        p = Particle([1.0, 0.0, pz * 0.1, pz * 0.9], [0.0, 0.0, 0.0],
                     {"PID": 11, "mass": m_electron})
        R = np.array(p.rotation_matrix())
        assert np.allclose(R @ R.T, np.eye(3), atol=1e-10)


# ---------------------------------------------------------------------------
# boost_matrix
# ---------------------------------------------------------------------------
class TestBoostMatrix:
    def test_rest_particle_gives_identity(self):
        """Particle at rest → boost_matrix = identity."""
        p = Particle([m_electron, 0.0, 0.0, 0.0], [0.0, 0.0, 0.0],
                     {"PID": 11, "mass": m_electron})
        B = np.array(p.boost_matrix())
        assert np.allclose(B, np.eye(4), atol=1e-10)

    def test_boost_preserves_minkowski_norm(self):
        """Boosting a rest-frame mass vector should give p^2 = m^2."""
        pz = np.sqrt(1.0**2 - m_electron**2)
        p = Particle([1.0, 0.0, 0.0, pz], [0.0, 0.0, 0.0],
                     {"PID": 11, "mass": m_electron})
        B = np.array(p.boost_matrix())
        p_rest = [m_electron, 0.0, 0.0, 0.0]
        p_lab = B @ p_rest
        m2 = p_lab[0]**2 - p_lab[1]**2 - p_lab[2]**2 - p_lab[3]**2
        assert abs(m2 - m_electron**2) < 1e-10

    # Bug #6 regression: when gamma==1 beta should be 0, not 1
    def test_beta_zero_for_rest_particle(self):
        """Rest particle: beta must be 0 (Bug #6 regression)."""
        p = Particle([m_electron, 0.0, 0.0, 0.0], [0.0, 0.0, 0.0],
                     {"PID": 11, "mass": m_electron})
        # If beta=1 erroneously, off-diagonal elements would be non-zero
        B = np.array(p.boost_matrix())
        assert np.allclose(B, np.eye(4), atol=1e-10)


# ---------------------------------------------------------------------------
# two_body_decay
# ---------------------------------------------------------------------------
class TestTwoBodyDecay:
    def test_pi0_to_two_photons_energy_conservation(self):
        p = Particle([m_pi0, 0.0, 0.0, 0.0], [0.0, 0.0, 0.0],
                     {"PID": 111, "mass": m_pi0, "stability": "short-lived"})
        daughters = p.two_body_decay({"PID": 22}, {"PID": 22})
        E_total = daughters[0].get_p0()[0] + daughters[1].get_p0()[0]
        assert abs(E_total - m_pi0) < 1e-10

    def test_pi0_to_two_photons_momentum_conservation(self):
        p = Particle([m_pi0, 0.0, 0.0, 0.0], [0.0, 0.0, 0.0],
                     {"PID": 111, "mass": m_pi0, "stability": "short-lived"})
        daughters = p.two_body_decay({"PID": 22}, {"PID": 22})
        p3_sum = (np.array(daughters[0].get_p0()[1:]) +
                  np.array(daughters[1].get_p0()[1:]))
        assert np.allclose(p3_sum, [0.0, 0.0, 0.0], atol=1e-10)

    def test_daughters_on_shell(self):
        p = Particle([m_pi0, 0.0, 0.0, 0.0], [0.0, 0.0, 0.0],
                     {"PID": 111, "mass": m_pi0, "stability": "short-lived"})
        for d in p.two_body_decay({"PID": 22}, {"PID": 22}):
            E, px, py, pz = d.get_p0()
            assert abs(E**2 - px**2 - py**2 - pz**2) < 1e-10

    def test_missing_mass_raises(self):
        p = Particle([m_pi0, 0.0, 0.0, 0.0], [0.0, 0.0, 0.0],
                     {"PID": 111, "mass": m_pi0, "stability": "short-lived"})
        with pytest.raises(ValueError):
            p.two_body_decay({"foo": "bar"}, {"PID": 22})


# ---------------------------------------------------------------------------
# decay_particle
# ---------------------------------------------------------------------------
class TestDecayParticle:
    def test_pi0_short_lived_produces_two_photons(self):
        p = Particle([m_pi0, 0.0, 0.0, 0.0], [0.0, 0.0, 0.0],
                     {"PID": 111, "mass": m_pi0, "stability": "short-lived"})
        daughters = p.decay_particle()
        assert len(daughters) == 2
        for d in daughters:
            assert d.get_ids()["PID"] == 22

    def test_decay_marks_particle_ended(self):
        p = Particle([m_pi0, 0.0, 0.0, 0.0], [0.0, 0.0, 0.0],
                     {"PID": 111, "mass": m_pi0, "stability": "short-lived"})
        p.decay_particle()
        assert p.get_ended() is True

    def test_unknown_pid_raises(self):
        p = Particle([1.0, 0.0, 0.0, 1.0], [0.0, 0.0, 0.0],
                     {"PID": 9999, "mass": 0.0})
        with pytest.raises(ValueError):
            p.decay_particle()

    # Bug #9: eta and eta' have 3-body channels that crash decay_particle.
    # These tests document the current (buggy) behaviour so they can be updated
    # once the bug is fixed.
    def test_eta_decay_channels_available(self):
        assert 221 in meson_decay_dict
        channels = meson_decay_dict[221]
        three_body = [c for c in channels if len(c[1]) > 2]
        assert len(three_body) > 0, "eta should have 3-body channels (bug #9 marker)"

    def test_eta_prime_decay_channels_available(self):
        assert 331 in meson_decay_dict
        channels = meson_decay_dict[331]
        three_body = [c for c in channels if len(c[1]) > 2]
        assert len(three_body) > 0, "eta' should have 3-body channels (bug #9 marker)"
