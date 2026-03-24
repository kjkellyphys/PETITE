"""
Integration tests for PETITE.shower.Shower and the detector_cut utility.
These tests require pre-computed data files in data/ and are skipped otherwise.
"""
import numpy as np
import pytest

from PETITE.particle import Particle
from PETITE.physical_constants import m_electron, m_muon, m_pi0
from PETITE.shower import detector_cut

from conftest import requires_data


# ---------------------------------------------------------------------------
# Shower initialisation
# ---------------------------------------------------------------------------
@requires_data
class TestShowerInit:
    def test_graphite_shower_creates(self, graphite_shower):
        """Fixture should instantiate without error."""
        assert graphite_shower is not None

    def test_min_kinetic_energy_stored(self, graphite_shower):
        assert graphite_shower.min_kinetic_energy == pytest.approx(0.050)

    def test_target_material_stored(self, graphite_shower):
        assert graphite_shower.get_target_material() == "graphite"

    def test_material_properties_set(self, graphite_shower):
        props = graphite_shower.get_material_properties()
        # Should return Z, A, rho, dEdx (at minimum)
        assert len(props) >= 3


# ---------------------------------------------------------------------------
# Mean free path / NSigma
# ---------------------------------------------------------------------------
@requires_data
class TestGetMFP:
    def test_photon_mfp_positive(self, graphite_shower):
        mfp = graphite_shower.get_mfp([22, 1.0])
        assert mfp > 0.0

    def test_electron_mfp_positive(self, graphite_shower):
        mfp = graphite_shower.get_mfp([11, 1.0])
        assert mfp > 0.0

    def test_mfp_decreases_with_energy(self, graphite_shower):
        """Higher energy → larger cross section → shorter mean free path."""
        mfp_low = graphite_shower.get_mfp([22, 0.1])
        mfp_high = graphite_shower.get_mfp([22, 5.0])
        # not strictly guaranteed but expected for reasonable energies
        assert mfp_low >= mfp_high


# ---------------------------------------------------------------------------
# draw_sample
# ---------------------------------------------------------------------------
@requires_data
class TestDrawSample:
    @pytest.mark.parametrize("process,pid", [
        ("Brem", 11),
        ("PairProd", 22),
        ("Ann", -11),
        ("Comp", 22),
    ])
    def test_returns_array(self, graphite_shower, process, pid):
        sample = graphite_shower.draw_sample(0.5, process=process)
        assert hasattr(sample, "__len__")
        assert len(sample) > 0

    def test_brem_sample_has_four_dimensions(self, graphite_shower):
        """Brem uses 4D phase space."""
        sample = np.array(graphite_shower.draw_sample(1.0, process="Brem"))
        assert len(sample) == 4


# ---------------------------------------------------------------------------
# generate_shower
# ---------------------------------------------------------------------------
@requires_data
class TestGenerateShower:
    def _photon(self, E):
        return Particle([E, 0.0, 0.0, E], [0.0, 0.0, 0.0],
                        {"PID": 22, "weight": 1.0, "mass": 0.0})

    def _electron(self, E):
        pz = np.sqrt(E**2 - m_electron**2)
        return Particle([E, 0.0, 0.0, pz], [0.0, 0.0, 0.0],
                        {"PID": 11, "weight": 1.0, "mass": m_electron})

    def test_photon_shower_returns_list(self, graphite_shower):
        parts = graphite_shower.generate_shower(self._photon(1.0))
        assert isinstance(parts, list)
        assert len(parts) >= 1

    def test_all_particles_ended(self, graphite_shower):
        parts = graphite_shower.generate_shower(self._photon(1.0))
        assert all(p.get_ended() for p in parts)

    def test_shower_contains_electrons_or_photons(self, graphite_shower):
        parts = graphite_shower.generate_shower(self._photon(1.0))
        pids = {p.get_ids()["PID"] for p in parts}
        # a photon shower must produce electrons/positrons or at least photons
        assert pids & {11, -11, 22}

    def test_electron_shower_runs(self, graphite_shower):
        parts = graphite_shower.generate_shower(self._electron(0.5))
        assert isinstance(parts, list)
        assert len(parts) >= 1

    def test_low_energy_photon_terminates_immediately(self, graphite_shower):
        """Photon below threshold: shower of length 1 (just the initial particle)."""
        E_below = graphite_shower.min_kinetic_energy * 0.5
        p0 = self._photon(E_below)
        parts = graphite_shower.generate_shower(p0)
        # generate_shower returns a list with one entry for below-threshold input
        assert len(parts) == 1

    def test_weights_non_negative(self, graphite_shower):
        parts = graphite_shower.generate_shower(self._photon(1.0))
        for p in parts:
            assert p.get_ids()["weight"] >= 0.0


# ---------------------------------------------------------------------------
# detector_cut
# ---------------------------------------------------------------------------
class TestDetectorCut:
    def _photon(self, pz_frac=1.0):
        """Photon pointing within angle pz_frac * 1 GeV in z direction."""
        pt = np.sqrt(1 - pz_frac**2)
        return Particle([1.0, pt, 0.0, pz_frac], [0.0, 0.0, 0.0],
                        {"PID": 22, "weight": 1.0, "mass": 0.0})

    def test_on_axis_photon_passes(self):
        # Slightly off-axis so rT > 0 (detector_cut uses strict > for inner radius)
        p = Particle([1.0, 0.001, 0.0, np.sqrt(1.0 - 0.001**2)], [0.0, 0.0, 0.0],
                     {"PID": 22, "weight": 1.0, "mass": 0.0})
        result = detector_cut([p], [300.0], detector_radius=2.0)
        assert len(result[0]) == 1

    def test_off_axis_photon_fails(self):
        """Photon at 45° will miss a small detector at 300 m."""
        p = Particle([1.0, 1.0, 0.0, 0.0], [0.0, 0.0, 0.0],
                     {"PID": 22, "weight": 1.0, "mass": 0.0})
        result = detector_cut([p], [300.0], detector_radius=2.0)
        assert len(result[0]) == 0

    def test_empty_list_returns_empty(self):
        result = detector_cut([], [300.0], detector_radius=2.0, method="Sample")
        assert result == [[]]

    def test_efficiency_method_in_range(self):
        p = self._photon(pz_frac=1.0)
        eff = detector_cut([p], [300.0], detector_radius=2.0, method="Efficiency")
        assert 0.0 <= eff[0] <= 1.0

    def test_total_weight_method(self):
        p = self._photon(pz_frac=1.0)
        tw = detector_cut([p], [300.0], detector_radius=2.0, method="TotalWeight")
        assert tw[0] >= 0.0

    def test_energy_cut_filters(self):
        """Photon with E=1 GeV should be removed by energy_cut=(0, 0.5)."""
        p = self._photon(pz_frac=1.0)
        result = detector_cut([p], [300.0], detector_radius=2.0,
                               energy_cut=(0.0, 0.5), method="Sample")
        assert len(result[0]) == 0
