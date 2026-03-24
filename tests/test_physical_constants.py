"""
Tests for PETITE.physical_constants — value sanity checks and unit consistency.
"""
import numpy as np
import pytest

import PETITE.physical_constants as pc


class TestUnitDefinitions:
    def test_gev_is_one(self):
        assert pc.GeV == 1

    def test_mev_is_1e_minus3(self):
        assert pc.MeV == 1e-3

    def test_kev_is_1e_minus6(self):
        assert pc.keV == 1e-6

    def test_ev_is_1e_minus9(self):
        assert pc.eV == 1e-9

    def test_tev_is_1e3(self):
        assert pc.TeV == 1e3


class TestParticleMasses:
    def test_electron_mass_kev(self):
        # PDG 2022: m_e = 510.998950 keV
        assert abs(pc.m_electron / pc.keV - 510.998950) < 1e-3

    def test_muon_mass_mev(self):
        # PDG 2022: m_mu = 105.6583755 MeV
        assert abs(pc.m_muon / pc.MeV - 105.6583755) < 1e-2

    def test_muon_heavier_than_electron(self):
        assert pc.m_muon > pc.m_electron

    def test_pi0_mass_less_than_pi_pm(self):
        assert pc.m_pi0 < pc.m_pi_pm

    def test_proton_mass_positive(self):
        assert pc.m_proton > 0

    def test_proton_heavier_than_electron(self):
        assert pc.m_proton > pc.m_electron

    def test_alpha_em_value(self):
        # fine structure constant ≈ 1/137
        assert abs(pc.alpha_em - 1 / 137.035999) < 1e-8

    def test_hbarc_positive(self):
        assert pc.hbarc > 0


class TestTargetInformation:
    REQUIRED_KEYS = {"Z_T", "A_T", "mT", "rho", "dEdx"}

    @pytest.mark.parametrize("material", ["graphite", "lead", "iron", "aluminum"])
    def test_has_required_keys(self, material):
        assert self.REQUIRED_KEYS.issubset(pc.target_information[material])

    def test_graphite_z_is_6(self):
        assert pc.target_information["graphite"]["Z_T"] == 6

    def test_lead_z_is_82(self):
        assert pc.target_information["lead"]["Z_T"] == 82

    @pytest.mark.parametrize("material", pc.target_information)
    def test_dedx_equals_twice_density(self, material):
        info = pc.target_information[material]
        assert np.isclose(info["dEdx"], 2.0 * info["rho"])
