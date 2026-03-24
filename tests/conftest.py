"""
Shared fixtures and helpers for the PETITE test suite.
"""
import os
import numpy as np
import pytest

from PETITE.particle import Particle
from PETITE.physical_constants import m_electron, m_muon, m_pi0, m_pi_pm

# ---------------------------------------------------------------------------
# Data directory
# ---------------------------------------------------------------------------
DATA_DIR = os.path.join(os.path.dirname(__file__), "..", "data_Te") + os.sep
_DATA_AVAILABLE = os.path.isdir(DATA_DIR) and os.path.isfile(DATA_DIR + "sm_maps.pkl")

requires_data = pytest.mark.skipif(
    not _DATA_AVAILABLE,
    reason="Pre-computed data files not found in data/",
)

# ---------------------------------------------------------------------------
# Global random seed fixture (makes stochastic tests reproducible)
# ---------------------------------------------------------------------------
@pytest.fixture(autouse=True)
def fix_random_seed():
    np.random.seed(42)


# ---------------------------------------------------------------------------
# Particle fixtures
# ---------------------------------------------------------------------------
@pytest.fixture
def electron_at_rest():
    return Particle([m_electron, 0.0, 0.0, 0.0], [0.0, 0.0, 0.0],
                    {"PID": 11, "mass": m_electron})

@pytest.fixture
def electron_1gev():
    pz = np.sqrt(1.0**2 - m_electron**2)
    return Particle([1.0, 0.0, 0.0, pz], [0.0, 0.0, 0.0],
                    {"PID": 11, "mass": m_electron})

@pytest.fixture
def electron_10gev():
    pz = np.sqrt(10.0**2 - m_electron**2)
    return Particle([10.0, 0.0, 0.0, pz], [0.0, 0.0, 0.0],
                    {"PID": 11, "mass": m_electron})

@pytest.fixture
def positron_1gev():
    pz = np.sqrt(1.0**2 - m_electron**2)
    return Particle([1.0, 0.0, 0.0, pz], [0.0, 0.0, 0.0],
                    {"PID": -11, "mass": m_electron})

@pytest.fixture
def photon_1gev():
    return Particle([1.0, 0.0, 0.0, 1.0], [0.0, 0.0, 0.0],
                    {"PID": 22, "mass": 0.0})

@pytest.fixture
def photon_10gev():
    return Particle([10.0, 0.0, 0.0, 10.0], [0.0, 0.0, 0.0],
                    {"PID": 22, "mass": 0.0})

@pytest.fixture
def muon_5gev():
    pz = np.sqrt(5.0**2 - m_muon**2)
    return Particle([5.0, 0.0, 0.0, pz], [0.0, 0.0, 0.0],
                    {"PID": 13, "mass": m_muon})

@pytest.fixture
def pi0_at_rest():
    return Particle([m_pi0, 0.0, 0.0, 0.0], [0.0, 0.0, 0.0],
                    {"PID": 111, "mass": m_pi0, "stability": "short-lived"})

@pytest.fixture
def pi0_boosted():
    pz = np.sqrt(1.0**2 - m_pi0**2)
    return Particle([1.0, 0.0, 0.0, pz], [0.0, 0.0, 0.0],
                    {"PID": 111, "mass": m_pi0, "stability": "short-lived"})

# ---------------------------------------------------------------------------
# Session-scoped shower fixtures (slow to build — load only once)
# ---------------------------------------------------------------------------
@pytest.fixture(scope="session")
def graphite_shower():
    if not _DATA_AVAILABLE:
        pytest.skip("data/ not available")
    from PETITE.shower import Shower
    return Shower(DATA_DIR, "graphite", min_kinetic_energy=0.050, seed=42)

@pytest.fixture(scope="session")
def dark_vector_shower():
    if not _DATA_AVAILABLE:
        pytest.skip("data/ not available")
    from PETITE.bsm_shower import BSMShower
    return BSMShower(
        DATA_DIR, "graphite", min_energy=0.050,
        model_name="dark_vector",
        model_keys={"mV": 0.050},
    )

@pytest.fixture(scope="session")
def mcp_shower():
    if not _DATA_AVAILABLE:
        pytest.skip("data/ not available")
    from PETITE.bsm_shower import BSMShower
    return BSMShower(
        DATA_DIR, "graphite", min_energy=0.030,
        model_name="mcp",
        model_keys={"m_chi": 0.010},
        active_processes=["mcpAnnihilation"],
    )

# ---------------------------------------------------------------------------
# Physics helpers
# ---------------------------------------------------------------------------
def minkowski_sq(p4):
    """p^mu p_mu in (+---) metric."""
    E, px, py, pz = p4
    return E**2 - px**2 - py**2 - pz**2
