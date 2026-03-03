import numpy as np
try:
    from PETITE.physical_constants import *
except:
    from physical_constants import *

def mcp_annihilation_fourvecs(p0, sampled_event, mV=0.0):
    """Reconstruct final SM/dark photon four vectors from 
    mc-sampled kinematic variables for mcp annihilation e+e- > chi chi-bar
    Args:
        p0: incoming positron Particle object
        sampled_event: list including cos(theta) of outgoing particle as zero'th element
        m_chi: mass of mcp being produced
    Returns:
        List of four four-vectors representing the two final state vectors: 
        two SM photons, or one SM photon and one dark photon
    """
    Ee = p0.get_pf()[0]
    ct = sampled_event[0]
    m_chi=mV #workaround to match referencing in bsm_shower.py

    s = 2*m_electron*(Ee+m_electron)
    EeCM = np.sqrt(s)/2.0

    Echi_CM = np.sqrt(s)/2.0
    pF = np.sqrt(Echi_CM**2 - m_chi**2)

    g0 = EeCM/m_electron
    b0 = 1.0/g0*np.sqrt(g0**2-1.0)

    ph = np.random.uniform(0.0, 2.0*np.pi)

    if ct < -1.0 or ct > 1.0:
        print("Error in MCP Annihiliation Calculation")
        print(Ee, m_electron, m_chi, ct)

    p14v = [g0*Echi_CM - b0*g0*pF*ct, -pF*np.sqrt(1-ct**2)*np.sin(ph), -pF*np.sqrt(1-ct**2)*np.cos(ph), b0*g0*Echi_CM - g0*pF*ct]
    p24v = [g0*Echi_CM + b0*g0*pF*ct, pF*np.sqrt(1-ct**2)*np.sin(ph), pF*np.sqrt(1-ct**2)*np.cos(ph), b0*g0*Echi_CM + g0*pF*ct]

    return [p14v, p24v]

