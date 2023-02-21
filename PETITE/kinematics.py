import numpy as np
from .physical_constants import *

def e_to_egamma_fourvecs(ep, me, w, ct, ctp, ph):
    """Reconstruct electron and photon four vectors from 
    mc-sampled kinematic variables for electron/positron 
    SM brem e N -> e N gamma
    Args:
        ep: incoming electron energy
        me: electron mass
        w: energy of emitted photon 
        ct: 
        ctp: 
        ph: 
    Returns:
        List of four four-vectors representing initial electron, final electron and photon 
        momenta in that order
    """
    epp = ep - w
    p, pp = np.sqrt(ep**2 - me**2), np.sqrt(epp**2 - me**2)

    Em4v = [ep, 0, 0, p] #Four-vector of electron
    al = np.random.uniform(0, 2.0*np.pi)
    cal, sal = np.cos(al), np.sin(al)
    st, stp = np.sqrt(1.0 - ct**2), np.sqrt(1.0 - ctp**2)
    sp, cp = np.sin(ph), np.cos(ph)
    g4v = [w, w*cal*st, w*sal*st, w*ct] #Four-vector of photon

    Ep4v = [epp, pp*(sal*sp*stp + cal*(ctp*st - cp*ct*stp)), pp*(ctp*sal*st - (cp*ct*sal + cal*sp)*stp), pp*(ct*ctp + cp*st*stp)] #Four-vector of positron

    return [Em4v, Ep4v, g4v]

def e_to_eV_fourvecs(ep, me, w, MV, ct, ctp, ph):
    """Reconstruct electron and photon four vectors from 
    mc-sampled kinematic variables for electron/positron 
    dark sector brem e N -> e N V
    Args:
        ep, me, w, ct, ctp, ph 
    Returns:
        List of four four-vectors representing initial electron, final electron
        and dark vecotor in that order
    """

    epp = ep - w
    p, pp, k = np.sqrt(ep**2 - me**2), np.sqrt(epp**2 - me**2), np.sqrt(w**2 - MV**2)

    Em4v = [ep, 0, 0, p] #Four-vector of electron
    al = np.random.uniform(0, 2.0*np.pi)
    cal, sal = np.cos(al), np.sin(al)
    st, stp = np.sqrt(1.0 - ct**2), np.sqrt(1.0 - ctp**2)
    sp, cp = np.sin(ph), np.cos(ph)
    V4v = [w, k*cal*st, k*sal*st, k*ct] #Four-vector of photon

    Ep4v = [epp, pp*(sal*sp*stp + cal*(ctp*st - cp*ct*stp)), pp*(ctp*sal*st - (cp*ct*sal + cal*sp)*stp), pp*(ct*ctp + cp*st*stp)] #Four-vector of positron

    return [Em4v, Ep4v, V4v]

def gamma_to_epem_fourvecs(w, me, epp, ctp, ctm, ph):
    """Reconstruct photon, electron and positron four vectors from 
    mc-sampled kinematic variables for pair production 
    gamma Z -> e- e+ Z
    Args:
        w, me, epp, ctp, ctm, ph
    Returns:
        List of four four-vectors representing the initial photon, positron and electron
        momenta in that order
    """

    epm = w - epp
    pm, pp = np.sqrt(epm**2 - me**2), np.sqrt(epp**2 - me**2)

    Eg4v = [w, 0, 0, w]
    al = np.random.uniform(0, 2.0*np.pi)

    cal, sal = np.cos(al), np.sin(al)
    stp, stm = np.sqrt(1.0 - ctp**2), np.sqrt(1.0 - ctm**2)
    spal, cpal = np.sin(ph+al), np.cos(ph+al)

    pp4v = [epp, pp*stp*cal, pp*stp*sal, pp*ctp]
    pm4v = [epm, pm*stm*cpal, pm*stm*spal, pm*ctm]

    return [Eg4v, pp4v, pm4v]
    
def compton_fourvecs(Eg, me, mV, ct):
    """Reconstruct final electron and photon four vectors from 
    mc-sampled kinematic variables for SM Compton  gamma e > gamma e 
    or dark Compoton gamma e > V e
    Args:
        Eg, me, mV, ct
    Returns:
        List of four four-vectors representing the final state electron and 
        vector (SM or dark photon)
    """
    s = me**2 + 2*Eg*me
    Ee0 = (s + me**2)/(2.0*np.sqrt(s))
    Ee = (s - mV**2 + me**2)/(2*np.sqrt(s))
    EV = (s + mV**2 - me**2)/(2*np.sqrt(s))
    pF = np.sqrt(Ee**2 - me**2)

    g0 = Ee0/me
    b0 = 1.0/g0*np.sqrt(g0**2 - 1.0)

    ph = np.random.uniform(0, 2.0*np.pi)
    pe4v = [g0*Ee - b0*g0*pF*ct, -pF*np.sqrt(1-ct**2)*np.sin(ph), -pF*np.sqrt(1-ct**2)*np.cos(ph), b0*g0*Ee-g0*pF*ct]
    pV4v = [g0*EV + b0*g0*pF*ct, pF*np.sqrt(1-ct**2)*np.sin(ph), pF*np.sqrt(1-ct**2)*np.cos(ph), b0*g0*EV + g0*pF*ct]

    return [pe4v, pV4v]

def annihilation_fourvecs(Ee, me, mV, ct):
    """Reconstruct final SM/dark photon four vectors from 
    mc-sampled kinematic variables for SM annihilation e+e- > gamma gamma
    or dark annihilation e+e- > gamma V
    Args:
        Ee, me, mV, ct
    Returns:
        List of four four-vectors representing the two final state vectors: 
        two SM photons, or one SM photon and one dark photon
    """

    s = 2*me*(Ee+me)
    EeCM = np.sqrt(s)/2.0
    Eg = (s - mV**2)/(2*np.sqrt(s))
    EV = (s + mV**2)/(2*np.sqrt(s))
    pF = Eg

    g0 = EeCM/me
    b0 = 1.0/g0*np.sqrt(g0**2-1.0)

    ph = np.random.uniform(0.0, 2.0*np.pi)

    if ct < -1.0 or ct > 1.0:
        print("Error in Annihiliation Calculation")
        print(Ee, me, mV, ct)

    pg4v = [g0*Eg - b0*g0*pF*ct, -pF*np.sqrt(1-ct**2)*np.sin(ph), -pF*np.sqrt(1-ct**2)*np.cos(ph), b0*g0*Eg - g0*pF*ct]
    pV4v = [g0*EV + b0*g0*pF*ct, pF*np.sqrt(1-ct**2)*np.sin(ph), pF*np.sqrt(1-ct**2)*np.cos(ph), b0*g0*EV + g0*pF*ct]

    return [pg4v, pV4v]

