import numpy as np
try:
    from .physical_constants import *
    from .radiative_return import boost, invariant_mass
except:
    from physical_constants import *
    from radiative_return import boost, invariant_mass

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
    pe4v = [g0*Ee + b0*g0*pF*ct, -pF*np.sqrt(1-ct**2)*np.sin(ph), -pF*np.sqrt(1-ct**2)*np.cos(ph), b0*g0*Ee+g0*pF*ct]
    pV4v = [g0*EV - b0*g0*pF*ct, pF*np.sqrt(1-ct**2)*np.sin(ph), pF*np.sqrt(1-ct**2)*np.cos(ph), b0*g0*EV - g0*pF*ct]

    return [pe4v, pV4v]

def ee_to_ee_fourvecs(Einc, me, ct):
    """Reconstruct final electron and electron (positron) four vectors from 
    mc-sampled kinematic variables for SM Moller/Bhabha  e e > e e 
    Args:
        Eg, me, ct
    Returns:
        List of four four-vectors representing the final state electron and 
        positron/electron
    """
    s = 2*me**2 + 2*Einc*me
    Ee0 = np.sqrt(s)/2.0
    pF = np.sqrt(Ee0**2 - me**2)

    g0 = Ee0/me
    b0 = 1.0/g0*np.sqrt(g0**2 - 1.0)

    ph = np.random.uniform(0, 2.0*np.pi)
    outgoing_particle_fourvector = [g0*Ee0 + b0*g0*pF*ct, -pF*np.sqrt(1-ct**2)*np.sin(ph), -pF*np.sqrt(1-ct**2)*np.cos(ph), b0*g0*Ee0+g0*pF*ct]
    new_electron_fourvector = [g0*Ee0 - b0*g0*pF*ct, pF*np.sqrt(1-ct**2)*np.sin(ph), pF*np.sqrt(1-ct**2)*np.cos(ph), b0*g0*Ee0 - g0*pF*ct]

    return [outgoing_particle_fourvector, new_electron_fourvector]

def radiative_return_fourvecs(pe, mV, x1):
    """
    Reconstruct V four-momentum in the radiative return process e^+ e^- > gamma V working in the 
    collinear emission approximation for the ISR photons. 
    Args:
        pe : four momentum of the incoming positron in the lab frame
        mV : vector mass
        x1 : energy fraction of electron or positron after it radiated (the energy fraction of the other particle is mV^2/(x s) ) 
    Returns:
        pV : four momentum
    """
    ml = m_electron 
    s = 2.*ml*(ml + pe[0])

    pCM_in_lab = pe + np.array([ml,0.,0.,0.]) 


    
    x2 = mV**2/(x1*s)
    
    if x2 > 1.:
        print("you're bad and you should feel bad")

    print("x1, x2, x1*x2*s,  mV^2 = ", x1, "\t", x2,"\t",x1*x2*s, "\t", mV**2)

    E1 = x1*np.sqrt(s)/2.
    E2 = x2*np.sqrt(s)/2.
    
    # CM four-momenta after the beam particles have radiated to bring the parton interaction energy on resonance
    p1 = np.array([E1, 0., 0., E1])
    #p1 = np.array([E1, 0., 0., np.sqrt(E1**2 - ml**2)])
    p2 = np.array([E2, 0., 0., -E2])
    #p2 = np.array([E2, 0., 0., -np.sqrt(E2**2 - ml**2)])
    pV = p1 + p2
    # we boost to the "lab frame", which is the frame where the electron (before radiation) is at rest
    pV_lab = boost(np.array([np.sqrt(s)/2., 0.,0., -np.sqrt(s/4. - ml**2)]), pV) 
    pV3_lab_rotated = np.linalg.norm(pV_lab[1:])*pCM_in_lab[1:]/np.linalg.norm(pCM_in_lab[1:])
    pV_lab_rotated = np.array([pV_lab[0], pV3_lab_rotated[0],pV3_lab_rotated[1],pV3_lab_rotated[2]]) 
    return(pV_lab_rotated)

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

