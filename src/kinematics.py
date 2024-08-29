import numpy as np
try:
    from .physical_constants import *
    from .radiative_return import boost, invariant_mass
except:
    from physical_constants import *
    from radiative_return import boost, invariant_mass

Egamma_min = 0.001
def e_to_egamma_fourvecs(p0, sampled_event):
    """Reconstruct electron and photon four vectors from 
    mc-sampled kinematic variables for electron/positron 
    SM brem e N -> e N gamma
    Args:
        p0: incoming electron/positron Particle object
        sampled_event: MC event sample of outgoing kinematics
    Returns:
        List of four four-vectors representing final electron and photon 
        momenta in that order
    """
    ep = p0.get_pf()[0]
    x1, x2, x3, x4 = sampled_event[:4]
    w = Egamma_min + x1*(ep - m_electron - Egamma_min)
    ct = np.cos((x2+x3)/2)
    ctp = np.cos((x2-x3)*ep/(2*(ep-w)))
    ph = (x4-1/2)*2.0*np.pi

    epp = ep - w
    p, pp = np.sqrt(ep**2 - m_electron**2), np.sqrt(epp**2 - m_electron**2)

    Em4v = [ep, 0, 0, p] #Four-vector of electron
    al = np.random.uniform(0, 2.0*np.pi)
    cal, sal = np.cos(al), np.sin(al)
    st, stp = np.sqrt(1.0 - ct**2), np.sqrt(1.0 - ctp**2)
    sp, cp = np.sin(ph), np.cos(ph)
    g4v = [w, w*cal*st, w*sal*st, w*ct] #Four-vector of photon

    Ep4v = [epp, pp*(sal*sp*stp + cal*(ctp*st - cp*ct*stp)), pp*(ctp*sal*st - (cp*ct*sal + cal*sp)*stp), pp*(ct*ctp + cp*st*stp)] #Four-vector of positron

    return [Ep4v, g4v]

def e_to_eV_fourvecs(p0, sampled_event, mV=0.0):
    """Reconstruct electron and photon four vectors from 
    mc-sampled kinematic variables for electron/positron 
    dark sector brem e N -> e N V
    Args:
        p0: incoming electron/positron Particle object
        sampled_event: MC event sample of outgoing kinematics
        optional mV: dark vector mass
    Returns:
        List of four four-vectors representing final electron
        and dark vecotor in that order
    """

    ep = p0.get_pf()[0]
    w = sampled_event[0]*ep
    ct = (1 - 10**sampled_event[1])
    p, k = np.sqrt(ep**2 - m_electron**2), np.sqrt(w**2 - mV**2)

    Em4v = [ep, 0, 0, p] #Four-vector of electron
    al = np.random.uniform(0, 2.0*np.pi)
    cal, sal = np.cos(al), np.sin(al)
    st = np.sqrt(1.0 - ct**2)
    V4v = [w, k*cal*st, k*sal*st, k*ct] #Four-vector of photon

    return [Em4v, V4v]

def gamma_to_epem_fourvecs(p0, sampled_event):
    """Reconstruct photon, electron and positron four vectors from 
    mc-sampled kinematic variables for pair production 
    gamma Z -> e- e+ Z
    Args:
        p0: incoming photon Particle object
        sampled_event: MC event sample of outgoing kinematics
    Returns:
        List of four four-vectors representing the outgoing positron and electron
        momenta in that order
    """

    w = p0.get_pf()[0]
    x1, x2, x3, x4 = sampled_event[:4]
    epp = m_electron + x1*(w-2*m_electron)
    ctp = np.cos(w*(x2+x3)/(2*epp))
    ctm = np.cos(w*(x2-x3)/(2*(w-epp)))
    ph = x4*2*np.pi

    epm = w - epp
    pm, pp = np.sqrt(epm**2 - m_electron**2), np.sqrt(epp**2 - m_electron**2)

    Eg4v = [w, 0, 0, w]
    al = np.random.uniform(0, 2.0*np.pi)

    cal, sal = np.cos(al), np.sin(al)
    stp, stm = np.sqrt(1.0 - ctp**2), np.sqrt(1.0 - ctm**2)
    spal, cpal = np.sin(ph+al), np.cos(ph+al)

    pp4v = [epp, pp*stp*cal, pp*stp*sal, pp*ctp]
    pm4v = [epm, pm*stm*cpal, pm*stm*spal, pm*ctm]

    return [pp4v, pm4v]
    
def compton_fourvecs(p0, sampled_event, mV=0.0):
    """Reconstruct final electron and photon four vectors from 
    mc-sampled kinematic variables for SM Compton  gamma e > gamma e 
    or dark Compoton gamma e > V e
    Args:
        p0: incoming photon Particle object
        sampled_event: list including cos(theta) of outgoing particle as zero'th element
        optional mV: mass of outgoing dark vector
    Returns:
        List of four four-vectors representing the final state electron and 
        vector (SM or dark photon)
    """
    Eg = p0.get_pf()[0]
    ct = sampled_event[0]

    s = m_electron**2 + 2*Eg*m_electron
    Ee0 = (s + m_electron**2)/(2.0*np.sqrt(s))
    Ee = (s - mV**2 + m_electron**2)/(2*np.sqrt(s))
    EV = (s + mV**2 - m_electron**2)/(2*np.sqrt(s))
    pF = np.sqrt(Ee**2 - m_electron**2)

    g0 = Ee0/m_electron
    b0 = 1.0/g0*np.sqrt(g0**2 - 1.0)

    ph = np.random.uniform(0, 2.0*np.pi)
    pe4v = [g0*Ee + b0*g0*pF*ct, -pF*np.sqrt(1-ct**2)*np.sin(ph), -pF*np.sqrt(1-ct**2)*np.cos(ph), b0*g0*Ee+g0*pF*ct]
    pV4v = [g0*EV - b0*g0*pF*ct, pF*np.sqrt(1-ct**2)*np.sin(ph), pF*np.sqrt(1-ct**2)*np.cos(ph), b0*g0*EV - g0*pF*ct]

    return [pe4v, pV4v]

def ee_to_ee_fourvecs(p0, sampled_event):
    """Reconstruct final electron and electron (positron) four vectors from 
    mc-sampled kinematic variables for SM Moller/Bhabha  e e > e e 
    Args:
        p0: incoming electron/positron Particle object
        sampled_event: list including cos(theta) of outpoing particle as zero'th element
    Returns:
        List of four four-vectors representing the final state electron and 
        positron/electron
    """
    Einc = p0.get_pf()[0]
    ct = sampled_event[0]

    s = 2*m_electron**2 + 2*Einc*m_electron
    Ee0 = np.sqrt(s)/2.0
    pF = np.sqrt(Ee0**2 - m_electron**2)

    g0 = Ee0/m_electron
    b0 = 1.0/g0*np.sqrt(g0**2 - 1.0)

    ph = np.random.uniform(0, 2.0*np.pi)
    outgoing_particle_fourvector = [g0*Ee0 + b0*g0*pF*ct, -pF*np.sqrt(1-ct**2)*np.sin(ph), -pF*np.sqrt(1-ct**2)*np.cos(ph), b0*g0*Ee0+g0*pF*ct]
    new_electron_fourvector = [g0*Ee0 - b0*g0*pF*ct, pF*np.sqrt(1-ct**2)*np.sin(ph), pF*np.sqrt(1-ct**2)*np.cos(ph), b0*g0*Ee0 - g0*pF*ct]

    return [outgoing_particle_fourvector, new_electron_fourvector]

def radiative_return_fourvecs(pe, sampled_event, mV=0.0):
    """
    Reconstruct V four-momentum in the radiative return process e^+ e^- > gamma V working in the 
    collinear emission approximation for the ISR photons. 
    Args:
        pe : Particle() of the incoming positron in the lab frame
        mV : vector mass
        x1 : energy fraction of electron or positron after it radiated (the energy fraction of the other particle is mV^2/(x s) ) 
    Returns:
        pV : four momentum
    """
    s = 2.*m_electron*(m_electron + pe.get_pf()[0])

    beta = (2.*alpha_em/np.pi) * (np.log(s/m_electron**2) - 1.)
    umax = np.power(1.-mV**2/s,beta/2.)

    x1 = 1.- np.power(sampled_event[0]*umax,2./beta)
    x2 = mV**2/(x1*s)
    
    if x2 >= 1.:
        print("wrong kinematics...")
        print("x1, x2, x1*x2*s,  mV^2 = ", x1, "\t", x2,"\t",x1*x2*s, "\t", mV**2)

    E1 = x1*np.sqrt(s)/2.
    E2 = x2*np.sqrt(s)/2.
    
    # CM four-momenta after the beam particles have radiated to bring the parton interaction energy on resonance
    p1 = np.array([E1, 0., 0., E1])
    p2 = np.array([E2, 0., 0., -E2])
    pV = p1 + p2
    # we boost to the "lab frame", which is the frame where the electron (before radiation) is at rest
    pV_lab = boost(np.array([np.sqrt(s)/2., 0.,0., -np.sqrt(s/4. - m_electron**2)]), pV) 
    return(pV_lab, pV_lab)#returning two four-vectors just for proper handling in dark_shower.py

def annihilation_fourvecs(p0, sampled_event, mV=0.0):
    """Reconstruct final SM/dark photon four vectors from 
    mc-sampled kinematic variables for SM annihilation e+e- > gamma gamma
    or dark annihilation e+e- > gamma V
    Args:
        p0: incoming positron Particle object
        sampled_event: list including cos(theta) of outgiong particle as zero'th element
        optional mV: mass of dark vector being produced
    Returns:
        List of four four-vectors representing the two final state vectors: 
        two SM photons, or one SM photon and one dark photon
    """
    Ee = p0.get_pf()[0]
    ct = sampled_event[0]

    s = 2*m_electron*(Ee+m_electron)
    EeCM = np.sqrt(s)/2.0
    Eg = (s - mV**2)/(2*np.sqrt(s))
    EV = (s + mV**2)/(2*np.sqrt(s))
    pF = Eg

    g0 = EeCM/m_electron
    b0 = 1.0/g0*np.sqrt(g0**2-1.0)

    ph = np.random.uniform(0.0, 2.0*np.pi)

    if ct < -1.0 or ct > 1.0:
        print("Error in Annihiliation Calculation")
        print(Ee, m_electron, mV, ct)

    pg4v = [g0*Eg - b0*g0*pF*ct, -pF*np.sqrt(1-ct**2)*np.sin(ph), -pF*np.sqrt(1-ct**2)*np.cos(ph), b0*g0*Eg - g0*pF*ct]
    pV4v = [g0*EV + b0*g0*pF*ct, pF*np.sqrt(1-ct**2)*np.sin(ph), pF*np.sqrt(1-ct**2)*np.cos(ph), b0*g0*EV + g0*pF*ct]

    return [pg4v, pV4v]

