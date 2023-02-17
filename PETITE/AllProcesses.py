import numpy as np
import vegas as vg
import functools
import random as rnd
from physical_constants import *

#--------------------------------------------------------------------------
#Functions for atomic form factors for incident photons/electrons/positrons
#--------------------------------------------------------------------------
def Unity(Z, me, t):
    return(1.0)

def dummy(x,y,z):
    return(0)

def PPQSq(xx, me, w):
    """Computes momentum transfer squared for photon-scattering pair production
    Args:
        xx: tuple consisting of kinematic rescaled kinematic variables 
            epsilon_plus, delta_plus, delta_minus, phi (see ... for definitions) 
        me: electron mass 
        w:  frequency of emitted virtual photon, equal to sum of energies of outgoing e+- pair
    Returns:
        nuclear momentum transfer squared
    """
    epp, dp, dm, ph = xx
    epm = w - epp
    return me**2*((dp**2 + dm**2 + 2.0*dp*dm*np.cos(ph)) + me**2*((1.0 + dp**2)/(2.0*epp) + (1.0+dm**2)/(2.0*epm))**2) 

def BremQSq(xx, me, ep):
    """Momentum Transfer Squared for electron/positron bremsstrahlung
    Args:
        w: frequency of radiated photon 
        d: rescaled emission angle, equal to theta * epsilon/m, 
           where epsilon energy of incoming lepton
        dp: rescaled emission angle, equal to theta * epsilon'/m, 
           where epsilon' energy of outgoing lepton
        ph: angle phi
        me: electron mass
        ep: epsilon', e+/e- energy after radiation
    Returns:
        nuclear momentum transfer squared
    """
    w, d, dp, ph = xx
    epp = ep - w
    return me**2*((d**2 + dp**2 - 2*d*dp*np.cos(ph)) + me**2*((1 + d**2)/(2*ep) - (1 + dp**2)/(2*epp))**2)

def aa(Z, me):
    return 184.15*(2.718)**-0.5*Z**(-1./3.)/me
def aap(Z, me):
    return 1194*(2.718)**-0.5*Z**(-2./3.)/me

def G2el(Z, me, t):
    a0 = aa(Z, me)
    return Z**2*a0**4*t**2/(1 + a0**2*t)**2

def G2inel(Z, me, t):
    ap0 = aap(Z, me)
    return Z*ap0**4*t**2/(1 + ap0**2*t)**2

#--------------------------------------------------------------------
#Differential Cross Sections for Incident Electrons/Positrons/Photons
#--------------------------------------------------------------------

def dSBrem_dP_T(EvtInfo, varthTildeV):
    """Standard Model Bremsstrahlung in the Small-Angle Approximation
       e (ep) + Z -> e (epp) + gamma (w) + Z
       Outgoing kinematics given by w, d (delta), dp (delta'), and ph (phi)

       Input parameters needed:
            ep (incident electron energy)
            me (electron mass)
            Z (Target Atomic Number)
            al (electro-weak fine-structure constant)
    """
    ep=EvtInfo['E_inc']
    Z =EvtInfo['Z_T']
    mV=0

    if len(np.shape(varthTildeV)) == 1:
        varthTildeV = np.array([varthTildeV])
    dSigs = []
    for varthTilde in varthTildeV:
        w, d, dp, ph = varthTilde

        epp = ep - w

        qsqT = m_electron**2*((d**2 + dp**2 - 2*d*dp*np.cos(ph)) + m_electron**2*((1 + d**2)/(2*ep) - (1 + dp**2)/(2*epp))**2)
        PF = 8.0/np.pi*Z**2*alpha_em*(alpha_em/m_electron)**2*(epp*m_electron**4)/(w*ep*qsqT**2)*d*dp

        T1 = d**2/(1 + d**2)**2
        T2 = dp**2/(1 + dp**2)**2
        T3 = w**2/(2*ep*epp)*(d**2 + dp**2)/((1 + d**2)*(1 + dp**2))
        T4 = -(epp/ep + ep/epp)*(d*dp*np.cos(ph))/((1 + d**2)*(1 + dp**2))
        dSig0 = PF*(T1+T2+T3+T4)

        if dSig0 < 0.0 or np.isnan(dSig0):
            print([dSig0, PF, T1, T2, T3, T4, qsqT])
        dSigs.append(dSig0)

    if len(dSigs) == 1:
        return dSigs[0]
    else:
        return dSigs

def dSDBrem_dP_T(EvtInfo, varthV):
    """Dark Vector Bremsstrahlung in the Small-Angle Approximation
       e (ep) + Z -> e (epp) + V (w) + Z
       Outgoing kinematics given by w, d (delta), dp (delta'), and ph (phi)

       Input parameters needed:
            ep (incident electron energy)
            me (electron mass)
            MV (Dark Vector Mass)
            Z (Target Atomic Number)
            al (electro-weak fine-structure constant)
    """
    ep=EvtInfo['E_inc']
    Z =EvtInfo['Z_T']
    mV=EvtInfo['m_V']

    if len(np.shape(varthV)) == 1:
        varthV = np.array([varthV])
    dSigs = []
    for varth in varthV:
        w, d, dp, ph = varth

        epp = ep - w

        xsq = (d**2 + dp**2 - 2.0*d*dp*np.cos(ph)) + mV**2/(4.0*ep*epp)*(1 + 0.5*(d**2 + dp**2))**2
        PF = 4.0*alpha_em**3*Z**2/(np.pi*mV**2)*w**3/ep**3*1/(xsq**2*epp)*(d*dp)/((1+d**2)*(1+dp**2))
        T1 = (ep**2+epp**2)/w**2*xsq
        T2 = -(d**2 - dp**2)**2/((1+d**2)*(1+dp**2))
        T3 = -mV**2/(4.0*w**2)*(ep/epp*(1+dp**2) + epp/ep*(1+d**2))
        dSig0 = PF*(T1+T2+T3)
        if dSig0 < 0.0 or np.isnan(dSig0):
            print(dSig0, varth, ep, epp, PF, T1, T2, T3)
        dSigs.append(dSig0)
    if len(dSigs) == 1:
        return dSigs[0]
    else:
        return dSigs

def dAnn_dCT(EvtInfo, varthV):
    """Annihilation of a Positron and Electron into a Photon and a (Dark) Photon
       e+ (Ee) + e- (me) -> gamma + gamma/V

       Input parameters needed:
            Ee (incident positron energy)
            me (electron mass)
            mV (Dark Vector Mass -- can be set to zero for SM Case)
            al (electro-weak fine-structure constant)
    """
    Ee=EvtInfo['E_inc']
    mV=EvtInfo['m_V']

    if len(np.shape(varthV)) == 1:
        varthV = np.array([varthV])
    dSigs = []
    for varth in varthV:
        ct = varth[0]

        if Ee < (mV**2-2*m_electron**2)/(2*m_electron):
            return 0.0

        s = 2.0*m_electron*(Ee + m_electron)
        b = np.sqrt(1.0 - 4.0*m_electron**2/s)
        dSigs.append(4.0*np.pi*alpha_em**2/(s*(1 - b**2*ct**2))*((s-mV**2)/(2*s)*(1+ct**2) + 2.0*mV**2/(s-mV**2)))
    if len(dSigs) == 1:
        return dSigs[0]
    else:
        return dSigs

def dSPairProd_dP_T(EvtInfo, varthTildeV):
    """Standard Model Pair Production in the Small-Angle Approximation
       gamma (w) + Z -> e+ (epp) + e- (epm) + Z
       Outgoing kinematics given by epp, dp (delta+), dm (delta-), and ph (phi)

       Input parameters needed:
            w (incident photon energy)
            me (electron mass)
            Z (Target Atomic Number)
            al (electro-weak fine-structure constant)
    """
    w=EvtInfo['E_inc']
    Z =EvtInfo['Z_T']
    mV=0
    
    if len(np.shape(varthTildeV)) == 1:
        varthTildeV = np.array([varthTildeV])
    dSigs = []
    for varthTilde in varthTildeV:
        epp, dp, dm, ph = varthTilde

        epm = w - epp

        qsqT = (dp**2 + dm**2 + 2.0*dp*dm*np.cos(ph)) + m_electron**2*((1.0 + dp**2)/(2.0*epp) + (1.0+dm**2)/(2.0*epm))**2
        PF = 8.0/np.pi*Z**2*alpha_em*(alpha_em/m_electron)**2*epp*epm/(w**3*qsqT**2)*dp*dm
        
        T1 = -1.0*dp**2/(1.0 + dp**2)**2
        T2 = -1.0*dm**2/(1.0 + dm**2)**2
        T3 = w**2/(2.0*epp*epm)*(dp**2 + dm**2)/((1.0 + dp**2)*(1.0 + dm**2))
        T4 = (epp/epm + epm/epp)*(dp*dm*np.cos(ph))/((1.0 + dp**2)*(1.0+dm**2))

        dSig0 = PF*(T1+T2+T3+T4)

        if dSig0 < 0.0 or np.isnan(dSig0):
            print([dSig0, PF, T1, T2, T3, T4, qsqT])
        dSigs.append(dSig0)
    if len(dSigs) == 1:
        return dSigs[0]
    else:
        return dSigs

def dSCompton_dCT(EvtInfo, varthV):
    """Compton Scattering of a Photon off an at-rest Electron, producing either a photon or a Dark Vector
        gamma (Eg) + e- (me) -> e- + gamma/V

       Input parameters needed:
            Eg (incident photon energy)
            me (electron mass)
            MV (Dark Vector Mass -- can be set to zero for SM Case)
            al (electro-weak fine-structure constant)
    """
    Eg=EvtInfo['E_inc']
    Z =EvtInfo['Z_T']
    mV=EvtInfo['m_V']

    if len(np.shape(varthV)) == 1:
        varthV = np.array([varthV])
    dSigs = []
    for varth in varthV:
        ct = varth[0]

        s = m_electron**2 + 2*Eg*m_electron
        JacFac = 0.5*(s**2-m_electron**4)/s
        t = -1/2*(m_electron**4 + s*(-mV**2 + s + ct*np.sqrt(m_electron**4 + (mV**2 - s)**2 - 2*m_electron**2*(mV**2 + s))) - m_electron**2*(mV**2 + 2*s + ct*np.sqrt(m_electron**4 + (mV**2 - s)**2 - 2*m_electron**2*(mV**2 + s))))/s
        PF = 2.0*np.pi*alpha_em**2/(s-m_electron**2)**2

        if mV == 0.:
            T1 = (6.0*m_electron**2*s + 3.0*m_electron**4 - s**2)/((m_electron**2-s)*(-m_electron**2+s+t))
            T2 = 4*m_electron**4/(s+t-m_electron**2)**2
            T3 = (t*(s-m_electron**2) + (s+m_electron**2)**2)/(s-m_electron**2)**2
        else:
            T1 = (2.0*m_electron**2*(mV**2-3*s)-3*m_electron**4-2*mV**2*s+2*mV**4+s**2)/((m_electron**2-s)*(m_electron**2+mV**2-s-t))
            T2 = (2*m_electron**2*(2*m_electron**2+mV**2))/(m_electron**2+mV**2-s-t)**2
            T3 = ((m_electron**2+s)*(m_electron**2+mV**2+s)+t*(s-m_electron**2))/(m_electron**2-s)**2

        dSig0 = PF*JacFac*(T1+T2+T3)
        if np.isnan(dSig0):
            print(dSig0, PF, JacFac, T1, T2, T3, ct, s, t, varth)
        dSigs.append(dSig0)

    if len(dSigs) == 1:
        return dSigs[0]
    else:
        return dSigs

#Function for drawing unweighted events from a weighted distribution
def GetPts(Dist, npts):
    """If weights are too cumbersome, this function returns a properly-weighted sample from Dist"""
    ret = []
    MW = np.max(np.transpose(Dist)[-1])

    tochoosefrom = [pis for pis in range(len(Dist))]
    choicesgetter = rnd.choices(tochoosefrom, np.transpose(Dist)[-1], k=npts)
    for cg in choicesgetter:
        ret.append(Dist[cg][0:-1])

    return ret

#------------------------------------------------------------------------------
#Total Cross Sections and Sample Draws for Incident Electrons/Positrons/Photons
#------------------------------------------------------------------------------
NPts = 10000 #Default number of points to draw for unweighted samples

diff_xsection_options={"PairProd" : dSPairProd_dP_T,
                       "Comp"     : dSCompton_dCT,
                       "Brem"     : dSBrem_dP_T,
                       "Ann"      : dAnn_dCT }
nitn_options={"PairProd":500,
              "Brem":500,
              "DarkBrem":500,
              "Comp":5000,
              "Ann":5000}
nstrat_options={"PairProd":[15, 25, 25, 15],
                "Brem":[15, 25, 25, 15],
                "DarkBrem":[15, 25, 25, 15],
                "Comp":[300],
                "Ann":[300]}

FourD = {"PairProd", "Brem", "DarkBrem"}
TwoD = {"Comp", "Ann"}
def IGRange(EI, Process):
    EInc=EI['E_inc']
    Egmin=EI['Eg_min']
    mV=EI['m_V']
    if Process in FourD:
        if Process == "PairProd":
            minE = Egmin
            maxdel = np.sqrt(EInc/m_electron)
        else:
            minE = np.max([Egmin,mV])
            maxdel = np.sqrt(EInc/np.max([m_electron,mV]))
        return [[minE, EInc-m_electron], [0., maxdel], [0., maxdel], [0., 2*np.pi]]
    elif Process in TwoD:
        if Process == "Comp":
            return [[-1., 1.0]]
        elif Process == "Ann":
            EVMin = Egmin + mV
            ctmaxV = np.sqrt(2.0)*(2.0*m_electron*(EInc-EVMin)*(EInc+m_electron)+EInc*mV**2)/(np.sqrt((EInc-m_electron)*(2.0*EInc+m_electron))*(2*m_electron*(EInc+m_electron)-mV**2))
            ctMaxmV = np.sqrt(2.0)*(EInc*mV**2 - 2.0*m_electron*(EInc - Egmin)*(EInc+m_electron))/(np.sqrt((EInc-m_electron)*(2*EInc+m_electron))*(2*m_electron*(EInc+m_electron)-mV**2))
            if ctmaxV < 0.0 or ctMaxmV > 0.0:
                ctMax = 0.0
            else:
                ctMax = np.min([np.abs(ctmaxV), np.abs(ctMaxmV)])
            return [[-ctMax, ctMax]]
    else:
        raise Exception("Your process is not in the list")

def VEGASIntegration(EI, Process, VB=False, mode='XSec'):
    """Function for Integration of Various SM/BSM Differential
       Cross Sections.

       Available Processes ('Process'):
        -- 'Brem': Standard Model e + Z -> e + gamma + Z
        -- 'DarkBrem': BSM e + Z -> e + V + Z
        -- 'PairProd': SM gamma + Z -> e^+ + e^- + Z
        -- 'Comp': SM/BSM gamma + e -> e + gamma/V
        -- 'Ann': SM/BSM e^+ e^- -> gamma + gamma/V

        ('Brem', 'DarkBrem', 'PairProd' calculated in 
         small-angle approximation)

       Input parameters needed:
            EI: dictionary containing
              -- 'E_inc' (incident electron/positron energy)
              -- 'm_e' (electron mass)
              -- 'Z_T' (Target Atomic Number)
              -- 'alpha_FS' (electro-weak fine-structure constant)
              -- 'm_V' (Dark vector mass, assumed to be zero if absent)
              -- 'Eg_min': minimum lab-frame energy (GeV) of outgoing photons

        Optional arguments:
            VB: verbose flag for printing some status updates
            mode: Options for how to integrate/return information:
             -- 'XSec': return total integrated cross section (default)
             -- 'Pickle': return VEGAS integrator object
             -- 'Sample': return VEGAS sample (including weights)
             -- 'UnweightedSample' return unweighted sample of events
    """
    if Process in diff_xsection_options:
        if ('m_V' in EI.keys()) == False:
            EI['m_V'] = 0.0
        if ('Eg_min' in EI.keys()) == False:
            EI['Eg_min'] = 0.0
        igrange = IGRange(EI, Process)
        diff_xsec_func = diff_xsection_options[Process] 
    else:
        raise Exception("You process is not in the list")
    integrand = vg.Integrator(igrange)
    if mode == 'Pickle' or mode == 'XSec':
        if VB:
            print("Integrator set up", Process, EI)
        integrand(functools.partial(diff_xsec_func, EI), nitn=nitn_options[Process], nstrat=nstrat_options[Process])
        if VB:
            print("Burn-in complete", EI)
        result = integrand(functools.partial(diff_xsec_func, EI), nitn=nitn_options[Process], nstrat=nstrat_options[Process])
        if VB:
            print("Fully Integrated", EI, result.mean)
        if mode == 'Pickle':
            return integrand
        else:
            return result.mean
    elif mode == 'Sample' or mode == 'UnweightedSample':
        integrand(functools.partial(diff_xsec_func, EI), nitn=nitn_options[Process], nstrat=nstrat_options[Process])
        result = integrand(functools.partial(diff_xsec_func, EI), nitn=nitn_options[Process], nstrat=nstrat_options[Process])

        integral, pts = 0.0, []
        for x, wgt in integrand.random_batch():
            integral += wgt.dot(diff_xsec_func(EI, x))
        if VB:
            print(integral)
        NSamp = 1
        for kc in range(NSamp):
            for x, wgt in integrand.random():
                M0 = wgt*diff_xsec_func(EI, x)
                pts.append(np.concatenate([list(x), [M0]]))
        if mode == 'Sample':
            tr = np.array([integral, pts], dtype=object)
        elif mode == 'UnweightedSample':
            tr = np.array([integral, GetPts(pts, NPts)], dtype=object)
        return tr
        