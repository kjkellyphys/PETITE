import numpy as np
import vegas as vg
import functools
import random as rnd
try:
    from .physical_constants import *
except:
    from physical_constants import *

#--------------------------------------------------------------------------
#Functions for atomic form factors for incident photons/electrons/positrons
#--------------------------------------------------------------------------
def unity(Z, me, t):
    return(1.0)

def dummy(x,y):
    return(0)

def pair_production_q_sq_dimensionless(xx, EI):
    """Computes momentum transfer squared for photon-scattering pair production
    Args:
        xx: tuple consisting of kinematic rescaled kinematic variables 
            epsilon_plus, delta_plus, delta_minus, phi (see ... for definitions) 
        me: electron mass 
        w:  frequency of emitted virtual photon, equal to sum of energies of outgoing e+- pair
    Returns:
        nuclear momentum transfer squared
    """
    x1, x2, x3, x4 = xx
    w = EI['E_inc']
    epp, dp, dm, ph = m_electron + x1*(w-2*m_electron), w/(2*m_electron)*(x2+x3), w/(2*m_electron)*(x2-x3), x4*2*np.pi

    epm = w - epp
    return m_electron**2*((dp**2 + dm**2 + 2.0*dp*dm*np.cos(ph)) + m_electron**2*((1.0 + dp**2)/(2.0*epp) + (1.0+dm**2)/(2.0*epm))**2) 

def brem_q_sq_dimensionless(xx, EI):
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
    x1, x2, x3, x4 = xx
    Egamma_min = EI['Eg_min']
    ep = EI['E_inc']
    w, d, dp, ph = Egamma_min + x1*(ep - m_electron - Egamma_min), ep/(2*m_electron)*(x2+x3), ep/(2*m_electron)*(x2-x3), x4*2*np.pi

    epp = ep - w
    return m_electron**2*((d**2 + dp**2 - 2*d*dp*np.cos(ph)) + m_electron**2*((1 + d**2)/(2*ep) - (1 + dp**2)/(2*epp))**2)

def aa(Z, me):
    return 184.15*(2.718)**-0.5*Z**(-1./3.)/me
def aap(Z, me):
    return 1194*(2.718)**-0.5*Z**(-2./3.)/me

def g2_elastic(Z, me, t):
    a0 = aa(Z, me)
    return Z**2*a0**4*t**2/(1 + a0**2*t)**2

def g2_inelastic(Z, me, t):
    ap0 = aap(Z, me)
    return Z*ap0**4*t**2/(1 + ap0**2*t)**2

#--------------------------------------------------------------------
#Differential Cross Sections for Incident Electrons/Positrons/Photons
#--------------------------------------------------------------------

def dsigma_brem_dimensionless(event_info, phase_space_par_list):
    """Standard Model Bremsstrahlung in the Small-Angle Approximation
       e (ep) + Z -> e (epp) + gamma (w) + Z
       Outgoing kinematics given by w, d (delta), dp (delta'), and ph (phi)

       Input parameters needed:
            ep (incident electron energy)
            Z (Target Atomic Number)
    """
    ep=event_info['E_inc']
    Z =event_info['Z_T']
    Egamma_min = event_info['Eg_min']
    mV=0

    if len(np.shape(phase_space_par_list)) == 1:
        phase_space_par_list = np.array([phase_space_par_list])
    dSigs = []
    for variables in phase_space_par_list:
        x1, x2, x3, x4 = variables
        w, d, dp, ph = Egamma_min + x1*(ep - m_electron - Egamma_min), ep/(2*m_electron)*(x2+x3), ep/(2*m_electron)*(x2-x3), x4*2*np.pi

        epp = ep - w

        if not((Egamma_min < w < ep - m_electron) and (m_electron < epp < ep) and (d > 0.) and (dp > 0.)):
            dSigs.append(0.0)
        else:
            qsqT = m_electron**2*((d**2 + dp**2 - 2*d*dp*np.cos(ph)) + m_electron**2*((1 + d**2)/(2*ep) - (1 + dp**2)/(2*epp))**2)
            PF = 8.0/np.pi*alpha_em*(alpha_em/m_electron)**2*(epp*m_electron**4)/(w*ep*qsqT**2)*d*dp
            jacobian_factor = np.pi*ep**2*(ep - m_electron - Egamma_min)/m_electron**2
            FF_hydrogen = g2_elastic(1.0, m_electron, qsqT)

            T1 = d**2/(1 + d**2)**2
            T2 = dp**2/(1 + dp**2)**2
            T3 = w**2/(2*ep*epp)*(d**2 + dp**2)/((1 + d**2)*(1 + dp**2))
            T4 = -(epp/ep + ep/epp)*(d*dp*np.cos(ph))/((1 + d**2)*(1 + dp**2))
            dSig0 = PF*(T1+T2+T3+T4)*jacobian_factor*FF_hydrogen

            if dSig0 < 0.0 or np.isnan(dSig0):
                print([dSig0, PF, T1, T2, T3, T4, qsqT, jacobian_factor, FF_hydrogen])
                print([x1,x2,x3,x4])
                print([w,d,dp,ph])
            dSigs.append(dSig0)

    if len(dSigs) == 1:
        return dSigs[0]
    else:
        return dSigs

def dsigma_darkbrem_dP_T(event_info, phase_space_par_list):
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
    ep=event_info['E_inc']
    Z =event_info['Z_T']
    mV=event_info['m_V']

    if len(np.shape(phase_space_par_list)) == 1:
        phase_space_par_list = np.array([phase_space_par_list])
    dSigs = []
    for varth in phase_space_par_list:
        w, d, dp, ph = varth

        epp = ep - w

        xsq = (d**2 + dp**2 - 2.0*d*dp*np.cos(ph)) + mV**2/(4.0*ep*epp)*(1 + 0.5*(d**2 + dp**2))**2
        PF = 1#+4.0*alpha_em**3*Z**2/(np.pi*mV**2)*w**3/ep**3*1/(xsq**2*epp)*(d*dp)/((1+d**2)*(1+dp**2)) #FIXME
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

def dsigma_annihilation_dCT(event_info, phase_space_par_list):
    """Annihilation of a Positron and Electron into a Photon and a (Dark) Photon
       e+ (Ee) + e- (me) -> gamma + gamma/V

       Input parameters needed:
            Ee (incident positron energy)
            me (electron mass)
            mV (Dark Vector Mass -- can be set to zero for SM Case)
            al (electro-weak fine-structure constant)
    """
    Ee=event_info['E_inc']
    mV=event_info['m_V']

    if len(np.shape(phase_space_par_list)) == 1:
        phase_space_par_list = np.array([phase_space_par_list])
    dSigs = []
    for varth in phase_space_par_list:
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

def dsigma_pairprod_dimensionless(event_info, phase_space_par_list):
    """Standard Model Pair Production in the Small-Angle Approximation
       gamma (w) + Z -> e+ (epp) + e- (epm) + Z
       Outgoing kinematics given by epp, dp (delta+), dm (delta-), and ph (phi)

       Input parameters needed:
            w (incident photon energy)
            Z (Target Atomic Number)
    """
    w=event_info['E_inc']
    Z =event_info['Z_T']
    mV=0
    
    if len(np.shape(phase_space_par_list)) == 1:
        phase_space_par_list = np.array([phase_space_par_list])
    dSigs = []
    for variables in phase_space_par_list:
        x1, x2, x3, x4 = variables
        epp, dp, dm, ph = m_electron + x1*(w-2*m_electron), w/(2*m_electron)*(x2+x3), w/(2*m_electron)*(x2-x3), x4*2*np.pi

        epm = w - epp

        if not((m_electron < epm < w) and (m_electron < epp < w) and (dm > 0.) and (dp > 0.)):
            dSigs.append(0.0)
        else:
            qsqT = (dp**2 + dm**2 + 2.0*dp*dm*np.cos(ph)) + m_electron**2*((1.0 + dp**2)/(2.0*epp) + (1.0+dm**2)/(2.0*epm))**2
            PF = 8.0/np.pi*alpha_em*(alpha_em/m_electron)**2*epp*epm/(w**3*qsqT**2)*dp*dm
            jacobian_factor = np.pi*w**2*(w-2*m_electron)/m_electron**2
            FF_hydrogen = g2_elastic(1.0, m_electron, qsqT)

            T1 = -1.0*dp**2/(1.0 + dp**2)**2
            T2 = -1.0*dm**2/(1.0 + dm**2)**2
            T3 = w**2/(2.0*epp*epm)*(dp**2 + dm**2)/((1.0 + dp**2)*(1.0 + dm**2))
            T4 = (epp/epm + epm/epp)*(dp*dm*np.cos(ph))/((1.0 + dp**2)*(1.0+dm**2))

            dSig0 = PF*(T1+T2+T3+T4)*jacobian_factor*FF_hydrogen

            if dSig0 < 0.0 or np.isnan(dSig0):
                print([dSig0, PF, T1, T2, T3, T4, qsqT, jacobian_factor, FF_hydrogen])
            dSigs.append(dSig0)
    if len(dSigs) == 1:
        return dSigs[0]
    else:
        return dSigs

def dsigma_compton_dCT(event_info, phase_space_par_list):
    """Compton Scattering of a Photon off an at-rest Electron, producing either a photon or a Dark Vector
        gamma (Eg) + e- (me) -> e- + gamma/V

       Input parameters needed:
            Eg (incident photon energy)
            MV (Dark Vector Mass -- can be set to zero for SM Case)
    """
    Eg=event_info['E_inc']
    Z =event_info['Z_T']
    mV=event_info['m_V']

    if len(np.shape(phase_space_par_list)) == 1:
        phase_space_par_list = np.array([phase_space_par_list])
    dSigs = []
    for varth in phase_space_par_list:
        ct = varth[0]

        s = m_electron**2 + 2*Eg*m_electron
        jacobian = 0.5*(s**2-m_electron**4)/s
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

        dSig0 = PF*jacobian*(T1+T2+T3)
        if np.isnan(dSig0):
            print(dSig0, PF, jacobian, T1, T2, T3, ct, s, t, varth)
        dSigs.append(dSig0)

    if len(dSigs) == 1:
        return dSigs[0]
    else:
        return dSigs

#Function for drawing unweighted events from a weighted distribution
def get_points(distribution, npts):
    """If weights are too cumbersome, this function returns a properly-weighted sample from Dist"""
    ret = []
    MW = np.max(np.transpose(distribution)[-1])

    tochoosefrom = [pis for pis in range(len(distribution))]
    choicesgetter = rnd.choices(tochoosefrom, np.transpose(distribution)[-1], k=npts)
    for cg in choicesgetter:
        ret.append(distribution[cg][0:-1])

    return ret

#------------------------------------------------------------------------------
#Total Cross Sections and Sample Draws for Incident Electrons/Positrons/Photons
#------------------------------------------------------------------------------
n_points = 10000 #Default number of points to draw for unweighted samples

diff_xsection_options={"PairProd" : dsigma_pairprod_dimensionless,
                       "Comp"     : dsigma_compton_dCT,
                       "Brem"     : dsigma_brem_dimensionless,
                       "Ann"      : dsigma_annihilation_dCT, 
                       "DarkBrem" : dsigma_darkbrem_dP_T }
nitn_options={"PairProd":10,
              "Brem":10,
              "DarkBrem":10,
              "Comp":500,
              "Ann":500}
nstrat_options={"PairProd":[40, 40, 40, 40],
                "Brem":[40, 40, 40, 40],
                "DarkBrem":[15, 25, 25, 15],
                "Comp":[300],
                "Ann":[300]}

four_dim = {"PairProd", "Brem", "DarkBrem"}
two_dim = {"Comp", "Ann"}

def integration_range(event_info, process):
    EInc=event_info['E_inc']
    Egmin=event_info['Eg_min']
    mV=event_info['m_V']
    if process in four_dim:
        if process == "PairProd" or process == 'Brem':
            return [[0, 1], [0, 2], [-2, 2], [0, 1]]
        else:
            minE = np.max([Egmin,mV])
            maxdel = np.sqrt(EInc/np.max([m_electron,mV]))
        return [[minE, EInc-m_electron], [0., maxdel], [0., maxdel], [0., 2*np.pi]]
    elif process in two_dim:
        if process == "Comp":
            return [[-1., 1.0]]
        elif process == "Ann":
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

def vegas_integration(event_info, process, verbose=False, mode='XSec'):
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
              -- 'Z (Target Atomic Number)
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
    if process in diff_xsection_options:
        if ('m_V' in event_info.keys()) == False:
            event_info['m_V'] = 0.0
        if ('Eg_min' in event_info.keys()) == False:
            event_info['Eg_min'] = 0.001
        igrange = integration_range(event_info, process)
        diff_xsec_func = diff_xsection_options[process] 
    else:
        raise Exception("You process is not in the list")
    integrand = vg.Integrator(igrange)
    if mode == 'Pickle' or mode == 'XSec':
        if verbose:
            print("Integrator set up", process, event_info)
        integrand(functools.partial(diff_xsec_func, event_info), nitn=nitn_options[process], nstrat=nstrat_options[process])
        if verbose:
            print("Burn-in complete", event_info)
        result = integrand(functools.partial(diff_xsec_func, event_info), nitn=nitn_options[process], nstrat=nstrat_options[process])
        if verbose:
            print("Fully Integrated", event_info, result.mean)
        if mode == 'Pickle':
            return integrand
        else:
            return result.mean
    elif mode == 'Sample' or mode == 'UnweightedSample':
        integrand(functools.partial(diff_xsec_func, event_info), nitn=nitn_options[process], nstrat=nstrat_options[process])
        result = integrand(functools.partial(diff_xsec_func, event_info), nitn=nitn_options[process], nstrat=nstrat_options[process])

        integral, pts = 0.0, []
        for x, wgt in integrand.random_batch():
            integral += wgt.dot(diff_xsec_func(event_info, x))
        if verbose:
            print(integral)
        NSamp = 1
        for kc in range(NSamp):
            for x, wgt in integrand.random():
                M0 = wgt*diff_xsec_func(event_info, x)
                pts.append(np.concatenate([list(x), [M0]]))
        if mode == 'Sample':
            tr = np.array([integral, pts], dtype=object)
        elif mode == 'UnweightedSample':
            tr = np.array([integral, get_points(pts, n_points)], dtype=object)
        return tr
        
