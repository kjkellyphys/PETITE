import numpy as np
import vegas as vg
import random as rnd

from PETITE.physical_constants import m_electron, m_muon, alpha_em

sm_process_info = {}
sm_process_counter = 0

#--------------------------------------------------------------------------
#Functions for atomic form factors for incident photons/electrons/positrons
#--------------------------------------------------------------------------
def unity(EI, t):
    return(1.0)

def dummy(x,y):
    return(0)

def pair_production_q_sq_dimensionless(xx, EI):
    """Computes momentum transfer squared for photon-scattering pair production
    Args:
        xx: tuple consisting of kinematic rescaled kinematic variables
            epsilon_plus, delta_plus, delta_minus, phi (see ... for definitions)
        EI: dictionary with incident kinetic energy 'T_inc'
    Returns:
        nuclear momentum transfer squared
    """
    x1, x2, x3, x4 = xx
    w = EI['T_inc']
    epp, dp, dm, ph = (
        m_electron + x1 * (w - 2 * m_electron),
        w / (2 * m_electron) * (x2 + x3),
        w / (2 * m_electron) * (x2 - x3),
        x4 * 2 * np.pi,
    )
    
    epm = w - epp
    return m_electron**2 * (
        (dp**2 + dm**2 + 2.0 * dp * dm * np.cos(ph))
        + m_electron**2
        * ((1.0 + dp**2) / (2.0 * epp) + (1.0 + dm**2) / (2.0 * epm)) ** 2
    )

def brem_q_sq_dimensionless(xx, EI):
    """Momentum Transfer Squared for lepton/antilepton bremsstrahlung
    Args:
        xx: tuple consisting of kinematic rescaled kinematic variables
        EI: dictionary with incident kinetic energy 'T_inc' and minimum photon energy 'Eg_min'
    Returns:
        nuclear momentum transfer squared
    """
    x1, x2, x3, x4 = xx
    Egamma_min = EI["Eg_min"]
    m_lep = EI["m_lepton"] if "m_lepton" in EI.keys() else m_electron
    ep = EI["T_inc"] + m_lep
    w, d, dp, ph = (
        Egamma_min + x1 * (ep - m_lep - Egamma_min),
        ep / (2 * m_lep) * (x2 + x3),
        ep / (2 * m_lep) * (x2 - x3),
        (x4 - 1 / 2) * 2 * np.pi,
    )

    epp = ep - w
    return m_lep**2 * (
        (d**2 + dp**2 - 2 * d * dp * np.cos(ph))
        + m_lep**2 * ((1 + d**2) / (2 * ep) - (1 + dp**2) / (2 * epp)) ** 2
    )

def aa(Z, me):
    """Support function for atomic form factors"""
    return 184.15 * (2.718) ** -0.5 * Z ** (-1.0 / 3.0) / me
def aap(Z, me):
    """Support function for atomic form factors"""
    return 1194 * (2.718) ** -0.5 * Z ** (-2.0 / 3.0) / me

def g2_elastic(EI, t):
    """Elastic form factor"""
    Z = EI["Z_T"]
    a0 = aa(Z, m_electron)
    return Z**2 * a0**4 * t**2 / (1 + a0**2 * t) ** 2

#--------------------------------------------------------------------
#Differential Cross Sections for Incident Electrons/Positrons/Photons
#--------------------------------------------------------------------
@vg.lbatchintegrand
class dsigma_brem_dimensionless:
    def __init__(self, event_info, ndim, batch_mode=False):
        """Standard Model Bremsstrahlung in the Small-Angle Approximation
        e (ep) + Z -> e (epp) + gamma (w) + Z
        Outgoing kinematics given by w, d (delta), dp (delta'), and ph (phi)
        Input parameters needed:
            ep (incident electron energy)
            Z (Target Atomic Number)
        """
        self.event_info = event_info
        self.ndim = ndim
        self.batch_mode = batch_mode
    
    def __call__(self, phase_space_par_list):
        ep = self.event_info["T_inc"] + self.event_info["m_lepton"]
        Egamma_min = self.event_info["Eg_min"]
        m_lepton = self.event_info["m_lepton"] 
        
        if self.batch_mode:
            x1, x2, x3, x4 = phase_space_par_list.T
        else:
            x1, x2, x3, x4 = phase_space_par_list
        w, d, dp, ph = (
            Egamma_min + x1 * (ep - m_lepton - Egamma_min),
            ep / (2 * m_lepton) * (x2 + x3),
            ep / (2 * m_lepton) * (x2 - x3),
            (x4 - 1 / 2) * 2 * np.pi,
        )

        epp = ep - w
        allowed_kinematics = (
            (Egamma_min < w)
            & (w < ep - m_lepton)
            & (m_lepton < epp)
            & (epp < ep)
            & (d > 0.0)
            & (dp > 0.0)
        )
        qsq = m_lepton**2 * (
            (d**2 + dp**2 - 2 * d * dp * np.cos(ph))
            + m_lepton**2 * ((1 + d**2) / (2 * ep) - (1 + dp**2) / (2 * epp)) ** 2
        )
        PF = (
            8.0
            / np.pi
            * alpha_em
            * (alpha_em / m_lepton) ** 2
            * (epp * m_lepton**4)
            / (w * ep * qsq**2)
            * d
            * dp
        )
        jacobian_factor = np.pi * ep**2 * (ep - m_lepton - Egamma_min) / m_lepton**2
        FF = g2_elastic(self.event_info, qsq)
        T1 = d**2 / (1 + d**2) ** 2
        T2 = dp**2 / (1 + dp**2) ** 2
        T3 = w**2 / (2 * ep * epp) * (d**2 + dp**2) / ((1 + d**2) * (1 + dp**2))
        T4 = -(epp / ep + ep / epp) * (d * dp * np.cos(ph)) / ((1 + d**2) * (1 + dp**2))
        dSigs = allowed_kinematics * PF * (T1 + T2 + T3 + T4) * jacobian_factor * FF
        if np.isnan(dSigs).sum() > 0:
            print(dSigs, PF, T1, T2, T3, T4, qsq, jacobian_factor, FF)
            print([x1, x2, x3, x4])
            print([w, d, dp, ph])
        return dSigs

@vg.lbatchintegrand
class dsigma_annihilation_dCT:
    def __init__(self, event_info, ndim, batch_mode=False):
        self.event_info = event_info
        self.ndim = ndim
        self.batch_mode = batch_mode

    def __call__(self, phase_space_par_list):
        """Annihilation of a Positron and Electron into a Photon and a Photon
        e+ (Ee) + e- (me) -> gamma + gamma

        Input parameters needed:
            T_inc (incident positron kinetic energy)
            me (electron mass)
            al (electro-weak fine-structure constant)
        """
        Ee = self.event_info["T_inc"] + m_electron
        s = 2.0 * m_electron * (Ee + m_electron)

        if "Eg_min" in self.event_info.keys():
            EgMin = self.event_info["Eg_min"]
        else:
            EgMin = 0.0
        ctMax = (
            np.sqrt((Ee + m_electron) / (Ee - m_electron))
            * (2 * m_electron * (Ee - 2 * EgMin + m_electron))
            / (2 * m_electron * (Ee + m_electron))
        )

        b = np.sqrt(1.0 - 4.0 * m_electron**2 / s)

        if self.batch_mode:
            ct = phase_space_par_list[:, 0]
        else:
            ct = phase_space_par_list[0]
        forbidden_kinematics = ct > ctMax

        dSigs = np.where(
            forbidden_kinematics,
            0.0,
            4.0
            * np.pi
            * alpha_em**2
            / (s * (1 - b**2 * ct**2))
            * ((s) / (2 * s) * (1 + ct**2)),
        )

        return dSigs

@vg.lbatchintegrand
class dsigma_pairprod_dimensionless:
    def __init__(self, event_info, ndim, batch_mode=False):
        """Standard Model Pair Production in the Small-Angle Approximation
        gamma (w) + Z -> e+ (epp) + e- (epm) + Z
        Outgoing kinematics given by epp, dp (delta+), dm (delta-), and ph (phi)

        Input parameters needed:
            w (incident photon energy)
            Z (Target Atomic Number)
        """
        self.event_info = event_info
        self.ndim = ndim
        self.batch_mode = batch_mode

    def __call__(self, phase_space_par_list):

        w = self.event_info["T_inc"]

        if self.batch_mode:
            x1, x2, x3, x4 = phase_space_par_list.T
        else:
            x1, x2, x3, x4 = phase_space_par_list
        epp, dp, dm, ph = (
            m_electron + x1 * (w - 2 * m_electron),
            w / (2 * m_electron) * (x2 + x3),
            w / (2 * m_electron) * (x2 - x3),
            x4 * 2 * np.pi,
        )

        epm = w - epp

        allowed_kinematics = (
            (m_electron < epm)
            & (epm < w)
            & (m_electron < epp)
            & (epp < w)
            & (dm > 0.0)
            & (dp > 0.0)
        )

        qsq_over_m_electron_sq = (
            dp**2 + dm**2 + 2.0 * dp * dm * np.cos(ph)
        ) + m_electron**2 * (
            (1.0 + dp**2) / (2.0 * epp) + (1.0 + dm**2) / (2.0 * epm)
        ) ** 2
        PF = (
            8.0
            / np.pi
            * alpha_em
            * (alpha_em / m_electron) ** 2
            * epp
            * epm
            / (w**3 * qsq_over_m_electron_sq**2)
            * dp
            * dm
        )
        jacobian_factor = np.pi * w**2 * (w - 2 * m_electron) / m_electron**2
        FF = g2_elastic(self.event_info, m_electron**2 * qsq_over_m_electron_sq)

        T1 = -1.0 * dp**2 / (1.0 + dp**2) ** 2
        T2 = -1.0 * dm**2 / (1.0 + dm**2) ** 2
        T3 = (
            w**2 / (2.0 * epp * epm) * (dp**2 + dm**2) / ((1.0 + dp**2) * (1.0 + dm**2))
        )
        T4 = (
            (epp / epm + epm / epp)
            * (dp * dm * np.cos(ph))
            / ((1.0 + dp**2) * (1.0 + dm**2))
        )

        dSigs = np.where(
            allowed_kinematics, PF * (T1 + T2 + T3 + T4) * jacobian_factor * FF, 0
        )

        if np.isnan(dSigs).sum() > 0:
            bad_entries = np.isnan(dSigs)
            print(
                [
                    dSigs[bad_entries],
                    PF[bad_entries],
                    T1[bad_entries],
                    T2[bad_entries],
                    T3[bad_entries],
                    T4[bad_entries],
                    qsq_over_m_electron_sq[bad_entries],
                    jacobian_factor[bad_entries],
                    FF[bad_entries],
                ]
            )

        return dSigs

@vg.lbatchintegrand
class dsigma_compton_dCT:
    def __init__(self, event_info, ndim, batch_mode=False):
        """Compton Scattering of a Photon off an at-rest Electron, producing a photon
        gamma (Eg) + e- (me) -> e- + gamma

        Input parameters needed:
            Eg (incident photon energy)
            MV (Dark Vector Mass -- can be set to zero for SM Case)
        """
        self.event_info = event_info
        self.ndim = ndim
        self.batch_mode = batch_mode

    def __call__(self, phase_space_par_list):

        Eg = self.event_info["T_inc"]

        s = m_electron**2 + 2 * Eg * m_electron
        
        if self.batch_mode:
            ct = phase_space_par_list[:, 0]
        else:
            ct = phase_space_par_list[0]
        #ct = np.asarray(phase_space_par_list)
        #if self.batch_mode:
        #    ct = ct[:, 0]

        jacobian = (
            (s - m_electron**2)
            / (2 * s)
            * np.sqrt(
                (s) ** 2
                - 2 * m_electron**2 * (s)
                + m_electron**4
            )
        )

        t = (
            -1
            / 2
            * (
                m_electron**4
                + s
                * (
                    + s
                    + ct
                    * np.sqrt(
                        m_electron**4
                        + (-s) ** 2
                        - 2 * m_electron**2 * (s)
                    )
                )
                - m_electron**2
                * (
                    2 * s
                    + ct
                    * np.sqrt(
                        m_electron**4
                        + (- s) ** 2
                        - 2 * m_electron**2 * (s)
                    )
                )
            )
            / s
        )
        PF = 2.0 * np.pi * alpha_em**2 / (s - m_electron**2) ** 2

        T1 = (6.0 * m_electron**2 * s + 3.0 * m_electron**4 - s**2) / (
            (m_electron**2 - s) * (-(m_electron**2) + s + t)
        )
        T2 = 4 * m_electron**4 / (s + t - m_electron**2) ** 2
        T3 = (t * (s - m_electron**2) + (s + m_electron**2) ** 2) / (
            s - m_electron**2
        ) ** 2

        # NOTE: AFAI can tell, there's no hardcoded kinematic condition here
        dSigs = PF * jacobian * (T1 + T2 + T3)

        if np.isnan(dSigs).sum() > 0:
            bad_entries = np.isnan(dSigs)
            print(
                dSigs[bad_entries],
                PF[bad_entries],
                jacobian[bad_entries],
                T1[bad_entries],
                T2[bad_entries],
                T3[bad_entries],
                ct[bad_entries],
                s[bad_entries],
                t[bad_entries],
                phase_space_par_list[bad_entries],
            )

        return dSigs
    
def dsigma_compton_dxdpt(event_info, phase_space_par_list):
    """Compton Scattering of a Photon off a bound Electron, producing either a photon or a Dark Vector
        gamma (Eg) + e- (me) -> e- + gamma/V
       Input parameters needed:
            Eg (incident photon energy)
            MV (Dark Vector Mass -- can be set to zero for SM Case)
       phase_space_par_list: list of tuples with x, pt values
    """
    Eg=event_info['T_inc']
    if 'mV' in event_info.keys():
        mV=event_info['mV']
    else:
        mV = 0.0

    s = m_electron**2 + 2*Eg*m_electron
    if s < (m_electron + mV)**2:
        if len(np.shape(phase_space_par_list)) == 1:
            return 0.0
        else:
            return np.zeros(shape=len(phase_space_par_list))
    Zeff = event_info['Z_eff']
    Lambda = Zeff*alpha_em*m_electron

    if len(np.shape(phase_space_par_list)) == 1:
        phase_space_par_list = np.array([phase_space_par_list])
    dSigs = []
    for variables in phase_space_par_list:
        a = m_electron/Lambda
        b = mV**2/(2*Lambda*x*s)
        x = variables[0]
        pts = variables[1]
        dSig0 = alpha_em/np.pi * 1/pts * (x**2+(1-x)**2)*(mV**2+2*m_electron**2)*2/ (3*m_electron*Lambda) \
        * (1/(x**2*s**2)) * (1/(x**2*((a-b/x)**2+1)**3))
        if np.isnan(dSig0):
            print(dSig0, x, pts, s, Lambda, a, b)
        dSigs.append(dSig0)

    if len(dSigs) == 1:
        return dSigs[0]
    else:
        return dSigs

@vg.lbatchintegrand
class dsigma_moller_dCT:
    def __init__(self, event_info, ndim, batch_mode=False):
        """Moller Scattering of an Electron off an at-rest Electron
        e- (Einc) + e- (me) -> e- + e-

        Input parameters needed:
            T_inc (incident electron kinetic energy)
        """
        self.event_info = event_info
        self.ndim = ndim
        self.batch_mode = batch_mode

    def __call__(self, phase_space_par_list):

        Ee = self.event_info["T_inc"] + m_electron
        if "Te_min" in self.event_info.keys():
            DE = self.event_info["Te_min"] + m_electron
        else:
            DE = 0.010

        delta_ct_limit = 2.0 * DE / (Ee - m_electron)

        if self.batch_mode:
            ct = phase_space_par_list[:, 0]
        else:
            ct = phase_space_par_list[0]

        allowed_kinematics = (ct > -1 + delta_ct_limit) & (ct < 1.0 - delta_ct_limit)
        s = m_electron**2 + 2 * Ee * m_electron
        dSigs = np.where(
            allowed_kinematics,
            16
            * np.pi**2
            * alpha_em**2
            * (
                s**2 * (3 + ct**2) ** 2
                - 8 * m_electron**2 * s * (7 + ct**4)
                + 16 * m_electron**4 * (6 - 3 * ct**2 + ct**4)
            )
            / (
                8
                * np.pi
                * s
                * (s - 4 * m_electron**2) ** 2
                * (1 - ct) ** 2
                * (1 + ct) ** 2
            ),
            0,
        )

        return dSigs
    
@vg.lbatchintegrand
class dsigma_muonelectron_dCT:
    def __init__(self, event_info, ndim, batch_mode=False):
        """Moller Scattering of a Muon off an at-rest Electron
        mu- (Einc) + e- (me) -> mu- + e-

        Input parameters needed:
            T_inc (incident muon kinetic energy)
        """
        self.event_info = event_info
        self.ndim = ndim
        self.batch_mode = batch_mode

    def __call__(self, phase_space_par_list):

        Emu = self.event_info["T_inc"] + m_muon
        s = m_electron**2 + m_muon**2 + 2*m_electron*Emu
        if "Te_min" in self.event_info.keys():
            DE = self.event_info["Te_min"] + m_electron
        else:
            DE = 0.010

        t_limit = 2.0*m_electron*(m_electron - DE)

        if self.batch_mode:
            ct = phase_space_par_list[:, 0]
        else:
            ct = phase_space_par_list[0]

        t = - 2.0*(1 - ct)*((s + m_electron**2 - m_muon**2)**2/(4.0*s) - m_electron**2)

        allowed_kinematics = (t < t_limit)
        dSigs = np.where(
            allowed_kinematics,
            16*np.pi**2*alpha_em**2*(s**2 + 2*(m_electron**2 + m_muon**2)*(2*t + m_muon**2 - 3*m_electron**2) + (s + t - 4*m_electron**2)**2)/(16*np.pi*s*t**2),
            0,
        )
        
        return dSigs

def sigma_muone(event_info):
    """Total cross section for Muon-electron scattering"""
    Emu = event_info["T_inc"] + m_muon
    EeMIN = event_info["Te_min"] + m_electron

    threshold = 1.0/(2.0*m_electron)*(m_electron*(EeMIN - m_electron) + np.sqrt(m_electron*(EeMIN+m_electron)*(m_electron*(EeMIN-m_electron) + 2*m_muon**2)))

    s = m_electron**2 + m_muon**2 + 2*m_electron*Emu
    t_max = 2.0*m_electron*(m_electron - EeMIN)
    t_min = -4.0*((s + m_electron**2 - m_muon**2)**2/(4*s) - m_electron**2)

    PF = 16*np.pi**2*alpha_em**2/(8.0*np.pi*((s-m_muon**2)**2 + m_electron**4 - 2*(s+m_muon**2)*m_electron**2))

    T1 = -2.0*(s**2 + m_muon**4 + 5*m_electron**4 - 2*m_electron**2*(2*s+m_muon**2))*(1.0/t_max - 1.0/t_min)
    T2 = 2.0*(s + 2*m_muon**2 - 2*m_electron**2)*np.log(t_max/t_min)
    T3 = t_max - t_min

    return PF*(T1+T2+T3)*np.heaviside(Emu - threshold, 1)

def sigma_moller(event_info):
    """Total cross section for Moller scattering"""

    Ee = event_info["T_inc"] + m_electron
    TeMIN = event_info["Te_min"]
    threshold = 3 * m_electron + 4 * TeMIN

    PF = 2 * np.pi * alpha_em**2 / (m_electron * (Ee**2 - m_electron**2))
    T1 = (
        Ee
        - 3 * m_electron
        - 4 * TeMIN
        + 2
        * Ee**2
        * (
            -2 / (Ee - 3 * m_electron - 2 * TeMIN)
            + 1 / TeMIN
            + 1 / (-Ee + m_electron + TeMIN)
            + 2 / (Ee + m_electron + 2 * TeMIN)
        )
    )
    T2 = (
        2
        * m_electron
        * (m_electron - 2 * Ee)
        / (Ee - m_electron)
        * np.log(
            (
                (-Ee + m_electron + TeMIN)
                * (-Ee + 3 * m_electron + 2 * TeMIN)
                / (TeMIN * (Ee + m_electron + 2 * TeMIN))
            )
            * np.heaviside(Ee - threshold, 1)
            + np.heaviside(threshold - Ee, 1)
        )
    )

    return PF * (T1 + T2) * np.heaviside(Ee - threshold, 1)


@vg.lbatchintegrand
class dsigma_bhabha_dCT:
    def __init__(self, event_info, ndim, batch_mode=False):
        """Bhabha Scattering of a Positron off an at-rest Electron
        e+ (Einc) + e- (me) -> e+ + e-

        Input parameters needed:
            T_inc (incident positron kinetic energy)
        """
        self.event_info = event_info
        self.ndim = ndim
        self.batch_mode = batch_mode

    def __call__(self, phase_space_par_list):

        Ee = self.event_info["T_inc"] + m_electron
        if "Te_min" in self.event_info.keys():
            DE = self.event_info["Te_min"] + m_electron
        else:
            DE = 0.010
        delta_ct_limit = 2.0 * DE / (Ee - m_electron)

        if self.batch_mode:
            ct = phase_space_par_list[:, 0]
        else:
            ct = phase_space_par_list[0]

        allowed_kinematics = (ct > -1 + delta_ct_limit) & (ct < 1.0 - delta_ct_limit)
        s = m_electron**2 + 2 * Ee * m_electron

        dSigs = np.where(
            allowed_kinematics,
            (
                alpha_em**2
                * np.pi
                * (
                    256 * (-1 + ct) ** 2 * ct**2 * m_electron**8
                    - 128
                    * (-1 + ct)
                    * (1 + ct * (1 + ct) * (-3 + 2 * ct))
                    * m_electron**6
                    * s
                    + 16
                    * (7 + ct * (2 + ct * (-5 + 6 * (-1 + ct) * ct)))
                    * m_electron**4
                    * s**2
                    - 8
                    * (7 + ct * (-3 + ct * (3 + ct * (-1 + 2 * ct))))
                    * m_electron**2
                    * s**3
                    + (3 + ct**2) ** 2 * s**4
                )
            )
            / (2 * (-1 + ct) ** 2 * s**3 * (-4 * m_electron**2 + s) ** 2),
            0,
        )

        return dSigs


def sigma_bhabha(event_info):
    """Total cross section for Bhabha scattering"""

    Ee = event_info["T_inc"] + m_electron
    TeMIN = event_info["Te_min"]
    threshold = 3 * m_electron + 4 * TeMIN

    PF = (
        np.pi
        * alpha_em**2
        / (
            12
            * (Ee - m_electron)
            * m_electron
            * (Ee + m_electron) ** 3
            * (Ee - 3 * m_electron - 2 * TeMIN)
            * TeMIN
        )
    )
    T1 = (Ee - 3 * m_electron - 4 * TeMIN) * (
        24 * Ee**2 * (Ee + m_electron) ** 2
        + (Ee - 3 * m_electron)
        * (31 * Ee**2 + 84 * Ee * m_electron + 57 * m_electron**2)
        * TeMIN
        - 4 * (16 * Ee**2 + 39 * Ee * m_electron + 33 * m_electron**2) * TeMIN**2
        + 8 * (Ee - 3 * m_electron) * TeMIN**3
        - 8 * TeMIN**4
    )
    T2 = (
        24
        * (Ee + m_electron)
        * (2 * Ee**2 + 4 * Ee * m_electron + m_electron**2)
        * (Ee - 3 * m_electron - 2 * TeMIN)
        * TeMIN
        * np.log(
            (2 * TeMIN / (Ee - 3 * m_electron - 2 * TeMIN))
            * np.heaviside(Ee - threshold, 1)
            + np.heaviside(threshold - Ee, 1)
        )
    )

    return PF * (T1 + T2) * np.heaviside(Ee - threshold, 1)

#Function for drawing unweighted events from a weighted distribution
def get_points(distribution, npts):
    """If weights are too cumbersome, this function returns a properly-weighted sample from Dist"""
    ret = []

    tochoosefrom = [pis for pis in range(len(distribution))]
    choicesgetter = rnd.choices(tochoosefrom, np.transpose(distribution)[-1], k=npts)
    for cg in choicesgetter:
        ret.append(distribution[cg][0:-1])

    return ret

#------------------------------------------------------------------------------
#Total Cross Sections and Sample Draws for Incident Electrons/Positrons/Photons
#------------------------------------------------------------------------------
n_points = 10000 #Default number of points to draw for unweighted samples

diff_xsection_options = {
    "PairProd": dsigma_pairprod_dimensionless,
    "Comp": dsigma_compton_dCT,
    "Moller": dsigma_moller_dCT,
    "MuonE": dsigma_muonelectron_dCT,
    "Bhabha": dsigma_bhabha_dCT,
    "Brem": dsigma_brem_dimensionless,
    "MuonBrem": dsigma_brem_dimensionless,
    "Ann": dsigma_annihilation_dCT,
    }

vegas_integrator_options = {"PairProd":{"nitn":10, "nstrat":[60, 50, 40, 50]},
                            "Brem":{"nitn":10, "nstrat":[60, 50, 50, 50]},
                            "MuonBrem":{"nitn":10, "nstrat":[60, 50, 50, 50]},
                            "Comp":{"nitn":20, "nstrat":[1000]},
                            "Moller":{"nitn":20, "nstrat":[1000]},
                            "MuonE":{"nitn":20, "nstrat":[1000]},
                            "Bhabha":{"nitn":20, "nstrat":[1000]},
                            "Ann":{"nitn":20, "nstrat":[1000]},
                            }
      
four_dim = {"PairProd", "Brem", "MuonBrem"}
one_dim = {"Comp", "Ann","Moller","MuonE","Bhabha"}

def integration_range(event_info, process):
    """Defines the integration range for the VEGAS integrator given a specific process
    Args:
        event_info - dictionary with parameter needed to evaluate the cross-section:
            T_inc - incident particle kinetic energy
            mV - vector mass
            Eg_min - minimum lab-frame energy (GeV) of outgoing photons
            Te_min - minimum lab-frame kinetic energy (GeV) of outgoing electrons
            costheta_min - minimum lab-frame cosine of the angle between outgoing electrons
            xmin - optional minimum value of x (dimensionless variable related to energy) for DarkBrem if only interested in higher energy part of phase space
        process - process to be integrated over
    Returns:
        list of integration ranges for each dimension (note that number of dimensions depend on process)
    """
    if process in four_dim:
        return [[0, 1], [0, 2], [-2, 2], [0, 1]]
    elif process in one_dim:
        if process == "Comp" or process == "Ann" or process == "MuonE":
            return [[-1., 1.0]]
        else:
            if 'Te_min' in event_info.keys():
                DE = event_info['Te_min'] + m_electron
            else:
                DE = 0.005
            delta_ct_limit = 2.0*DE/(event_info['T_inc'] - m_electron)
            return [[-1.0+delta_ct_limit, 1.0-delta_ct_limit]]

    else:
        raise Exception("Your process is not in the list")

def vegas_integration(event_info, process, verbose=False, mode="XSec"):
    """Function for Integration of Various SM/BSM Differential
    Cross Sections.

    Available Processes ('Process'):
     -- 'Brem': Standard Model e + Z -> e + gamma + Z
     -- 'MuonBrem': Standard Model mu + Z -> mu + gamma + Z
     -- 'PairProd': SM gamma + Z -> e^+ + e^- + Z
     -- 'Comp': SM/BSM gamma + e -> e + gamma/V
     -- 'Ann': SM/BSM e^+ e^- -> gamma + gamma/V

     ('Brem', 'PairProd' calculated in
      small-angle approximation)

    Input parameters needed:
         EI: dictionary containing
           -- 'T_inc' (incident particle kinetic energy)
           -- 'Z (Target Atomic Number)
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
        if not ("Eg_min" in event_info.keys()):
            event_info["Eg_min"] = 0.001
        if not ("Te_min" in event_info.keys()):
            event_info["Te_min"] = 0.005
        if not ("m_lepton" in event_info.keys()):
            event_info["m_lepton"] = m_electron
        if process == "MuonBrem":
            event_info["m_lepton"] = m_muon
        igrange = integration_range(event_info, process)
        diff_xsec_func = diff_xsection_options[process]
    else:
        raise Exception(
            f"Could not find process {process} in available processes: {diff_xsection_options.keys()}"
        )
    integ = vg.Integrator(igrange)
    f_integrand = diff_xsec_func(event_info=event_info, ndim=len(igrange), batch_mode=True)

    if mode == "Pickle" or mode == "XSec":
        if verbose:
            print("Integrator set up", process, event_info)
        integ(f_integrand, **vegas_integrator_options[process])
        if verbose:
            print("Burn-in complete", event_info)
        result = integ(f_integrand, **vegas_integrator_options[process])

        if verbose:
            print("Fully Integrated", event_info, result.mean)
        if mode == "Pickle":
            return integ
        else:
            return result.mean
    elif mode == "Sample" or mode == "UnweightedSample":
        integ(f_integrand, **vegas_integrator_options[process])
        result = integ(f_integrand, **vegas_integrator_options[process])

        integral, pts = 0.0, []
        for x, wgt in integ.random_batch():
            integral += wgt.dot(f_integrand(x))
        if verbose:
            print(integral)
        NSamp = 1
        for kc in range(NSamp):
            for x, wgt in integ.random():
                M0 = wgt * f_integrand(x)
                pts.append(np.concatenate([list(x), [M0]]))
        if mode == "Sample":
            tr = np.array([integral, pts], dtype=object)
        elif mode == "UnweightedSample":
            tr = np.array([integral, get_points(pts, n_points)], dtype=object)
        return tr
        
