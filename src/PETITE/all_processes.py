import numpy as np
import vegas as vg
import random as rnd

from PETITE.physical_constants import m_electron, m_proton, GeV, alpha_em
from PETITE.radiative_return import lepton_luminosity_integrand
from PETITE.radiative_return import transformed_lepton_luminosity_integrand

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
        EI: dictionary with incident energy 'E_inc'
    Returns:
        nuclear momentum transfer squared
    """
    x1, x2, x3, x4 = xx
    w = EI['E_inc']
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
    """Momentum Transfer Squared for electron/positron bremsstrahlung
    Args:
        xx: tuple consisting of kinematic rescaled kinematic variables
        EI: dictionary with incident energy 'E_inc' and minimum photon energy 'Eg_min'
    Returns:
        nuclear momentum transfer squared
    """
    x1, x2, x3, x4 = xx
    Egamma_min = EI["Eg_min"]
    ep = EI["E_inc"]
    w, d, dp, ph = (
        Egamma_min + x1 * (ep - m_electron - Egamma_min),
        ep / (2 * m_electron) * (x2 + x3),
        ep / (2 * m_electron) * (x2 - x3),
        (x4 - 1 / 2) * 2 * np.pi,
    )

    epp = ep - w
    return m_electron**2 * (
        (d**2 + dp**2 - 2 * d * dp * np.cos(ph))
        + m_electron**2 * ((1 + d**2) / (2 * ep) - (1 + dp**2) / (2 * epp)) ** 2
    )

def darkbrem_qsq(xx, EI):
    """Momentum Transfer Squared for dark photon bremsstrahlung
    Args:
        xx: tuple consisting of kinematic rescaled kinematic variables
        EI: dictionary with incident energy 'E_inc' and dark photon mass 'mV'
    Returns:
        q^2 (dimensionful)
    """
    x, l1mct, ttilde = xx

    Ebeam = EI["E_inc"]
    MTarget = EI["mT"]
    
    tconv = (
        2
        * MTarget
        * (MTarget + Ebeam)
        * np.sqrt(Ebeam**2 + m_electron**2)
        / (MTarget * (MTarget + 2 * Ebeam) + m_electron**2)
    ) ** 2
    return ttilde * tconv


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

def g2_inelastic(EI, t):
    """Inelastic form factor"""
    Z = EI["Z_T"]
    ap0 = aap(Z, m_electron)
    return Z * ap0**4 * t**2 / (1 + ap0**2 * t) ** 2

mu_p = 2.79  # https://journals.aps.org/prd/pdf/10.1103/PhysRevD.8.3109
def Gelastic_inelastic_over_tsquared(EI, t):
    """
    Form factor squared used for elastic/inelastic contributions to Dark Bremsstrahlung Calculation
    Rescaled by 1/t^2 to make it easier to integrate over t
    (Scales like Z^2 in the small-t limit)
    See Eq. (9) of Gninenko et al (Phys. Lett. B 782 (2018) 406-411)
    """
    Z = EI["Z_T"]
    A = EI["A_T"]
    c1 = (111 * Z ** (-1 / 3) / m_electron) ** 2
    c2 = 0.164 * GeV**2 * A ** (-2 / 3)
    Gel = (1.0 / (1.0 + c1 * t)) ** 2 * (1 + t / c2) ** (-2)
    ap2 = (773.0 * Z ** (-2.0 / 3) / m_electron) ** 2
    Ginel = (
        Z
        / ((c1**2 * Z**2))
        * np.power((ap2 / (1.0 + ap2 * t)), 2.0)
        * ((1.0 + (mu_p**2 - 1.0) * t / (4.0 * m_proton**2)) / (1.0 + t / 0.71) ** 4)
    )
    return Z**2 * c1**2 * (Gel + Ginel)

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
        # return c_dsigma_brem_dimensionless(
        #     phase_space_par_list, self.event_info["E_inc"], self.event_info["Eg_min"]
        # )
        ep = self.event_info["E_inc"]
        Egamma_min = self.event_info["Eg_min"]
        #if len(np.shape(phase_space_par_list)) == 1:
        #    phase_space_par_list = np.array([phase_space_par_list])
        # for variables in phase_space_par_list:
        if self.batch_mode:
            x1, x2, x3, x4 = phase_space_par_list.T
        else:
            x1, x2, x3, x4 = phase_space_par_list
        w, d, dp, ph = (
            Egamma_min + x1 * (ep - m_electron - Egamma_min),
            ep / (2 * m_electron) * (x2 + x3),
            ep / (2 * m_electron) * (x2 - x3),
            (x4 - 1 / 2) * 2 * np.pi,
        )

        epp = ep - w
        allowed_kinematics = (
            (Egamma_min < w)
            & (w < ep - m_electron)
            & (m_electron < epp)
            & (epp < ep)
            & (d > 0.0)
            & (dp > 0.0)
        )
        qsq = m_electron**2 * (
            (d**2 + dp**2 - 2 * d * dp * np.cos(ph))
            + m_electron**2 * ((1 + d**2) / (2 * ep) - (1 + dp**2) / (2 * epp)) ** 2
        )
        PF = (
            8.0
            / np.pi
            * alpha_em
            * (alpha_em / m_electron) ** 2
            * (epp * m_electron**4)
            / (w * ep * qsq**2)
            * d
            * dp
        )
        jacobian_factor = np.pi * ep**2 * (ep - m_electron - Egamma_min) / m_electron**2
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
            # dSigs.append(dSig0) # NOTE: not needed?
        return dSigs

class dsig_dx_dcostheta_dark_brem_exact_tree_level:
    def __init__(self, event_info, ndim, batch_mode=False):
        """
        Exact Tree-Level Dark Photon Bremsstrahlung
        e (ep) + Z -> e (epp) + V (w) + Z
        result it dsigma/dx/dcostheta where x=E_darkphoton/E_beam and theta is angle between beam and dark photon
        Input parameters needed:
            x0, x1, x2:  kinematic parameters related to energy of emitted vector, cosine of its angle
                            and the momentum transfer to the nucleus (precise relation depends on event_info['Method'] see below.
            me (mass of electron)
            mV (mass of dark photon)
            Ebeam (incident electron energy)
            ZTarget (Target charge)
            ATarget (Target Atomic mass number)
            MTarget (Target mass)
        """
        self.event_info = event_info
        self.ndim = ndim
        self.batch_mode = batch_mode

    def __call__(self, phase_space_par_list):
        if self.batch_mode:
            x0, x1, x2 = phase_space_par_list.T
        else:
            x0, x1, x2 = phase_space_par_list

        me = m_electron
        mV = self.event_info["mV"]
        Ebeam = self.event_info["E_inc"]
        MTarget = self.event_info["mT"]

        if not ("Method" in self.event_info.keys()):
            self.event_info["Method"] = "Log"
        if self.event_info["Method"] == "Log":
            x, l1mct, lttilde = x0, x1, x2
            one_minus_costheta = 10**l1mct
            costheta = 1.0 - one_minus_costheta
            ttilde = 10**lttilde
            Jacobian = one_minus_costheta * ttilde * np.log(10.0) ** 2
        elif self.event_info["Method"] == "Standard":
            x, costheta, ttilde = x0, x1, x2
            Jacobian = 1.0

        k = np.sqrt((x * Ebeam) ** 2 - mV**2)
        p = np.sqrt(Ebeam**2 - me**2)
        V = np.sqrt(p**2 + k**2 - 2 * p * k * costheta)

        utilde = -2 * (x * Ebeam**2 - k * p * costheta) + mV**2
        discr = (
            utilde**2
            + 4 * MTarget * utilde * ((1 - x) * Ebeam + MTarget)
            + 4 * MTarget**2 * V**2
        )

        # NOTE: safe sqrt -- points with discr < 0 are killed afterwards
        # NOTE: a better way to specify the kinematical bounds is to do it within the integrand

        Qplus = V * (utilde + 2 * MTarget * ((1 - x) * Ebeam + MTarget)) + (
            (1 - x) * Ebeam + MTarget
        ) * np.sqrt(np.abs(discr))
        Qplus = Qplus / (2 * ((1 - x) * Ebeam + MTarget) ** 2 - 2 * V**2)
        Qminus = V * (utilde + 2 * MTarget * ((1 - x) * Ebeam + MTarget)) - (
            (1 - x) * Ebeam + MTarget
        ) * np.sqrt(np.abs(discr))
        Qminus = Qminus / (2 * ((1 - x) * Ebeam + MTarget) ** 2 - 2 * V**2)

        Qplus = np.fabs(Qplus)
        Qminus = np.fabs(Qminus)

        tplus = 2 * MTarget * (np.sqrt(MTarget**2 + Qplus**2) - MTarget)
        tminus = 2 * MTarget * (np.sqrt(MTarget**2 + Qminus**2) - MTarget)

        tconv = (
            2
            * MTarget
            * (MTarget + Ebeam)
            * np.sqrt(Ebeam**2 + m_electron**2)
            / (MTarget * (MTarget + 2 * Ebeam) + m_electron**2)
        ) ** 2

        t = ttilde * tconv

        q0 = -t / (2 * MTarget)
        q = np.sqrt(t**2 / (4 * MTarget**2) + t)

        costhetaq = -(V**2 + q**2 + me**2 - (Ebeam + q0 - x * Ebeam) ** 2) / (2 * V * q)

        mVsq2mesq = mV**2 + 2 * me**2

        Am2 = (
            -8
            * MTarget
            * (4 * Ebeam**2 * MTarget - t * (2 * Ebeam + MTarget))
            * mVsq2mesq
        )
        A1 = 8 * MTarget**2 / utilde
        Am1 = (8 / utilde) * (
            MTarget**2
            * (
                2 * t * utilde
                + utilde**2
                + 4 * Ebeam**2 * (2 * (x - 1) * mVsq2mesq - t * ((x - 2) * x + 2))
                + 2 * t * (-(mV**2) + 2 * me**2 + t)
            )
            - 2 * Ebeam * MTarget * t * ((1 - x) * utilde + (x - 2) * (mVsq2mesq + t))
            + t**2 * (utilde - mV**2)
        )
        A0 = (8 / utilde**2) * (
            MTarget**2
            * (2 * t * utilde + (t - 4 * Ebeam**2 * (x - 1) ** 2) * mVsq2mesq)
            + 2 * Ebeam * MTarget * t * (utilde - (x - 1) * mVsq2mesq)
        )
        Y = -t + 2 * q0 * Ebeam - 2 * q * p * (p - k * costheta) * costhetaq / V
        W = (
            Y**2
            - 4 * q**2 * p**2 * k**2 * (1 - costheta**2) * (1 - costhetaq**2) / V**2
        )

        # if W == 0.0:
        #     print("x, costheta, t = ", [x, costheta, t])
        #     print(
        #         "Y, q, p, k, costheta, costhetaq, V",
        #         [Y, q, p, k, costheta, costhetaq, V],
        #     )

        phi_integral = (A0 + Y * A1 + Am1 / np.sqrt(W) + Y * Am2 / W**1.5) / (
            8 * MTarget**2
        )

        formfactor_separate_over_tsquared = Gelastic_inelastic_over_tsquared(
            self.event_info, t
        )

        ans = (
            formfactor_separate_over_tsquared
            * np.power(alpha_em, 3)
            * k
            * Ebeam
            * phi_integral
            / (p * np.sqrt(k**2 + p**2 - 2 * p * k * costheta))
        )

        # kinematic boundaries
        allowed_kinematics = (
            (x * Ebeam >= mV)
            & (discr >= 0)
            & (tplus > tminus)
            & (t > tminus)
            & (t < tplus)
            & (np.fabs(costhetaq) <= 1.0)
            & (W > 0)
        )

        return np.where(allowed_kinematics, ans * tconv * Jacobian, 0)

def dsigma_radiative_return_dx(event_info, x):
    """
    Radiative return cross-section e^+ e^- > V differential with respect to the longitudinal momentum fraction
    carried by one of beam particles
    Args:
        event_info - dictionary with parameter needed to evaluate the cross-section:
            E_inc - incoming positron energy
            mV - vector mass
        x - fraction of initial CM momentum carried by one of the beam particles
    Returns:
        radiative return cross-section in GeV^-2
    """
    mV = event_info["mV"]
    Ee = event_info["E_inc"]

    s = 2.0 * m_electron * (Ee + m_electron)
    betaf = np.sqrt(1.0 - 4.0 * (m_electron**2) / (mV**2))
    
    # Changed betaf -> (1.0/betaf) on 26/08/2024, fix relative to original published result.
    prefac = (
        (4.0 * np.pi**2) * alpha_em * (1.0 / betaf) * (3.0 / 2.0 - betaf**2 / 2.0) / s
    )
    
     # this needs to be integrated over x in [y, 1], where y=mV^2/s
    return prefac * lepton_luminosity_integrand(s, mV**2 / s, x)

@vg.lbatchintegrand
class dsigma_radiative_return_du:
    def __init__(self, event_info, ndim, batch_mode=False):
        """
        Radiative return cross-section e^+ e^- > V differential with respect to the longitudinal momentum fraction
        carried by one of beam particles
        Args:
            event_info - dictionary with parameter needed to evaluate the cross-section:
                E_inc - incoming positron energy
                mV - vector mass
            u - (1-x)^(beta/2) where x is the fraction of initial CM momentum carried by one of the beam particles
        Returns:
            radiative return cross-section in GeV^-2
        """
        self.event_info = event_info
        self.ndim = ndim
            
    def __call__(self, phase_space_par_list):

        mV = self.event_info["mV"]
        Ee = self.event_info["E_inc"]

        s = 2.0 * m_electron * (Ee + m_electron)
        if s < mV**2:
            if len(np.shape(phase_space_par_list)) <= 1:
                return 0.0
            else:
                return np.zeros(shape=len(phase_space_par_list))

        beta = (2.0 * alpha_em / np.pi) * (np.log(s / m_electron**2) - 1.0)
        umax = np.power(1.0 - mV**2 / s, beta / 2.0)

        betaf = np.sqrt(1.0 - 4.0 * (m_electron**2) / (mV**2))
        prefac = (
            (4.0 * np.pi**2)
            * alpha_em
            * betaf
            * (3.0 / 2.0 - betaf**2 / 2.0)
            / s
            * umax
        )

        """
            this needs to be integrated over x in [sqrt(y), 1], where y=mV^2/s and multiplied by 2

            the factor of 2 comes from splitting the [y,1] integration into [y,sqrt(y)] + [sqrt(y),1], 
            and using x-> y/x in the first part comes from splitting the [y,1] integration into [y,sqrt(y)] + [sqrt(y),1]
        """
        u0 = np.array(phase_space_par_list)
        x1 = 1.0 - np.power(u0 * umax, 2.0 / beta)
        x2 = mV**2 / (x1 * s)

        allowed_kinematics = (x2 < 1.0) & (x1 > 0.0) & (u0 < 1.0)
        dSigs = np.where(
            allowed_kinematics,
            2.0
            * prefac
            * transformed_lepton_luminosity_integrand(s, mV**2 / s, u0 * umax),
            0,
        )

        return dSigs

@vg.lbatchintegrand
class dsigma_annihilation_dCT:
    def __init__(self, event_info, ndim, batch_mode=False):
        self.event_info = event_info
        self.ndim = ndim
        self.batch_mode = batch_mode

    def __call__(self, phase_space_par_list):
        """Annihilation of a Positron and Electron into a Photon and a (Dark) Photon
        e+ (Ee) + e- (me) -> gamma + gamma/V

        Input parameters needed:
            Ee (incident positron energy)
            me (electron mass)
            mV (Dark Vector Mass -- can be set to zero for SM Case)
            al (electro-weak fine-structure constant)
        """
        Ee = self.event_info["E_inc"]
        if "mV" in self.event_info.keys():
            mV = self.event_info["mV"]
        else:
            mV = 0.0
        s = 2.0 * m_electron * (Ee + m_electron)

        if "Eg_min" in self.event_info.keys():
            EgMin = self.event_info["Eg_min"]
        else:
            EgMin = 0.0
        ctMax = (
            np.sqrt((Ee + m_electron) / (Ee - m_electron))
            * (2 * m_electron * (Ee - 2 * EgMin + m_electron) - mV**2)
            / (2 * m_electron * (Ee + m_electron) - mV**2)
        )

        if s < mV**2:
            if len(np.shape(phase_space_par_list)) == 1:
                return 0.0
            else:
                return np.zeros(shape=len(phase_space_par_list))

        b = np.sqrt(1.0 - 4.0 * m_electron**2 / s)

        if self.batch_mode:
            ct = phase_space_par_list[:, 0]
        else:
            ct = phase_space_par_list[0]
        #ct = np.asarray(phase_space_par_list)
        #if self.batch_mode:
        #    ct = ct[:, 0]
        forbidden_kinematics = ct > ctMax

        dSigs = np.where(
            forbidden_kinematics,
            0.0,
            4.0
            * np.pi
            * alpha_em**2
            / (s * (1 - b**2 * ct**2))
            * ((s - mV**2) / (2 * s) * (1 + ct**2) + 2.0 * mV**2 / (s - mV**2)),
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

        w = self.event_info["E_inc"]

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
        """Compton Scattering of a Photon off an at-rest Electron, producing either a photon or a Dark Vector
        gamma (Eg) + e- (me) -> e- + gamma/V

        Input parameters needed:
            Eg (incident photon energy)
            MV (Dark Vector Mass -- can be set to zero for SM Case)
        """
        self.event_info = event_info
        self.ndim = ndim
        self.mV = self.event_info["mV"] if "mV" in self.event_info.keys() else 0
        self.batch_mode = batch_mode

    def __call__(self, phase_space_par_list):

        Eg = self.event_info["E_inc"]

        s = m_electron**2 + 2 * Eg * m_electron
        if s < (m_electron + self.mV) ** 2:
            if len(np.shape(phase_space_par_list)) == 1:
                return 0.0
            else:
                return np.zeros(shape=len(phase_space_par_list))

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
                (s - self.mV**2) ** 2
                - 2 * m_electron**2 * (s + self.mV**2)
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
                    -(self.mV**2)
                    + s
                    + ct
                    * np.sqrt(
                        m_electron**4
                        + (self.mV**2 - s) ** 2
                        - 2 * m_electron**2 * (self.mV**2 + s)
                    )
                )
                - m_electron**2
                * (
                    self.mV**2
                    + 2 * s
                    + ct
                    * np.sqrt(
                        m_electron**4
                        + (self.mV**2 - s) ** 2
                        - 2 * m_electron**2 * (self.mV**2 + s)
                    )
                )
            )
            / s
        )
        PF = 2.0 * np.pi * alpha_em**2 / (s - m_electron**2) ** 2

        if self.mV == 0.0:
            T1 = (6.0 * m_electron**2 * s + 3.0 * m_electron**4 - s**2) / (
                (m_electron**2 - s) * (-(m_electron**2) + s + t)
            )
            T2 = 4 * m_electron**4 / (s + t - m_electron**2) ** 2
            T3 = (t * (s - m_electron**2) + (s + m_electron**2) ** 2) / (
                s - m_electron**2
            ) ** 2
        else:
            T1 = (
                2.0 * m_electron**2 * (self.mV**2 - 3 * s)
                - 3 * m_electron**4
                - 2 * self.mV**2 * s
                + 2 * self.mV**4
                + s**2
            ) / ((m_electron**2 - s) * (m_electron**2 + self.mV**2 - s - t))
            T2 = (2 * m_electron**2 * (2 * m_electron**2 + self.mV**2)) / (
                m_electron**2 + self.mV**2 - s - t
            ) ** 2
            T3 = (
                (m_electron**2 + s) * (m_electron**2 + self.mV**2 + s)
                + t * (s - m_electron**2)
            ) / (m_electron**2 - s) ** 2

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
    Eg=event_info['E_inc']
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
            Einc (incident electron energy)
        """
        self.event_info = event_info
        self.ndim = ndim
        self.batch_mode = batch_mode

    def __call__(self, phase_space_par_list):

        Ee = self.event_info["E_inc"]
        if "Ee_min" in self.event_info.keys():
            DE = self.event_info["Ee_min"]
        else:
            DE = 0.010

        delta_ct_limit = 2.0 * DE / (Ee - m_electron)

        if self.batch_mode:
            ct = phase_space_par_list[:, 0]
        else:
            ct = phase_space_par_list[0]
        #ct = np.asarray(phase_space_par_list)
        #if self.batch_mode:
        #    ct = ct[:, 0]

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


def sigma_moller(event_info):
    """Total cross section for Moller scattering"""

    Ee = event_info["E_inc"]
    TeMIN = event_info["Ee_min"] - m_electron
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
            Einc (incident positron energy)
        """
        self.event_info = event_info
        self.ndim = ndim
        self.batch_mode = batch_mode

    def __call__(self, phase_space_par_list):

        Ee = self.event_info["E_inc"]
        if "Ee_min" in self.event_info.keys():
            DE = self.event_info["Ee_min"]
        else:
            DE = 0.010
        delta_ct_limit = 2.0 * DE / (Ee - m_electron)

        if self.batch_mode:
            ct = phase_space_par_list[:, 0]
        else:
            ct = phase_space_par_list[0]
        #ct = np.asarray(phase_space_par_list)
        #if self.batch_mode:
        #    ct = ct[:, 0]

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

    Ee = event_info["E_inc"]
    TeMIN = event_info["Ee_min"] - m_electron
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
    "Bhabha": dsigma_bhabha_dCT,
    "Brem": dsigma_brem_dimensionless,
    "Ann": dsigma_annihilation_dCT,
    "DarkAnn": dsigma_radiative_return_du,  # dsigma_radiative_return_dx,
    "DarkComp": dsigma_compton_dCT,
    "DarkBrem": dsig_dx_dcostheta_dark_brem_exact_tree_level,
}

vegas_integrator_options = {"PairProd":{"nitn":10, "nstrat":[60, 50, 40, 50]},
                            "Brem":{"nitn":10, "nstrat":[60, 50, 50, 50]},
                            "DarkBrem":{"nitn":20, "nstrat":[100, 100, 40]},
                            "Comp":{"nitn":20, "nstrat":[1000]},
                            "Moller":{"nitn":20, "nstrat":[1000]},
                            "Bhabha":{"nitn":20, "nstrat":[1000]},
                            "Ann":{"nitn":20, "nstrat":[1000]},
                            "DarkAnn":{"nitn":10, "neval":10000},
                            "DarkComp":{"nitn":20, "nstrat":[1000]}}
      
four_dim = {"PairProd", "Brem"}
three_dim = {"DarkBrem"}
one_dim = {"Comp", "Ann","Moller","Bhabha", "DarkAnn", "DarkComp"}

def integration_range(event_info, process):
    """Defines the integration range for the VEGAS integrator given a specific process
    Args:
        event_info - dictionary with parameter needed to evaluate the cross-section:
            E_inc - incident positron energy
            mV - vector mass
            Eg_min - minimum lab-frame energy (GeV) of outgoing photons
            Ee_min - minimum lab-frame energy (GeV) of outgoing electrons
            costheta_min - minimum lab-frame cosine of the angle between outgoing electrons
            xmin - optional minimum value of x (dimensionless variable related to energy) for DarkBrem if only interested in higher energy part of phase space
        process - process to be integrated over
    Returns:
        list of integration ranges for each dimension (note that number of dimensions depend on process)
    """
    EInc=event_info['E_inc']
    mV=event_info['mV']
    s = 2.0*m_electron*(EInc+m_electron)

    if process in four_dim:
        if process == "PairProd" or process == 'Brem':
            return [[0, 1], [0, 2], [-2, 2], [0, 1]]
        else:
            Egmin=event_info['Eg_min']
            minE = np.max([Egmin,mV])
            maxdel = np.sqrt(EInc/np.max([m_electron,mV]))
        return [[minE, EInc-m_electron], [0., maxdel], [0., maxdel], [0., 2*np.pi]]
    elif process in three_dim:
        if 'costheta_min' in event_info:
            l1mct_max = np.log10(1 - event_info['costheta_min'])
        else:
            l1mct_max = np.log10(2.0)

        if 'xmin' in event_info: # DarkBrem
            xmin = event_info['xmin']
        else:
            xmin = 0.
        return [[max(xmin, mV/EInc), 1.-m_electron/EInc],[-12.0, l1mct_max], [-20.0, 0.0]]
    elif process in one_dim:
        if process == "Comp" or process == "Ann" or process == "DarkComp":
            return [[-1., 1.0]]
        elif process == "DarkAnn":
            #beta = (2.*alpha_em/np.pi) * (np.log(s/m_electron**2) - 1.)
            if s > mV**2:
                #return [[0., np.power(1.-mV**2/s,beta/2.)]]
                return [[0., 1.0]]
            else:
                return [[0.,0.]]
        else:
            if 'Ee_min' in event_info.keys():
                DE = event_info['Ee_min']
            else:
                DE = 0.005
            delta_ct_limit = 2.0*DE/(event_info['E_inc'] - m_electron)
            return [[-1.0+delta_ct_limit, 1.0-delta_ct_limit]]

    else:
        raise Exception("Your process is not in the list")

def vegas_integration(event_info, process, verbose=False, mode="XSec"):
    """Function for Integration of Various SM/BSM Differential
    Cross Sections.

    Available Processes ('Process'):
     -- 'Brem': Standard Model e + Z -> e + gamma + Z
     -- 'DarkBrem': BSM e + Z -> e + V + Z
     -- 'PairProd': SM gamma + Z -> e^+ + e^- + Z
     -- 'Comp': SM/BSM gamma + e -> e + gamma/V
     -- 'Ann': SM/BSM e^+ e^- -> gamma + gamma/V
     -- 'DarkAnn': e^+ e^- -> V

     ('Brem', 'DarkBrem', 'PairProd' calculated in
      small-angle approximation)

    Input parameters needed:
         EI: dictionary containing
           -- 'E_inc' (incident electron/positron energy)
           -- 'm_e' (electron mass)
           -- 'Z (Target Atomic Number)
           -- 'alpha_FS' (electro-weak fine-structure constant)
           -- 'mV' (Dark vector mass, assumed to be zero if absent)
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
        if not ("mV" in event_info.keys()):
            event_info["mV"] = 0.0
        if not ("Eg_min" in event_info.keys()):
            event_info["Eg_min"] = 0.001
        if not ("Ee_min" in event_info.keys()):
            event_info["Ee_min"] = 0.005
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
        
