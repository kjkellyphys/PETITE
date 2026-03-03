import numpy as np
import vegas as vg
import random as rnd

from PETITE.physical_constants import m_electron, m_muon, m_proton, GeV, alpha_em
from PETITE.dark_vector.radiative_return import transformed_lepton_luminosity_integrand

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
    Note: m_electron factors here are empirically determined and related to atomic form factors
          --they do *not* change when considering muon scattering off nucleus
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

class dsig_dx_dcostheta_dark_brem_exact_tree_level:
    def __init__(self, event_info, ndim, batch_mode=False):
        """
        Exact Tree-Level Dark Photon Bremsstrahlung
        ell (ep) + Z -> ell (epp) + V (w) + Z
        result it dsigma/dx/dcostheta where x=E_darkphoton/E_beam and theta is angle between beam and dark photon
        Input parameters needed:
            x0, x1, x2:  kinematic parameters related to energy of emitted vector, cosine of its angle
                            and the momentum transfer to the nucleus (precise relation depends on event_info['Method'] see below.
            mlepton (mass of incident lepton)
            mV (mass of dark photon)
            Ebeam (incident lepton energy)
            ZTarget (Target charge)
            ATarget (Target Atomic mass number)
            MTarget (Target mass)
        """
        self.event_info = event_info
        if "m_lepton" not in self.event_info.keys() and "process" in self.event_info.keys():
            if self.event_info["process"] == "DarkBrem":
                self.event_info["m_lepton"] = m_electron
            elif self.event_info["process"] == "DarkMuonBrem":
                self.event_info["m_lepton"] = m_muon
        self.ndim = ndim
        self.batch_mode = batch_mode

    def kinetic_energy_list(self, Te_max, Npts):
        minimum_KE = 1.25*self.event_info["mV"]
        return np.geomspace(minimum_KE, Te_max, Npts)

    def __call__(self, phase_space_par_list):
        if self.batch_mode:
            x0, x1, x2 = phase_space_par_list.T
        else:
            x0, x1, x2 = phase_space_par_list

        m_lepton = self.event_info["m_lepton"] 
        mV = self.event_info["mV"]
        T_beam = self.event_info["T_inc"]
        Ebeam = T_beam + m_lepton
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

        # Kevin 7/7/25: including fabs to avoid RuntimeWarning: invalid value encountered in sqrt
        # Situations where x * Ebeam < mV lead to negative values under the sqrt
        # This kinematic space is killed off later, so this is just a workaround
        k = np.sqrt(np.fabs((x * Ebeam) ** 2 - mV**2))
        p = np.sqrt(Ebeam**2 - m_lepton**2)
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
            * np.sqrt(Ebeam**2 + m_lepton**2)
            / (MTarget * (MTarget + 2 * Ebeam) + m_lepton**2)
        ) ** 2

        t = ttilde * tconv

        q0 = -t / (2 * MTarget)
        q = np.sqrt(t**2 / (4 * MTarget**2) + t)

        costhetaq = -(V**2 + q**2 + m_lepton**2 - (Ebeam + q0 - x * Ebeam) ** 2) / (2 * V * q)

        mVsq2mlepsq = mV**2 + 2 * m_lepton**2

        Am2 = (
            -8
            * MTarget
            * (4 * Ebeam**2 * MTarget - t * (2 * Ebeam + MTarget))
            * mVsq2mlepsq
        )
        A1 = 8 * MTarget**2 / utilde
        Am1 = (8 / utilde) * (
            MTarget**2
            * (
                2 * t * utilde
                + utilde**2
                + 4 * Ebeam**2 * (2 * (x - 1) * mVsq2mlepsq - t * ((x - 2) * x + 2))
                + 2 * t * (-(mV**2) + 2 * m_lepton**2 + t)
            )
            - 2 * Ebeam * MTarget * t * ((1 - x) * utilde + (x - 2) * (mVsq2mlepsq + t))
            + t**2 * (utilde - mV**2)
        )
        A0 = (8 / utilde**2) * (
            MTarget**2
            * (2 * t * utilde + (t - 4 * Ebeam**2 * (x - 1) ** 2) * mVsq2mlepsq)
            + 2 * Ebeam * MTarget * t * (utilde - (x - 1) * mVsq2mlepsq)
        )
        Y = -t + 2 * q0 * Ebeam - 2 * q * p * (p - k * costheta) * costhetaq / V
        #taking the absolute value of W for safer square rooting later -- Kevin 7/7/25
        #points where W < 0 are killed by allowed_kinematics
        W = (
            np.fabs(
                Y**2
                - 4 * q**2 * p**2 * k**2 * (1 - costheta**2) * (1 - costhetaq**2) / V**2
            )
        )

        # if W == 0.0:
        #     print("x, costheta, t = ", [x, costheta, t])
        #     print(
        #         "Y, q, p, k, costheta, costhetaq, V",
        #         [Y, q, p, k, costheta, costhetaq, V],
        #     )

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
        phi_integral = np.where(allowed_kinematics, 
                                (A0 + Y * A1 + Am1 / np.sqrt(W) + Y * Am2 / W**1.5) / (
                                8 * MTarget**2
                                ),
                                0.0)
        #phi_integral = (A0 + Y * A1 + Am1 / np.sqrt(W) + Y * Am2 / W**1.5) / (
        #    8 * MTarget**2
        #)

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

        return np.where(allowed_kinematics, ans * tconv * Jacobian, 0)


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
        self.batch_mode = batch_mode

    def kinetic_energy_list(self, Te_max, Npts):
        resonant_energy = (self.event_info["mV"]**2 - 2*m_electron**2)/(2*m_electron)
        Ee_max = Te_max + m_electron
        if resonant_energy > Ee_max:
            print("Updating Ee_max to be higher than resonant energy for this process")
            Ee_max = resonant_energy * 100.0
        kinetic_energy_list = resonant_energy * (1.0 + np.geomspace(1e-4, (Ee_max - resonant_energy)/resonant_energy, Npts)) - m_electron
        return kinetic_energy_list
        
    def __call__(self, phase_space_par_list):

        mV = self.event_info["mV"]
        Te = self.event_info["T_inc"]
        Ee = Te + m_electron
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
        if self.batch_mode:
            u0 = phase_space_par_list[:, 0]
        else:
            u0 = phase_space_par_list[0]
        #u0 = np.array(phase_space_par_list)
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
        Te = self.event_info["T_inc"]
        Ee = Te + m_electron
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

    def kinetic_energy_list(self, Te_max, Npts):
        resonant_energy = self.event_info["mV"]*(1.0 + self.event_info["mV"]/(2.0*m_electron))
        Eg_max_process = Te_max
        if resonant_energy > Eg_max_process:
            print("Updating Eg_max_process to be higher than resonant energy for this process")
            Eg_max_process = resonant_energy * 100.0
        kinetic_energy_list = resonant_energy * (1.0 + np.geomspace(1.0e-4, (Eg_max_process - resonant_energy)/resonant_energy, Npts))
        return kinetic_energy_list
    
    def __call__(self, phase_space_par_list):

        Eg = self.event_info["T_inc"]

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

bsm_diff_xsection_options = {"DarkBrem": dsig_dx_dcostheta_dark_brem_exact_tree_level,
                              "DarkMuonBrem": dsig_dx_dcostheta_dark_brem_exact_tree_level,
                              "DarkAnn": dsigma_radiative_return_du,
                              "DarkComp": dsigma_compton_dCT}

bsm_vegas_integrator_options = {"DarkBrem":{"nitn":20, "nstrat":[100, 100, 40]},
                                "DarkMuonBrem":{"nitn":20, "nstrat":[100, 100, 40]},
                                "DarkComp":{"nitn":20, "nstrat":[1000]},
                                "DarkAnn":{"nitn":10, "neval":10000}}
      
three_dim = {"DarkBrem", "DarkMuonBrem"}
one_dim = {"DarkAnn", "DarkComp"}

def bsm_integration_range(event_info, process):
    """Defines the integration range for the VEGAS integrator given a specific process
    Args:
        event_info - dictionary with parameter needed to evaluate the cross-section:
            T_inc - incident particle kinetic energy
            mV - vector mass
            Eg_min - minimum lab-frame energy (GeV) of outgoing photons
            Ee_min - minimum lab-frame energy (GeV) of outgoing electrons
            costheta_min - minimum lab-frame cosine of the angle between outgoing electrons
            xmin - optional minimum value of x (dimensionless variable related to energy) for DarkBrem if only interested in higher energy part of phase space
        process - process to be integrated over
    Returns:
        list of integration ranges for each dimension (note that number of dimensions depend on process)
    """
    TInc=event_info['T_inc']
    mV=event_info['mV']

    if process in three_dim:
        m_lepton = event_info['m_lepton'] if 'm_lepton' in event_info.keys() else m_electron
        EInc = TInc + m_lepton
        if 'costheta_min' in event_info:
            l1mct_max = np.log10(1 - event_info['costheta_min'])
        else:
            l1mct_max = np.log10(2.0)

        if 'xmin' in event_info: # DarkBrem
            xmin = event_info['xmin']
        else:
            xmin = 0.
        return [[max(xmin, mV/EInc), 1.-m_lepton/EInc],[-12.0, l1mct_max], [-20.0, 0.0]]
    elif process in one_dim:
        if process == "DarkComp":
            return [[-1., 1.0]]
        elif process == "DarkAnn":
            s = 2.0 * m_electron * ((TInc + m_electron) + m_electron)
            if s > mV**2:
                return [[0., 1.0]]
            else:
                return [[0.,0.]]
    else:
        raise Exception("Your process for BSM production is not in the list")

def bsm_vegas_integration(event_info, process, verbose=False, mode="XSec"):
    """Function for Integration of Various SM/BSM Differential
    Cross Sections.

    Available Processes ('Process'):
     -- 'Dark(Muon)Brem': BSM e + Z -> e + V + Z
     -- 'DarkComp': BSM gamma + e -> e + gamma/V
     -- 'DarkAnn': e^+ e^- -> V

    Input parameters needed:
         EI: dictionary containing
           -- 'T_inc' (incident particle kinetic energy)
           -- 'm_e' (electron mass)
           -- 'Z (Target Atomic Number)
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
    if process in bsm_diff_xsection_options.keys():
        if not ("mV" in event_info.keys()):
            event_info["mV"] = 0.0
        if not ("Eg_min" in event_info.keys()):
            event_info["Eg_min"] = 0.001
        if not ("Te_min" in event_info.keys()):
            event_info["Te_min"] = 0.005
        if not ("m_lepton" in event_info.keys()):
            event_info["m_lepton"] = m_electron
        if process == "DarkMuonBrem":
            event_info["m_lepton"] = m_muon
        igrange = bsm_integration_range(event_info, process)
        bsm_diff_xsec_func = bsm_diff_xsection_options[process]
    else:
        raise Exception(
            f"Could not find BSM process {process} in available processes: {bsm_diff_xsection_options.keys()}"
        )
    integ = vg.Integrator(igrange)
    f_integrand = bsm_diff_xsec_func(event_info=event_info, ndim=len(igrange), batch_mode=True)

    if mode == "Pickle" or mode == "XSec":
        if verbose:
            print("Integrator set up: ", process, " -- mV ", event_info["mV"], " -- T_inc ", event_info["T_inc"])
        integ(f_integrand, **bsm_vegas_integrator_options[process])
        if verbose:
            print("Burn-in complete: ", process, " -- mV ", event_info["mV"], " -- T_inc ", event_info["T_inc"])
        result = integ(f_integrand, **bsm_vegas_integrator_options[process])

        if verbose:
            print("Fully Integrated: ", process, result.mean)
        if mode == "Pickle":
            return integ
        else:
            return result.mean
    elif mode == "Sample" or mode == "UnweightedSample":
        integ(f_integrand, **bsm_vegas_integrator_options[process])
        result = integ(f_integrand, **bsm_vegas_integrator_options[process])

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
        
