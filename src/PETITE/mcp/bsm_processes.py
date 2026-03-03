import numpy as np
import vegas as vg
import random as rnd

from PETITE.physical_constants import m_electron, m_muon, m_proton, GeV, alpha_em

@vg.lbatchintegrand
class dsigma_mcp_annihilation_dcostheta:
    def __init__(self, event_info, ndim, batch_mode=False):
        """
        Annihilation of electron/positron pair into a pair of millicharged particles
        by s-channel photon exchange.
    
        """

        self.event_info = event_info
        self.ndim = ndim
        self.batch_mode = batch_mode

    def kinetic_energy_list(self, Te_max, Npts):
        threshold_energy = 2.0*self.event_info["m_chi"]**2/m_electron - m_electron
        Ee_max = Te_max + m_electron
        if threshold_energy > Ee_max:
            print("Updating Te_max to be higher than threshold energy for this process")
            Ee_max = threshold_energy * 100.0
        kinetic_energy_list = threshold_energy * (1.0 + np.geomspace(1e-4, (Ee_max - threshold_energy)/threshold_energy, Npts)) - m_electron
        return kinetic_energy_list

    def __call__(self, phase_space_par_list):
        if self.batch_mode:
            ct = phase_space_par_list[:, 0]
        else:
            ct = phase_space_par_list[0]

        Te = self.event_info["T_inc"]
        Ee = Te + m_electron
        s = 2.0 * m_electron * (Ee + m_electron)
        m_chi = self.event_info["m_chi"]

        if s < 4.0 * m_chi**2:
            if len(np.shape(phase_space_par_list)) == 1:
                return 0.0
            else:
                return np.zeros(shape=len(phase_space_par_list))
        
        inner_term = 1.0 + 4.0*(m_electron**2 + m_chi**2)/s + (1.0 - 4.0*m_electron**2/s)*(1.0 - 4.0*m_chi**2/s)*ct**2
        prefactor = np.pi*alpha_em**2/(2.0*s)*np.sqrt((s - 4.0*m_chi**2)/(s - 4.0*m_electron**2))
        dSigs = prefactor*inner_term
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

bsm_diff_xsection_options = {"mcpAnnihilation": dsigma_mcp_annihilation_dcostheta}

bsm_vegas_integrator_options = {"mcpAnnihilation":{"nitn":20, "nstrat":[1000]}}
      
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
    return [[-1.0,1.0]]

def bsm_vegas_integration(event_info, process, verbose=False, mode="XSec"):
    """Function for Integration of Various SM/BSM Differential
    Cross Sections.

    Available Processes ('Process'):
        -- 'mcpAnnihilation': Annihilation of electron/positron pair into a pair of millicharged particles
    Input parameters needed:
         EI: dictionary containing
           -- 'T_inc' (incident particle kinetic energy)
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
    if process in bsm_diff_xsection_options.keys():
        if not ("m_chi" in event_info.keys()):
            event_info["m_chi"] = 0.0
        if not ("Eg_min" in event_info.keys()):
            event_info["Eg_min"] = 0.001
        if not ("Te_min" in event_info.keys()):
            event_info["Te_min"] = 0.005

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
            print("Integrator set up: ", process, " -- m_chi ", event_info["m_chi"], " -- T_inc ", event_info["T_inc"])
        integ(f_integrand, **bsm_vegas_integrator_options[process])
        if verbose:
            print("Burn-in complete: ", process, " -- m_chi ", event_info["m_chi"], " -- T_inc ", event_info["T_inc"])
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
        
