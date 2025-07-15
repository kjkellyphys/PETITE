import numpy as np
import vegas as vg
import pickle 
import os

from scipy.interpolate import interp1d
from scipy.integrate import quad

from PETITE.moliere import get_scattered_momentum_fast, get_scattered_momentum_Bethe
from PETITE.particle import Particle, meson_twobody_branchingratios
from PETITE.atomic_annihilation import sigma_atomic
from PETITE.atomic_compton import sigma_atomic_comp
from PETITE.kinematics import (
    l_to_lV_fourvecs,
    compton_fourvecs,
    radiative_return_fourvecs,
    compton_fourvecs_boundelectron,
)
from PETITE.shower import Shower
import PETITE.all_processes as proc
from copy import deepcopy
from numpy.random import random as draw_U
from PETITE.physical_constants import (
    GeVsqcm2,
    cmtom,
    m_electron,
    m_muon,
    alpha_em,
)

class interpolate1d(interp1d):
    """Extend scipy interp1d to interpolate/extrapolate per axis in log space"""
    
    def __init__(self, x, y, *args, xspace='linear', yspace='linear', **kwargs):
        self.xspace = xspace
        self.yspace = yspace
        if self.xspace == 'log': x = np.log10(x + 1e-20)
        if self.yspace == 'log': y = np.log10(y + 1e-20)
        super().__init__(x, y, *args, **kwargs)
        
    def __call__(self, x, *args, **kwargs):
        if self.xspace == 'log': x = np.log10(x)
        if self.yspace == 'log':
            return 10**super().__call__(x, *args, **kwargs)
        else:
            return super().__call__(x, *args, **kwargs)
        

dark_process_codes = ["DarkBrem", "DarkAnn", "DarkComp", "TwoBody_BSMDecay", "DarkMuonBrem"]

dark_kinematic_function = {
    "DarkBrem": l_to_lV_fourvecs,
    "DarkAnn": radiative_return_fourvecs,
    "DarkComp": compton_fourvecs,
    "DarkMuonBrem": l_to_lV_fourvecs,
}
dark_diff_xsection_options = {
    "DarkComp": proc.dsigma_compton_dCT,
    "DarkBrem": proc.dsig_dx_dcostheta_dark_brem_exact_tree_level,
    "DarkAnn": proc.dsigma_radiative_return_du,
    "DarkMuonBrem": proc.dsig_dx_dcostheta_dark_brem_exact_tree_level,
}
dimensionalities_dark = {"DarkComp": 1, "DarkBrem": 3, "DarkAnn": 1, "DarkMuonBrem": 3}

class DarkShower(Shower):
    """ A class to reprocess an existing EM shower to generate dark photons
    """

    def __init__(self,
                 dict_dir,
                 target_material,
                 min_energy,
                 mV_in_GeV ,
                 mode="exact",
                 maxF_fudge_global=1,
                 max_n_integrators=int(1e4),
                 kinetic_mixing=1.0,
                 Zeff=29.508,
                 bound_electron=True,
                 g_e=None,
                 active_processes=None,
                 fast_MCS_mode=True ,
                 rescale_MCS=1,
    ):
        super().__init__(dict_dir, target_material, min_energy)
        """Initializes the dark shower object.
        Args:
            dict_dir: directory containing the pre-computed MC samples of various shower processes
            target_material: string label of the homogeneous material through which
            particles propagate (available materials are the dict keys of
            Z, A, rho, etc)
            min_energy: minimum particle energy in GeV at which the particle
            finishes its propagation through the target
            mV_in_GeV: vector mass in GeV
            mode: determines whether mV is set to MV_in_GeV or the nearest value for which integrators have been trained
        """

        self.active_processes = active_processes
        if self.active_processes is None:
            self.active_processes = dark_process_codes

        self.set_dark_dict_dir(dict_dir)
        self.set_target_material(target_material)
        self.min_energy = min_energy
        self.kinetic_mixing = kinetic_mixing
        self.bound_electron = bound_electron
        self.Zeff = Zeff

        self.g_e = g_e
        if self.g_e is None:
            self.g_e = self.kinetic_mixing * np.sqrt(4 * np.pi * alpha_em)

        self.set_material_properties()
        self.set_n_targets()
        self.set_mV_list(dict_dir)
        self.set_mV(mV_in_GeV, mode)

        self.set_dark_cross_sections()
        self.set_DarkAnnXSec()
        self.set_DarkCompXSec()
        self.set_dark_NSigmas()
        self.set_weight_arrays()
        self.set_drate_dE()
        self.set_dark_samples()
        self.set_MCS_momentum(fast_MCS_mode)
        self.set_MCS_rescale_factor(rescale_MCS)

        self._maxF_fudge_global = maxF_fudge_global
        self._max_n_integrators = max_n_integrators

    def set_MCS_rescale_factor(self, rescale_MCS):
        self._MCS_rescale_factor = rescale_MCS

    def set_MCS_momentum(self, fast_MCS_mode):
        if fast_MCS_mode:
            self._get_MCS_p = get_scattered_momentum_fast
        else:
            self._get_MCS_p = get_scattered_momentum_Bethe
    def set_dark_dict_dir(self, value):
        """Set the directory containing pre-simulated MC events for processes involing target nuclei"""
        self._dark_dict_dir = value

    def get_dark_dict_dir(self):
        """Get the top level directory containing pre-computed MC pickles""" 
        return self._dark_dict_dir
   

    def set_mV_list(self,dict_dir):
        sample_file=open(dict_dir + "dark_maps.pkl", 'rb')
        outer_dict=pickle.load(sample_file)
        sample_file.close()

        mass_list=list(outer_dict.keys())

        self._mV_list=mass_list


    def closest_lesser_value(self, input_list, input_value):
        arr = np.asarray(input_list)
        index = (np.abs(arr - input_value)).argmin()

        if arr[index]<=input_value:
            return(arr[index])
        else:
            return(arr[index-1])

    def set_mV(self, value, mode):
        """Set MVStr to value and extract the corresponding numerical mass of the dark photon"""
        if mode=='exact':
            self._mV= self.closest_lesser_value(self._mV_list, value)
            self._mV_estimator = self._mV
        elif mode=='approx':
            self._mV= value
            self._mV_estimator = self.closest_lesser_value(self._mV_list, value)
        else:
            raise Exception("Mode not valid. Chose exact or approx.")

    
    def get_mV(self):
        """Get the numerical value of the dark vector mass"""
        return self._mV
    
    def load_dark_sample(self, dict_dir, process): 
        sample_file=open(dict_dir + "dark_maps.pkl", 'rb')
        outer_dict=pickle.load(sample_file)
        sample_file.close()

        sample_dict=outer_dict[self._mV_estimator]
        if process in sample_dict.keys():
            return(sample_dict[process])
        else:
            print(process)
            raise Exception("Process String does not match library")
        

    def set_dark_samples(self):
        self._loaded_dark_samples={}
        #for process in dark_diff_xsection_options.keys():
        for process in self.active_processes:
            self._loaded_dark_samples[process]= \
                self.load_dark_sample(self._dict_dir, process)
            
    def load_dark_cross_section(self, dict_dir, process, target_material):
        dark_cross_section_file=open( dict_dir + "dark_xsec.pkl", 'rb')
        outer_dict=pickle.load(dark_cross_section_file)
        dark_cross_section_file.close()

        dark_cross_section_dict=outer_dict[self._mV_estimator]

        if process not in dark_cross_section_dict:
            raise Exception("Process String does not match library")
        
        if target_material in dark_cross_section_dict[process]:
            return(dark_cross_section_dict[process][target_material])

        else:
            raise Exception("Target Material is not in library")

    def set_dark_cross_sections(self):
        """Loads the pre-computed cross-sections for various shower processes 
        and extracts the minimum/maximum values of initial energies
        """

        # These contain only the cross sections for the chosen target material
        self._dark_brem_cross_section = self.load_dark_cross_section(self._dict_dir, 'DarkBrem', self._target_material)
        while self._dark_brem_cross_section[0][1] == 0.0:
            self._dark_brem_cross_section = self._dark_brem_cross_section[1:] #remove first element if it has zero cross-section
        self._dark_annihilation_cross_section  = self.load_dark_cross_section(self._dict_dir, 'DarkAnn', self._target_material) 
        self._dark_compton_cross_section = self.load_dark_cross_section(self._dict_dir, 'DarkComp', self._target_material) 
        self._dark_muon_brem_cross_section = self.load_dark_cross_section(self._dict_dir, 'DarkMuonBrem', self._target_material)
        while self._dark_muon_brem_cross_section[0][1] == 0.0:
            self._dark_muon_brem_cross_section = self._dark_muon_brem_cross_section[1:] #remove first element if it has zero cross-section

        self._resonant_annihilation_energy = (self._mV**2-2*m_electron**2)/(2*m_electron)
        self._compton_threshold_energy = self._mV**2/(2*m_electron) + self._mV
        self._minimum_calculable_dark_energy = {11:{"DarkBrem":self._dark_brem_cross_section[0][0]},
                                                -11:{"DarkBrem":self._dark_brem_cross_section[0][0], "DarkAnn":self._resonant_annihilation_energy},
                                                22:{"DarkComp":self._dark_compton_cross_section[0][0]},
                                                111:{"TwoBody_BSMDecay":-1},
                                                13:{"DarkMuonBrem":self._dark_muon_brem_cross_section[0][0]},
                                                -13:{"DarkMuonBrem":self._dark_muon_brem_cross_section[0][0]}}
        
        if self.bound_electron:
            self._minimum_calculable_dark_energy[-11]["DarkAnn"] = self._resonant_annihilation_energy/1000.0
            self._minimum_calculable_dark_energy[22]["DarkComp"] = self._compton_threshold_energy/1000.0

    def get_DarkBremXSec(self):
        """ Returns array of [energy,cross-section] values for brem """ 
        return self._dark_brem_cross_section 
    def get_DarkMuonBremXSec(self):
        """ Returns array of [energy,cross-section] values for muon brem """ 
        return self._dark_muon_brem_cross_section

    def set_DarkAnnXSec(self):
        """ Returns array of [energy,cross-section] values for e+e- annihilation with a bound electron """ 
        E_min = np.max([self._minimum_calculable_energy[-11], 0.001*self._resonant_annihilation_energy])       
        E_max = self._dark_brem_cross_section[-1][0]
        energy_list = np.logspace(np.log10(E_min), np.log10(E_max), 200)
        #energy_list = np.logspace(np.log10(0.0016), np.log10(1000), 200)
        #ER0 = ((self._mV**2 - 2*m_electron**2)/(2*m_electron))
        #Emax = np.max([100.0, 100*ER0])
        #energy_list = ER0*(1 + np.logspace(-4, np.log10((Emax - ER0)/ER0), 100))
        xsec = [sigma_atomic(e, self._mV, self.Zeff) for e in energy_list]
        self._dark_annihilation_cross_section_bound = np.column_stack((energy_list, xsec))
    def get_DarkAnnXSec(self):
        """ Returns array of [energy,cross-section] values for e+e- annihilation """ 
        if self.bound_electron:
            return self._dark_annihilation_cross_section_bound
        elif self.bound_electron == False:
            return self._dark_annihilation_cross_section
        else:
            raise Exception("Bound must be True or False")

    def set_DarkCompXSec(self):
        """ Returns array of [energy,cross-section] values for Compton scattering with a bound electron""" 
        E_min = np.max([self._minimum_calculable_energy[22], 0.001*self._compton_threshold_energy])       
        E_max = self._dark_compton_cross_section[-1][0]
        energy_list = np.logspace(np.log10(E_min), np.log10(E_max), 200)
        #energy_list = np.logspace(np.log10(0.0016), np.log10(1000), 200)
        #ER0 = ((self._mV**2 - 2*m_electron**2)/(2*m_electron))
        #Emax = np.max([100.0, 100*ER0])
        #energy_list = ER0*(1 + np.logspace(-4, np.log10((Emax - ER0)/ER0), 100))
        xsec = [sigma_atomic_comp(e, self._mV, self.Zeff) for e in energy_list]
        self._dark_compton_cross_section_bound = np.column_stack((energy_list, xsec))
    def get_DarkCompXSec(self):
        """ Returns array of [energy,cross-section] values for Compton """ 
        if self.bound_electron:
            return self._dark_compton_cross_section_bound
        elif self.bound_electron == False:
            return self._dark_compton_cross_section
        else:
            raise Exception("Bound must be True or False")

    def set_dark_NSigmas(self):
        """Constructs interpolations of n_T sigma (in 1/cm) as a functon of 
        incoming particle energy for each process
        """
        DBS, DAnnS, DCS = self.get_DarkBremXSec(), self.get_DarkAnnXSec(), self.get_DarkCompXSec()
        nZ, ne = self.get_n_targets()

        self._NSigmaDarkBrem = interpolate1d(np.transpose(DBS)[0], nZ*GeVsqcm2*np.transpose(DBS)[1], xspace='log', yspace='log', fill_value=-20.0, bounds_error=False)
        if self.bound_electron:
            self._NSigmaDarkAnn = interpolate1d(np.transpose(DAnnS)[0], ne*GeVsqcm2*np.transpose(DAnnS)[1], xspace='log', yspace='log', fill_value=-20.0, bounds_error=False)
        elif self.bound_electron == False:
            self._NSigmaDarkAnn = interpolate1d(np.transpose(DAnnS)[0] - self._resonant_annihilation_energy, ne*GeVsqcm2*np.transpose(DAnnS)[1], xspace='log', yspace='log', fill_value=-20.0, bounds_error=False)
        self._NSigmaDarkComp = interpolate1d(np.transpose(DCS)[0], ne*GeVsqcm2*np.transpose(DCS)[1], xspace='log', yspace='log', fill_value=-20.0, bounds_error=False)

        DMBSD = self.get_DarkMuonBremXSec()
        self._NSigmaDarkMuonBrem = interpolate1d(np.transpose(DMBSD)[0], nZ*GeVsqcm2*np.transpose(DMBSD)[1], xspace='log', yspace='log', fill_value=-20.0, bounds_error=False)

    def _dark_ann_integrand(self, E, Ei):
        dEdxT_GeVpercm = self.get_material_properties()[3]*(0.1)*cmtom #Converting MeV/cm to GeV/m to GeV/cm
        if self.bound_electron:
            return self._NSigmaDarkAnn(E)/dEdxT_GeVpercm*self._positron_exponential_factor(E, Ei)
        elif self.bound_electron == False:
            return self._NSigmaDarkAnn(E - self._resonant_annihilation_energy)/dEdxT_GeVpercm*self._positron_exponential_factor(E, Ei)
    def _dark_brem_integrand_elec(self, E, Ei):
        dEdxT_GeVpercm = self.get_material_properties()[3]*(0.1)*cmtom #Converting MeV/cm to GeV/m to GeV/cm
        nsigma = self._NSigmaDarkBrem(E)
        if nsigma < 1.0e-18:
            return 0.0
        return nsigma/dEdxT_GeVpercm*self._electron_exponential_factor(E, Ei)
    def _dark_brem_integrand_positron(self, E, Ei):
        dEdxT_GeVpercm = self.get_material_properties()[3]*(0.1)*cmtom #Converting MeV/cm to GeV/m to GeV/cm
        nsigma = self._NSigmaDarkBrem(E)
        if nsigma < 1.0e-18:
            return 0.0
        return nsigma/dEdxT_GeVpercm*self._positron_exponential_factor(E, Ei)

    def _dark_muon_brem_integrand(self, E, Ei):
        dEdxT_GeVpercm = self.get_material_properties()[3]*(0.1)*cmtom #Converting MeV/cm to GeV/m to GeV/cm
        nsigma = self._NSigmaDarkMuonBrem(E)
        if nsigma < 1.0e-18:
            return 0.0
        return nsigma/dEdxT_GeVpercm*self._muon_exponential_factor(E, Ei)

    def construct_brem_weight_array(self):
        DBS = self.get_DarkBremXSec()
        initial_energies = np.transpose(DBS)[0]

        dEdxT_GeVperm = self.get_material_properties()[3]*(0.1)
        mfp_electron_EI = np.array([self.get_mfp([11, Ei]) for Ei in initial_energies])
        mfp_positron_EI = np.array([self.get_mfp([-11, Ei]) for Ei in initial_energies])
        energy_loss_ten_mfp_electron = 10*mfp_electron_EI*dEdxT_GeVperm
        energy_loss_ten_mfp_positron = 10*mfp_positron_EI*dEdxT_GeVperm
        minimum_energy_0 = initial_energies[0]
        energy_break_electrons = np.ones(len(initial_energies))*minimum_energy_0
        energy_break_positrons = np.ones(len(initial_energies))*minimum_energy_0
        for i in range(len(energy_break_electrons)):
            Eb_electron = initial_energies[i] - energy_loss_ten_mfp_electron[i]
            Eb_positron = initial_energies[i] - energy_loss_ten_mfp_positron[i]
            if Eb_electron > minimum_energy_0:
                energy_break_electrons[i] = Eb_electron
            if Eb_positron > minimum_energy_0:
                energy_break_positrons[i] = Eb_positron

        brem_elec_weight_array =        np.array([quad(self._dark_brem_integrand_elec, minimum_energy_0, energy_break_electrons[i], args=(initial_energies[i]), full_output=1)[0] + 
                                                  quad(self._dark_brem_integrand_elec, energy_break_electrons[i], initial_energies[i], args=(initial_energies[i]), full_output=1)[0] for i in range(len(initial_energies))])
        brem_positron_weight_array =    np.array([quad(self._dark_brem_integrand_positron, minimum_energy_0, energy_break_positrons[i], args=(initial_energies[i]), full_output=1)[0] +
                                                  quad(self._dark_brem_integrand_positron, energy_break_positrons[i], initial_energies[i], args=(initial_energies[i]), full_output=1)[0] for i in range(len(initial_energies))])

        return [initial_energies, brem_elec_weight_array, brem_positron_weight_array]

    def construct_muon_brem_weight_array(self):
        DMBSD = self.get_DarkMuonBremXSec()
        initial_energies = np.transpose(DMBSD)[0]

        dEdxT_GeVperm = self.get_material_properties()[3]*(0.1)
        mfp_muon_EI = np.array([self.get_mfp([13, Ei]) for Ei in initial_energies])
        energy_loss_ten_mfp_muon = 10*mfp_muon_EI*dEdxT_GeVperm
        minimum_energy_0 = initial_energies[0]
        energy_break_muons = np.ones(len(initial_energies))*minimum_energy_0
        for i in range(len(energy_break_muons)):
            Eb_muon = initial_energies[i] - energy_loss_ten_mfp_muon[i]
            if Eb_muon > minimum_energy_0:
                energy_break_muons[i] = Eb_muon

        dark_brem_weight_array = np.array([quad(self._dark_muon_brem_integrand, minimum_energy_0, energy_break_muons[i], args=(initial_energies[i]), full_output=1)[0] +
                                                quad(self._dark_muon_brem_integrand, energy_break_muons[i], initial_energies[i], args=(initial_energies[i]), full_output=1)[0] for i in range(len(initial_energies))])
        return [initial_energies, dark_brem_weight_array]

    def construct_annihilation_weight_array(self):
        DAnnS = self.get_DarkAnnXSec()
        initial_energies = np.transpose(DAnnS)[0]

        dEdxT_GeVperm = self.get_material_properties()[3]*(0.1)
        mfp_positron_EI = np.array([self.get_mfp([-11, Ei]) for Ei in initial_energies])
        energy_loss_ten_mfp_positron = 10*mfp_positron_EI*dEdxT_GeVperm
        minimum_energy_0 = initial_energies[0]
        energy_break_positrons = np.ones(len(initial_energies))*minimum_energy_0
        for i in range(len(energy_break_positrons)):
            Eb_positron = initial_energies[i] - energy_loss_ten_mfp_positron[i]
            if Eb_positron > minimum_energy_0:
                energy_break_positrons[i] = Eb_positron

        annihilation_weight_array = np.array([quad(self._dark_ann_integrand, minimum_energy_0, energy_break_positrons[i], args=(initial_energies[i]), full_output=1)[0] +
                                              quad(self._dark_ann_integrand, energy_break_positrons[i], initial_energies[i], args=(initial_energies[i]), full_output=1)[0] for i in range(len(initial_energies))])
        
        return [initial_energies, annihilation_weight_array]

    def set_weight_arrays(self):
        dict_dir = self.get_dark_dict_dir()
        weights_file_name = dict_dir + "dark_weights.pkl"
        #check if file named weights_file_name exists
        files_set = False
        outer_dict = {}
        if os.path.exists(weights_file_name):
            sample_file=open(weights_file_name, 'rb')
            outer_dict=pickle.load(sample_file)
            sample_file.close()
            if self._mV_estimator in outer_dict.keys():
                if self._target_material in outer_dict[self._mV_estimator].keys():
                    initial_energies_brem_elec, brem_elec_weight_array = np.transpose(outer_dict[self._mV_estimator][self._target_material]['brem_elec_weights'])
                    initial_energies_brem_positron, brem_positron_weight_array = np.transpose(outer_dict[self._mV_estimator][self._target_material]['brem_positron_weights'])
                    initial_energies_muon_brem, muon_brem_weight_array = np.transpose(outer_dict[self._mV_estimator][self._target_material]['muon_brem_weights'])
                    if self.bound_electron is False and 'annihilation_weights' in outer_dict[self._mV_estimator][self._target_material].keys():
                        initial_energies_annihilation, annihilation_weight_array = np.transpose(outer_dict[self._mV_estimator][self._target_material]['annihilation_weights'])
                    elif self.bound_electron is False and 'annihilation_weights' not in outer_dict[self._mV_estimator][self._target_material].keys():
                        initial_energies_annihilation, annihilation_weight_array = self.construct_annihilation_weight_array()
                        outer_dict[self._mV_estimator][self._target_material]['annihilation_weights'] = np.transpose([initial_energies_annihilation, annihilation_weight_array])
                        sample_file = open(weights_file_name, 'wb')
                        pickle.dump(outer_dict, sample_file)
                        sample_file.close()
                    files_set = True
        if files_set == False:
            print("Weights not previously calculated, calculating now...")
            initial_energies_brem_elec, brem_elec_weight_array, brem_positron_weight_array = self.construct_brem_weight_array()
            initial_energies_brem_positron = initial_energies_brem_elec
            initial_energies_annihilation, annihilation_weight_array = self.construct_annihilation_weight_array()
            initial_energies_muon_brem, muon_brem_weight_array = self.construct_muon_brem_weight_array()
            if self._mV_estimator not in outer_dict.keys():
                outer_dict[self._mV_estimator] = {}
            if self.bound_electron:
                outer_dict[self._mV_estimator][self._target_material] = {'brem_elec_weights':np.transpose([initial_energies_brem_elec, brem_elec_weight_array]),
                                                                        'brem_positron_weights':np.transpose([initial_energies_brem_positron, brem_positron_weight_array]),
                                                                        'muon_brem_weights': np.transpose([initial_energies_muon_brem, muon_brem_weight_array])}
            else: #only save the annihilation case if dealing with a free-electron calculation
                outer_dict[self._mV_estimator][self._target_material] = {'brem_elec_weights':np.transpose([initial_energies_brem_elec, brem_elec_weight_array]),
                                                                        'brem_positron_weights':np.transpose([initial_energies_brem_positron, brem_positron_weight_array]),
                                                                        'annihilation_weights':np.transpose([initial_energies_annihilation, annihilation_weight_array]),
                                                                        'muon_brem_weights':np.transpose([initial_energies_muon_brem, muon_brem_weight_array])}
            
            sample_file=open(weights_file_name, 'wb')
            pickle.dump(outer_dict, sample_file)
            sample_file.close()

        self._brem_elec_numerical_weight = interp1d(initial_energies_brem_elec, brem_elec_weight_array, fill_value=0.0, bounds_error=False)
        self._brem_positron_numerical_weight = interp1d(initial_energies_brem_positron, brem_positron_weight_array, fill_value=0.0, bounds_error=False)
        self._muon_brem_numerical_weight = interp1d(initial_energies_muon_brem, muon_brem_weight_array, fill_value=0.0, bounds_error=False)
        if self.bound_electron: #explicitly calculate annihilation weight arrays if dealing with a bound-electron case
            initial_energies_annihilation, annihilation_weight_array = self.construct_annihilation_weight_array()
        self._annihilation_numerical_weight = interp1d(initial_energies_annihilation, annihilation_weight_array, fill_value=0.0, bounds_error=False)            
    
    def _d_rate_d_E_elec_brem(self, Ei):
        dEdxT_GeVperm = self.get_material_properties()[3]*(0.1)
        mfp_electron_EI = self.get_mfp([11, Ei])
        energy_loss_ten_mfp = 10*mfp_electron_EI*dEdxT_GeVperm
        energy_array = np.linspace(np.max([Ei-energy_loss_ten_mfp, self.get_DarkBremXSec()[0][0]]), Ei, 11)
        energy_center_array = np.array([(energy_array[i] + energy_array[i+1])/2. for i in range(len(energy_array)-1)])

        brem_elec_weights = np.array([quad(self._dark_brem_integrand_elec, energy_array[i], energy_array[i+1], args=(Ei), full_output=1)[0] for i in range(len(energy_array)-1)])
        return np.transpose([energy_center_array, brem_elec_weights])
    
    def _d_rate_d_E_positron_brem(self, Ei):
        dEdxT_GeVperm = self.get_material_properties()[3]*(0.1)
        mfp_positron_EI = self.get_mfp([-11, Ei])
        energy_loss_ten_mfp = 10*mfp_positron_EI*dEdxT_GeVperm
        energy_array = np.linspace(np.max([Ei-energy_loss_ten_mfp, self.get_DarkBremXSec()[0][0]]), Ei, 11)
        energy_center_array = np.array([(energy_array[i] + energy_array[i+1])/2. for i in range(len(energy_array)-1)])

        brem_positron_weights = np.array([quad(self._dark_brem_integrand_positron, energy_array[i], energy_array[i+1], args=(Ei), full_output=1)[0] for i in range(len(energy_array)-1)])
        return np.transpose([energy_center_array, brem_positron_weights])

    def _d_rate_d_E_muon_brem(self, Ei):
        dEdxT_GeVperm = self.get_material_properties()[3]*(0.1)
        mfp_muon_EI = self.get_mfp([13, Ei])
        energy_loss_ten_mfp = 10*mfp_muon_EI*dEdxT_GeVperm
        energy_array = np.linspace(np.max([Ei-energy_loss_ten_mfp, self.get_DarkMuonBremXSec()[0][0]]), Ei, 11)
        energy_center_array = np.array([(energy_array[i] + energy_array[i+1])/2. for i in range(len(energy_array)-1)])

        muon_brem_weights = np.array([quad(self._dark_muon_brem_integrand, energy_array[i], energy_array[i+1], args=(Ei), full_output=1)[0] for i in range(len(energy_array)-1)])
        return np.transpose([energy_center_array, muon_brem_weights])


    def _d_rate_d_E_positron_ann_bound(self, Ei):
        dEdxT_GeVperm = self.get_material_properties()[3]*(0.1)
        mfp_positron_EI = self.get_mfp([-11, Ei])
        energy_loss_ten_mfp = 10*mfp_positron_EI*dEdxT_GeVperm
        energy_array = np.linspace(np.max([Ei-energy_loss_ten_mfp, self.get_DarkAnnXSec()[0][0]]), Ei, 11)
        energy_center_array = np.array([(energy_array[i] + energy_array[i+1])/2. for i in range(len(energy_array)-1)])

        darkann_weights = np.array([quad(self._dark_ann_integrand, energy_array[i], energy_array[i+1], args=(Ei), full_output=1)[0] for i in range(len(energy_array)-1)])
        return np.transpose([energy_center_array, darkann_weights])

    def _d_rate_d_E_positron_ann(self, Ei):
        if Ei < self._resonant_annihilation_energy:
            return [[[0., 0.]], [0., 1.]]
        minimum_saved_energy = self.get_DarkAnnXSec()[0][0]

        dEdxT_GeVperm = self.get_material_properties()[3]*(0.1)
        mfp_positron_EI = self.get_mfp([-11, Ei])
        energy_loss_ten_mfp = 10*mfp_positron_EI*dEdxT_GeVperm
        energy_array = np.linspace(np.max([Ei-energy_loss_ten_mfp, minimum_saved_energy]), Ei, 11)
        energy_center_array = np.array([(energy_array[i] + energy_array[i+1])/2. for i in range(len(energy_array)-1)])

        darkann_weights = np.array([quad(self._dark_ann_integrand, energy_array[i], energy_array[i+1], args=(Ei), full_output=1)[0] for i in range(len(energy_array)-1)])

        resonant_bin_center = 0.5*(self._resonant_annihilation_energy + np.min([minimum_saved_energy, Ei]))
        sMAX = 2*(m_electron*np.min([minimum_saved_energy, Ei]) + m_electron**2)
        beta = (2.*alpha_em/np.pi) * (np.log(sMAX/m_electron**2) - 1.)
        dEdxT_GeVpercm = self.get_material_properties()[3]*(0.1)*cmtom #Converting MeV/cm to GeV/m to GeV/cm
        weight_analytic = (1/dEdxT_GeVpercm)*(2*np.pi**2*alpha_em/m_electron)*(self.get_n_targets()[1])*GeVsqcm2*(sMAX - self._mV**2)**beta*self._positron_exponential_factor(self._resonant_annihilation_energy, Ei)
        darkann_weights = np.concatenate([[weight_analytic], darkann_weights])
        energy_center_array = np.concatenate([[resonant_bin_center], energy_center_array])
        energy_array = np.concatenate([[self._resonant_annihilation_energy, np.min([minimum_saved_energy, Ei])], energy_array])

        return [np.transpose([energy_center_array, darkann_weights]), energy_array]
    
    def _d_rate_d_E_elec_brem_array(self):
        Ei_samp = np.transpose(self.get_DarkBremXSec())[0]
        return {Ei:self._d_rate_d_E_elec_brem(Ei) for Ei in Ei_samp}
    def _d_rate_d_E_muon_brem_array(self):
        Ei_samp = np.transpose(self.get_DarkMuonBremXSec())[0]
        return {Ei:self._d_rate_d_E_muon_brem(Ei) for Ei in Ei_samp}
    def _d_rate_d_E_positron_brem_array(self):
        Ei_samp = np.transpose(self.get_DarkBremXSec())[0]
        return {Ei:self._d_rate_d_E_positron_brem(Ei) for Ei in Ei_samp}
    def _d_rate_d_E_positron_ann_bound_array(self):
        Ei_samp = np.transpose(self.get_DarkAnnXSec())[0]
        return {Ei:self._d_rate_d_E_positron_ann_bound(Ei) for Ei in Ei_samp}
    def _d_rate_d_E_positron_ann_array(self):
        Ei_samp = np.transpose(self.get_DarkAnnXSec())[0]
        return {Ei:self._d_rate_d_E_positron_ann(Ei)[0] for Ei in Ei_samp}

    def set_drate_dE(self):
        dict_dir = self.get_dark_dict_dir()
        drate_file_name = dict_dir + "dark_drate.pkl"

        files_set = False
        outer_dict = {}
        if os.path.exists(drate_file_name):
            sample_file=open(drate_file_name, 'rb')
            outer_dict=pickle.load(sample_file)
            sample_file.close()
            if self._mV_estimator in outer_dict.keys():
                if self._target_material in outer_dict[self._mV_estimator].keys():
                    d_rate_dict_elec_brem = outer_dict[self._mV_estimator][self._target_material]['brem_elec_drate']
                    d_rate_dict_positron_brem = outer_dict[self._mV_estimator][self._target_material]['brem_positron_drate']
                    d_rate_dict_muon_brem = outer_dict[self._mV_estimator][self._target_material]['muon_brem_drate']
                    if self.bound_electron is False and 'annihilation_drate' in outer_dict[self._mV_estimator['self._target_material']].keys():
                        d_rate_dict_positron_ann = outer_dict[self._mV_estimator][self._target_material]['annihilation_drate']
                    elif self.bound_electron is False and 'annihilation_drate' not in outer_dict[self._mV_estimator][self._target_material].keys():
                        d_rate_dict_positron_ann = self._d_rate_d_E_positron_ann_array()
                        outer_dict[self._mV_estimator][self._target_material]['annihilation_drate'] = d_rate_dict_positron_ann
                        sample_file = open(drate_file_name, 'wb')
                        pickle.dump(outer_dict, sample_file)
                        sample_file.close()
                    elif self.bound_electron:
                        d_rate_dict_positron_ann = self._d_rate_d_E_positron_ann_bound_array()
                    elif self.bound_electron == False:
                        d_rate_dict_positron_ann = outer_dict[self._mV_estimator][self._target_material]['annihilation_drate']
                    files_set = True
        if files_set == False:
            print("dRate not previously calculated, calculating now...")
            d_rate_dict_elec_brem = self._d_rate_d_E_elec_brem_array()
            d_rate_dict_positron_brem = self._d_rate_d_E_positron_brem_array()
            d_rate_dict_muon_brem = self._d_rate_d_E_muon_brem_array()
            if self.bound_electron:
                d_rate_dict_positron_ann = self._d_rate_d_E_positron_ann_bound_array()
            elif self.bound_electron == False:
                d_rate_dict_positron_ann = self._d_rate_d_E_positron_ann_array()

            if self._mV_estimator not in outer_dict.keys():
                outer_dict[self._mV_estimator] = {}
            
            if self.bound_electron:
                outer_dict[self._mV_estimator][self._target_material] = {'brem_elec_drate':d_rate_dict_elec_brem,
                                                                        'brem_positron_drate':d_rate_dict_positron_brem,
                                                                        'muon_brem_drate':d_rate_dict_muon_brem}
            else: #only save the annihilation case if dealing with a free-electron calculation
                outer_dict[self._mV_estimator][self._target_material] = {'brem_elec_drate':d_rate_dict_elec_brem,
                                                                         'brem_positron_drate':d_rate_dict_positron_brem,
                                                                         'muon_brem_drate':d_rate_dict_muon_brem,
                                                                          'annihilation_drate':d_rate_dict_positron_ann}

            sample_file=open(drate_file_name, 'wb')
            pickle.dump(outer_dict, sample_file)
            sample_file.close()

        self._d_rate_dict_elec_brem = d_rate_dict_elec_brem
        self._d_rate_dict_positron_brem = d_rate_dict_positron_brem
        self._d_rate_dict_positron_ann = d_rate_dict_positron_ann
        self._d_rate_dict_muon_brem = d_rate_dict_muon_brem

    def GetBSMWeights(self, particle, process):
        if isinstance(particle, list) or isinstance(particle, np.ndarray):
            PID, energy_initial = particle
        else:
            PID, energy_initial = particle.get_ids()["PID"], particle.get_p0()[0]

        if PID not in [-11, 11, 13, -13, 22, 111, 221, 331]:
            return 0.0

        if process not in (self._minimum_calculable_dark_energy[PID]).keys():
            return 0.0
        if energy_initial < self._minimum_calculable_dark_energy[PID][process]:
            return 0.0
        if PID == 22:
            if process != "DarkComp" or energy_initial < self._minimum_calculable_energy[22]:
                return 0.0
            return (self.g_e**2/(4*np.pi*alpha_em))*self._NSigmaDarkComp(energy_initial)/(self._NSigmaPP(energy_initial) + self._NSigmaComp(energy_initial))
        if process == "DarkBrem":
            if np.abs(PID) != 11:
                return 0.0
            if PID == 11:
                return (self.g_e**2/(4*np.pi*alpha_em))*self._brem_elec_numerical_weight(energy_initial)
            else:
                return (self.g_e**2/(4*np.pi*alpha_em))*self._brem_positron_numerical_weight(energy_initial)
        if PID == -11 and process == "DarkAnn":
            if self.bound_electron:
                return (self.g_e**2/(4*np.pi*alpha_em))*self._annihilation_numerical_weight(energy_initial)
            else:
                minimum_saved_energy = self.get_DarkAnnXSec()[0][0]
                sMAX = 2*(m_electron*np.min([minimum_saved_energy, energy_initial]) + m_electron**2)
                beta = (2.*alpha_em/np.pi) * (np.log(sMAX/m_electron**2) - 1.)
                dEdxT_GeVpercm = self.get_material_properties()[3]*(0.1)*cmtom #Converting MeV/cm to GeV/m to GeV/cm
                weight_analytic = (1/dEdxT_GeVpercm)*(2*np.pi**2*alpha_em/m_electron)*(self.get_n_targets()[1])*GeVsqcm2*(sMAX - self._mV**2)**beta*self._positron_exponential_factor(self._resonant_annihilation_energy, energy_initial)
                if energy_initial > minimum_saved_energy:
                    weight_numerical = self._annihilation_numerical_weight(energy_initial)
                else:
                    weight_numerical = 0.0
                return (self.g_e**2/(4*np.pi*alpha_em))*(weight_numerical + weight_analytic)
        if PID == 111 or PID == 221 or PID == 331:
            if process == "TwoBody_BSMDecay":
                mass_ratio = self._mV/particle.get_ids()["mass"]
                if mass_ratio >= 1.0:
                    return 0.0
                return 2*(self.kinetic_mixing)**2*(1.0 - mass_ratio**2)**3*meson_twobody_branchingratios[particle.get_ids()["PID"]]
            else:
                return 0.0
        #if np.abs(PID) == 13 and process == "DarkMuonBrem":
        #    return (self.g_e**2/(4*np.pi*alpha_em))*self._muon_brem_numerical_weight(energy_initial)
        else:
            if np.abs(PID) == 13 and process == "DarkMuonBrem":
                return (self.g_e**2/(4*np.pi*alpha_em))*self._muon_brem_numerical_weight(energy_initial)
            else:      
                return 0.0

    def draw_dark_sample(self,Einc,LU_Key=-1,process="DarkBrem",VB=False):
        dark_sample_list=self._loaded_dark_samples 
        if LU_Key<0 or LU_Key > len(dark_sample_list[process]):
            energies = dark_sample_list[process][0:]
            energies = np.array([x[0] for x in energies])

            LU_Key = np.argmin(np.abs(energies - Einc)) + 1
            if LU_Key < 0:
                LU_Key = 0
                print("Warning: sampling below minimum energy for process" + str(process) + " , Energy: " + str(Einc))
            if LU_Key >= len(dark_sample_list[process]):
                LU_Key = len(dark_sample_list[process]) - 1
                print("Warning: sampling above maximum energy for process" + str(process) + " , Energy: " + str(Einc))

        # this grabs the dictionary part rather than the energy. 
        dark_sample_dict=dark_sample_list[process][LU_Key][1]

        integrand = dark_sample_dict["adaptive_map"]
        max_F      = dark_sample_dict["max_F"][self._target_material]*self._maxF_fudge_global
        neval_vegas= dark_sample_dict["neval"]
        integrand=vg.Integrator(map=integrand, max_nhcube=1, nstrat=np.ones(dimensionalities_dark[process]), neval=neval_vegas)


        event_info={'E_inc': Einc, 'm_e': m_electron, 'Z_T': self._ZTarget, 'A_T':self._ATarget, 'mT':self._ATarget, 'alpha_FS': alpha_em, 'mV': self._mV, 'Eg_min':self._Egamma_min, 'Zeff':self.Zeff}
        if process == "DarkBrem":
            event_info['m_lepton'] = m_electron
        elif process == "DarkMuonBrem":
            event_info['m_lepton'] = m_muon

        if process in dark_diff_xsection_options:
            diff_xsec_func = dark_diff_xsection_options[process]
        else:
            raise Exception("Your process is not in the list")
        f_integrand = diff_xsec_func(event_info=event_info, ndim=dimensionalities_dark[process])

        if VB:
            sampcount = 0
        n_integrators_used = 0
        sample_found = False
        while sample_found is False and n_integrators_used < self._max_n_integrators:
            n_integrators_used += 1
            for x,wgt in integrand.random():
                if VB:
                    sampcount += 1  
                #if  max_F*draw_U()<wgt*diff_xsec_func(event_info,x):
                if  max_F*draw_U()<wgt*f_integrand(x):
                    sample_found = True
                    break
        if sample_found is False:
            #raise Exception("No Sample Found", process, Einc, LU_Key)
            print("No Sample Found", process, Einc, LU_Key)
            return None
        if VB:
            return np.concatenate([list(x), [sampcount]])
        else:
            return x

    def electron_wave_function(self, pe):
        Lambda = alpha_em*self.Zeff*m_electron
        return 32/np.pi*Lambda**5*pe**2/(pe**2 + Lambda**2)**4

    def draw_pe_sample(self):
        # max value of the electron wave function
        c = self.electron_wave_function(alpha_em*self.Zeff*m_electron/np.sqrt(3))
        pe_max = 1e-3
        while True:
            x = np.random.uniform(0, pe_max)
            u = np.random.uniform(0, 1)
            # accept-reject sampling
            if u < self.electron_wave_function(x) / (c):
                return x

    def produce_bsm_particle(self, p_original, process, weight=None, VB=False):
        p0 = deepcopy(p_original)
        if weight is None:
            wg = self.GetBSMWeights(p0, process)
        else:
            wg = weight

        dict_samp = None
        if process == "DarkAnn" and p0.get_ids()["PID"] == -11:
            dict_samp = self._d_rate_dict_positron_ann
        elif process == "DarkBrem":
            if p0.get_ids()["PID"] == 11:
                dict_samp = self._d_rate_dict_elec_brem
            else:
                dict_samp = self._d_rate_dict_positron_brem
        elif process == "DarkMuonBrem":
            dict_samp = self._d_rate_dict_muon_brem
        if dict_samp is not None:
            energies_saved = np.array(list(dict_samp.keys()))

            E0 = p0.get_p0()[0]
            if E0 < np.min(energies_saved):
                Ei = np.min(energies_saved)
            elif E0 > np.max(energies_saved):
                Ei = np.max(energies_saved)
            else:
                Ei = np.max([Ei for Ei in energies_saved if Ei <= E0])
            energies, relative_probabilities = np.transpose(dict_samp[Ei])
            if np.sum(relative_probabilities) == 0.0:
                return None
            relative_probabilities = relative_probabilities/np.sum(relative_probabilities)
            E_interact = np.random.choice(energies, p=relative_probabilities) + (E0-Ei) #correct for difference between true energy and energy for which samples were saved
            dEdxT = self.get_material_properties()[3]*(0.1)
            dist = (p0.get_p0()[0] - E_interact)/dEdxT
            p_scat = self._get_MCS_p(p0.get_p0(), self._rhoTarget*(dist/cmtom),
                                     self._ATarget, self._ZTarget,
                                     self._MCS_rescale_factor)
            p0.set_pf(p_scat)
            p0.lose_energy(E0 - E_interact)

        E0 = p0.get_pf()[0]
        RM = p0.rotation_matrix()
        if process == "DarkAnn" and self.bound_electron and E0 <= self._resonant_annihilation_energy:
            EVf, pVxfZF, pVyfZF, pVzfZF = np.sqrt(E0**2 - m_electron**2 + self._mV**2), 0, 0, np.sqrt(E0**2 - m_electron**2)
        elif process == "DarkComp" and self.bound_electron and E0 <= self._compton_threshold_energy:
            EVf, pVxfZF, pVyfZF, pVzfZF = np.sqrt(E0**2 + self._mV**2), 0, 0, E0
        elif process == "DarkAnn" and not self.bound_electron and E0 <= self.get_DarkAnnXSec()[0][0]:
            EVf, pVxfZF, pVyfZF, pVzfZF = self._resonant_annihilation_energy, 0, 0, np.sqrt(self._resonant_annihilation_energy**2 - self._mV**2)
        else:
            sample_event = self.draw_dark_sample(E0, process=process, VB=VB)
            if sample_event is None:
                return None
            #dark-production is estabilished such that the last particle returned corresponds to the dark vector
            if process == "DarkComp" and self.bound_electron:
                pe = self.draw_pe_sample()
                c0 = np.random.uniform(-1, 1)
                #check if the event is still above threshold with the electron's momentum included
                s = m_electron**2 + 2*E0*(np.sqrt(m_electron**2 + pe**2) - c0*pe)
                Ee = (s - self._mV**2 + m_electron**2)/(2*np.sqrt(s))
                if Ee < m_electron:
                    EVf, pVxfZF, pVyfZF, pVzfZF = self._mV, 0, 0, 0
                    wg = 0.0
                else:
                    EVf, pVxfZF, pVyfZF, pVzfZF = compton_fourvecs_boundelectron(p0, sample_event, mV=self._mV, Pe=pe, cte=c0)[-1]
            else:
                EVf, pVxfZF, pVyfZF, pVzfZF = dark_kinematic_function[process](p0, sample_event, mV=self._mV)[-1]
        pV4LF = np.concatenate([[EVf], np.dot(RM, [pVxfZF, pVyfZF, pVzfZF])])

        init_IDs = p0.get_ids()
        V_dict = {}
        V_dict["PID"] = 4900022
        V_dict["parent_PID"] = init_IDs["PID"]
        V_dict["ID"] = 2*(init_IDs["ID"]) + 0
        V_dict["parent_ID"] = init_IDs["ID"]
        V_dict["generation_number"] = init_IDs["generation_number"] + 1
        V_dict["generation_process"] = process
        V_dict["weight"] = wg*init_IDs["weight"]

        if process == "DarkAnn" and self.bound_electron:
            V_dict["generation_process"] = "DarkAnn_bound"
        if process == "DarkComp" and self.bound_electron:
            V_dict["generation_process"] = "DarkComp_bound"
            
        return Particle(pV4LF, p0.get_rf(), V_dict)

    def generate_dark_shower(self, ExDir=None, SParams=None):
        """ Process an existing SM shower (or produce a new one) by interating 
        through its particles and generating possible dark photon emissions using 
        all available processes.
        Args:
            ExDir: path to file containing existing SM shower OR an actual shower (list of Particle objects)
            SParamas: if no path provided, incident particle of a new SM shower to generate, 
            consisting of a "Particle" object
        Returns:
            [ShowerToSamp, NewShower]: where ShowerToSamp is the initial SM shower and NewShower 
            is the list of possible dark photon emissions generated from it
        """
        if ExDir is None and SParams is None:
            print("Need an existing SM shower-file directory or SM incident particle to run dark shower")
            return None
        
        if ExDir is not None and isinstance(ExDir, str):
            ShowerToSamp = np.load(ExDir, allow_pickle=True)
        elif ExDir is not None and isinstance(ExDir, list):
            ShowerToSamp = ExDir
        elif isinstance(SParams, Particle):
            ShowerToSamp = self.generate_shower(SParams)
        else:
            raise ValueError("Provided SParams must be a `Particle' class object")

        NewShower = []
        for ap in ShowerToSamp:
            for process_code in self.active_processes:
                wg = self.GetBSMWeights(ap, process=process_code)
                if wg > 0.0:
                    if process_code == "TwoBody_BSMDecay":
                        gamma_dict = {"mass":0, "PID":22}
                        V_dict = {"mass":self._mV, "PID":4900022,
                                  "weight":ap.get_ids()["weight"]*wg,
                                  "parent_PID":ap.get_ids()["PID"], "parent_ID":ap.get_ids()["ID"],
                                  "ID":2*(ap.get_ids()["ID"])+1, "generation_number":ap.get_ids()["generation_number"]+1,
                                  "generation_process":process_code}
                        npart = ap.two_body_decay(gamma_dict, V_dict)[1]
                        NewShower.append(npart)
                    else:    
                        npart = self.produce_bsm_particle(ap, process=process_code, weight=wg)
                        if npart is not None:
                            NewShower.append(npart)
        return ShowerToSamp, NewShower
