import numpy as np
import vegas as vg
import pickle 
import os

from scipy.interpolate import interp1d
from scipy.integrate import quad

from PETITE.sm.moliere import get_scattered_momentum_fast, get_scattered_momentum_Bethe
from PETITE.particle import Particle

from PETITE.shower import Shower
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
        
from PETITE.dark_vector import bsm_process_information as bsm_info
bsm_process_codes = bsm_info.bsm_process_info.keys()

incident_particle_PIDs_BSM = []
for proc_name in bsm_process_codes:
    i_p_PID = bsm_info.bsm_process_info[proc_name]['incoming_particle_PID']
    if type(i_p_PID) == int:
        i_p_PID = np.array([i_p_PID])
    for pid in i_p_PID:
        incident_particle_PIDs_BSM.append(pid)
incident_particle_PIDs_BSM = np.unique(incident_particle_PIDs_BSM)

class BSMShower(Shower):
    """ A class to reprocess an existing EM shower to generate dark photons
    """

    def __init__(self,
                 dict_dir,
                 target_material,
                 min_energy,
                 mV_in_GeV ,
                 active_processes=None,
                 mode="exact",
                 maxF_fudge_global=1,
                 max_n_integrators=int(1e4),
                 kinetic_mixing=1.0,
                 g_e=None,
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
            self.active_processes = bsm_process_codes

        self.active_processes_by_incident_PID = {}
        for pid in incident_particle_PIDs_BSM:
            self.active_processes_by_incident_PID[pid] = []
            for proc_name in self.active_processes:
                i_p_PID = bsm_info.bsm_process_info[proc_name]['incoming_particle_PID']
                if type(i_p_PID) == int:
                    i_p_PID = np.array([i_p_PID])
                if pid in i_p_PID:
                    self.active_processes_by_incident_PID[pid].append(proc_name)

        self.set_bsm_dict_dir(dict_dir)
        self.set_target_material(target_material)
        self.min_energy = min_energy
        self.kinetic_mixing = kinetic_mixing

        self.g_e = g_e
        if self.g_e is None:
            self.g_e = self.kinetic_mixing * np.sqrt(4 * np.pi * alpha_em)

        self.set_material_properties()
        self.set_n_targets()
        self.set_mV_list(dict_dir)
        self.set_mV(mV_in_GeV, mode)
 
        self.set_bsm_cross_sections()
        self.set_bsm_NSigmas()

        self.set_weight_arrays()
        self.set_drate_dE()
        self.set_bsm_samples()
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
    def set_bsm_dict_dir(self, value):
        """Set the directory containing pre-simulated MC events for processes involing target nuclei"""
        self._bsm_dict_dir = value

    def get_bsm_dict_dir(self):
        """Get the top level directory containing pre-computed MC pickles""" 
        return self._bsm_dict_dir
   
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
    
    def load_bsm_sample(self, dict_dir, process): 
        sample_file=open(dict_dir + "dark_maps.pkl", 'rb')
        outer_dict=pickle.load(sample_file)
        sample_file.close()

        sample_dict=outer_dict[self._mV_estimator]
        if process in sample_dict.keys():
            return(sample_dict[process])
        else:
            print(process)
            raise Exception("Process String does not match library")
        
    def set_bsm_samples(self):
        self._loaded_bsm_samples={}
        for process in self.active_processes:
            self._loaded_bsm_samples[process]= \
                self.load_bsm_sample(self._dict_dir, process)
            
    def load_bsm_cross_section(self, dict_dir, process, target_material):
        bsm_cross_section_file=open( dict_dir + "dark_xsec.pkl", 'rb')
        outer_dict=pickle.load(bsm_cross_section_file)
        bsm_cross_section_file.close()

        bsm_cross_section_dict=outer_dict[self._mV_estimator]

        if process not in bsm_cross_section_dict:
            raise Exception("Process String does not match library")
        
        if target_material in bsm_cross_section_dict[process]:
            return(bsm_cross_section_dict[process][target_material])
        else:
            raise Exception("Target Material is not in library")

    def set_bsm_cross_sections(self):
        """Loads the pre-computed cross-sections for various shower processes 
        and extracts the minimum/maximum values of initial energies
        """
        self._bsm_cross_sections = {}
        for process in self.active_processes:
            self._bsm_cross_sections[process] = self.load_bsm_cross_section(self._dict_dir, process, self._target_material)
            while self._bsm_cross_sections[process][0][1] == 0.0:
                self._bsm_cross_sections[process] = self._bsm_cross_sections[process][1:] #remove first element if it has zero cross-section

        self._minimum_calculable_dark_kinetic_energy = {}
        for pid in incident_particle_PIDs_BSM:
            self._minimum_calculable_dark_kinetic_energy[pid] = np.min([self._bsm_cross_sections[process][0][0] for process in self.active_processes_by_incident_PID[pid]])
        #self._resonant_annihilation_energy = (self._mV**2-2*m_electron**2)/(2*m_electron)
        #self._compton_threshold_energy = self._mV**2/(2*m_electron) + self._mV
        #self._minimum_calculable_dark_energy = {11:{"DarkBrem":self._dark_brem_cross_section[0][0]},
        #                                        -11:{"DarkBrem":self._dark_brem_cross_section[0][0], "DarkAnn":self._resonant_annihilation_energy},
        #                                        22:{"DarkComp":self._dark_compton_cross_section[0][0]},
        #                                        111:{"TwoBody_BSMDecay":-1},
        #                                        13:{"DarkMuonBrem":self._dark_muon_brem_cross_section[0][0]},
        #                                        -13:{"DarkMuonBrem":self._dark_muon_brem_cross_section[0][0]}}
        
        #if self.bound_electron:
        #    self._minimum_calculable_dark_energy[-11]["DarkAnn"] = self._resonant_annihilation_energy/1000.0
        #    self._minimum_calculable_dark_energy[22]["DarkComp"] = self._compton_threshold_energy/1000.0

    def get_bsmXSec(self, process):
        """ Returns array of [kinetic_energy,cross-section] values for the given process """
        if process in self._bsm_cross_sections:
            return self._bsm_cross_sections[process]
        else:
            raise Exception("Process String does not match library")

    def set_bsm_NSigmas(self):
        """Constructs interpolations of n_T sigma (in 1/cm) as a functon of 
        incoming particle energy for each process
        """
        nZ, ne = self.get_n_targets()

        self._bsm_NSigma = {}
        for process in self.active_processes:
            x, y = np.transpose(self.get_bsmXSec(process))
            target_particle = bsm_info.bsm_process_info[process]['scattering_target'] #string of either 'electron' or 'nucleus'
            if target_particle == 'electron':
                n_target = ne
            else:
                n_target = nZ
            #bsm_nsigma are interpolated in terms of kinetic energies *above minimum calculable energy* for the given process, for better stability
            self._bsm_NSigma[process] = interpolate1d(x - x[0], n_target*GeVsqcm2*y, xspace='log', yspace='log', fill_value=-20.0, bounds_error=False)

    def _bsm_integrand(self, T_interact, T_incident, process, incident_PID):
        #Calculating the integram for the BSM scattering weight for incident particles 
        #that lose energy as propagating. (Function not called for incident photons)

        dEdxT_GeVpercm = self.get_material_properties()[3]*(0.1)*cmtom #Converting MeV/cm to GeV/m to GeV/cm
        T_min = self.get_bsmXSec(process)[0][0]
        return self._bsm_NSigma[process](T_interact - T_min)/dEdxT_GeVpercm*self._particle_exponential_factor(incident_PID, T_interact, T_incident)

    def construct_weight_array(self, process, incident_PID):
        initial_kinetic_energies = np.transpose(self.get_bsmXSec(process))[0]
        if incident_PID == 22:
            raise Exception("Photons do not lose ionization energy, so weight array construction is not necessary")
        dEdxT_GeVperm = self.get_material_properties()[3]*(0.1)

        #Calculating the incident particle's (SM) mean-free path for initial energies of interest.
        mfp_TI = np.array([self.get_mfp([incident_PID, Ti]) for Ti in initial_kinetic_energies])

        energy_loss_ten_mfp = 10*mfp_TI*dEdxT_GeVperm
        minimum_energy_0 = initial_kinetic_energies[0]
        kinetic_energy_break = np.ones(len(initial_kinetic_energies))*minimum_energy_0
        for i in range(len(kinetic_energy_break)):
            T_break = initial_kinetic_energies[i] - energy_loss_ten_mfp[i]
            if T_break > minimum_energy_0:
                kinetic_energy_break[i] = T_break
        
        weight_array = np.array([quad(self._bsm_integrand, minimum_energy_0, kinetic_energy_break[i],
                                    args=(initial_kinetic_energies[i], process, incident_PID), full_output=1)[0] +
                                 quad(self._bsm_integrand, kinetic_energy_break[i], initial_kinetic_energies[i],
                                      args=(initial_kinetic_energies[i], process, incident_PID), full_output=1)[0] for i in range(len(initial_kinetic_energies))])

        return [initial_kinetic_energies, weight_array]

    def set_weight_arrays(self):
        self._numerical_weights = {}
        for pid in incident_particle_PIDs_BSM:
            if pid == 22:
                continue #skipping numerical weight construction for incident photons
            self._numerical_weights[pid] = {}
            for process in self.active_processes_by_incident_PID[pid]:
                x, y = self.construct_weight_array(process, pid)
                self._numerical_weights[pid][process] = interpolate1d(x, y, fill_value=0.0, bounds_error=False, xspace='log', yspace='log')

    def _d_rate_d_E(self, T_initial, process, incident_PID):
        dEdxT_GeVperm = self.get_material_properties()[3]*(0.1)
        mfp_TI = self.get_mfp([incident_PID, T_initial])
        energy_loss_ten_mfp = 10*mfp_TI*dEdxT_GeVperm
        kinetic_energy_min = self.get_bsmXSec(process)[0][0]
        kinetic_energy_array = np.linspace(np.max([T_initial-energy_loss_ten_mfp, kinetic_energy_min]), T_initial, 11)
        kinetic_energy_center_array = np.array([(kinetic_energy_array[i] + kinetic_energy_array[i+1])/2. for i in range(len(kinetic_energy_array)-1)])

        bsm_weights = np.array([quad(self._bsm_integrand, kinetic_energy_array[i], kinetic_energy_array[i+1],
                                      args=(T_initial, process, incident_PID), full_output=1)[0] for i in range(len(kinetic_energy_array)-1)])

        return np.transpose([kinetic_energy_center_array, bsm_weights])
    
    def _d_rate_d_E_array(self, process, incident_PID):
        T_i_samp = np.transpose(self.get_bsmXSec(process))[0]
        return {Ti:self._d_rate_d_E(Ti, process, incident_PID) for Ti in T_i_samp}

    def set_drate_dE(self):
        self._d_rate_dict = {}
        for pid in incident_particle_PIDs_BSM:
            if pid == 22:
                continue #skipping d_rate_d_E construction for incident photons
            self._d_rate_dict[pid] = {}
            for process in self.active_processes_by_incident_PID[pid]:
                self._d_rate_dict[pid][process] = self._d_rate_d_E_array(process, pid)

    def get_bsm_weight(self, particle, process):
        if isinstance(particle, list) or isinstance(particle, np.ndarray):
            PID, kinetic_energy_initial = particle
        else:
            PID, kinetic_energy_initial = particle.get_ids()["PID"], particle.get_p0()[0]

        if process not in self.active_processes_by_incident_PID[PID]:
            return 0.0 #process and incident particle PID not compatible, so weight is zero.

        if PID == 22:
            rescale_factor = (self.g_e**2/(4.0*np.pi*alpha_em))
            kinetic_energy_minimum = self.get_bsmXSec(process)[0][0]
            return rescale_factor*self._bsm_NSigma[process](kinetic_energy_initial - kinetic_energy_minimum)/self._NSigmaParticle(PID, kinetic_energy_initial)

        rescale_factor = (self.g_e**2/(4.0*np.pi*alpha_em))
        return rescale_factor * self._numerical_weights[PID][process](kinetic_energy_initial)

    def draw_bsm_sample(self, Tinc, process, LU_Key=-1):
        bsm_sample_list = self._loaded_bsm_samples
        if LU_Key < 0 or LU_Key > len(bsm_sample_list[process]):
            kinetic_energies = bsm_sample_list[process][0:]
            kinetic_energies = np.array([x[0] for x in kinetic_energies])

            LU_Key = np.argmin(np.abs(kinetic_energies - Tinc)) + 1
            if LU_Key < 0:
                LU_Key = 0
                print("Warning: sampling below minimum kinetic energy for process" + str(process) + " , Kinetic Energy: " + str(Tinc))
            if LU_Key >= len(bsm_sample_list[process]):
                LU_Key = len(bsm_sample_list[process]) - 1
                print("Warning: sampling above maximum kinetic energy for process" + str(process) + " , Kinetic Energy: " + str(Tinc))

        # this grabs the dictionary part rather than the kinetic energy.
        bsm_sample_dict = bsm_sample_list[process][LU_Key][1]

        max_F     = bsm_sample_dict["max_F"][self._target_material]*self._maxF_fudge_global
        integrand = vg.Integrator(map = bsm_sample_dict["adaptive_map"], 
                                  max_nhcube = 1, 
                                  nstrat = np.ones(bsm_info.bsm_process_info[process]['dimensionalities']),
                                  neval=bsm_sample_dict["neval"])

        event_info = {'T_inc': Tinc, 'm_e': m_electron, 'Z_T': self._ZTarget, 
                      'A_T': self._ATarget, 'mT': self._ATarget, 'alpha_FS': alpha_em,
                      'mV': self._mV, 'Eg_min':self._Egamma_min}
        if process == "DarkBrem":
            event_info['m_lepton'] = m_electron
        elif process == "DarkMuonBrem":
            event_info['m_lepton'] = m_muon
        
        if process in self.active_processes:
            diff_xsec_func = bsm_info.bsm_process_info[process]['differential_cross_section']
        else:
            raise Exception("Your process is not in the list of active processes for this shower")
        f_integrand = diff_xsec_func(event_info=event_info, ndim=bsm_info.bsm_process_info[process]['dimensionalities'])

        n_integrators_used = 0
        sample_found = False
        while sample_found is False and n_integrators_used < self._max_n_integrators:
            n_integrators_used += 1
            for x, wgt in integrand.random():
                if max_F*draw_U() < wgt*f_integrand(x):
                    sample_found = True
                    break
        if sample_found is False:
            raise Exception("No Sample Found", process, Tinc, LU_Key)
        else:
            return x

    '''
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

        adaptive_map = dark_sample_dict["adaptive_map"]
        max_F      = dark_sample_dict["max_F"][self._target_material]*self._maxF_fudge_global
        neval_vegas= dark_sample_dict["neval"]
        integrand=vg.Integrator(map=adaptive_map, max_nhcube=1, nstrat=np.ones(dimensionalities_dark[process]), neval=neval_vegas)


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
            raise Exception("No Sample Found", process, Einc, LU_Key)
        if VB:
            return np.concatenate([list(x), [sampcount]])
        else:
            return x
    '''

    def produce_bsm_particle(self, p_original, process, weight=None):
        p0 = deepcopy(p_original)
        if weight is None:
            wg = self.get_bsm_weight(p0, process)
        else:
            wg = weight

        dict_samp = None
        if p0.get_ids()["PID"] not in self._d_rate_dict or process not in self._d_rate_dict[p0.get_ids()["PID"]]:
            dict_samp = None
        else:
            dict_samp = self._d_rate_dict[p0.get_ids()["PID"]][process]
            kinetic_energies_saved = np.array(list(dict_samp.keys()))

            T_start = p0.get_p0()[0] - p0.get_ids()['mass']
            if T_start < np.min(kinetic_energies_saved):
                T_initial = np.min(kinetic_energies_saved) 
            elif T_start > np.max(kinetic_energies_saved):
                raise ValueError("Particle kinetic energy above maximum calculable energy for this process, so BSM particle production cannot be sampled")
            else:
                T_initial = np.max([T for T in kinetic_energies_saved if T <= T_start])

            T_interact_options, relative_probabilities = np.transpose(dict_samp[T_initial])
            if np.sum(relative_probabilities) == 0.0:
                return None
            relative_probabilities = relative_probabilities/np.sum(relative_probabilities)
            T_interact = np.random.choice(T_interact_options, p=relative_probabilities) + (T_start - T_initial) #correct for difference between true kinetic energy and kinetic energy for which samples were saved

            dEdxT = self.get_material_properties()[3]*(0.1)
            distance_to_interact = (T_start - T_interact)/dEdxT
            p_scat = self._get_MCS_p(p0.get_p0(), self._rhoTarget*(distance_to_interact/cmtom), 
                                     self._ATarget, self._ZTarget, self._MCS_rescale_factor)

            p0.set_pf(p_scat)
            p0.lose_energy(T_start - T_interact)

        T_interact = p0.get_pf()[0] - p0.get_ids()['mass']
        RM = p0.rotation_matrix()

        sample_event = self.draw_bsm_sample(T_interact, process=process)
        #BSM-production is estabilished such that the last particle returned corresponds to the dark vector
        EVf, pVxfZF, pVyfZF, pVzfZF = bsm_info.bsm_process_info[process]['kinematic_function'](p0, sample_event, mV=self._mV)[-1]
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
            
        return Particle(pV4LF, p0.get_rf(), V_dict)

    def generate_bsm_shower(self, ExDir=None, SParams=None):
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
                wg = self.get_bsm_weight(ap, process=process_code)
                if wg > 0.0:
                    npart = self.produce_bsm_particle(ap, process=process_code, weight=wg)
                    if npart is not None:
                        NewShower.append(npart)
        return ShowerToSamp, NewShower
