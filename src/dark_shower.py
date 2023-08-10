import numpy as np
import pickle 

from scipy.interpolate import interp1d
from scipy.integrate import quad

from .moliere import get_scattered_momentum 
from .particle import Particle, meson_twobody_branchingratios
from .kinematics import e_to_egamma_fourvecs, e_to_eV_fourvecs, gamma_to_epem_fourvecs, compton_fourvecs, radiative_return_fourvecs
from .shower import Shower
from .all_processes import *

import sys
from numpy.random import random as draw_U
        
Z = {'hydrogen':1.0, 'graphite':6.0, 'lead':82.0, 'iron':26.0} #atomic number of different targets
A = {'hydrogen':1.0, 'graphite':12.0, 'lead':207.2, 'iron':56.0} #atomic mass of different targets

GeVsqcm2 = 1.0/(5.06e13)**2 #Conversion between cross sections in GeV^{-2} to cm^2
cmtom = 0.01
mp0 = 1.673e-24 #g

#dark_process_codes = ["DarkBrem", "Ann", "Comp", "TwoBody_BSMDecay"]
dark_process_codes = ["DarkBrem", "DarkAnn", "DarkComp", "TwoBody_BSMDecay"]

dark_kinematic_function = {"DarkBrem" : e_to_eV_fourvecs,
                           "DarkAnn"      : radiative_return_fourvecs,
                           "DarkComp"     : compton_fourvecs}
diff_xsection_options={"DarkComp"      : dsigma_compton_dCT,
                        "DarkBrem" : dsig_etl_helper,
                        "DarkAnn"      : dsigma_radiative_return_du }

class DarkShower(Shower):
    """ A class to reprocess an existing EM shower to generate dark photons
    """

    def __init__(self, dict_dir, target_material, min_energy, mV_in_GeV , \
                          mode="exact", maxF_fudge_global=1,max_n_integrators=int(1e4), \
                          kinetic_mixing=1.0, g_e=None, active_processes=None):
        super().__init__(dict_dir, target_material, min_energy)
        """Initializes the shower object.
        Args:
            PickDir: directory containing the pre-computed MC samples of various shower processes
            TargetMaterial: string label of the homogeneous material through which 
            particles propagate (available materials are the dict keys of 
            Z, A, rho, etc)
            MinEnergy: minimum particle energy in GeV at which the particle 
            finishes its propagation through the target
            mV_string: str which determines the pre-computed MC sample of massive
            vector events to use (see MVLib variable for available choices)
        """

        self.active_processes = active_processes
        if self.active_processes is None:
            self.active_processes = dark_process_codes

        self.set_dark_dict_dir(dict_dir)
        self.set_target_material(target_material)
        self.min_energy = min_energy
        self.kinetic_mixing = kinetic_mixing
        self.g_e = g_e
        if self.g_e is None:
            self.g_e = self.kinetic_mixing*np.sqrt(4*np.pi*alpha_em)

        self.set_material_properties()
        self.set_n_targets()
        self.set_mV_list(dict_dir)
        self.set_mV(mV_in_GeV, mode)
        
        self.set_dark_cross_sections()
        self.set_dark_NSigmas()
        self.set_dark_samples()


        self._maxF_fudge_global=maxF_fudge_global
        self._max_n_integrators=max_n_integrators
  
    
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
        for process in diff_xsection_options.keys():
            self._loaded_dark_samples[process]= \
                self.load_dark_sample(self._dict_dir, process)
            
    def load_dark_cross_section(self, dict_dir, process, target_material):
        #dark_cross_section_file=open( dict_dir + "dark_xsecs.pkl", 'rb')
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
        self._dark_annihilation_cross_section  = self.load_dark_cross_section(self._dict_dir, 'DarkAnn', self._target_material) 
        self._dark_compton_cross_section = self.load_dark_cross_section(self._dict_dir, 'DarkComp', self._target_material) 

        self._minimum_calculable_dark_energy = {11:self._dark_brem_cross_section[0][0],
                                                -11:np.min([self._dark_brem_cross_section[0][0], self._dark_annihilation_cross_section[0][0]]),
                                                22:self._dark_compton_cross_section[0][0],
                                                111:-1}

    def get_DarkBremXSec(self):
        """ Returns array of [energy,cross-section] values for brem """ 
        return self._dark_brem_cross_section 
    def get_DarkAnnXSec(self):
        """ Returns array of [energy,cross-section] values for e+e- annihilation """ 
        return self._dark_annihilation_cross_section
    def get_DarkCompXSec(self):
        """ Returns array of [energy,cross-section] values for Compton """ 
        return self._dark_compton_cross_section

    def set_dark_NSigmas(self):
        """Constructs interpolations of n_T sigma (in 1/cm) as a functon of 
        incoming particle energy for each process
        """
        DBS, DAnnS, DCS = self.get_DarkBremXSec(), self.get_DarkAnnXSec(), self.get_DarkCompXSec()
        nZ, ne = self.get_n_targets()
        self._NSigmaDarkBrem = interp1d(np.transpose(DBS)[0], nZ*GeVsqcm2*np.transpose(DBS)[1], fill_value=0.0, bounds_error=False)
        self._NSigmaDarkAnn = interp1d(np.transpose(DAnnS)[0], ne*GeVsqcm2*np.transpose(DAnnS)[1], fill_value=0.0, bounds_error=False)
        self._NSigmaDarkComp = interp1d(np.transpose(DCS)[0], ne*GeVsqcm2*np.transpose(DCS)[1], fill_value=0.0, bounds_error=False)

        II_y_DarkBrem = np.array([quad(self._NSigmaDarkBrem, DBS[0][0],   DBS[i][0], full_output=1)[0]   for i in range(len(DBS))])
        II_y_DarkAnn  = np.array([quad(self._NSigmaDarkAnn,  DAnnS[0][0], DAnnS[i][0], full_output=1)[0] for i in range(len(DAnnS))])
        II_y_DarkComp = np.array([quad(self._NSigmaDarkComp, DCS[0][0],   DCS[i][0], full_output=1)[0]   for i in range(len(DCS))])

        self._interaction_integral_DarkBrem = interp1d(np.transpose(DBS)[0],   II_y_DarkBrem, fill_value=0.0, bounds_error=False)
        self._interaction_integral_DarkAnn  = interp1d(np.transpose(DAnnS)[0], II_y_DarkAnn, fill_value=0.0, bounds_error=False)
        self._interaction_integral_DarkComp = interp1d(np.transpose(DCS)[0],   II_y_DarkComp, fill_value=0.0, bounds_error=False)

    def GetBSMWeights(self, particle, process):
        """Compute relative weight of dark photon emission to the available SM processes
        Args: 
            PID: incoming PDG ID of the particle 
            Energy: its energy
        Returns:
            float, representing probability of V emission (for a fixed kinetic mixing) 
            divided by the probabilities of available SM processes

        """
        PID, energy_final, energy_initial = particle.get_ids()["PID"], particle.get_pf()[0], particle.get_p0()[0]
        if energy_initial < self._minimum_calculable_dark_energy[PID]:
            return 0.0

        if PID == 22:
            if process != "DarkComp":
                return 0.0
            return (self.g_e**2/(4*np.pi*alpha_em))*self._NSigmaDarkComp(energy_final)/(self._NSigmaPP(energy_final) + self._NSigmaComp(energy_final))
        elif PID == 11:
            if process != "DarkBrem":
                return 0.0
            numerator = (self._interaction_integral_DarkBrem(energy_initial) - self._interaction_integral_DarkBrem(energy_final))
            denominator1 = (self._interaction_integral_Brem(energy_initial) - self._interaction_integral_Brem(energy_final))
            denominator2 = (self._interaction_integral_Moller(energy_initial) - self._interaction_integral_Moller(energy_final))
            return (self.g_e**2/(4*np.pi*alpha_em))*numerator/(denominator1 + denominator2)
        elif PID == -11:
            if process == "DarkBrem":
                numerator = (self._interaction_integral_DarkBrem(energy_initial) - self._interaction_integral_DarkBrem(energy_final))
                denominator1 = (self._interaction_integral_Brem(energy_initial) - self._interaction_integral_Brem(energy_final))
                denominator2 = (self._interaction_integral_Bhabha(energy_initial) - self._interaction_integral_Bhabha(energy_final))
                denominator3 = (self._interaction_integral_Ann(energy_initial) - self._interaction_integral_Ann(energy_final))
                return (self.g_e**2/(4*np.pi*alpha_em))*numerator/(denominator1+denominator2+denominator3)

            elif process == "DarkAnn":
                if self._NSigmaDarkAnn(energy_initial) == 0.0:
                    return 0.0
                else:
                    numerator = (self._interaction_integral_DarkAnn(energy_initial) - self._interaction_integral_DarkAnn(energy_final))
                    denominator1 = (self._interaction_integral_Brem(energy_initial) - self._interaction_integral_Brem(energy_final))
                    denominator2 = (self._interaction_integral_Bhabha(energy_initial) - self._interaction_integral_Bhabha(energy_final))
                    denominator3 = (self._interaction_integral_Ann(energy_initial) - self._interaction_integral_Ann(energy_final))
                
                    E_res = (self._mV**2 - 2*m_electron**2)/(2*m_electron)
                    if energy_final < E_res:
                        particle.lose_energy((energy_final - 1.1*E_res))

                    return (self.g_e**2/(4*np.pi*alpha_em))*numerator/(denominator1+denominator2+denominator3)                
            else:
                return 0.0
        elif PID == 111 or PID == 221 or PID == 331:
            if process == "TwoBody_BSMDecay":
                mass_ratio = self._mV/particle.get_ids()["mass"]
                if mass_ratio >= 1.0:
                    return 0.0
                return 2*(self.kinetic_mixing)**2*(1.0 - mass_ratio**2)**3*meson_twobody_branchingratios[particle.get_ids()["PID"]]
            else:
                return 0.0
        else:
            return 0.0


    def draw_dark_sample(self,Einc,LU_Key=-1,process="DarkBrem",VB=False):
        dark_sample_list=self._loaded_dark_samples 
        if LU_Key<0 or LU_Key > len(dark_sample_list[process]):
            # Get the LU_Key corresponding to the closest incoming energy
            energies = dark_sample_list[process][0:]
            energies = np.array([x[0] for x in energies])
            # Get index of nearest (higher) energy
            LU_Key = np.argmin(np.abs(energies - Einc)) + 1

        # this grabs the dictionary part rather than the energy. 
        dark_sample_dict=dark_sample_list[process][LU_Key][1]

        integrand = dark_sample_dict["adaptive_map"]
        max_F      = dark_sample_dict["max_F"][self._target_material]*self._maxF_fudge_global
        neval_vegas= dark_sample_dict["neval"]
        integrand=vg.Integrator(map=integrand, max_nhcube=1, neval=neval_vegas)

        event_info={'E_inc': Einc, 'm_e': m_electron, 'Z_T': self._ZTarget, 'A_T':self._ATarget, 'mT':self._ATarget, 'alpha_FS': alpha_em, 'mV': self._mV, 'Eg_min':self._Egamma_min}
        
        if process in diff_xsection_options:
            diff_xsec_func = diff_xsection_options[process]
        else:
            raise Exception("Your process is not in the list")

        if VB:
            sampcount = 0
        n_integrators_used = 0
        sample_found = False
        while sample_found is False and n_integrators_used < self._max_n_integrators:
            n_integrators_used += 1
            for x,wgt in integrand.random():
                if VB:
                    sampcount += 1  
                if  max_F*draw_U()<wgt*diff_xsec_func(event_info,x):
                    sample_found = True
                    break
        if sample_found is False:
            raise Exception("No Sample Found", process, Einc, LU_Key)
        if VB:
            return np.concatenate([list(x), [sampcount]])
        else:
            return(x)

    def GetPositronDarkBF(self, Energy):
        """Branching fraction for a positron to undergo dark brem vs dark 
        annihilation"""
        if Energy < (self._mV**2 - m_electron**2)/(2*m_electron) + 2*self._Egamma_min:
            return 1.0
        else:
            return self._NSigmaDarkBrem(Energy)/(self._NSigmaDarkBrem(Energy) + self._NSigmaDarkAnn(Energy))

    def produce_bsm_particle(self, p0, process, VB=False):
        E0 = p0.get_pf()[0]
        RM = p0.rotation_matrix()
        sample_event = self.draw_dark_sample(E0, process=process, VB=VB)

        #dark-production is estabilished such that the last particle returned corresponds to the dark vector
        EVf, pVxfZF, pVyfZF, pVzfZF = dark_kinematic_function[process](p0, sample_event, mV=self._mV)[-1] 
        pV4LF = np.concatenate([[EVf], np.dot(RM, [pVxfZF, pVyfZF, pVzfZF])])

        wg = self.GetBSMWeights(p0, process)

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
        
        if ExDir is not None and type(ExDir)==str:
            ShowerToSamp = np.load(ExDir, allow_pickle=True)
        elif ExDir is not None and type(ExDir)==list:
            ShowerToSamp = ExDir
        elif type(SParams)==Particle:
            ShowerToSamp = self.generate_shower(SParams)
        else:
            raise ValueError("Provided SParams must be a `Particle' class object")

        NewShower = []
        for ap in ShowerToSamp:
            for process_code in self.active_processes:
                if self.GetBSMWeights(ap, process=process_code) > 0.0:
                    if process_code == "TwoBody_BSMDecay":
                        gamma_dict = {"mass":0, "PID":22}
                        V_dict = {"mass":self._mV, "PID":4900022,
                                  "weight":ap.get_ids()["weight"]*self.GetBSMWeights(ap, process=process_code),
                                  "parent_PID":ap.get_ids()["PID"], "parent_ID":ap.get_ids()["ID"],
                                  "ID":2*(ap.get_ids()["ID"])+1, "generation_number":ap.get_ids()["generation_number"]+1,
                                  "generation_process":process_code}
                        npart = ap.two_body_decay(gamma_dict, V_dict)[1]
                        NewShower.append(npart)
                    else:    
                        npart = self.produce_bsm_particle(ap, process=process_code)
                        NewShower.append(npart)
        return ShowerToSamp, NewShower