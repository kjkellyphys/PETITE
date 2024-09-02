import numpy as npA
import pickle 

from scipy.interpolate import interp1d
from scipy.integrate import quad

from .moliere import get_scattered_momentum_fast, get_scattered_momentum_Bethe
from .particle import Particle, mass_dict
from .kinematics import e_to_egamma_fourvecs, gamma_to_epem_fourvecs, compton_fourvecs, annihilation_fourvecs, ee_to_ee_fourvecs
from .all_processes import *
from .physical_constants import *
from datetime import datetime

#np.random.seed(int(datetime.now().timestamp()))

from numpy.random import random as draw_U
import copy

process_code = {'Brem':0, 'Ann': 1, 'PairProd': 2, 'Comp': 3, "Moller":4, "Bhabha":5}
diff_xsection_options={"PairProd" : dsigma_pairprod_dimensionless,
                        "Comp"     : dsigma_compton_dCT,
                        "Brem"     : dsigma_brem_dimensionless,
                        "Ann"      : dsigma_annihilation_dCT,
                        "Moller"   : dsigma_moller_dCT,
                        "Bhabha"   : dsigma_bhabha_dCT }
dimensionalities={"PairProd":4,
                  "Comp":1,
                  "Brem":4,
                  "Ann":1,
                  "Moller":1,
                  "Bhabha":1}
     
formfactor_dict      ={"PairProd" : g2_elastic,
                        "Comp"     : unity,
                        "Brem"     : g2_elastic,
                        "Ann"      : unity,
                        "Moller"   : unity,
                        "Bhabha"   : unity }

QSq_dict             ={"PairProd" : pair_production_q_sq_dimensionless, 
                       "Brem"     : brem_q_sq_dimensionless, 
                       "Comp": dummy, 
                       "Ann": dummy, 
                       "Moller":dummy, 
                       "Bhabha":dummy }

kinematic_function   ={"PairProd" : gamma_to_epem_fourvecs,
                       "Brem"     : e_to_egamma_fourvecs,
                       "Comp"     : compton_fourvecs,
                       "Ann"      : annihilation_fourvecs,
                       "Moller"   : ee_to_ee_fourvecs,
                       "Bhabha"   : ee_to_ee_fourvecs}

process_PIDS         ={"PairProd" : [-11, 11],
                       "Brem"     : [0, 22],
                       "Comp"     : [11, 22],
                       "Ann"      : [22, 22],
                       "Moller"   : [0, 11],
                       "Bhabha"   : [0, 11]}

class Shower:
    """ Representation of a shower

    """
    def __init__(self, dict_dir, target_material, min_energy, maxF_fudge_global=1,max_n_integrators=int(1e4), fast_MCS_mode=True, seed=None,rescale_MCS=1 ):
        """Initializes the shower object.
        Args:
            dict_dir: directory containing the pre-computed VEGAS integrators and auxillary info.
            target_material: string label of the homogeneous material through which 
            particles propagate (available materials are the dict keys of 
            Z, A, rho, etc)
            min_Energy: minimum particle energy in GeV at which the particle 
            finishes its propagation through the target
        """
        if seed is not None:
            np.random.seed(seed)

        self.set_dict_dir(dict_dir)
        self.set_target_material(target_material)
        self.min_energy = min_energy

        self.set_material_properties()
        self.set_n_targets()
        self.set_cross_sections()
        self.set_samples()
        self.set_NSigmas()
        self.set_MCS_momentum(fast_MCS_mode)
        self.set_MCS_rescale_factor(rescale_MCS)
        
        self._maxF_fudge_global=maxF_fudge_global
        self._max_n_integrators=max_n_integrators

    def set_MCS_rescale_factor(self, rescale_MCS):
        self._MCS_rescale_factor=rescale_MCS
        
    def set_MCS_momentum(self, fast_MCS_mode):
        if fast_MCS_mode:
            self._get_MCS_p=get_scattered_momentum_fast
        else:
            self._get_MCS_p=get_scattered_momentum_Bethe       
        
    def load_sample(self, dict_dir, process):
        sample_file=open(dict_dir + "sm_maps.pkl", 'rb')
        sample_dict=pickle.load(sample_file)
        sample_file.close()

        if process in sample_dict.keys():
            return(sample_dict[process])
        else:
            print(process)
            raise Exception("Process String does not match library")
    
    def load_cross_section(self, dict_dir, process, target_material):
        cross_section_file=open( dict_dir + "sm_xsec.pkl", 'rb')
        cross_section_dict=pickle.load(cross_section_file)
        cross_section_file.close()

        if process not in cross_section_dict:
            raise Exception("Process String does not match library")
        
        if target_material in cross_section_dict[process]:
            return(cross_section_dict[process][target_material])
        else:
            raise Exception("Target Material is not in library")
        
    def set_dict_dir(self, value):
        """Set the top level directory containing pre-computed MC pickles to value"""
        self._dict_dir = value
    def get_dict_dir(self):
        """Get the top level directory containing pre-computed MC pickles""" 
        return self._dict_dir
    def set_target_material(self, value):
        """Set the string representing the target material to value"""
        self._target_material = value
    def get_target_material(self):
        """Get the string representing the target material"""
        return self._target_material
    def set_material_properties(self):
        """Defines material properties (Z, A, rho, etc) based on the target 
        material label
        """
        self._ZTarget = target_information[self.get_target_material()]["Z_T"]
        self._ATarget = target_information[self.get_target_material()]["A_T"]
        self._rhoTarget = target_information[self.get_target_material()]["rho"]
        self._dEdx = target_information[self.get_target_material()]["dEdx"]

    def get_material_properties(self):
        """Returns target material properties: Z, A, rho, dE/dx"""
        return self._ZTarget, self._ATarget, self._rhoTarget, self._dEdx

    def set_n_targets(self):
        """Determines nuclear and electron target densities for the 
           target material
        """
        ZT, AT, rhoT, dEdxT = self.get_material_properties()
        self._nTarget = rhoT/m_proton_grams/AT
        self._nElecs = self._nTarget*ZT

    def set_samples(self):
        self._loaded_samples={}
        for Process in process_code.keys():
            self._loaded_samples[Process]= \
                self.load_sample(self._dict_dir, Process)
        self._Egamma_min = self._loaded_samples['Brem'][0][1]['Eg_min']
        self._Ee_min = self._loaded_samples['Brem'][0][1]['Ee_min']
        
    def get_n_targets(self):
        """Returns nuclear and electron target densities for the 
           target material in 1/cm^3
        """

        return self._nTarget, self._nElecs

    def set_cross_sections(self):
        """Loads the pre-computed cross-sections for various shower processes 
        and extracts the minimum/maximum values of initial energies
        """

        # These contain only the cross sections for the chosen target material
        self._brem_cross_section = self.load_cross_section(self._dict_dir, 'Brem', self._target_material)
        self._pair_production_cross_section   = self.load_cross_section(self._dict_dir, 'PairProd', self._target_material)
        self._annihilation_cross_section  = self.load_cross_section(self._dict_dir, 'Ann', self._target_material) 
        self._compton_cross_section = self.load_cross_section(self._dict_dir, 'Comp', self._target_material) 
        self._moller_cross_section = self.load_cross_section(self._dict_dir, 'Moller', self._target_material) 
        self._bhabha_cross_section = self.load_cross_section(self._dict_dir, 'Bhabha', self._target_material) 

        self._minimum_calculable_energy = {11:np.min([self._brem_cross_section[0][0], self._moller_cross_section[0][0]]),
                                           -11:np.min([self._brem_cross_section[0][0], self._bhabha_cross_section[0][0], self._annihilation_cross_section[0][0]]),
                                           22:np.min([self._pair_production_cross_section[0][0], self._compton_cross_section[0][0]])}

    def get_brem_cross_section(self):
        """ Returns array of [energy,cross-section] values for brem """ 
        return self._brem_cross_section
    def get_pairprod_cross_section(self):
        """ Returns array of [energy,cross-section] values for pair production """ 
        return self._pair_production_cross_section
    def get_annihilation_cross_section(self):
        """ Returns array of [energy,cross-section] values for e+e- annihilation """ 
        return self._annihilation_cross_section
    def get_compton_cross_section(self):
        """ Returns array of [energy,cross-section] values for Compton """ 
        return self._compton_cross_section
    def get_moller_cross_section(self):
        """ Returns array of [energy,cross-section] values for Moller """ 
        return self._moller_cross_section
    def get_bhabha_cross_section(self):
        """ Returns array of [energy,cross-section] values for Bhabha """ 
        return self._bhabha_cross_section

    def set_NSigmas(self):
        """Constructs interpolations of n_T sigma (in 1/cm) as a functon of 
        incoming particle energy for each process
        """
        BS, PPS, AnnS, CS, MS, BhS = self.get_brem_cross_section(), self.get_pairprod_cross_section(), self.get_annihilation_cross_section(), self.get_compton_cross_section(), self.get_moller_cross_section(), self.get_bhabha_cross_section()
        nZ, ne = self.get_n_targets()
        self._NSigmaBrem = interp1d(np.transpose(BS)[0], nZ*GeVsqcm2*np.transpose(BS)[1], fill_value=0.0, bounds_error=False)
        self._NSigmaPP = interp1d(np.transpose(PPS)[0], nZ*GeVsqcm2*np.transpose(PPS)[1], fill_value=0.0, bounds_error=False)
        self._NSigmaAnn = interp1d(np.transpose(AnnS)[0], ne*GeVsqcm2*np.transpose(AnnS)[1], fill_value=0.0, bounds_error=False)
        self._NSigmaComp = interp1d(np.transpose(CS)[0], ne*GeVsqcm2*np.transpose(CS)[1], fill_value=0.0, bounds_error=False)
        #bhabha_moller_energies = np.logspace(np.log10(3*m_electron + self.min_energy), 2, 101)
        bhabha_moller_energies = np.logspace(np.log10(3*m_electron + self._Ee_min), 2, 101)
        self._NSigmaMoller = interp1d(bhabha_moller_energies, ne*GeVsqcm2*sigma_moller({"E_inc":bhabha_moller_energies, "Ee_min":self._Ee_min}), fill_value=0.0, bounds_error=False)
        self._NSigmaBhabha = interp1d(bhabha_moller_energies, ne*GeVsqcm2*sigma_bhabha({"E_inc":bhabha_moller_energies, "Ee_min":self._Ee_min}), fill_value=0.0, bounds_error=False)

        II_y_Brem = np.array([quad(self._NSigmaBrem, BS[0][0],   BS[i][0], full_output=1)[0]   for i in range(len(BS))])
        self._interaction_integral_Brem = interp1d(np.transpose(BS)[0],   II_y_Brem, fill_value=0.0, bounds_error=False)

        II_y_PP = np.array([quad(self._NSigmaPP, PPS[0][0],   PPS[i][0], full_output=1)[0]   for i in range(len(PPS))])
        self._interaction_integral_PP = interp1d(np.transpose(PPS)[0],   II_y_PP, fill_value=0.0, bounds_error=False)

        II_y_Ann = np.array([quad(self._NSigmaAnn, AnnS[0][0],   AnnS[i][0], full_output=1)[0]   for i in range(len(AnnS))])
        self._interaction_integral_Ann = interp1d(np.transpose(AnnS)[0],   II_y_Ann, fill_value=0.0, bounds_error=False)

        II_y_Comp = np.array([quad(self._NSigmaComp, CS[0][0],   CS[i][0], full_output=1)[0]   for i in range(len(CS))])
        self._interaction_integral_Comp = interp1d(np.transpose(CS)[0],   II_y_Comp, fill_value=0.0, bounds_error=False)

        II_y_Moller = np.array([quad(self._NSigmaMoller, bhabha_moller_energies[0], bhabha_moller_energies[i], full_output=1)[0] for i in range(len(bhabha_moller_energies))])
        self._interaction_integral_Moller = interp1d(bhabha_moller_energies, II_y_Moller, fill_value=0.0, bounds_error=False)
        
        II_y_Bhabha = np.array([quad(self._NSigmaBhabha, bhabha_moller_energies[0], bhabha_moller_energies[i], full_output=1)[0] for i in range(len(bhabha_moller_energies))])
        self._interaction_integral_Bhabha = interp1d(bhabha_moller_energies, II_y_Bhabha, fill_value=0.0, bounds_error=False)
        
    def _positron_exponential_factor(self, E, Ei):
        """Returns the exponential factor for positron non-interaction probability
        over an energy interval [E, Ei]"""
        #this quantity has units of GeV/cm (assuming E, Ei given in GeV)
        n_sigma_diff = (self._interaction_integral_Brem(Ei) - self._interaction_integral_Brem(E)) \
                     + (self._interaction_integral_Ann(Ei) - self._interaction_integral_Ann(E))   \
                     + (self._interaction_integral_Bhabha(Ei) - self._interaction_integral_Bhabha(E))
        if n_sigma_diff < 0.0 or E > Ei:
            return 0.0
        #dEdxT has units of GeV/m
        dEdxT = self.get_material_properties()[3]*(0.1) #Converting MeV/cm to GeV/m
        #the ratio n_sigma_diff/dEdxT has units of (GeV/cm)/(GeV/m) = m/cm = 100
        return np.exp(-n_sigma_diff/dEdxT/cmtom)

    def _electron_exponential_factor(self, E, Ei):
        """Returns the exponential factor for electron non-interaction probability
        over an energy interval [E, Ei]"""
        n_sigma_diff = (self._interaction_integral_Brem(Ei) - self._interaction_integral_Brem(E)) \
                     + (self._interaction_integral_Moller(Ei) - self._interaction_integral_Moller(E))
        if n_sigma_diff < 0.0 or E > Ei:
            return 0.0
        dEdxT = self.get_material_properties()[3]*(0.1) #Converting MeV/cm to GeV/m
        return np.exp(-n_sigma_diff/dEdxT/cmtom)

    def _NSigmaElectron(self, E):
        """Returns n sigma for electrons as a function of energy in GeV"""
        return self._NSigmaBrem(E) + self._NSigmaMoller(E)
    def _NSigmaPhoton(self, E):
        """Returns n sigma for photons as a function of energy in GeV"""
        return self._NSigmaPP(E) + self._NSigmaComp(E)
    def _NSigmaPositron(self, E):
        """Returns n sigma for positrons as a function of energy in GeV"""
        return self._NSigmaBrem(E) + self._NSigmaBhabha(E) + self._NSigmaAnn(E)
    

    def get_mfp(self, particle): 
        """Returns particle mean free path in meters for PID=22 (photons), 
        11 (electrons) or -11 (positrons) as a function of energy in GeV"""
        if type(particle) is not Particle and (type(particle) is list or type(particle) is np.ndarray):
            PID, Energy = particle
        else:
            PID, Energy = particle.get_ids()["PID"], particle.get_pf()[0]
        if PID == 22:
            return cmtom*(self._NSigmaPP(Energy) + self._NSigmaComp(Energy))**-1
        elif PID == 11:
            return cmtom*(self._NSigmaBrem(Energy) + self._NSigmaMoller(Energy))**-1
        elif PID == -11:
            return cmtom*(self._NSigmaBrem(Energy) + self._NSigmaAnn(Energy) + self._NSigmaBhabha(Energy))**-1
        
    def BF_positron_brem(self, Energy):
        """Branching fraction for a positron to undergo brem vs annihilation"""
        b0, b1 = self._NSigmaBrem(Energy), self._NSigmaAnn(Energy)
        return b0/(b0+b1)
    def BF_photon_pairprod(self, Energy):
        """Branching fraction for a photon to undergo pair production vs compton"""
        b0, b1 = self._NSigmaPP(Energy), self._NSigmaComp(Energy)
        return b0/(b0+b1)


    def draw_sample(self,Einc,LU_Key=-1,process='PairProd',VB=False):
        """Draws a sample from the pre-computed VEGAS integrator for a given
        process and incoming energy.
        Inputs:
            Einc: incoming particle energy in GeV
            LU_Key: (look up key) index of the pre-computed VEGAS integrator corresponding to
            the closest incoming energy. If LU_Key is negative, the look up key identifies
            the closest incoming energy to Einc.
            process: string label of the process
            VB: boolean flag to print verbose output
        Returns:
            x: array of MC-sampled variables"""

        sample_list=self._loaded_samples 

        if LU_Key<0 or LU_Key > len(sample_list[process]):
            energies = sample_list[process][0:]
            energies = np.array([x[0] for x in energies])

            LU_Key = np.argmin(np.abs(energies - Einc)) + 1
            if LU_Key < 0:
                LU_Key = 0
                print("Warning: sampling below minimum energy for process" + str(process))
            if LU_Key >= len(sample_list[process]):
                LU_Key = len(sample_list[process]) - 1
                print("Warning: sampling above maximum energy for process" + str(process))

        sample_dict=sample_list[process][LU_Key][1]

        adaptive_map = sample_dict["adaptive_map"]
        max_F      = sample_dict["max_F"][self._target_material]*self._maxF_fudge_global
        neval_vegas= sample_dict["neval"]
        integrand=vg.Integrator(map=adaptive_map, max_nhcube=1, nstrat=np.ones(dimensionalities[process]), neval=neval_vegas)

        event_info={'E_inc': Einc, 'm_e': m_electron, 'Z_T': self._ZTarget, 'A_T':self._ATarget, 'mT':self._ATarget, 'alpha_FS': alpha_em, 'mV': 0, 'Eg_min':self._Egamma_min, 'Ee_min':self._Ee_min}
        
                
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
    
    def sample_scattering(self, p0, process, VB=False):
        E0 = p0.get_pf()[0]
        if E0 <= np.max([self._minimum_calculable_energy[p0.get_ids()["PID"]], self.min_energy, p0.get_ids()["mass"]]):
            return None
        RM = p0.rotation_matrix()
        sample_event = self.draw_sample(E0, process=process, VB=VB)

        NFVs = kinematic_function[process](p0, sample_event)

        E1f, p1xZF, p1yZF, p1zZF = NFVs[0]
        E2f, p2xZF, p2yZF, p2zZF = NFVs[1]

        p1_labframe = np.concatenate([[E1f], np.dot(RM, [p1xZF, p1yZF, p1zZF])])
        p2_labframe = np.concatenate([[E2f], np.dot(RM, [p2xZF, p2yZF, p2zZF])])

        init_IDs = p0.get_ids()
        p1_dict = {}
        p1_dict["PID"] = process_PIDS[process][0]
        p1_dict["parent_PID"] = init_IDs["PID"]
        p1_dict["ID"] = 2*(init_IDs["ID"]) + 0
        p1_dict["parent_ID"] = init_IDs["ID"]
        p1_dict["generation_number"] = init_IDs["generation_number"] + 1
        p1_dict["generation_process"] = process
        p1_dict["weight"] = init_IDs["weight"]

        p2_dict = p1_dict.copy()
        p2_dict["PID"] = process_PIDS[process][1]
        p2_dict["ID"] = 2*(init_IDs["ID"]) + 1

        if p1_dict["PID"] == 0:
            p1_dict["PID"] = init_IDs["PID"]
        if p2_dict["PID"] == 0:
            p2_dict["PID"] = init_IDs["PID"]

        p1_dict['mass'] = mass_dict[p1_dict["PID"]]
        p2_dict['mass'] = mass_dict[p2_dict["PID"]]

        new_particle1 = Particle(p1_labframe, p0.get_rf(), p1_dict)
        new_particle2 = Particle(p2_labframe, p0.get_rf(), p2_dict)
        
        return [new_particle1, new_particle2]

    def propagate_particle(self, Part0, Losses=False, MS=False):
        """Propagates a particle through material between hard scattering events, 
        possibly including multiple scattering and dE/dx losses
            Args:
                Part0: initial Particle object
                Losses: bool that indicates whether to include dE/dx losses
                MS: bool that indicates whether to include multiple scattering
            Returns:
                Part0: updated Particle object with new position and 
                (potentially) energy/momentum
        """
        delta_z=0
        
        if Part0.get_ended() is True:
            Part0.set_rf(Part0.get_rf())
            return Part0
        else:
            if (np.linalg.norm(Part0.get_p0() - Part0.get_pf()) > 0.0) \
               or (np.linalg.norm(Part0.get_r0() - Part0.get_rf()) > 0.0) \
               or (Part0.get_ended()):
                raise ValueError("propagate_particle() should only be called \
                for a particle with pf = p0 and rf = r0 and get_ended() == False")

            particle_min_energy = np.max([self._minimum_calculable_energy[Part0.get_ids()["PID"]],\
                                          self.min_energy, Part0.get_ids()["mass"]])
            if Part0.get_p0()[0] < particle_min_energy:
                Part0.set_ended(True)
                return Part0
            
            if Losses == False:
                mfp = self.get_mfp(Part0)
                distC = np.random.uniform(0.0, 1.0)
                dist = mfp*np.log(1.0/(1.0-distC))

                p0 = Part0.get_p0()[1:]
                if MS:
                    P0 = self._get_MCS_p(Part0.get_p0(), self._rhoTarget*(dist/cmtom), \
                                         self._ATarget, self._ZTarget,
                                         self._MCS_rescale_factor)
                    PHat = (p0 + P0[1:])/np.linalg.norm(p0+P0[1:])
                    Part0.set_pf(P0)
                else:
                    PHat = p0/np.linalg.norm(p0)
                x0, y0, z0 = Part0.get_r0()
                Part0.set_rf([x0 + PHat[0]*dist, y0 + PHat[1]*dist, z0 + PHat[2]*dist])

            else:
                z_travelled =0
                hard_scatter=False

                while hard_scatter == False and Part0.get_pf()[0] >= particle_min_energy:
                    mfp = self.get_mfp(Part0)
                    random_number = np.random.uniform(0.0, 1.0)
                    delta_z = mfp/np.random.uniform(low=6, high=20)
                
                    if random_number > np.exp( -delta_z/mfp):
                        hard_scatter = True
                    # If no hard scatter propagate particle and account for energy loss
                    else:
                        hard_scatter = False
                        Part0.lose_energy(Losses*delta_z)
                        z_travelled = z_travelled+delta_z

                        pfx, pfy, pfz = Part0.get_pf()[1:]
                        pf0 = np.linalg.norm([pfx, pfy, pfz])
                        if pf0 > 0.0:
                            x_current, y_current, z_current = Part0.get_rf()
                            Part0.set_rf([x_current + pfx/pf0*delta_z, \
                                          y_current + pfy/pf0*delta_z, z_current + pfz/pf0*delta_z])
                        if MS:
                            Part0.set_pf(self._get_MCS_p(Part0.get_pf(),\
                                                         self._rhoTarget*(delta_z/cmtom), \
                                                         self._ATarget, self._ZTarget, self._MCS_rescale_factor))

                distC = np.random.uniform(0.0, 1.0)
                if Part0._pf[0] < particle_min_energy:
                    last_increment = distC*delta_z
                else:
                    mfp = self.get_mfp(Part0)
                    last_increment = mfp*np.log(1.0/(1.0+(np.exp(-delta_z/mfp)-1)*distC))
                Part0.lose_energy(Losses*last_increment)
                pfx, pfy, pfz = Part0.get_pf()[1:]
                pf0 = np.linalg.norm([pfx, pfy, pfz])
                if pf0 > 0.0:
                    x_current, y_current, z_current = Part0.get_rf()
                    Part0.set_rf([x_current + pfx/pf0*last_increment, y_current + pfy/pf0*last_increment, z_current + pfz/pf0*last_increment])
                if MS:
                    Part0.set_pf(self._get_MCS_p(Part0.get_pf(),
                                                 self._rhoTarget*(last_increment/cmtom),
                                                 self._ATarget, self._ZTarget, self._MCS_rescale_factor) )

            Part0.set_ended(True)
            return Part0

    def generate_shower(self, p0, VB=False, GlobalMS=True):
        """
        Generates particle shower from an initial particle
        Args:
            p0: initial Particle 
            VB: bool to turn on/off verbose output
            GlobalMS: bool, multiple scattering flag. Set to false to disable multiple scattering of electrons and positrons

        Returns:
            AllParticles: a list of all particles generated in the shower
        """
        if VB:
            print("Starting shower, initial particle with ID Info")
            print(p0.get_ids())
            print("Initial four-momenta:")
            print(p0.get_p0())
        p0.set_ended(False)
        p0copy = copy.deepcopy(p0)
        all_particles = [p0copy]

        if GlobalMS==True:
            MS_e=True
            MS_g=False
        else:
            MS_e=False
            MS_g=False

        if p0.get_p0()[0] < self.min_energy:
            p0.set_ended(True)
            return all_particles

        while all([ap.get_ended() == True for ap in all_particles]) is False:
            for apI, ap in enumerate(all_particles):
                if ap.get_ended() is True:
                    continue
                else:
                    newparticles = None

                    if ap.get_ids()["stability"] == "short-lived":
                        newparticles = ap.decay_particle()
                    
                    elif ap.get_ids()["stability"] == "stable":
                        # Propagate particle until next hard interaction
                        if ap.get_ids()["PID"] == 22:
                            ap = self.propagate_particle(ap,MS=MS_g)
                        elif np.abs(ap.get_ids()["PID"]) == 11:
                            dEdxT = self.get_material_properties()[3]*(0.1) #Converting MeV/cm to GeV/m
                            ap = self.propagate_particle(ap, MS=MS_e, Losses=dEdxT)
                        
                        all_particles[apI] = ap

                        if (all([apC.get_ended() == True for apC in all_particles])\
                            is True and ap.get_pf()[0] < self.min_energy):
                            break

                        # Generate secondaries for the hard interaction
                        # Note: secondaries include the scattered parent particle 
                        # (i.e. the original the parent is not modified)
                        if ap.get_ids()["PID"] == 11:
                            choices0 = self._NSigmaBrem(ap.get_pf()[0]), self._NSigmaMoller(ap.get_pf()[0])
                            SC = np.sum(choices0)
                            if SC == 0.0 or np.isnan(SC):
                                continue
                            choices0 = choices0/SC
                            draw = np.random.choice(["Brem","Moller"], p=choices0)
                            newparticles = self.sample_scattering(ap, process=draw, VB=VB)
                        elif ap.get_ids()["PID"] == -11:
                            choices0 = self._NSigmaBrem(ap.get_pf()[0]), \
                                self._NSigmaAnn(ap.get_pf()[0]), self._NSigmaBhabha(ap.get_pf()[0])
                            SC = np.sum(choices0)
                            if SC == 0.0 or np.isnan(SC):
                                continue
                            choices0 = choices0/SC
                            draw = np.random.choice(["Brem","Ann","Bhabha"], p=choices0)
                            newparticles = self.sample_scattering(ap, process=draw, VB=VB)

                        elif ap.get_ids()["PID"] == 22:
                            choices0 = self._NSigmaPP(ap.get_pf()[0]), self._NSigmaComp(ap.get_pf()[0])
                            SC = np.sum(choices0)
                            if SC == 0.0 or np.isnan(SC):
                                continue
                            choices0 = choices0/SC
                            draw = np.random.choice(["PairProd", "Comp"], p=choices0)
                            newparticles = self.sample_scattering(ap, process=draw, VB=VB)

                    if newparticles is None:
                        continue
                    for dp in newparticles:
                        if dp.get_p0()[0] > self.min_energy:
                            all_particles.append(dp)
                    
        return all_particles

def event_display(all_particles):
    '''Draws event display for a list of particles'''
    import matplotlib
    from matplotlib import pyplot as plt
    from matplotlib.font_manager import FontProperties
    from matplotlib.ticker import FixedLocator, MaxNLocator

    # Set up fonts and figure specs
    font0 = FontProperties()
    font = font0.copy()
    font.set_size(24)
    font.set_family('serif')
    labelfont=font0.copy()
    labelfont.set_size(20)
    labelfont.set_weight('bold')
    legfont=font0.copy()
    legfont.set_size(18)
    legfont.set_weight('bold')

    figwid = 6.0*2.0
    fighei = 6.0*0.5
    lside = 3.0
    rside = 3.5
    wwspace = 1.25

    ncol = 1
    nrow = 2

    wid = lside + ncol*figwid + (ncol-1)*wwspace + rside

    bot = 3.77
    top = 3.5
    hhspace = 0.25

    hei = bot + nrow*fighei + (nrow-1)*hhspace + top

    lfactor = lside/wid
    rfactor = rside/wid
    bfactor = bot/hei
    tfactor = top/hei
    wfactor = wwspace/figwid
    hfactor = hhspace/fighei

    matplotlib.rcParams['axes.linewidth'] = 2.0
    fig, axes = plt.subplots(nrow, ncol, figsize=(wid, hei), facecolor='1.0');
    fig.subplots_adjust(left = lfactor, bottom=bfactor, right=(1.0-rfactor), top=(1.0-tfactor), wspace=wfactor, hspace=hfactor);

    # get max z and x or y values in all particles in the shower
    zmin = -0.02
    zmax = np.max([p.get_rf()[2] for p in all_particles])
    ymax = np.max([np.max([np.abs(p.get_rf()[0]) for p in all_particles]), np.max([np.abs(p.get_rf()[1]) for p in all_particles])])
    ymin = -ymax

    ax = axes[0]
    ax.axis([zmin, zmax, ymin, ymax])
    ax.plot([zmin, zmin, zmax, zmax, zmin], [ymin, ymax, ymax, ymin, ymin], ls='-', color='k', lw=4, zorder=50)
    ax.set_ylabel(r'$x\ [\mathrm{m}]$', fontproperties=font)        

    ax.tick_params(direction='in', zorder=30, length=20, width=2)
    ax.tick_params(direction='in', which='minor', zorder=30, length=15, width=1.5)
    [l.set_position((0.5, -0.015)) for l in ax.get_xticklabels()]
    [l.set_size((0)) for l in ax.get_xticklabels()]
    [l.set_size((labelfont.get_size())) for l in ax.get_yticklabels()]

    for ki0 in all_particles:
        ki = np.concatenate([ki0.get_r0(), ki0.get_rf()])
        if ki0.get_pid() == 22:
            with matplotlib.rc_context({'path.sketch': (5, 15, 1)}):
                ax.plot([ki[2], ki[5]], [ki[0], ki[3]], lw=1, ls='-', color='g', alpha=0.5)
        if ki0.get_pid() == 11:
            ax.plot([ki[2], ki[5]], [ki[0], ki[3]], lw=1, ls='-', color='r', alpha=0.5)
        if ki0.get_pid() == -11:
            ax.plot([ki[2], ki[5]], [ki[0], ki[3]], lw=1, ls='-', color='b', alpha=0.5)

    ax = axes[1]
    ax.axis([zmin, zmax, ymin, ymax])
    ax.plot([zmin, zmin, zmax, zmax, zmin], [ymin, ymax, ymax, ymin, ymin], ls='-', color='k', lw=4, zorder=50)
    ax.set_xlabel(r'$z\ [\mathrm{m}]$', fontproperties=font)        
    ax.set_ylabel(r'$y\ [\mathrm{m}]$', fontproperties=font)        

    ax.tick_params(direction='in', zorder=30, length=20, width=2)
    ax.tick_params(direction='in', which='minor', zorder=30, length=15, width=1.5)
    [l.set_position((0.5, -0.015)) for l in ax.get_xticklabels()]
    [l.set_size((labelfont.get_size())) for l in ax.get_xticklabels()]
    [l.set_size((labelfont.get_size())) for l in ax.get_yticklabels()]

    for ki0 in all_particles:
        ki = np.concatenate([ki0.get_r0(), ki0.get_rf()])
        if ki0.get_pid() == 22:
            with matplotlib.rc_context({'path.sketch': (5, 15, 1)}):
                ax.plot([ki[2], ki[5]], [ki[1], ki[4]], lw=1, ls='-', color='g', alpha=0.5)
        if ki0.get_pid() == 11:
            ax.plot([ki[2], ki[5]], [ki[1], ki[4]], lw=1, ls='-', color='r', alpha=0.5)
        if ki0.get_pid() == -11:
            ax.plot([ki[2], ki[5]], [ki[1], ki[4]], lw=1, ls='-', color='b', alpha=0.5)

def transverse_position(particle, z):
    '''Determines the transverse position of a particle at a given z'''

    x0, y0, z0 = particle.get_r0()
    E, px, py, pz = particle.get_p0()

    T = (z-z0)/pz #proxy for the time of propagation from z0 to z in the longitudinal direction
    xf, yf = x0 + T*px, y0 + T*py
    return [xf, yf]

def detector_cut(particle_list, detector_positions, detector_radius, method="Sample", energy_cut=None, detector_inner_radius=0.0):
    '''Places an imaginary detector of a certain size at a certain distance from the beam origin.
        A particle is assumed to have crossed the detector if it passes through a disk of radius detector_radius
        Input:
            -- particle_list: list of Particle objects
            -- detector_positions: list of positions of the detector centers (x,y,z) wrt the beam origin
            -- detector_radius: radius of the detector
            -- method: string that determines what is returned by the function.
                Determines which particles pass through the detector and returns a variety of options given by "method":
                -- "Sample": returns a list of lists of particles that pass through each detector
                -- "Efficiency": returns the fraction of particles (including weights) passing through the detector
                -- "TotalWeight": returns the total weight of particles passing through the detector
            -- energy_cut: tuple of minimum and maximum energies of particles to consider
            -- detector_inner_radius: inner radius of the detector
    '''
    particle_list = np.array(particle_list)

    if energy_cut is not None:
        energies = np.array([p0.get_p0()[0] for p0 in particle_list])
        particle_list = particle_list[np.where(energies < energy_cut[1],True,False)*np.where(energies>energy_cut[0],True,False)]

    if len(particle_list) == 0:
        if method == "Sample":
            return [[] for i in range(len(detector_positions))]
        else:
            return [0.0 for i in range(len(detector_positions))]

    rT = np.linalg.norm([transverse_position(p0, detector_positions) for p0 in particle_list], axis=1)

    pass_cuts_where = np.transpose(np.where(rT > detector_inner_radius, True, False)*np.where(rT < detector_radius, True, False))

    pass_cuts = [particle_list[pass_cuts_where[i]] for i in range(len(pass_cuts_where))]

    if method == "Sample":
        return pass_cuts
    elif method == "Efficiency":
        return [np.sum([p0.get_ids()["weight"] for p0 in pass_cuts[ii]])/np.sum([p0.get_ids()["weight"] for p0 in particle_list]) for ii in range(len(pass_cuts))]
    elif method == "TotalWeight":
        return [np.sum([p0.get_ids()["weight"] for p0 in pass_cuts[ii]]) for ii in range(len(pass_cuts))]