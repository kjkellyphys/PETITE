import numpy as np
import pickle 

from scipy.interpolate import interp1d

from .moliere import get_scattered_momentum 
from .particle import Particle
from .kinematics import e_to_egamma_fourvecs, e_to_eV_fourvecs, gamma_to_epem_fourvecs, compton_fourvecs, annihilation_fourvecs
from .shower import Shower
from .AllProcesses import *

import sys
from numpy.random import random as draw_U

me = 0.000511
        

Z = {'graphite':6.0, 'lead':82.0} #atomic number of different targets
A = {'graphite':12.0, 'lead':207.2} #atomic mass of different targets

GeVsqcm2 = 1.0/(5.06e13)**2 #Conversion between cross sections in GeV^{-2} to cm^2
cmtom = 0.01
mp0 = 1.673e-24 #g

process_code = {'ExactBrem':0, 'Ann': 1, 'Comp': 3}

class DarkShower(Shower):
    """ A class to reprocess an existing EM shower to generate dark photons
    """

    def __init__(self, dict_dir, target_material, min_energy, mV_in_GeV , \
                          mode="exact", maxF_fudge_global=1,max_n_integrators=int(1e4)):
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


        self.set_dark_dict_dir(dict_dir)
        self.set_target_material(target_material)
        self.min_energy = min_energy

        self.set_material_properties()
        self.set_n_targets()
        self.set_mV_list(dict_dir)
        self.set_mV(mV_in_GeV, mode)
        
        self.set_dark_cross_sections()
        self.set_dark_NSigmas()
        self.set_dark_samples()


        self._maxF_fudge_global=maxF_fudge_global

        #self._neval_vegas=neval
        self._max_n_integrators=max_n_integrators



  
    
    def set_dark_dict_dir(self, value):
        """Set the directory containing pre-simulated MC events for processes involing target nuclei"""
        self._dark_dict_dir = value

    def get_dark_dict_dir(self):
        """Get the top level directory containing pre-computed MC pickles""" 
        return self._dark_dict_dir
   

    def set_mV_list(self,dict_dir):
        sample_file=open(dict_dir + "dark_samp_dicts.pkl", 'rb')
        outer_dict=pickle.load(sample_file)
        sample_file.close()

        mass_list=list(outer_dict.keys())

        self._mV_list=mass_list


    def closest_lesser_value(self, input_list, input_value):
        arr = np.asarray(input_list)
        index = (np.abs(arr - input_value)).argmin()

        if arr[index]<input_value:
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
        sample_file=open(dict_dir + "dark_samp_dicts.pkl", 'rb')
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
        for process in process_code.keys():
            self._loaded_dark_samples[process]= \
                self.load_dark_sample(self._dict_dir, process)
            


    def load_dark_cross_section(self, dict_dir, process, target_material):
        dark_cross_section_file=open( dict_dir + "dark_xSec_Dicts.pkl", 'rb')
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
        self._dark_brem_cross_section = self.load_dark_cross_section(self._dict_dir, 'ExactBrem', self._target_material)
        self._dark_annihilation_cross_section  = self.load_dark_cross_section(self._dict_dir, 'Ann', self._target_material) 
        self._dark_compton_cross_section = self.load_dark_cross_section(self._dict_dir, 'Comp', self._target_material) 

        self._EeVecBrem = np.transpose(self._brem_cross_section)[0] #FIXME: not sure what these are
        self._EgVecPP = np.transpose(self._pair_production_cross_section)[0]
        self._EeVecAnn = np.transpose(self._annihilation_cross_section)[0]
        self._EgVecComp = np.transpose(self._compton_cross_section)[0]

        # log10s of minimum energes, energy spacing for the cross-section tables  
        self._logEeMinBrem, self._logEeSSBrem = np.log10(self._EeVecBrem[0]), np.log10(self._EeVecBrem[1]) - np.log10(self._EeVecBrem[0])
        self._logEeMinAnn, self._logEeSSAnn = np.log10(self._EeVecAnn[0]), np.log10(self._EeVecAnn[1]) - np.log10(self._EeVecAnn[0])
        self._logEgMinPP, self._logEgSSPP = np.log10(self._EgVecPP[0]), np.log10(self._EgVecPP[1]) - np.log10(self._EgVecPP[0])
        self._logEgMinComp, self._logEgSSComp= np.log10(self._EgVecComp[0]), np.log10(self._EgVecComp[1]) - np.log10(self._EgVecComp[0])



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
        self._NSigmaDarkBrem = interp1d(np.transpose(DBS)[0], nZ*GeVsqcm2*np.transpose(DBS)[1])
        self._NSigmaDarkAnn = interp1d(np.transpose(DAnnS)[0], ne*GeVsqcm2*np.transpose(DAnnS)[1])
        self._NSigmaDarkComp = interp1d(np.transpose(DCS)[0], ne*GeVsqcm2*np.transpose(DCS)[1])

    def GetBSMWeights(self, PID, Energy):
        """Compute relative weight of dark photon emission to the available SM processes
        Args: 
            PID: incoming PDG ID of the particle 
            Energy: its energy
        Returns:
            float, representing probability of V emission (for a fixed kinetic mixing) 
            divided by the probabilities of available SM processes

        """
        if PID == 22:
            if np.log10(Energy) < self._logEgMinDarkComp:
                return 0.0
            else:
                return self._NSigmaDarkComp(Energy)/(self._NSigmaPP(Energy) + self._NSigmaComp(Energy))
        elif PID == 11:
            if np.log10(Energy) < self._logEeMinDarkBrem:
                return 0.0
            else:
                return self._NSigmaDarkBrem(Energy)/self._NSigmaBrem(Energy)
        elif PID == -11:
            if np.log10(Energy) < self._logEeMinDarkBrem:
                BremPiece = 0.0
            else:
                BremPiece = self._NSigmaDarkBrem(Energy)
            if np.log10(Energy) < self._logEeMinDarkAnn:
                AnnPiece = 0.0
            else:
                AnnPiece = self._NSigmaDarkAnn(Energy)
            return (BremPiece + AnnPiece)/(self._NSigmaBrem(Energy) + self._NSigmaAnn(Energy))



    def draw_dark_sample(self,Einc,LU_Key,process,VB=False):

        dark_sample_list=self._loaded_dark_samples 

        # this grabs the dictionary part rather than the energy. 
        dark_sample_dict=dark_sample_list[process][LU_Key][1]

        integrand = dark_sample_dict["integrator"]
        max_F      = dark_sample_dict["max_F"]*self._maxF_fudge_global
        max_X      = dark_sample_dict["max_X"]
        max_wgt    = dark_sample_dict["max_wgt"]
        neval_vegas= dark_sample_dict["neval"]

        event_info={'E_inc': Einc, 'm_e': m_electron, 'Z_T': self._ZTarget, 'alpha_FS': alpha_em, 'm_V': self._mV}

        #### START FIXME
        # 
        # THESE ARE dummy FUNCTIONS
        diff_xsection_options={"dark_Comp"     : XX_dark_comp_func_XX,
                               "dark_Brem"     : XX_dark_brem_func_XX,
                               "dark_Ann"      : XX_dark_ann_func_XX }
        
        formfactor_dict      ={"dark_Comp"     : unity,
                               "dark_Brem"     : g2_elastic,
                               "dark_Ann"      : unity }

        QSq_dict             ={"dark_Brem"     : XX_dark_brem_q_sq_XX,
                                "dark_Comp"    : dummy, 
                                "dark_Ann"     : dummy }

        ###  END FIXME


        if process in diff_xsection_options:
            diff_xsec_func = diff_xsection_options[process]
            FF_func        = formfactor_dict[process]
            QSq_func       = QSq_dict[process]
        else:
            raise Exception("Your process is not in the list")

        integrand.set(max_nhcube=1, neval=neval_vegas)
        if VB:
            sampcount = 0
            n_integrators_used = 0
            sample_found = False
        while sample_found is False and n_integrators_used < self._max_n_integrators:
            n_integrators_used += 1
            for x,wgt in integrand.random():
                FF_eval=FF_func(event_info['Z_T'], m_electron, QSq_func(x, event_info ) )/event_info['Z_T']**2
                FF_H = FF_func(1.0, m_electron, QSq_func(x, event_info ) ) 
                if VB:
                    sampcount += 1  
                if  max_F*draw_U()<wgt*diff_xsec_func(event_info,x)*FF_eval/FF_H:
                    sample_found = True
                    break
        if sample_found is False:
            print("No Sample Found")
            #FIXME What to do when we end up here?
            #Coordinate with SM solution
        if VB:
            return np.concatenate([list(x), [sampcount]])
        else:
            return(x)


###      OLD CODE  |   CAN EVENTUALLY DELETE 
#        integrand.set(max_nhcube=1, neval=self._neval_vegas)
#        if VB:
#           sampcount = 0
#        for x,wgt in integrand.random():
#            FF_eval=FF_func(event_info['Z_T'], m_electron, QSq_func(x, m_electron, event_info['E_inc'] ) )
#            if VB:
#                sampcount += 1  
#            if  max_F*draw_U()<wgt*diff_xsec_func(event_info,x)*FF_eval:
#                break
#        if VB:
#            return np.concatenate([list(x), [sampcount]])
#        else:
#           return(x)


    def GetPositronDarkBF(self, Energy):
        """Branching fraction for a positron to undergo dark brem vs dark 
        annihilation"""
        if np.log10(Energy) < self._logEeMinDarkAnn:
            return 1.0
        else:
            return self._NSigmaDarkBrem(Energy)/(self._NSigmaDarkBrem(Energy) + self._NSigmaDarkAnn(Energy))

    def DarkElecBremSample(self, Elec0):
        """Generate a brem event from an initial electron/positron
            Args:
                Elec0: incoming electron/positron (instance of) Particle 
                in lab frame
            Returns:
                NewV: outgoing dark photon (instance of) Particle 
                in lab frame
        """
        Ee0, pex0, pey0, pez0 = Elec0.get_pf()

        ThZ = np.arccos(pez0/np.sqrt(pex0**2 + pey0**2 + pez0**2))
        PhiZ = np.arctan2(pey0, pex0)
        RM = [[np.cos(ThZ)*np.cos(PhiZ), -np.sin(PhiZ), np.sin(ThZ)*np.cos(PhiZ)],
            [np.cos(ThZ)*np.sin(PhiZ), np.cos(PhiZ), np.sin(ThZ)*np.sin(PhiZ)],
            [-np.sin(ThZ), 0, np.cos(ThZ)]]

        LUKey = int((np.log10(Ee0) - self._logEeMinDarkBrem)/self._logEeSSDarkBrem)
        LUKey = LUKey + 1
        
        ## FIXME Need right key name
        SampEvt = self.draw_sample(Ee0, LUKey, 'dark_Brem', VB=VB)


        ct = np.cos(self.get_mV()/(SampEvt[0])*np.sqrt((Ee0-SampEvt[0])/Ee0)*SampEvt[1])
        ctp =np.cos(self.get_mV()/(SampEvt[0])*np.sqrt(Ee0/(Ee0-SampEvt[0]))*SampEvt[2])
        NFVs = eeVFourVecs(Ee0, me, SampEvt[0], self.get_mV(), ct, ctp, SampEvt[3])

        EVf, pVxfZF, pVyfZF, pVzfZF = NFVs[2]
        pV3ZF = [pVxfZF, pVyfZF, pVzfZF]    
        pV3LF = np.dot(RM, pV3ZF)

        if EVf > Ee0:
            print("---------------------------------------------")
            print("High Energy V Found from Electron Samples:")
            print(Elec0.get_pf())
            print(EVf)
            print(SampEvt)
            print(LUKey)
            print(Ee0)
            print("---------------------------------------------")

        wg = self.GetBSMWeights(11, Ee0)

        GenType = process_code['dark_Brem']

        NewV = Particle(4900022, EVf, pV3LF[0], pV3LF[1], pV3LF[2], Elec0.get_rf()[0], Elec0.get_rf()[1], Elec0.get_rf()[2], 2*(Elec0.get_IDs()[1])+1, Elec0.get_IDs()[1], Elec0.get_IDs()[0], Elec0.get_IDs()[4]+1, GenType, wg)
        return NewV

    def DarkAnnihilationSample(self, Elec0):
        """Generate an annihilation event from an initial positron
            Args:
                Elec0: incoming positron (instance of) Particle in lab frame
            Returns:
                NewV: outgoing dark photon (instances of) Particle 
                in lab frame
        """

        Ee0, pex0, pey0, pez0 = Elec0.get_pf()

        ThZ = np.arccos(pez0/np.sqrt(pex0**2 + pey0**2 + pez0**2))
        PhiZ = np.arctan2(pey0, pex0)
        RM = [[np.cos(ThZ)*np.cos(PhiZ), -np.sin(PhiZ), np.sin(ThZ)*np.cos(PhiZ)],
            [np.cos(ThZ)*np.sin(PhiZ), np.cos(PhiZ), np.sin(ThZ)*np.sin(PhiZ)],
            [-np.sin(ThZ), 0, np.cos(ThZ)]]

        LUKey = int((np.log10(Ee0) - self._logEeMinDarkAnn)/self._logEeSSDarkAnn)
        LUKey = LUKey + 1
        
        # FIXME  need right key name 
        sample_event = self.draw_sample(Ee0, LUKey, 'dark_Ann', VB=VB)
        #NFVs = Ann_FVs(EeMod, meT, MVT, SampEvt[0])[1]
        NFVs = Ann_FVs(Ee0, me, self.get_mV(), SampEvt[0])[1]
        GenType = process_code['dark_Ann']

        EVf, pVxfZF, pVyfZF, pVzfZF = NFVs
        pV3ZF = [pVxfZF, pVyfZF, pVzfZF]    
        pV3LF = np.dot(RM, pV3ZF)
        wg = self.GetBSMWeights(-11, Ee0)

        if EVf > Ee0:
            print("---------------------------------------------")
            print("High Energy V Found from Positron Samples:")
            print(Elec0.get_pf())
            print(EVf)
            print(SampEvt)
            print(LUKey)
            print(EeMod)
            print(wg)
            print("---------------------------------------------")

        NewV = Particle(4900022, EVf, pV3LF[0], pV3LF[1], pV3LF[2], Elec0.get_rf()[0], Elec0.get_rf()[1], Elec0.get_rf()[2], 2*(Elec0.get_IDs()[1])+1, Elec0.get_IDs()[1], Elec0.get_IDs()[0], Elec0.get_IDs()[4]+1, GenType, wg)
        return NewV

    def DarkComptonSample(self, Phot0):
        """Generate a dark Compton event from an initial photon
            Args:
                Phot0: incoming photon (instance of) Particle in lab frame
            Returns:
                NewV: outgoing dark photon (instances of) Particle 
                in lab frame
        """
        Eg0, pgx0, pgy0, pgz0 = Phot0.get_pf()

        ThZ = np.arccos(pgz0/np.sqrt(pgx0**2 + pgy0**2 + pgz0**2))
        PhiZ = np.arctan2(pgy0, pgx0)
        RM = [[np.cos(ThZ)*np.cos(PhiZ), -np.sin(PhiZ), np.sin(ThZ)*np.cos(PhiZ)],
            [np.cos(ThZ)*np.sin(PhiZ), np.cos(PhiZ), np.sin(ThZ)*np.sin(PhiZ)],
            [-np.sin(ThZ), 0, np.cos(ThZ)]]

        LUKey = int((np.log10(Eg0) - self._logEgMinDarkComp)/self._logEgSSDarkComp)
        LUKey = LUKey + 1
        
        # FIXME Need right key name
        sample_event = self.draw_sample(Ee0, LUKey, 'dark_Comp', VB=VB)


        #NFVs = Compton_FVs(EgMod, meanniT, MVT, SampEvt[0])[1]
        NFVs = Compton_FVs(Eg0, me, self.get_mV(), SampEvt[0])[1]

        EVf, pVxfZF, pVyfZF, pVzfZF = NFVs
        pV3ZF = [pVxfZF, pVyfZF, pVzfZF]    
        pV3LF = np.dot(RM, pV3ZF)

        wg = self.GetBSMWeights(22, Eg0)
        GenType = process_code['dark_Comp']
        if EVf > Eg0:
            print("---------------------------------------------")
            print("High Energy V Found from Photon Samples:")
            print(Phot0.get_pf())
            print(EVf)
            print(SampEvt)
            print(LUKey)
            print(EgMod)
            print(wg)
            print("---------------------------------------------")

        NewV = Particle(4900022, EVf, pV3LF[0], pV3LF[1], pV3LF[2], Phot0.get_rf()[0], Phot0.get_rf()[1], Phot0.get_rf()[2], 2*(Phot0.get_IDs()[1])+0, Phot0.get_IDs()[1], Phot0.get_IDs()[0], Phot0.get_IDs()[4]+1, GenType, wg)
        return NewV

    def GenDarkShower(self, ExDir=None, SParams=None):
        """ Process an existing SM shower (or produce a new one) by interating 
        through its particles and generating possible dark photon emissions using 
        all available processes.
        Args:
            ExDir: path to file containing existing SM shower OR an actual shower (list of Particle objects)
            SParamas: if no path provided, parameters of a new SM shower to generate, 
            consisting of a tuple (PID0, p40, ParPID)
        Returns:
            [ShowerToSamp, NewShower]: where ShowerToSamp is the initial SM shower and NewShower 
            is the list of possible dark photon emissions generated from it
        """
        if ExDir is None and SParams is None:
            print("Need an existing SM shower-file directory or SM shower parameters to run dark shower")
            return None
        
        if ExDir is not None and type(ExDir)==str:
            ShowerToSamp = np.load(ExDir, allow_pickle=True)
        elif ExDir is not None and type(ExDir)==list:
            ShowerToSamp = ExDir
        else:
            PID0, p40, ParPID = SParams
            ShowerToSamp = self.GenShower(PID0, p40, ParPID)
        
        NewShower = []
        for ap in ShowerToSamp:
            if ap.get_IDs()[0] == 11:
                if np.log10(ap.get_pf()[0]) < self._logEeMinDarkBrem:
                    continue
                npart = self.DarkElecBremSample(ap)
            elif ap.get_IDs()[0] == -11:
                if np.log10(ap.get_pf()[0]) < self._logEeMinDarkBrem:
                    continue
                DarkBFEpBrem = self.GetPositronDarkBF(ap.get_pf()[0])
                ch = np.random.uniform(low=0., high=1.0)
                if ch < DarkBFEpBrem:
                    npart = self.DarkElecBremSample(ap)
                else:
                    npart = self.DarkAnnihilationSample(ap)
            elif ap.get_IDs()[0] == 22:
                if np.log10(ap.get_pf()[0]) < self._logEgMinDarkComp:
                    continue
                npart = self.DarkComptonSample(ap)
            NewShower.append(npart)

        return ShowerToSamp, NewShower
