import numpy as np
import pickle 

from scipy.interpolate import interp1d

from .moliere import get_scattered_momentum 
from .particle import Particle
from .kinematics import eegFourVecs, eeVFourVecs, gepemFourVecs, Compton_FVs, Ann_FVs
from .AllProcesses import *
from datetime import datetime

#np.random.seed(datetime.now().timestamp())
np.random.seed(19121974)

import sys
from numpy.random import random as draw_U


##  https://pdg.lbl.gov/2019/reviews/rpp2018-rev-phys-constants.pdf
me =0.51099895000 
alpha_FS =  1/137.035999084 

Z = {'graphite':6.0, 'lead':82.0} #atomic number of different targets
A = {'graphite':12.0, 'lead':207.2} #atomic mass of different targets
rho = {'graphite':2.210, 'lead':11.35} #g/cm^3
dEdx = {'graphite':2.0*rho['graphite'], 'lead':2.0*rho['lead']} #MeV per cm

GeVsqcm2 = 1.0/(5.06e13)**2 #Conversion between cross sections in GeV^{-2} to cm^2
cmtom = 0.01
mp0 = 1.673e-24 #g

process_code = {'Brem':0, 'Ann': 1, 'PairProd': 2, 'Comp': 3}

class Shower:
    """ Representation of a shower

    """
    def __init__(self, DictDir, TargetMaterial, MinEnergy, maxF_fudge_global=1):
        """Initializes the shower object.
        Args:
            DictDir: directory containing the pre-computed VEGAS integrators and auxillary info.
            TargetMaterial: string label of the homogeneous material through which 
            particles propagate (available materials are the dict keys of 
            Z, A, rho, etc)
            MinEnergy: minimum particle energy in GeV at which the particle 
            finishes its propagation through the target
        """

        
        ## Need to swap this out for integrator objects
        self.set_DictDir(DictDir)
        self.set_TargetMaterial(TargetMaterial)
        self.MinEnergy = MinEnergy

        self.set_MaterialProps()
        self.set_nTargets()
        self.set_CrossSections()
        self.set_NSigmas()
        self.set_samples()

        self._maxF_fudge_global=maxF_fudge_global


                
        
    def load_Samp(self, DictDir, Process, TargetMaterial):
        samp_file=open(DictDir + "samp_Dicts.pkl", 'rb')
        samp_Dict=pickle.load(samp_file)
        samp_file.close()

        if Process in samp_Dict.keys():
            return(samp_Dict[Process])
        else:
            print(Process)
            raise Exception("Process String does not match library")
    

    def load_xSec(self, DictDir, Process, TargetMaterial):
        xSec_file=open( DictDir + "xSec_Dicts.pkl", 'rb')
        xSec_Dict=pickle.load(xSec_file)
        xSec_file.close()

        if Process not in xSec_Dict:
            raise Exception("Process String does not match library")
        
        if TargetMaterial in xSec_Dict[Process]:
            return(xSec_Dict[Process][TargetMaterial])
        else:
            raise Exception("Target Material is not in library")



        
    def set_DictDir(self, value):
        """Set the top level directory containing pre-computed MC pickles to value"""
        self._DictDir = value
    def get_DictDir(self):
        """Get the top level directory containing pre-computed MC pickles""" 
        return self._DictDir
    def set_TargetMaterial(self, value):
        """Set the string representing the target material to value"""
        self._TargetMaterial = value
    def get_TargetMaterial(self):
        """Get the string representing the target material"""
        return self._TargetMaterial
    def set_MaterialProps(self):
        """Defines material properties (Z, A, rho, etc) based on the target 
        material label
        """
        self._ZTarget, self._ATarget, self._rhoTarget, self._dEdx = Z[self.get_TargetMaterial()], A[self.get_TargetMaterial()], rho[self.get_TargetMaterial()], dEdx[self.get_TargetMaterial()]

    def get_MaterialProps(self):
        """Returns target material properties: Z, A, rho, dE/dx"""
        return self._ZTarget, self._ATarget, self._rhoTarget, self._dEdx

    def set_nTargets(self):
        """Determines nuclear and electron target densities for the 
           target material
        """
        ZT, AT, rhoT, dEdxT = self.get_MaterialProps()
        self._nTarget = rhoT/mp0/AT
        self._nElecs = self._nTarget*ZT

    def set_samples(self):
        self._loaded_samples={}
        for Process in process_code.keys():
            self._loaded_samples[Process]= \
                self.load_Samp(self._DictDir, Process, self._TargetMaterial)

        
    def get_nTargets(self):
        """Returns nuclear and electron target densities for the 
           target material in 1/cm^3
        """

        return self._nTarget, self._nElecs

    def set_CrossSections(self):
        """Loads the pre-computed cross-sections for various shower processes 
        and extracts the minimum/maximum values of initial energies
        """

        # These contain only the cross sections for the chosen target material
        self._BremXSec = self.load_xSec(self._DictDir, 'Brem', self._TargetMaterial)
        self._PPXSec   = self.load_xSec(self._DictDir, 'PairProd', self._TargetMaterial)
        self._AnnXSec  = self.load_xSec(self._DictDir, 'Ann', self._TargetMaterial) 
        self._CompXSec = self.load_xSec(self._DictDir, 'Comp', self._TargetMaterial) 

        self._EeVecBrem = np.transpose(self._BremXSec)[0]
        self._EgVecPP = np.transpose(self._PPXSec)[0]
        self._EeVecAnn = np.transpose(self._AnnXSec)[0]
        self._EgVecComp = np.transpose(self._CompXSec)[0]

        # log10s of minimum energes, energy spacing for the cross-section tables  
        self._logEeMinBrem, self._logEeSSBrem = np.log10(self._EeVecBrem[0]), np.log10(self._EeVecBrem[1]) - np.log10(self._EeVecBrem[0])
        self._logEeMinAnn, self._logEeSSAnn = np.log10(self._EeVecAnn[0]), np.log10(self._EeVecAnn[1]) - np.log10(self._EeVecAnn[0])
        self._logEgMinPP, self._logEgSSPP = np.log10(self._EgVecPP[0]), np.log10(self._EgVecPP[1]) - np.log10(self._EgVecPP[0])
        self._logEgMinComp, self._logEgSSComp= np.log10(self._EgVecComp[0]), np.log10(self._EgVecComp[1]) - np.log10(self._EgVecComp[0])

    def get_BremXSec(self):
        """ Returns array of [energy,cross-section] values for brem """ 
        return self._BremXSec
    def get_PPXSec(self):
        """ Returns array of [energy,cross-section] values for pair production """ 
        return self._PPXSec
    def get_AnnXSec(self):
        """ Returns array of [energy,cross-section] values for e+e- annihilation """ 
        return self._AnnXSec
    def get_CompXSec(self):
        """ Returns array of [energy,cross-section] values for Compton """ 
        return self._CompXSec

    def set_NSigmas(self):
        """Constructs interpolations of n_T sigma (in 1/cm) as a functon of 
        incoming particle energy for each process
        """
        BS, PPS, AnnS, CS = self.get_BremXSec(), self.get_PPXSec(), self.get_AnnXSec(), self.get_CompXSec()
        nZ, ne = self.get_nTargets()
        self._NSigmaBrem = interp1d(np.transpose(BS)[0], nZ*GeVsqcm2*np.transpose(BS)[1])
        self._NSigmaPP = interp1d(np.transpose(PPS)[0], nZ*GeVsqcm2*np.transpose(PPS)[1])
        self._NSigmaAnn = interp1d(np.transpose(AnnS)[0], ne*GeVsqcm2*np.transpose(AnnS)[1])
        self._NSigmaComp = interp1d(np.transpose(CS)[0], ne*GeVsqcm2*np.transpose(CS)[1])

    def GetMFP(self, PID, Energy):
        """Returns particle mean free path in meters for PID=22 (photons), 
        11 (electrons) or -11 (positrons) as a function of energy in GeV"""
        if PID == 22:
            return cmtom*(self._NSigmaPP(Energy) + self._NSigmaComp(Energy))**-1
        elif PID == 11:
            return cmtom*(self._NSigmaBrem(Energy))**-1
        elif PID == -11:
            return cmtom*(self._NSigmaBrem(Energy) + self._NSigmaAnn(Energy))**-1
        
    def BF_Positron_Brem(self, Energy):
        """Branching fraction for a positron to undergo brem vs annihilation"""
        b0, b1 = self._NSigmaBrem(Energy), self._NSigmaAnn(Energy)
        return b0/(b0+b1)
    def BF_Photon_PP(self, Energy):
        """Branching fraction for a photon to undergo pair production vs compton"""
        b0, b1 = self._NSigmaPP(Energy), self._NSigmaComp(Energy)
        return b0/(b0+b1)




    def Draw_Sample(self,Einc,LU_Key,Process,VB=False):

        sample_list=self._loaded_samples 

        # this grabs the dictionary part rather than the energy. 
        sample_dict=sample_list[Process][LU_Key][1]

        integrand = sample_dict["integrator"]
        max_F      = sample_dict["max_F"]*self._maxF_fudge_global
        max_X      = sample_dict["max_X"]
        max_wgt    = sample_dict["max_wgt"]

        EvtInfo={'E_inc': Einc, 'm_e': me, 'Z_T': self._ZTarget, 'alpha_FS': alpha_FS, 'm_V': 0}
        diff_xsection_options={"PairProd" : dSPairProd_dP_T,
                               "Comp"     : dSCompton_dCT,
                               "Brem"     : dSBrem_dP_T,
                               "Ann"      : dAnn_dCT }
        
        FF_dict              ={"PairProd" : G2el,
                               "Comp"     : Unity,
                               "Brem"     : G2el,
                               "Ann"      : Unity }

        QSq_dict             ={"PairProd" : PPQSq, "Brem"     : BremQSq, "Comp": dummy, "Ann": dummy }

        
        if Process in diff_xsection_options:
            diff_xsec_func = diff_xsection_options[Process]
            FF_func        = FF_dict[Process]
        else:
            raise Exception("Your process is not in the list")

        integrand.set(max_nhcube=1, neval=300)
        if VB:
            sampcount = 0
        for x,wgt in integrand.random():
            FF_eval=FF_func(EvtInfo['Z_T'], me, QSq(x, me, EvtInfo['E_inc'] ) )
            if VB:
                sampcount += 1  
            if  max_F*draw_U()<wgt*diff_xsec_func(EvtInfo,x)*FF_eval:
                break
        if VB:
            return np.concatenate([list(x), [sampcount]])
        else:
            return(x)



    def ElecBremSample(self, Elec0, VB=False):
        """Generate a brem event from an initial electron/positron
            Args:
                Elec0: incoming electron/positron (instance of) Particle 
                in lab frame
            Returns:
                [NewE, NewG] where
                NewE: outgoing electron/positron (instance of) Particle 
                in lab frame
                NewG: outgoing photon (instance of) Particle 
                in lab frame
        """
        Ee0, pex0, pey0, pez0 = Elec0.get_pf()

        # Construct rotation matrix to rotate simulated event to lab frame
        ThZ = np.arccos(pez0/np.sqrt(pex0**2 + pey0**2 + pez0**2))
        PhiZ = np.arctan2(pey0, pex0)
        RM = [[np.cos(ThZ)*np.cos(PhiZ), -np.sin(PhiZ), np.sin(ThZ)*np.cos(PhiZ)],
            [np.cos(ThZ)*np.sin(PhiZ), np.cos(PhiZ), np.sin(ThZ)*np.sin(PhiZ)],
            [-np.sin(ThZ), 0, np.cos(ThZ)]]

        # Find the closest initial energy among the precomputed samples and get it
        LUKey = int((np.log10(Ee0) - self._logEeMinBrem)/self._logEeSSBrem)
        LUKey = LUKey + 1
        
        SampEvt = self.Draw_Sample(Ee0, LUKey, 'Brem', VB=VB)

                
        # reconstruct final electron and photon 4-momenta from the MC-sampled variables
        NFVs = eegFourVecs(Ee0, me, SampEvt[0], np.cos(me/Ee0*SampEvt[1]), np.cos(me/(Ee0-SampEvt[0])*SampEvt[2]), SampEvt[3])

        Eef, pexfZF, peyfZF, pezfZF = NFVs[1]
        Egf, pgxfZF, pgyfZF, pgzfZF = NFVs[2]

        pe3ZF = [pexfZF, peyfZF, pezfZF]
        pg3ZF = [pgxfZF, pgyfZF, pgzfZF]
        
        # Rotate back to lab frame
        pe3LF = np.dot(RM, pe3ZF)
        pg3LF = np.dot(RM, pg3ZF)
        
        pos = Elec0.get_rf()
        init_IDs = Elec0.get_IDs()

        if VB:
            newparticlewgt = SampEvt[-1]
        else:
            newparticlewgt = 1.0
        NewE = Particle(init_IDs[0], Eef, pe3LF[0], pe3LF[1], pe3LF[2], pos[0], pos[1], pos[2], 2*(init_IDs[1])+0, init_IDs[1], init_IDs[0], init_IDs[4]+1,process_code['Brem'], newparticlewgt)
        NewG = Particle(22, Egf, pg3LF[0], pg3LF[1], pg3LF[2], pos[0], pos[1], pos[2], 2*(init_IDs[1])+1, init_IDs[1], init_IDs[0], init_IDs[4]+1, process_code['Brem'], newparticlewgt)

        return [NewE, NewG]

    def AnnihilationSample(self, Elec0, VB=False):
        """Generate an annihilation event from an initial positron
            Args:
                Elec0: incoming positron (instance of) Particle in lab frame
            Returns:
                [NewG1, NewG2]: outgoing photons (instances of) Particle 
                in lab frame
        """

        Ee0, pex0, pey0, pez0 = Elec0.get_pf()

        # Construct rotation matrix to rotate simulated event to lab frame
        ThZ = np.arccos(pez0/np.sqrt(pex0**2 + pey0**2 + pez0**2))
        PhiZ = np.arctan2(pey0, pex0)
        RM = [[np.cos(ThZ)*np.cos(PhiZ), -np.sin(PhiZ), np.sin(ThZ)*np.cos(PhiZ)],
            [np.cos(ThZ)*np.sin(PhiZ), np.cos(PhiZ), np.sin(ThZ)*np.sin(PhiZ)],
            [-np.sin(ThZ), 0, np.cos(ThZ)]]

        # Find the closest initial energy among the precomputed samples and get it
        LUKey = int((np.log10(Ee0) - self._logEeMinAnn)/self._logEeSSAnn)
        LUKey = LUKey + 1
        
        SampEvt = self.Draw_Sample(Ee0, LUKey, 'Ann', VB=VB)

        # reconstruct final photon 4-momenta from the MC-sampled variables
        NFVs = Ann_FVs(Ee0, me, 0.0, SampEvt[0])

        Eg1f, pg1xfZF, pg1yfZF, pg1zfZF = NFVs[0]
        Eg2f, pg2xfZF, pg2yfZF, pg2zfZF = NFVs[1]

        pg3ZF1 = [pg1xfZF, pg1yfZF, pg1zfZF]
        pg3ZF2 = [pg2xfZF, pg2yfZF, pg2zfZF]
    
        pg3LF1 = np.dot(RM, pg3ZF1)
        pg3LF2 = np.dot(RM, pg3ZF2)   

        pos = Elec0.get_rf()
        init_IDs = Elec0.get_IDs()

        if VB:
            newparticlewgt = SampEvt[-1]
        else:
            newparticlewgt = 1.0

        NewG1 = Particle(22, Eg1f, pg3LF1[0], pg3LF1[1], pg3LF1[2], pos[0], pos[1], pos[2], 2*(init_IDs[1])+0, init_IDs[1], init_IDs[0], init_IDs[4]+1, process_code['Ann'], newparticlewgt)
        NewG2 = Particle(22, Eg2f, pg3LF2[0], pg3LF2[1], pg3LF2[2], pos[0], pos[1], pos[2], 2*(init_IDs[1])+1, init_IDs[1], init_IDs[0], init_IDs[4]+1, process_code['Ann'], newparticlewgt)

        return [NewG1, NewG2]

    def PairProdSample(self, Phot0, VB=False):
        """Generate a photon splitting event from an initial photon
            Args:
                Phot0: incoming positron (instance of) Particle in lab frame
            Returns:
                [NewEp, NewEm]: outgoing positron and electron (instances of) Particle 
                in lab frame
        """
        Eg0, pgx0, pgy0, pgz0 = Phot0.get_pf()

        # Construct rotation matrix to rotate simulated event to lab frame
        ThZ = np.arccos(pgz0/np.sqrt(pgx0**2 + pgy0**2 + pgz0**2))
        PhiZ = np.arctan2(pgy0, pgx0)
        RM = [[np.cos(ThZ)*np.cos(PhiZ), -np.sin(PhiZ), np.sin(ThZ)*np.cos(PhiZ)],
            [np.cos(ThZ)*np.sin(PhiZ), np.cos(PhiZ), np.sin(ThZ)*np.sin(PhiZ)],
            [-np.sin(ThZ), 0, np.cos(ThZ)]]

        # Find the closest initial energy among the precomputed samples and get it
        LUKey = int((np.log10(Eg0) - self._logEgMinPP)/self._logEgSSPP)
        LUKey = LUKey + 1
        
        SampEvt = self.Draw_Sample(Eg0, LUKey, 'PairProd', VB=VB)

        
        # reconstruct final electron and positron 4-momenta from the MC-sampled variables
        NFVs = gepemFourVecs(Eg0, me, SampEvt[0], np.cos(me/Eg0*SampEvt[1]), np.cos(me/Eg0*SampEvt[2]), SampEvt[3])
        Eepf, pepxfZF, pepyfZF, pepzfZF = NFVs[1]
        Eemf, pemxfZF, pemyfZF, pemzfZF = NFVs[2]

        pep3ZF = [pepxfZF, pepyfZF, pepzfZF]
        pem3ZF = [pemxfZF, pemyfZF, pemzfZF]

        pep3LF = np.dot(RM, pep3ZF)
        pem3LF = np.dot(RM, pem3ZF)

        pos = Phot0.get_rf()
        init_IDs = Phot0.get_IDs()

        if VB:
            newparticlewgt = SampEvt[-1]
        else:
            newparticlewgt = 1.0

        NewEp = Particle(-11,Eepf, pep3LF[0], pep3LF[1], pep3LF[2], pos[0], pos[1], pos[2], 2*(init_IDs[1])+0, init_IDs[1], init_IDs[0], init_IDs[4]+1, process_code['PairProd'], newparticlewgt)
        NewEm = Particle(11, Eemf, pem3LF[0], pem3LF[1], pem3LF[2], pos[0], pos[1], pos[2], 2*(init_IDs[1])+1, init_IDs[1], init_IDs[0], init_IDs[4]+1, process_code['PairProd'], newparticlewgt)

        return [NewEp, NewEm]

    def ComptonSample(self, Phot0, VB=False):
        """Generate a Compton event from an initial photon
            Args:
                Phot0: incoming photon (instance of) Particle in lab frame
            Returns:
                [NewE, NewG]: electron and photon (instances of) Particle 
                in lab frame
        """

        Eg0, pgx0, pgy0, pgz0 = Phot0.get_pf()

        # Construct rotation matrix to rotate simulated event to lab frame
        ThZ = np.arccos(pgz0/np.sqrt(pgx0**2 + pgy0**2 + pgz0**2))
        PhiZ = np.arctan2(pgy0, pgx0)
        RM = [[np.cos(ThZ)*np.cos(PhiZ), -np.sin(PhiZ), np.sin(ThZ)*np.cos(PhiZ)],
            [np.cos(ThZ)*np.sin(PhiZ), np.cos(PhiZ), np.sin(ThZ)*np.sin(PhiZ)],
            [-np.sin(ThZ), 0, np.cos(ThZ)]]

        # Find the closest initial energy among the precomputed samples and get it
        LUKey = int((np.log10(Eg0) - self._logEgMinComp)/self._logEgSSComp)        
        LUKey = LUKey + 1
        
        SampEvt = self.Draw_Sample(Eg0, LUKey, 'Comp', VB=VB)

        
        # reconstruct final electron and photon 4-momenta from the MC-sampled variables
        NFVs = Compton_FVs(Eg0, me, 0.0, SampEvt[0])

        Eef, pexfZF, peyfZF, pezfZF = NFVs[0]
        Egf, pgxfZF, pgyfZF, pgzfZF = NFVs[1]


        pe3LF = np.dot(RM, [pexfZF, peyfZF, pezfZF])
        pg3LF = np.dot(RM, [pgxfZF, pgyfZF, pgzfZF])

        pos = Phot0.get_rf()
        init_IDs = Phot0.get_IDs()

        if VB:
            newparticlewgt = SampEvt[-1]
        else:
            newparticlewgt = 1.0

        NewE = Particle(11, Eef, pe3LF[0], pe3LF[1], pe3LF[2], pos[0], pos[1], pos[2], 2*(init_IDs[1])+0, init_IDs[1], init_IDs[0], init_IDs[4]+1, process_code['Comp'], newparticlewgt)
        NewG = Particle(22, Egf, pg3LF[0], pg3LF[1], pg3LF[2], pos[0], pos[1], pos[2], 2*(init_IDs[1])+1, init_IDs[1], init_IDs[0], init_IDs[4]+1, process_code['Comp'], newparticlewgt)

        return [NewE, NewG]

    def PropagateParticle(self, Part0, Losses=False, MS=False):
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
        if Part0.get_Ended() is True:
            Part0.set_rf(Part0.get_rf())
            return Part0
        else:
            mfp = self.GetMFP(Part0.get_IDs()[0], Part0.get_p0()[0])
            distC = np.random.uniform(0.0, 1.0)
            dist = mfp*np.log(1.0/(1.0-distC))
            if np.abs(Part0.get_IDs()[0]) == 11:
                M0 = me
            elif Part0.get_IDs()[0] == 22:
                M0 = 0.0

            E0, px0, py0, pz0 = Part0.get_p0()
            if MS:
                ZT, AT, rhoT, dEdxT = self.get_MaterialProps()
                EF0, PxF0, PyF0, PzF0 = get_scattered_momentum(Part0.get_p0(), rhoT*(dist/cmtom), AT, ZT)
                PHatDenom = np.sqrt((PxF0 + px0)**2 + (PyF0 + py0)**2 + (PzF0 + pz0)**2)
                PHat = [(PxF0 + px0)/PHatDenom, (PyF0 + py0)/PHatDenom, (PzF0 + pz0)/PHatDenom]
            else:
                PHatDenom = np.sqrt(px0**2 + py0**2 + pz0**2)
                PHat = [(px0)/PHatDenom, (py0)/PHatDenom, (pz0)/PHatDenom]

            p30 = np.sqrt(px0**2 + py0**2 + pz0**2)

            x0, y0, z0 = Part0.get_r0()
            Part0.set_rf([x0 + PHat[0]*dist, y0 + PHat[1]*dist, z0 + PHat[2]*dist])

            if Losses is False:
                if MS:
                    Part0.set_pf(np.array([E0, PxF0, PyF0, PzF0]))
                else:
                    Part0.set_pf(Part0.get_p0())
            else:
                Ef = E0 - Losses*dist
                if Ef <= M0 or Ef < self.MinEnergy:
                    #print("Particle lost too much energy along path of propagation!")
                    Part0.set_Ended(True)
                    return Part0
                Part0.set_pf(np.array([Ef, px0/p30*np.sqrt(Ef**2-M0**2), py0/p30*np.sqrt(Ef**2-M0**2), pz0/p30*np.sqrt(Ef**2-M0**2)]))

            Part0.set_Ended(True)
            return Part0

    def GenShower(self, PID0, p40, ParPID, VB=False, GlobalMS=True):
        """
        Generates particle shower from an initial particle
        Args:
            PID0: PDG ID of the initial particle
            p40: four-momentum of the initial particle
            ParID: PDG ID of the parent of the initial particle
            VB: bool to turn on/off verbose output

        Returns:
            AllParticles: a list of all particles generated in the shower
        """
        p0 = Particle(PID0, p40[0], p40[1], p40[2], p40[3], 0.0, 0.0, 0.0, 1, 0, ParPID, 0, -1, 1.0)
        if VB:
            print("Starting shower, initial particle with ID Info")
            print(p0.get_IDs())
            print("Initial four-momenta:")
            print(p0.get_p0())

        AllParticles = [p0]

        if GlobalMS==True:
            MS_e=True
            MS_g=False
        else:
            MS_e=False
            MS_g=False

        if p0.get_p0()[0] < self.MinEnergy:
            p0.set_Ended(True)
            return AllParticles

        while all([ap.get_Ended() == True for ap in AllParticles]) is False:
            for apI, ap in enumerate(AllParticles):
                if ap.get_Ended() is True:
                    continue
                else:
                    # Propagate particle until next hard interaction
                    if ap.get_IDs()[0] == 22:
                        ap = self.PropagateParticle(ap,MS=MS_g)
                    elif np.abs(ap.get_IDs()[0]) == 11:
                        dEdxT = self.get_MaterialProps()[3]*(0.1) #Converting MeV/cm to GeV/m
                        ap = self.PropagateParticle(ap, MS=MS_e, Losses=dEdxT)

                    AllParticles[apI] = ap
                    
                    if (all([ap.get_Ended() == True for ap in AllParticles]) is True and ap.get_pf()[0] < self.MinEnergy):
                        break

                    # Generate secondaries for the hard interaction
                    # Note: secondaries include the scattered parent particle 
                    # (i.e. the original the parent is not modified)
                    if ap.get_IDs()[0] == 11:
                        npart = self.ElecBremSample(ap, VB=VB)
                    elif ap.get_IDs()[0] == -11:
                        BFEpBrem = self.BF_Positron_Brem(ap.get_pf()[0])
                        ch = np.random.uniform(low=0., high=1.0)
                        if ch < BFEpBrem:
                            npart = self.ElecBremSample(ap, VB=VB)
                        else:
                            npart = self.AnnihilationSample(ap, VB=VB)
                    elif ap.get_IDs()[0] == 22:
                        BFPhPP = self.BF_Photon_PP(ap.get_pf()[0])
                        ch = np.random.uniform(low=0., high=1.)
                        if ch < BFPhPP:
                            npart = self.PairProdSample(ap, VB=VB)
                        else:
                            npart = self.ComptonSample(ap, VB=VB)
                    if (npart[0]).get_p0()[0] > self.MinEnergy:
                        AllParticles.append(npart[0])
                    if (npart[1]).get_p0()[0] > self.MinEnergy:
                        AllParticles.append(npart[1])

        return AllParticles
