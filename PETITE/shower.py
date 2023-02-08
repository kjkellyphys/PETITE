import numpy as np

from scipy.interpolate import interp1d

from .moliere import get_scattered_momentum 
from .particle import Particle
from .kinematics import eegFourVecs, eeVFourVecs, gepemFourVecs, Compton_FVs, Ann_FVs

import sys

me = 0.000511

Z = {'graphite':6.0, 'lead':82.0} #atomic number of different targets
A = {'graphite':12.0, 'lead':207.2} #atomic mass of different targets
rho = {'graphite':2.210, 'lead':11.35} #g/cm^3
dEdx = {'graphite':2.0*rho['graphite'], 'lead':2.0*rho['lead']} #MeV per cm

GeVsqcm2 = 1.0/(5.06e13)**2 #Conversion between cross sections in GeV^{-2} to cm^2
cmtom = 0.01
mp0 = 1.673e-24 #g

process_code = {'brem':0, 'anni': 1, 'split': 2, 'compt': 3}

class Shower:
    """ Representation of a shower

    """
    def __init__(self, PickDir, TargetMaterial, MinEnergy):
        """Initializes the shower object.
        Args:
            PickDir: directory containing the pre-computed MC samples of various shower processes
            TargetMaterial: string label of the homogeneous material through which 
            particles propagate (available materials are the dict keys of 
            Z, A, rho, etc)
            MinEnergy: minimum particle energy in GeV at which the particle 
            finishes its propagation through the target
        """
        self.set_PickDir(PickDir)
        self.set_TargetMaterial(TargetMaterial)
        self.set_SampDir(PickDir + TargetMaterial + "/")
        self.set_SampDirE(PickDir + "electrons/")
        self.MinEnergy = MinEnergy

        self.set_MaterialProps()
        self.set_nTargets()
        self.set_samples()
        self.set_CrossSections()
        self.set_NSigmas()

    def set_PickDir(self, value):
        """Set the top level directory containing pre-computed MC pickles to value"""
        self._PickDir = value
    def get_PickDir(self):
        """Get the top level directory containing pre-computed MC pickles""" 
        return self._PickDir
    def set_TargetMaterial(self, value):
        """Set the string representing the target material to value"""
        self._TargetMaterial = value
    def get_TargetMaterial(self):
        """Get the string representing the target material"""
        return self._TargetMaterial
    def set_SampDir(self, value):
        """Set the directory containing pre-simulated MC events for processes involing target nuclei"""
        self._SampDir = value
    def get_SampDir(self):
        """Get the directory containing pre-simulated MC events for processes involing target nuclei"""
        return self._SampDir
    def set_SampDirE(self, value):
        """Set the directory containing pre-simulated MC events for processes involing target electrons"""
        self._SampDirE = value
    def get_SampDirE(self):
        """Get the directory containing pre-simulated MC events for processes involing target electrons"""
        return self._SampDirE

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

    def get_nTargets(self):
        """Returns nuclear and electron target densities for the 
           target material in 1/cm^3
        """

        return self._nTarget, self._nElecs

    def set_samples(self):
        """Loads the pre-computed MC pickles for shower processes"""
        self._BremSamples = np.load(self.get_SampDir()+"BremEvts.npy", allow_pickle=True)
        self._PPSamples = np.load(self.get_SampDir()+"PairProdEvts.npy", allow_pickle=True)
        self._AnnSamples = np.load(self.get_SampDirE()+"AnnihilationEvts.npy", allow_pickle=True)
        self._CompSamples = np.load(self.get_SampDirE()+"ComptonEvts.npy", allow_pickle=True)
    def get_BremSamples(self, ind):
        """Returns brem event from the sample with index ind"""
        return self._BremSamples[ind]
    def get_PPSamples(self, ind):
        """Returns pair production event from the sample with index ind"""
        return self._PPSamples[ind]
    def get_AnnSamples(self, ind):
        """Returns e+e- event from the sample with index ind"""
        return self._AnnSamples[ind]
    def get_CompSamples(self, ind):
        """Returns the Compton event from the sample with index ind"""
        return self._CompSamples[ind]

    def set_CrossSections(self):
        """Loads the pre-computed cross-sections for various shower processes 
        and extracts the minimum/maximum values of initial energies
        """

        # These files are arrays of [energy,cross-section] values
        self._BremXSec = np.load(self.get_SampDir()+"BremXSec.npy", allow_pickle=True)
        self._PPXSec = np.load(self.get_SampDir()+"PairProdXSec.npy", allow_pickle=True)
        self._AnnXSec = np.load(self.get_SampDirE()+"AnnihilationXSec.npy", allow_pickle=True)
        self._CompXSec = np.load(self.get_SampDirE()+"ComptonXSec.npy", allow_pickle=True)

        # lists of energies for which the cross-sections have been computed
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

    def ElecBremSample(self, Elec0):
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
        ts = self.get_BremSamples(LUKey)
        SampEvt = ts[np.random.randint(0, len(ts))]
        EeMod = self._EeVecBrem[LUKey]

        # reconstruct final electron and photon 4-momenta from the MC-sampled variables
        NFVs = eegFourVecs(Ee0, me, SampEvt[0]*Ee0/EeMod, np.cos(me/EeMod*SampEvt[1]), np.cos(me/(Ee0-SampEvt[0]*Ee0/EeMod)*SampEvt[2]), SampEvt[3])

        Eef, pexfZF, peyfZF, pezfZF = NFVs[1]
        Egf, pgxfZF, pgyfZF, pgzfZF = NFVs[2]

        pe3ZF = [pexfZF, peyfZF, pezfZF]
        pg3ZF = [pgxfZF, pgyfZF, pgzfZF]
        
        # Rotate back to lab frame
        pe3LF = np.dot(RM, pe3ZF)
        pg3LF = np.dot(RM, pg3ZF)
        
        pos = Elec0.get_rf()
        init_IDs = Elec0.get_IDs()

        NewE = Particle(init_IDs[0], Eef, pe3LF[0], pe3LF[1], pe3LF[2], pos[0], pos[1], pos[2], 2*(init_IDs[1])+0, init_IDs[1], init_IDs[0], init_IDs[4]+1,process_code['brem'], 1.0)
        NewG = Particle(22, Egf, pg3LF[0], pg3LF[1], pg3LF[2], pos[0], pos[1], pos[2], 2*(init_IDs[1])+1, init_IDs[1], init_IDs[0], init_IDs[4]+1, process_code['brem'], 1.0)

        return [NewE, NewG]

    def AnnihilationSample(self, Elec0):
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
        ts = self.get_AnnSamples(LUKey)
        SampEvt = ts[np.random.randint(0, len(ts))]
        EeMod = self._EeVecAnn[LUKey]

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

        NewG1 = Particle(22, Eg1f, pg3LF1[0], pg3LF1[1], pg3LF1[2], pos[0], pos[1], pos[2], 2*(init_IDs[1])+0, init_IDs[1], init_IDs[0], init_IDs[4]+1, process_code['anni'], 1.0)
        NewG2 = Particle(22, Eg2f, pg3LF2[0], pg3LF2[1], pg3LF2[2], pos[0], pos[1], pos[2], 2*(init_IDs[1])+1, init_IDs[1], init_IDs[0], init_IDs[4]+1, process_code['anni'], 1.0)

        return [NewG1, NewG2]

    def PhotonSplitSample(self, Phot0):
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
        ts = self.get_PPSamples(LUKey)
        SampEvt = ts[np.random.randint(0, len(ts))]
        EgMod = self._EgVecPP[LUKey]

        # reconstruct final electron and positron 4-momenta from the MC-sampled variables
        NFVs = gepemFourVecs(Eg0, me, SampEvt[0]*Eg0/EgMod, np.cos(me/EgMod*SampEvt[1]), np.cos(me/EgMod*SampEvt[2]), SampEvt[3])
        Eepf, pepxfZF, pepyfZF, pepzfZF = NFVs[1]
        Eemf, pemxfZF, pemyfZF, pemzfZF = NFVs[2]

        pep3ZF = [pepxfZF, pepyfZF, pepzfZF]
        pem3ZF = [pemxfZF, pemyfZF, pemzfZF]

        pep3LF = np.dot(RM, pep3ZF)
        pem3LF = np.dot(RM, pem3ZF)

        pos = Phot0.get_rf()
        init_IDs = Phot0.get_IDs()

        NewEp = Particle(-11,Eepf, pep3LF[0], pep3LF[1], pep3LF[2], pos[0], pos[1], pos[2], 2*(init_IDs[1])+0, init_IDs[1], init_IDs[0], init_IDs[4]+1, process_code['split'], 1.0)
        NewEm = Particle(11, Eemf, pem3LF[0], pem3LF[1], pem3LF[2], pos[0], pos[1], pos[2], 2*(init_IDs[1])+1, init_IDs[1], init_IDs[0], init_IDs[4]+1, process_code['split'], 1.0)

        return [NewEp, NewEm]

    def ComptonSample(self, Phot0):
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
        ts = self.get_CompSamples(LUKey)
        SampEvt = ts[np.random.randint(0, len(ts))]
        EgMod = self._EgVecComp[LUKey]

        # reconstruct final electron and photon 4-momenta from the MC-sampled variables
        NFVs = Compton_FVs(Eg0, me, 0.0, SampEvt[0])

        Eef, pexfZF, peyfZF, pezfZF = NFVs[0]
        Egf, pgxfZF, pgyfZF, pgzfZF = NFVs[1]


        pe3LF = np.dot(RM, [pexfZF, peyfZF, pezfZF])
        pg3LF = np.dot(RM, [pgxfZF, pgyfZF, pgzfZF])

        pos = Phot0.get_rf()
        init_IDs = Phot0.get_IDs()

        NewE = Particle(11, Eef, pe3LF[0], pe3LF[1], pe3LF[2], pos[0], pos[1], pos[2], 2*(init_IDs[1])+0, init_IDs[1], init_IDs[0], init_IDs[4]+1, process_code['compt'], 1.0)
        NewG = Particle(22, Egf, pg3LF[0], pg3LF[1], pg3LF[2], pos[0], pos[1], pos[2], 2*(init_IDs[1])+1, init_IDs[1], init_IDs[0], init_IDs[4]+1, process_code['compt'], 1.0)

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
                        npart = self.ElecBremSample(ap)
                    elif ap.get_IDs()[0] == -11:
                        BFEpBrem = self.BF_Positron_Brem(ap.get_pf()[0])
                        ch = np.random.uniform(low=0., high=1.0)
                        if ch < BFEpBrem:
                            npart = self.ElecBremSample(ap)
                        else:
                            npart = self.AnnihilationSample(ap)
                    elif ap.get_IDs()[0] == 22:
                        BFPhPP = self.BF_Photon_PP(ap.get_pf()[0])
                        ch = np.random.uniform(low=0., high=1.)
                        if ch < BFPhPP:
                            npart = self.PhotonSplitSample(ap)
                        else:
                            npart = self.ComptonSample(ap)
                    if (npart[0]).get_p0()[0] > self.MinEnergy:
                        AllParticles.append(npart[0])
                    if (npart[1]).get_p0()[0] > self.MinEnergy:
                        AllParticles.append(npart[1])

        return AllParticles