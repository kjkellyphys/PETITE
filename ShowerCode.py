import numpy as np
from scipy.interpolate import interp1d
from moliere import get_scattered_momentum

me = 0.000511
MinEnergy = 0.010
ZMax = 10.0

class Particle:
    def __init__(self, PID, E0, px0, py0, pz0, x0, y0, z0, ID, ParID, ParPID, GenID, GenProcess):
        self.set_IDs(np.array([PID, ID, ParPID, ParID, GenID, GenProcess]))

        self.set_p0(np.array([E0, px0, py0, pz0]))
        self.set_r0(np.array([x0, y0, z0]))

        self.set_Ended(False)

        self.set_pf(np.array([E0,px0,py0,pz0]))
        self.set_rf(np.array([x0, y0, z0]))

    def set_IDs(self, value):
        self._IDs = value
    def get_IDs(self):
        return self._IDs

    def set_p0(self, value):
        self._p0 = value
    def get_p0(self):
        return self._p0
    def set_pf(self, value):
        self._pf = value
    def get_pf(self):
        return self._pf

    def set_r0(self, value):
        self._r0 = value
    def get_r0(self):
        return self._r0
    def set_rf(self, value):
        self._rf = value
    def get_rf(self):
        return self._rf

    def set_Ended(self, value):    
        if value != True and value != False:
            raise ValueError("Ended property must be a boolean.")
        self._Ended = value
    def get_Ended(self):
        return self._Ended
        
def eegFourVecs(ep, me, w, ct, ctp, ph):
    epp = ep - w
    p, pp = np.sqrt(ep**2 - me**2), np.sqrt(epp**2 - me**2)

    Em4v = [ep, 0, 0, p] #Four-vector of electron
    al = np.random.uniform(0, 2.0*np.pi)
    cal, sal = np.cos(al), np.sin(al)
    st, stp = np.sqrt(1.0 - ct**2), np.sqrt(1.0 - ctp**2)
    sp, cp = np.sin(ph), np.cos(ph)
    g4v = [w, w*cal*st, w*sal*st, w*ct] #Four-vector of photon

    Ep4v = [epp, pp*(sal*sp*stp + cal*(ctp*st - cp*ct*stp)), pp*(ctp*sal*st - (cp*ct*sal + cal*sp)*stp), pp*(ct*ctp + cp*st*stp)] #Four-vector of positron

    return [Em4v, Ep4v, g4v]

def gepemFourVecs(w, me, epp, ctp, ctm, ph):
    epm = w - epp
    pm, pp = np.sqrt(epm**2 - me**2), np.sqrt(epp**2 - me**2)

    Eg4v = [w, 0, 0, w]
    al = np.random.uniform(0, 2.0*np.pi)

    cal, sal = np.cos(al), np.sin(al)
    stp, stm = np.sqrt(1.0 - ctp**2), np.sqrt(1.0 - ctm**2)
    spal, cpal = np.sin(ph+al), np.cos(ph+al)

    pp4v = [epp, pp*stp*cal, pp*stp*sal, pp*ctp]
    pm4v = [epm, pm*stm*cpal, pm*stm*spal, pm*ctm]

    return [Eg4v, pp4v, pm4v]
    
def Compton_FVs(Eg, me, mV, ct):
    s = me**2 + 2*Eg*me
    Ee0 = (s + me**2)/(2.0*np.sqrt(s))
    Ee = (s - mV**2 + me**2)/(2*np.sqrt(s))
    EV = (s + mV**2 - me**2)/(2*np.sqrt(s))
    pF = np.sqrt(Ee**2 - me**2)

    g0 = Ee0/me
    b0 = 1.0/g0*np.sqrt(g0**2 - 1.0)

    ph = np.random.uniform(0, 2.0*np.pi)
    pe4v = [g0*Ee - b0*g0*pF*ct, -pF*np.sqrt(1-ct**2)*np.sin(ph), -pF*np.sqrt(1-ct**2)*np.cos(ph), b0*g0*Ee-g0*pF*ct]
    pV4v = [g0*EV + b0*g0*pF*ct, pF*np.sqrt(1-ct**2)*np.sin(ph), pF*np.sqrt(1-ct**2)*np.cos(ph), b0*g0*EV + g0*pF*ct]

    return [pe4v, pV4v]

def Ann_FVs(Ee, me, mV, ct):
    s = 2*me*(Ee+me)
    EeCM = np.sqrt(s)/2.0
    Eg = (s - mV**2)/(2*np.sqrt(s))
    EV = (s + mV**2)/(2*np.sqrt(s))
    pF = Eg

    g0 = EeCM/me
    b0 = 1.0/g0*np.sqrt(g0**2-1.0)

    ph = np.random.uniform(0.0, 2.0*np.pi)

    if ct < -1.0 or ct > 1.0:
        print("Error in Annihiliation Calculation")
        print(Ee, me, mV, ct)

    pg4v = [g0*Eg - b0*g0*pF*ct, -pF*np.sqrt(1-ct**2)*np.sin(ph), -pF*np.sqrt(1-ct**2)*np.cos(ph), b0*g0*Eg - g0*pF*ct]
    pV4v = [g0*EV + b0*g0*pF*ct, pF*np.sqrt(1-ct**2)*np.sin(ph), pF*np.sqrt(1-ct**2)*np.cos(ph), b0*g0*EV + g0*pF*ct]

    return [pg4v, pV4v]

Z = {'graphite':6.0, 'lead':82.0} #atomic number of different targets
A = {'graphite':12.0, 'lead':207.2} #atomic mass of different targets
rho = {'graphite':2.210, 'lead':11.35} #g/cm^3

GeVsqcm2 = 1.0/(5.06e13)**2 #Conversion between cross sections in GeV^{-2} to cm^2
cmtom = 0.01
mp0 = 1.673e-24 #g

class Shower:
    def __init__(self, PickDir, TargetMaterial):
        self.set_PickDir(PickDir)
        self.set_TargetMaterial(TargetMaterial)
        self.set_SampDir(PickDir + TargetMaterial + "/")

        self.set_MaterialProps()
        self.set_nTargets()
        self.set_samples()
        self.set_CrossSections()
        self.set_NSigmas()

    def set_PickDir(self, value):
        self._PickDir = value
    def get_PickDir(self):
        return self._PickDir
    def set_TargetMaterial(self, value):
        self._TargetMaterial = value
    def get_TargetMaterial(self):
        return self._TargetMaterial
    def set_SampDir(self, value):
        self._SampDir = value
    def get_SampDir(self):
        return self._SampDir

    def set_MaterialProps(self):
        self._ZTarget, self._ATarget, self._rhoTarget = Z[self.get_TargetMaterial()], A[self.get_TargetMaterial()], rho[self.get_TargetMaterial()]
    def get_MaterialProps(self):
        return self._ZTarget, self._ATarget, self._rhoTarget

    def set_nTargets(self):
        ZT, AT, rhoT = self.get_MaterialProps()
        self._nTarget = rhoT/mp0/AT
        self._nElecs = self._nTarget*ZT
    def get_nTargets(self):
        return self._nTarget, self._nElecs

    def set_samples(self):
        self._BremSamples = np.load(self.get_SampDir()+"BremEvts.npy", allow_pickle=True)
        self._PPSamples = np.load(self.get_SampDir()+"PairProdEvts.npy", allow_pickle=True)
        self._AnnSamples = np.load(self.get_SampDir()+"AnnihilationEvts.npy", allow_pickle=True)
        self._CompSamples = np.load(self.get_SampDir()+"ComptonEvts.npy", allow_pickle=True)
    def get_BremSamples(self, ind):
        return self._BremSamples[ind]
    def get_PPSamples(self, ind):
        return self._PPSamples[ind]
    def get_AnnSamples(self, ind):
        return self._AnnSamples[ind]
    def get_CompSamples(self, ind):
        return self._CompSamples[ind]

    def set_CrossSections(self):
        self._BremXSec = np.load(self.get_SampDir()+"BremXSec.npy", allow_pickle=True)
        self._PPXSec = np.load(self.get_SampDir()+"PairProdXSec.npy", allow_pickle=True)
        self._AnnXSec = np.load(self.get_SampDir()+"AnnihilationXSec.npy", allow_pickle=True)
        self._CompXSec = np.load(self.get_SampDir()+"ComptonXSec.npy", allow_pickle=True)

        self._EeVecBrem = np.transpose(self._BremXSec)[0]
        self._EgVecPP = np.transpose(self._PPXSec)[0]
        self._EeVecAnn = np.transpose(self._AnnXSec)[0]
        self._EgVecComp = np.transpose(self._CompXSec)[0]

        self._logEeMinBrem, self._logEeSSBrem = np.log10(self._EeVecBrem[0]), np.log10(self._EeVecBrem[1]) - np.log10(self._EeVecBrem[0])
        self._logEeMinAnn, self._logEeSSAnn = np.log10(self._EeVecAnn[0]), np.log10(self._EeVecAnn[1]) - np.log10(self._EeVecAnn[0])
        self._logEgMinPP, self._logEgSSPP = np.log10(self._EgVecPP[0]), np.log10(self._EgVecPP[1]) - np.log10(self._EgVecPP[0])
        self._logEgMinComp, self._logEgSSComp= np.log10(self._EgVecComp[0]), np.log10(self._EgVecComp[1]) - np.log10(self._EgVecComp[0])

    def get_BremXSec(self):
        return self._BremXSec
    def get_PPXSec(self):
        return self._PPXSec
    def get_AnnXSec(self):
        return self._AnnXSec
    def get_CompXSec(self):
        return self._CompXSec

    def set_NSigmas(self):
        BS, PPS, AnnS, CS = self.get_BremXSec(), self.get_PPXSec(), self.get_AnnXSec(), self.get_CompXSec()
        nZ, ne = self.get_nTargets()
        self._NSigmaBrem = interp1d(np.transpose(BS)[0], nZ*GeVsqcm2*np.transpose(BS)[1])
        self._NSigmaPP = interp1d(np.transpose(PPS)[0], nZ*GeVsqcm2*np.transpose(PPS)[1])
        self._NSigmaAnn = interp1d(np.transpose(AnnS)[0], ne*GeVsqcm2*np.transpose(AnnS)[1])
        self._NSigmaComp = interp1d(np.transpose(CS)[0], ne*GeVsqcm2*np.transpose(CS)[1])

    def GetMFP(self, PID, Energy):
        if PID == 22:
            return cmtom*(self._NSigmaPP(Energy) + self._NSigmaComp(Energy))**-1
        elif PID == 11:
            return cmtom*(self._NSigmaBrem(Energy))**-1
        elif PID == -11:
            return cmtom*(self._NSigmaBrem(Energy) + self._NSigmaAnn(Energy))**-1
        
    def BF_Positron_Brem(self, Energy):
        b0, b1 = self._NSigmaBrem(Energy), self._NSigmaAnn(Energy)
        return b0/(b0+b1)
    def BF_Photon_PP(self, Energy):
        b0, b1 = self._NSigmaPP(Energy), self._NSigmaComp(Energy)
        return b0/(b0+b1)

    def ElecBremSample(self, Elec0):
        Ee0, pex0, pey0, pez0 = Elec0.get_pf()

        ThZ = np.arccos(pez0/np.sqrt(pex0**2 + pey0**2 + pez0**2))
        PhiZ = np.arctan2(pey0, pex0)
        RM = [[np.cos(ThZ)*np.cos(PhiZ), -np.sin(PhiZ), np.sin(ThZ)*np.cos(PhiZ)],
            [np.cos(ThZ)*np.sin(PhiZ), np.cos(PhiZ), np.sin(ThZ)*np.sin(PhiZ)],
            [-np.sin(ThZ), 0, np.cos(ThZ)]]

        LUKey = int((np.log10(Ee0) - self._logEeMinBrem)/self._logEeSSBrem)
        ts = self.get_BremSamples(LUKey)
        SampEvt = ts[np.random.randint(0, len(ts))]
        EeMod = self._EeVecBrem[LUKey]

        NFVs = eegFourVecs(Ee0, me, SampEvt[0]*Ee0/EeMod, np.cos(me/EeMod*SampEvt[1]), np.cos(me/(Ee0-SampEvt[0]*Ee0/EeMod)*SampEvt[2]), SampEvt[3])

        Eef, pexfZF, peyfZF, pezfZF = NFVs[1]
        Egf, pgxfZF, pgyfZF, pgzfZF = NFVs[2]

        pe3ZF = [pexfZF, peyfZF, pezfZF]
        pg3ZF = [pgxfZF, pgyfZF, pgzfZF]
        
        pe3LF = np.dot(RM, pe3ZF)
        pg3LF = np.dot(RM, pg3ZF)
        
        NewE = Particle(Elec0.get_IDs()[0], Eef, pe3LF[0], pe3LF[1], pe3LF[2], Elec0.get_rf()[0], Elec0.get_rf()[1], Elec0.get_rf()[2], 2*Elec0.get_IDs()[1]+0, Elec0.get_IDs()[1], Elec0.get_IDs()[0], Elec0.get_IDs()[4]+1, 0)
        NewG = Particle(22, Egf, pg3LF[0], pg3LF[1], pg3LF[2], Elec0.get_rf()[0], Elec0.get_rf()[1], Elec0.get_rf()[2], 2*Elec0.get_IDs()[1]+1, Elec0.get_IDs()[1], Elec0.get_IDs()[0], Elec0.get_IDs()[4]+1, 0)

        return [NewE, NewG]

    def AnnihilationSample(self, Elec0):
        Ee0, pex0, pey0, pez0 = Elec0.get_pf()

        ThZ = np.arccos(pez0/np.sqrt(pex0**2 + pey0**2 + pez0**2))
        PhiZ = np.arctan2(pey0, pex0)
        RM = [[np.cos(ThZ)*np.cos(PhiZ), -np.sin(PhiZ), np.sin(ThZ)*np.cos(PhiZ)],
            [np.cos(ThZ)*np.sin(PhiZ), np.cos(PhiZ), np.sin(ThZ)*np.sin(PhiZ)],
            [-np.sin(ThZ), 0, np.cos(ThZ)]]

        LUKey = int((np.log10(Ee0) - self._logEeMinAnn)/self._logEeSSAnn)
        ts = self.get_AnnSamples(LUKey)
        SampEvt = ts[np.random.randint(0, len(ts))]
        EeMod = self._EeVecAnn[LUKey]

        #NFVs = Ann_FVs(Ee0, me, 0.0, SampEvt[0])
        NFVs = Ann_FVs(Ee0, me, 0.0, SampEvt[0])

        Eg1f, pg1xfZF, pg1yfZF, pg1zfZF = NFVs[0]
        Eg2f, pg2xfZF, pg2yfZF, pg2zfZF = NFVs[1]

        pg3ZF1 = [pg1xfZF, pg1yfZF, pg1zfZF]
        pg3ZF2 = [pg2xfZF, pg2yfZF, pg2zfZF]
    
        pg3LF1 = np.dot(RM, pg3ZF1)
        pg3LF2 = np.dot(RM, pg3ZF2)   

        NewG1 = Particle(22, Eg1f, pg3LF1[0], pg3LF1[1], pg3LF1[2], Elec0.get_rf()[0], Elec0.get_rf()[1], Elec0.get_rf()[2], 2*Elec0.get_IDs()[1]+0, Elec0.get_IDs()[1], Elec0.get_IDs()[0], Elec0.get_IDs()[4]+1, 1)
        NewG2 = Particle(22, Eg2f, pg3LF2[0], pg3LF2[1], pg3LF2[2], Elec0.get_rf()[0], Elec0.get_rf()[1], Elec0.get_rf()[2], 2*Elec0.get_IDs()[1]+1, Elec0.get_IDs()[1], Elec0.get_IDs()[0], Elec0.get_IDs()[4]+1, 1)

        return [NewG1, NewG2]

    def PhotonSplitSample(self, Phot0):
        Eg0, pgx0, pgy0, pgz0 = Phot0.get_pf()

        ThZ = np.arccos(pgz0/np.sqrt(pgx0**2 + pgy0**2 + pgz0**2))
        PhiZ = np.arctan2(pgy0, pgx0)
        RM = [[np.cos(ThZ)*np.cos(PhiZ), -np.sin(PhiZ), np.sin(ThZ)*np.cos(PhiZ)],
            [np.cos(ThZ)*np.sin(PhiZ), np.cos(PhiZ), np.sin(ThZ)*np.sin(PhiZ)],
            [-np.sin(ThZ), 0, np.cos(ThZ)]]

        LUKey = int((np.log10(Eg0) - self._logEgMinPP)/self._logEgSSPP)
        ts = self.get_PPSamples(LUKey)
        SampEvt = ts[np.random.randint(0, len(ts))]
        EgMod = self._EgVecPP[LUKey]

        NFVs = gepemFourVecs(Eg0, me, SampEvt[0]*Eg0/EgMod, np.cos(me/EgMod*SampEvt[1]), np.cos(me/EgMod*SampEvt[2]), SampEvt[3])
        Eepf, pepxfZF, pepyfZF, pepzfZF = NFVs[1]
        Eemf, pemxfZF, pemyfZF, pemzfZF = NFVs[2]

        pep3ZF = [pepxfZF, pepyfZF, pepzfZF]
        pem3ZF = [pemxfZF, pemyfZF, pemzfZF]

        pep3LF = np.dot(RM, pep3ZF)
        pem3LF = np.dot(RM, pem3ZF)

        NewEp = Particle(-11,Eepf, pep3LF[0], pep3LF[1], pep3LF[2], Phot0.get_rf()[0], Phot0.get_rf()[1], Phot0.get_rf()[2], 2*Phot0.get_IDs()[1]+0, Phot0.get_IDs()[1], Phot0.get_IDs()[0], Phot0.get_IDs()[4]+1, 2)
        NewEm = Particle(11, Eemf, pem3LF[0], pem3LF[1], pem3LF[2], Phot0.get_rf()[0], Phot0.get_rf()[1], Phot0.get_rf()[2], 2*Phot0.get_IDs()[1]+1, Phot0.get_IDs()[1], Phot0.get_IDs()[0], Phot0.get_IDs()[4]+1, 2)

        return [NewEp, NewEm]

    def ComptonSample(self, Phot0):
        Eg0, pgx0, pgy0, pgz0 = Phot0.get_pf()

        ThZ = np.arccos(pgz0/np.sqrt(pgx0**2 + pgy0**2 + pgz0**2))
        PhiZ = np.arctan2(pgy0, pgx0)
        RM = [[np.cos(ThZ)*np.cos(PhiZ), -np.sin(PhiZ), np.sin(ThZ)*np.cos(PhiZ)],
            [np.cos(ThZ)*np.sin(PhiZ), np.cos(PhiZ), np.sin(ThZ)*np.sin(PhiZ)],
            [-np.sin(ThZ), 0, np.cos(ThZ)]]

        LUKey = int((np.log10(Eg0) - self._logEgMinComp)/self._logEgSSComp)
        ts = self.get_CompSamples(LUKey)
        SampEvt = ts[np.random.randint(0, len(ts))]
        EgMod = self._EgVecComp[LUKey]

        #NFVs = Compton_FVs(Eg0, me, 0.0, SampEvt[0])
        NFVs = Compton_FVs(Eg0, me, 0.0, SampEvt[0])

        Eef, pexfZF, peyfZF, pezfZF = NFVs[0]
        Egf, pgxfZF, pgyfZF, pgzfZF = NFVs[1]


        pe3LF = np.dot(RM, [pexfZF, peyfZF, pezfZF])
        pg3LF = np.dot(RM, [pgxfZF, pgyfZF, pgzfZF])

        NewE = Particle(11, Eef, pe3LF[0], pe3LF[1], pe3LF[2], Phot0.get_rf()[0], Phot0.get_rf()[1], Phot0.get_rf()[2], 2*Phot0.get_IDs()[1]+0, Phot0.get_IDs()[1], Phot0.get_IDs()[0], Phot0.get_IDs()[4]+1, 3)
        NewG = Particle(22, Egf, pg3LF[0], pg3LF[1], pg3LF[2], Phot0.get_rf()[0], Phot0.get_rf()[1], Phot0.get_rf()[2], 2*Phot0.get_IDs()[1]+1, Phot0.get_IDs()[1], Phot0.get_IDs()[0], Phot0.get_IDs()[4]+1, 3)

        return [NewE, NewG]

    def PropagateParticle(self, Part0, Losses=False, MS=False):
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
                ZT, AT, rhoT = self.get_MaterialProps()
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
                if Ef <= M0 or Ef < MinEnergy:
                    print("Particle lost too much energy along path of propagation!")
                    Part0.set_Ended(True)
                    return Part0
                Part0.set_pf(np.array([Ef, px0/p30*np.sqrt(Ef**2-M0**2), py0/p30*np.sqrt(Ef**2-M0**2), pz0/p30*np.sqrt(Ef**2-M0**2)]))

            Part0.set_Ended(True)
            return Part0

    def GenShower(self, PID0, p40, ParPID):
        p0 = Particle(PID0, p40[0], p40[1], p40[2], p40[3], 0.0, 0.0, 0.0, 1, 0, ParPID, 0, -1)
        AllParticles = [p0]

        while all([ap.get_Ended() == True for ap in AllParticles]) is False:
            for apI, ap in enumerate(AllParticles):
                if ap.get_Ended() is True:
                    continue
                else:
                    if ap.get_IDs()[0] == 22:
                        ap = self.PropagateParticle(ap)
                    elif np.abs(ap.get_IDs()[0]) == 11:
                        ap = self.PropagateParticle(ap, MS=True)
                    AllParticles[apI] = ap
                    if (all([ap.get_Ended() == True for ap in AllParticles]) is True and ap.get_pf()[0] < MinEnergy) or (all([ap.get_Ended() == True for ap in AllParticles]) is True and ap.get_r0()[2] > ZMax):
                        break
                    if ap.get_r0()[2] > ZMax:
                        ap.set_Ended(True)
                        continue
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
                    if (npart[0]).get_p0()[0] > MinEnergy and (npart[0]).get_r0()[2] < ZMax:
                        AllParticles.append(npart[0])
                    if (npart[1]).get_p0()[0] > MinEnergy and (npart[1]).get_r0()[2] < ZMax:
                        AllParticles.append(npart[1])

        return AllParticles
'''
def Shower(PID0, p40, ParPID):
    p0 = Particle(PID0, p40[0], p40[1], p40[2], p40[3], 0.0, 0.0, 0.0, 1, 0, ParPID, 0, -1)

    AllParticles = [p0]

    while all([ap.get_Ended() == True for ap in AllParticles]) is False:
        for apI, ap in enumerate(AllParticles):
            if ap.get_Ended() is True:
                continue
            else:
                if ap.get_IDs()[0] == 22:
                    ap = PropagateParticle(ap)
                elif np.abs(ap.get_IDs()[0]) == 11:
                    ap = PropagateParticle(ap, MS=True)
                    #ap = PropagateParticle(ap)
                AllParticles[apI] = ap
                if (all([ap.get_Ended() == True for ap in AllParticles]) is True and ap.get_pf()[0] < MinEnergy) or (all([ap.get_Ended() == True for ap in AllParticles]) is True and ap.get_r0()[2] > ZMax):
                    break
                if ap.get_r0()[2] > ZMax:
                    ap.set_Ended(True)
                    continue
                if ap.get_IDs()[0] == 11:
                    npart = ElecBremSample(ap)
                elif ap.get_IDs()[0] == -11:
                    BFEpBrem = BF_Positron_Brem(ap.get_pf()[0])
                    ch = np.random.uniform(low=0., high=1.0)
                    if ch < BFEpBrem:
                        npart = ElecBremSample(ap)
                    else:
                        npart = AnnihilationSample(ap)
                elif ap.get_IDs()[0] == 22:
                    BFPhPP = BF_Photon_PP(ap.get_pf()[0])
                    ch = np.random.uniform(low=0., high=1.)
                    if ch < BFPhPP:
                        npart = PhotonSplitSample(ap)
                    else:
                        npart = ComptonSample(ap)
                if (npart[0]).get_p0()[0] > MinEnergy and (npart[0]).get_r0()[2] < ZMax:
                    AllParticles.append(npart[0])
                if (npart[1]).get_p0()[0] > MinEnergy and (npart[1]).get_r0()[2] < ZMax:
                    AllParticles.append(npart[1])

    return AllParticles
'''