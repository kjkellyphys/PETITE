""" Generate samples of Standard Model EM shower events and save them.

    Uses saved VEGAS integrators to generate e+/- pair production and annihilation, 
    bremstrahlung and compton events for a range of initial particle energies 
    and target materials.
    The events are unweighted and saved for use in constructing a realistic shower.

    Typical usage:

    python Gen_Samples.py
"""

from PETITE.AllProcesses import *
import pickle as pk
import numpy as np
import os
import random as rnd

Dir0 = os.getcwd()
PickDir = Dir0 + "/NBP/"
PPSamp0 = np.load(PickDir+"Photon_PairProdPickles.npy", allow_pickle=True)
CompSamp0 = np.load(PickDir+"ComptonPickles.npy", allow_pickle=True)
BremSamp0 = np.load(PickDir+"ElectronPositron_BremPickles.npy", allow_pickle=True)
AnnSamp0 = np.load(PickDir+"AnnihilationPickles.npy", allow_pickle=True)

def PPQSq(xx, me, w):
    """Computes momentum transfer squared for photon-scattering pair production
    Args:
        xx: tuple consisting of kinematic rescaled kinematic variables 
            epsilon_plus, delta_plus, delta_minus, phi (see ... for definitions) 
        me: electron mass 
        w:  frequency of emitted virtual photon, equal to sum of energies of outgoing e+- pair
    Returns:
        nuclear momentum transfer squared
    """
    epp, dp, dm, ph = xx
    epm = w - epp
    return me**2*((dp**2 + dm**2 + 2.0*dp*dm*np.cos(ph)) + me**2*((1.0 + dp**2)/(2.0*epp) + (1.0+dm**2)/(2.0*epm))**2)


def BremQSq(w, d, dp, ph, me, ep):
    """Momentum Transfer Squared for electron/positron bremsstrahlung
    Args:
        w: frequency of radiated photon 
        d: rescaled emission angle, equal to theta * epsilon/m, 
           where epsilon energy of incoming lepton
        dp: rescaled emission angle, equal to theta * epsilon'/m, 
           where epsilon' energy of outgoing lepton
        ph: angle phi
        me: electron mass
        ep: epsilon', e+/e- energy after radiation
    Returns:
        nuclear momentum transfer squared
    """
    epp = ep - w
    return me**2*((d**2 + dp**2 - 2*d*dp*np.cos(ph)) + me**2*((1 + d**2)/(2*ep) - (1 + dp**2)/(2*epp))**2)

def aa(Z, me):
    """Elastic screening length from Eq. B. 42 in 
    https://journals.aps.org/rmp/pdf/10.1103/RevModPhys.46.815
    """
    return 184.15*(2.718)**-0.5*Z**(-1./3.)/me

def G2el(Z, me, t):
    """Elastic atomic form-factor that accounts for screening at low momentum transfers, see Eq. B.38 in 
    https://journals.aps.org/rmp/pdf/10.1103/RevModPhys.46.815
    """
    a0 = aa(Z, me)
    return Z**2*a0**4*t**2/(1 + a0**2*t)**2

def GetPts(Dist, npts, WgtIndex=4, LenRet=4):
    """If weights are too cumbersome, this function returns a properly-weighted sample from Dist"""
    """Returns an unweighted sample from a distribution of weighted events
    Args:
        Dist: weighted distribution sample (N, at least WgtIndex+1)
        npts: number of unweighted events to sample 
        WgtIndex: index of weights 
        LenRet: length of the sub-array to return in the unweighted sample 
            (i.e., LenRet<=WgtIndex throws away the weights in the return)
    Returns:
        unweighted sample of the distribution
    """
    ret = []
    tochoosefrom = [pis for pis in range(len(Dist))]
    choicesgetter = rnd.choices(tochoosefrom, np.transpose(Dist)[WgtIndex], k=npts)
    for cg in choicesgetter:
        ret.append(Dist[cg][:LenRet])
    return ret

TargetMaterials = ['graphite','lead']
Z = {'graphite':6.0, 'lead':82.0}

meT, alT = 0.000511, 1.0/137.0

for tm in TargetMaterials:
    SvDir = PickDir + tm + "/"
    ZT = Z[tm]
    if os.path.exists(SvDir) == False:
        os.system("mkdir " + SvDir)

    print("Working on pair production for ", tm)
    UnWS, XSecPP = [], []
    NPts = 30000
    for ki in range(len(PPSamp0)):
        Eg, integrand = PPSamp0[ki]
        pts = []

        xs0 = 0.0
        for x, wgt in integrand.random():
            MM0 = wgt*dSPairProd_dP_T([Eg, meT, ZT, alT], x)
            FF = G2el(ZT, meT, PPQSq(x, meT, Eg))/ZT**2
            xs0 += MM0*FF
            pts.append(np.concatenate([x, [MM0, MM0*FF]]))

        print("The length of the sample prior to unweighing is ",len(pts))
        UnWeightedScreening = GetPts(pts, NPts, WgtIndex=5, LenRet=4)

        UnWS.append(UnWeightedScreening)
        XSecPP.append([Eg, xs0])
        print(Eg, len(pts), len(UnWS[ki]), xs0)
    np.save(SvDir + "PairProdXSec", XSecPP)
    np.save(SvDir + "PairProdEvts", UnWS)

    print("Working on brem for ", tm)
    UnWS_Brem, XSecBrem = [], []
    NPts = 30000
    for ki in range(len(BremSamp0)):
        Ee, integrand = BremSamp0[ki]
        pts = []

        xs0 = 0.0
        for x, wgt in integrand.random():
            MM0 = wgt*dSBrem_dP_T([Ee, meT, ZT, alT], x)
            FF = G2el(ZT, meT, BremQSq(x[0], x[1], x[2], x[3], meT, Ee))/ZT**2
            xs0 += MM0*FF
            pts.append(np.concatenate([x, [MM0, MM0*FF]]))
        
        UnWeightedScreening = GetPts(pts, NPts, WgtIndex=5, LenRet=4)
        UnWS_Brem.append(UnWeightedScreening)
        XSecBrem.append([Ee, xs0])
        print(Ee, len(pts), len(UnWS_Brem[ki]), xs0)
    np.save(SvDir + "BremXSec", XSecBrem)
    np.save(SvDir + "BremEvts", UnWS_Brem)

SvDirE = PickDir + "/electrons/"
if os.path.exists(SvDirE) == False:
    os.system("mkdir " + SvDirE)

print("Working on Compton...")
UnWComp, XSecComp = [], []
NPts = 30000
for ki in range(len(CompSamp0)):
    Eg, integrand = CompSamp0[ki]

    xs0 = 0.0
    pts = []
    for x, wgt in integrand.random():
        MM0 = wgt*dSCompton_dCT([Eg, meT, 0.0, alT], x)
        xs0 += MM0
        pts.append(np.concatenate([x, [MM0]]))
    
    UnWeightedNoScreening = GetPts(pts, NPts, WgtIndex=1, LenRet=1)
    UnWComp.append(UnWeightedNoScreening)
    XSecComp.append([Eg, xs0])
    print(Eg, len(pts), len(UnWComp[ki]), xs0)
np.save(SvDirE+"ComptonXSec", XSecComp)
np.save(SvDirE+"ComptonEvts", UnWComp)


print("Working on e+ e- annihilation")
UnWAnn, XSecAnn = [], []
NPts = 30000
for ki in range(len(AnnSamp0)):
    Ee, integrand = AnnSamp0[ki]

    xs0 = 0.0
    pts = []
    for x, wgt in integrand.random():
        MM0 = wgt*dAnn_dCT([Ee, meT, alT, 0.0], x)
        xs0 += MM0
        pts.append(np.concatenate([x, [MM0]]))
    
    UnWeightedNoScreening = GetPts(pts, NPts, WgtIndex=1, LenRet=1)
    UnWAnn.append(UnWeightedNoScreening)
    XSecAnn.append([Ee, xs0])

    print(Ee, len(pts), len(UnWAnn[ki]), xs0)

np.save(SvDirE+"AnnihilationXSec", XSecAnn)
np.save(SvDirE+"AnnihilationEvts", UnWAnn)
