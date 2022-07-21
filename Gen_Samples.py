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

#Momentum Transfer Squared for photon-scattering Pair Production
def PPQSq(xx, me, w):
    epp, dp, dm, ph = xx
    epm = w - epp
    return me**2*((dp**2 + dm**2 + 2.0*dp*dm*np.cos(ph)) + me**2*((1.0 + dp**2)/(2.0*epp) + (1.0+dm**2)/(2.0*epm))**2)

#Momentum Transfer Squared for electron/positron bremsstrahlung
def BremQSq(w, d, dp, ph, me, ep):
    epp = ep - w
    return me**2*((d**2 + dp**2 - 2*d*dp*np.cos(ph)) + me**2*((1 + d**2)/(2*ep) - (1 + dp**2)/(2*epp))**2)

def aa(Z, me):
    return 184.15*(2.718)**-0.5*Z**(-1./3.)/me
#Form factor for screening effects
def G2el(Z, me, t):
    a0 = aa(Z, me)
    return Z**2*a0**4*t**2/(1 + a0**2*t)**2

def GetPts(Dist, npts, WgtIndex=4, LenRet=4):
    """If weights are too cumbersome, this function returns a properly-weighted sample from Dist"""
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

        UnWeightedScreening = GetPts(pts, NPts, WgtIndex=5, LenRet=4)
        UnWS.append(UnWeightedScreening)
        XSecPP.append([Eg, xs0])
        print(Eg, len(pts), len(UnWS[ki]), xs0)
    np.save(SvDir + "PairProdXSec", XSecPP)
    np.save(SvDir + "PairProdEvts", UnWS)

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