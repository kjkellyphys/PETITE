from AllProcesses import *
import pickle as pk
import numpy as np
import os
import random as rnd

Dir0 = os.getcwd()
PickDir = Dir0 + "/NBP/DarkV/"
DarkVMass = '10MeV'
BremSamp0 = np.load(PickDir+"/ElectronPositron_BremPickles_"+DarkVMass+".npy", allow_pickle=True)
MV = {'10MeV':0.010}

TargetMaterial = 'graphite'
Z = {'graphite':6.0}
PickDir0 = Dir0 + "/NBP/"
SvDir = PickDir0 + TargetMaterial + "/DarkV/"
if os.path.exists(SvDir) == False:
    os.system("mkdir " + SvDir)

meT, ZT, MVT, alT = 0.000511, Z[TargetMaterial], MV[DarkVMass], 1.0/137.0

#Momentum Transfer Squared for photon-scattering Pair Production
def PPQSq(epp, dp, dm, ph, me, w):
    epm = w - epp
    return me**2*((dp**2 + dm**2 + 2.0*dp*dm*np.cos(ph)) + me**2*((1.0 + dp**2)/(2.0*epp) + (1.0+dm**2)/(2.0*epm))**2)
#Momentum Transfer Squared for electron/positron bremsstrahlung
def BremQSq(w, d, dp, ph, me, ep):
    epp = ep - w
    return me**2*((d**2 + dp**2 - 2*d*dp*np.cos(ph)) + me**2*((1 + d**2)/(2*ep) - (1 + dp**2)/(2*epp))**2)
def DarkBremQsq(w, d, dp, ph, me, MV, ep):
    epp = ep - w
    PF0 = MV**2*ep*epp/w**2
    return PF0*((d**2 + dp**2 - 2.0*d*dp*np.cos(ph)) + MV**2/(4.0*ep*epp)*(1 + 0.5*(d**2 + dp**2))**2)

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

UnWS_Brem, XSecBrem = [], []
NPts = 30000
for ki in range(len(BremSamp0)):
    Ee, integrand = BremSamp0[ki]
    pts = []

    xs0 = 0.0
    for x, wgt in integrand.random():
        MM0 = wgt*dSDBrem_dP_T([Ee, meT, MVT, ZT, alT], x)
        xs0 += MM0
        FF = G2el(ZT, meT, DarkBremQsq(x[0], x[1], x[2], x[3], meT, MVT, Ee))/ZT**2
        pts.append(np.concatenate([x, [MM0, MM0*FF]]))
    
    UnWeightedScreening = GetPts(pts, NPts, WgtIndex=5, LenRet=4)
    UnWS_Brem.append(UnWeightedScreening)
    XSecBrem.append([Ee, xs0])
    print(Ee, len(pts), len(UnWS_Brem[ki]), xs0)
np.save(SvDir + "DarkBremXSec_"+DarkVMass, XSecBrem)
np.save(SvDir + "DarkBremEvts_"+DarkVMass, UnWS_Brem)

'''
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
np.save(SvDir+"ComptonXSec", XSecComp)
np.save(SvDir+"ComptonEvts", UnWComp)

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

np.save(SvDir+"AnnihilationXSec", XSecAnn)
np.save(SvDir+"AnnihilationEvts", UnWAnn)
'''