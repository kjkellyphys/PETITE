from PETITE.AllProcesses import *
import pickle as pk
import numpy as np
import os
import random as rnd

Dir0 = os.getcwd()
PickDir = Dir0 + "/NBP/DarkV/"
MV = {'10MeV':0.010, '100MeV':0.100}
#Momentum Transfer Squared for electron/positron bremsstrahlung of dark photon


def doc_string_format(arg_1,arg_2):
    '''
    Description of function

    
    Args:
            arg_1 (type): describe argument
            arg_2 (type): describe argument

    Returns: 
            (type): describe thing that is returned. 

    Raises: 
            optional section for functions that raise exceptions.

    
    Example: 
            Words
            go
    `       here
            
    '''

    return(0)



def DarkBremQsq(w, d, dp, ph, me, MV, ep):
    '''
    Function that computes the three momentum transfered squared for the bremmstrahlung 
    of a dark photon off an electron scattering in a static Coulomb field. All variables
    are defined in the lab frame. 

    See Landau and Lifshitz Vol 4, between Eqs. (93.8)  and (93.9) generalize to include
    a dark photon, and take the small angle limit (see Eq (93.14) for the SM case). 

    Args:
            w    (float): energy of dark photon (\omega)
            d    (float): angle between dark photon and initial electron momentum (\delta)
            dp   (float): angle between dark photon and final electron momentum (\delta')
            ph   (float): relative azimuthal angle (\phi)
            me   (float): mass of electron/lepton 
            MV   (float): mass of vector bosoon radiated
            ep   (float): energy of initial electron/lepton (including rest mass)
          

    Returns: 
            Qsq (type): magnitude of three momentum transfer squared (p_e' + k' - p_e)^2 

    Raises: 
            Value Error: MV>ep (production is energetically forbidden)
                         MV>w  (energy transfer must be bigger than MV)
                         w>ep  (energy transfer cannot exceed initial energy)
                         me or MV < 0 (masses must be positive)
                         Qsq<0  (momentum transfer squared must be positive)
    
    '''
    if MV<0 or me<0:
        raise ValueError('Masses must be positive semi-definite') 
    if MV>ep:
        raise ValueError('Mass of dark photon is too big, production is energetically forbidden')
    if w>ep:
        raise ValueError('Energy transfer exceeds energy of lepton')
    if w<MV:
        raise ValueError('Energy transfer is smaller than mass of dark photon')
    
    epp = ep - w
    PF0 = MV**2*ep*epp/w**2

    Qsq=PF0*((d**2 + dp**2 - 2.0*d*dp*np.cos(ph)) + MV**2/(4.0*ep*epp)*(1 + 0.5*(d**2 + dp**2))**2)

    if Qsq<0:
        raise ValueError('Momentum transfered squared is negative')
    return Qsq


def aa(Z, me):
    '''
    Computes screening parameter for charge form factor. 
    See Tsai Rev. Mod. Phys. 46, 815 (1974)

    
    Args:
            Z  (float): charge of atomic nucleus, typically integer. 
            me (float): electron mass

    Returns: 
            a (float): screening length. Dimensions set by electron mass. 

    Raises: 
            Value error: me<0 or Z<0
    '''

    if me<0 or Z<0:
        raise ValueError('Parameters must be positive')
    
    return 184.15*(2.718)**-0.5*Z**(-1./3.)/me



def G2el(Z, me, t):
    '''
    Computes atomic screening form factor 
    See Tsai Rev. Mod. Phys. 46, 815 (1974)

    
    Args:
            Z  (float)  : charge of atomic nucleus, typiically integer
            me (float): electron mass
            t  (float): three-momentum transfer squared

    ### SUGGESTION SWITCH TO Qsq as variable for clarity and consistency

    Returns: 
            a (float): screening length. Dimensions set by electron mass. 

    Raises: 
            Value error: me<0 or Z<0
    '''
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

DarkVMasses = ['10MeV','100MeV']
TargetMaterials = ['graphite']
Z = {'graphite':6.0}
PickDir0 = Dir0 + "/NBP/"

for dvm in DarkVMasses:
    BremSamp0 = np.load(PickDir+"/ElectronPositron_BremPickles_"+dvm+".npy", allow_pickle=True)
    CompSamp0 = np.load(PickDir+"/ComptonPickles_"+dvm+".npy", allow_pickle=True)
    AnnSamp0 = np.load(PickDir+"/AnnihilationPickles_"+dvm+".npy", allow_pickle=True)
    MVT = MV[dvm]

    for tm in TargetMaterials:
        SvDir = PickDir0 + tm + "/DarkV/"
        if os.path.exists(SvDir) == False:
            os.system("mkdir " + SvDir)
        ZT = Z[tm]

        UnWS_Brem, XSecBrem = [], []
        NPts = 30000
        for ki in range(len(BremSamp0)):
            Ee, integrand = BremSamp0[ki]
            pts = []

            xs0 = 0.0
            for x, wgt in integrand.random():
                MM0 = wgt*dSDBrem_dP_T([Ee, m_electron, MVT, ZT, alpha_em], x)
                FF = G2el(ZT, m_electron, DarkBremQsq(x[0], x[1], x[2], x[3], m_electron, MVT, Ee))/ZT**2
                xs0 += MM0*FF
                pts.append(np.concatenate([x, [MM0, MM0*FF]]))
            
            UnWeightedScreening = GetPts(pts, NPts, WgtIndex=5, LenRet=4)
            UnWS_Brem.append(UnWeightedScreening)
            XSecBrem.append([Ee, xs0])
            print(Ee, len(pts), len(UnWS_Brem[ki]), xs0)
        np.save(SvDir + "DarkBremXSec_"+dvm, XSecBrem)
        np.save(SvDir + "DarkBremEvts_"+dvm, UnWS_Brem)

    SvDirE = PickDir0 + '/electrons/DarkV/'
    if os.path.exists(SvDirE) == False:
        os.system("mkdir " + SvDirE)

    UnWComp, XSecComp = [], []
    NPts = 30000
    for ki in range(len(CompSamp0)):
        Eg, integrand = CompSamp0[ki]

        xs0 = 0.0
        pts = []
        for x, wgt in integrand.random():
            MM0 = wgt*dSCompton_dCT([Eg, m_electron, MVT, alpha_em], x)
            xs0 += MM0
            pts.append(np.concatenate([x, [MM0]]))
        
        UnWeightedNoScreening = GetPts(pts, NPts, WgtIndex=1, LenRet=1)
        UnWComp.append(UnWeightedNoScreening)
        XSecComp.append([Eg, xs0])
        print(Eg, len(pts), len(UnWComp[ki]), xs0)
    np.save(SvDirE+"ComptonXSec_"+dvm, XSecComp)
    np.save(SvDirE+"ComptonEvts_"+dvm, UnWComp)

    UnWAnn, XSecAnn = [], []
    NPts = 30000
    for ki in range(len(AnnSamp0)):
        Ee, integrand = AnnSamp0[ki]

        xs0 = 0.0
        pts = []
        for x, wgt in integrand.random():
            MM0 = wgt*dAnn_dCT([Ee, m_electron, alpha_em, MVT], x)
            xs0 += MM0
            pts.append(np.concatenate([x, [MM0]]))
        
        UnWeightedNoScreening = GetPts(pts, NPts, WgtIndex=1, LenRet=1)
        UnWAnn.append(UnWeightedNoScreening)
        XSecAnn.append([Ee, xs0])

        print(Ee, len(pts), len(UnWAnn[ki]), xs0)

    np.save(SvDirE+"AnnihilationXSec_"+dvm, XSecAnn)
    np.save(SvDirE+"AnnihilationEvts_"+dvm, UnWAnn)
