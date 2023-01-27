""" Generate samples of Standard Model EM shower events and save them.

    Uses saved VEGAS integrators to generate e+/- pair production and annihilation, 
    bremstrahlung and compton events for a range of initial particle energies 
    and target materials.
    The events are unweighted and saved for use in constructing a realistic shower.

    Typical usage:

    python Gen_Samples.py
"""

from PETITE.AllProcesses import *
import pickle
import copy
import numpy as np
import os
import random as rnd
from datetime import datetime
startTime = datetime.now()


Dir0 = os.getcwd()
PickDir = Dir0 + "/NBP/"
SvDir  = Dir0 + "/RyanDicts/"
PPSamp0 = np.load(PickDir+"Photon_PairProdPickles.npy", allow_pickle=True)
CompSamp0 = np.load(PickDir+"ComptonPickles.npy", allow_pickle=True)
BremSamp0 = np.load(PickDir+"ElectronPositron_BremPickles.npy", allow_pickle=True)
AnnSamp0 = np.load(PickDir+"AnnihilationPickles.npy", allow_pickle=True)



#just a placeholder
def dummy(x,y,z):
    return(0) 


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
 

def BremQSq(xx, me, ep):
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
    w, d, dp, ph = xx
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

def Unity(Z, me, t):
    return(1.0)

TargetMaterials = ['graphite','lead']
Z = {'graphite':6.0, 'lead':82.0}

Process_Files={"PairProd" : PPSamp0,
               "Comp": CompSamp0,
               "Brem" : BremSamp0,
               "Ann": AnnSamp0}

diff_xsections={"PairProd" : dSPairProd_dP_T,
                "Comp"     : dSCompton_dCT,    
                "Brem"     : dSBrem_dP_T,
                "Ann"      : dAnn_dCT }

FF_dict =      {"PairProd" : G2el,
                "Comp"     : Unity,
                "Brem"     : G2el,
                "Ann"      : Unity }

QSq_functions={"PairProd" : PPQSq, "Brem"     : BremQSq, "Comp": dummy, "Ann": dummy }

meT, alT = 0.000511, 1.0/137.0

UnWS, XSecPP = [], []
NPts = 30000

xSec_dict={}
samp_dict ={}

for process_key in Process_Files.keys():
    process_file=Process_Files[process_key]
    diff_xsec  =diff_xsections[process_key]
    QSq        =QSq_functions[process_key]
    xSec_dict[process_key]={}
    samp_dict[process_key]=[]


    for tm in TargetMaterials:
        xSec_dict[process_key][tm]=[]

    counter=0
    for ki in range(len(process_file)):

        ## This is just to speed up the code (for debugging)
        ## by not doing all the energies
        #if counter>2:
        #    break
        #print(counter)
        #counter=counter+1
        
        E_inc, integrand = process_file[ki]
        save_copy_integrand=copy.deepcopy(integrand)
        
        pts = []
        max_wgtTimesF=0

        
    
        xSec={}
        for tm in TargetMaterials:
            xSec[tm]=0.0

        for x, wgt in integrand.random():

            Z_H=1
             
            EvtInfo={'E_inc': E_inc, 'm_e': meT, 'Z_T': Z_H, 'alpha_FS': alT, 'm_V': 0}
            MM_H_0 = wgt*diff_xsec(EvtInfo, x)
            EvtInfo['E_inc'] = E_inc*1.2
            MM_H_higher = wgt*diff_xsec(EvtInfo, x)
            EvtInfo['E_inc'] = E_inc*0.8
            MM_H_lower = wgt*diff_xsec(EvtInfo, x)

            EvtInfo['E_inc'] = E_inc

            MM_H= max(MM_H_0, MM_H_higher, MM_H_lower)
            if MM_H > max_wgtTimesF:
                max_F=MM_H
                max_x = np.asarray(x)
                max_wgt= wgt 

            
            for tm in TargetMaterials:

                ZT = Z[tm]
                FF_func = FF_dict[process_key]

                FF= FF_func(ZT, meT, QSq(x, meT, E_inc) )
                xSec[tm] += MM_H_0*FF


        samp_dict[process_key].append([E_inc, \
                                      {"max_F": max_F, "max_X": max_x, "max_wgt": max_wgt, \
                                       "integrator": save_copy_integrand}])
        for tm in TargetMaterials:
            xSec_dict[process_key][tm].append([E_inc, xSec[tm] ] ) 
            
    xSec_dict[process_key][tm]= np.asarray(xSec_dict[process_key][tm] ) 
    samp_dict[process_key]    = samp_dict[process_key]

f_xSecs = open(SvDir + "xSec_Dicts.pkl","wb")
f_samps = open(SvDir + "samp_Dicts.pkl","wb")

pickle.dump(xSec_dict,f_xSecs)
pickle.dump(samp_dict,f_samps)

f_xSecs.close()
f_samps.close()


print("Run Time of Script:  ", datetime.now() - startTime)
