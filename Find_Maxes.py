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
CompSamp0 = np.load(PickDir+"ComptonPickles_Old.npy", allow_pickle=True)
BremSamp0 = np.load(PickDir+"ElectronPositron_BremPickles.npy", allow_pickle=True)
AnnSamp0 = np.load(PickDir+"AnnihilationPickles.npy", allow_pickle=True)


TargetMaterials = ['graphite','lead']
Z = {'graphite':6.0, 'lead':82.0}

Process_Files={"PairProd" : PPSamp0,
               "Comp": CompSamp0,
               "Brem" : BremSamp0,
               "Ann": AnnSamp0}

diff_xsections={"PairProd" : dsigma_pairprod_dP_T,
                "Comp"     : dsigma_compton_dCT,    
                "Brem"     : dsigma_brem_dP_T,
                "Ann"      : dsigma_annihilation_dCT }

FF_dict =      {"PairProd" : g2_elastic,
                "Comp"     : unity,
                "Brem"     : g2_elastic,
                "Ann"      : unity }

QSq_functions={"PairProd" : pair_production_q_sq, "Brem"  : brem_q_sq, "Comp": dummy, "Ann": dummy }


UnWS, XSecPP = [], []
#n_points = 30000

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

        # This is just to speed up the code (for debugging)
        # by not doing all the energies
        if counter>1:
           break
        print(counter)
        counter=counter+1
        
        E_inc, integrand = process_file[ki]
        save_copy_integrand=copy.deepcopy(integrand)
        
        pts = []
        max_wgtTimesF=0

        
    
        xSec={}
        for tm in TargetMaterials:
            xSec[tm]=0.0

        for x, wgt in integrand.random():

            Z_H=1
             
            EvtInfo={'E_inc': E_inc, 'm_e': m_electron, 'Z_T': Z_H, 'alpha_FS': alpha_em, 'm_V': 0}
            MM_H_0 = wgt*diff_xsec(EvtInfo, x)
            #EvtInfo['E_inc'] = E_inc*1.2
            #MM_H_higher = wgt*diff_xsec(EvtInfo, x)
            #EvtInfo['E_inc'] = E_inc*0.8
            #MM_H_lower = wgt*diff_xsec(EvtInfo, x)

            EvtInfo['E_inc'] = E_inc

            MM_H= MM_H_0   #max(MM_H_0, MM_H_higher, MM_H_lower)
            if MM_H > max_wgtTimesF:
                max_F=MM_H
                max_x = np.asarray(x)
                max_wgt= wgt 

            
            for tm in TargetMaterials:

                ZT = Z[tm]
                FF_func = FF_dict[process_key]

                FF= FF_func(ZT, m_electron, QSq(x, m_electron, E_inc) )
                xSec[tm] += MM_H_0*FF


        samp_dict[process_key].append([E_inc, \
                                      {"max_F": max_F, "max_X": max_x, "max_wgt": max_wgt, \
                                       "integrator": save_copy_integrand}])
        for tm in TargetMaterials:
            xSec_dict[process_key][tm].append([E_inc, xSec[tm] ] ) 
            
    xSec_dict[process_key][tm]= np.asarray(xSec_dict[process_key][tm] ) 
    samp_dict[process_key]    = samp_dict[process_key]

f_xSecs = open(SvDir + "Feb13_xSec_Dicts.pkl","wb")
f_samps = open(SvDir + "Feb13_samp_Dicts.pkl","wb")

pickle.dump(xSec_dict,f_xSecs)
pickle.dump(samp_dict,f_samps)

f_xSecs.close()
f_samps.close()


print("Run Time of Script:  ", datetime.now() - startTime)
