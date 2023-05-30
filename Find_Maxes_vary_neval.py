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
from tqdm import tqdm
startTime = datetime.now()


Dir0 = os.getcwd()
PickDir = Dir0 + "/data/VEGAS_backend/SM/"
SvDir  = Dir0 + "/data/VEGAS_dictionaries/"

PPSamp0 = np.load(PickDir+"PairProduction_AdaptiveMaps.npy", allow_pickle=True)
BremSamp0 = np.load(PickDir+"Bremsstrahlung_AdaptiveMaps.npy", allow_pickle=True)
CompSamp0 = np.load(PickDir+"Compton_AdaptiveMaps.npy", allow_pickle=True)
AnnSamp0 = np.load(PickDir+"Annihilation_AdaptiveMaps.npy", allow_pickle=True)
MollerSamp0 = np.load(PickDir+"Moller_AdaptiveMaps.npy", allow_pickle=True)
BhabhaSamp0 = np.load(PickDir+"Bhabha_AdaptiveMaps.npy", allow_pickle=True)

#Brem Samples were generated with Egamma_min = 0.001 GeV = 1 MeV
Egamma_min = 0.001

TargetMaterials = ['graphite','lead']
Z = {'graphite':6.0, 'lead':82.0}

Process_Files={"PairProd" : PPSamp0,
               "Comp": CompSamp0,
               "Brem" : BremSamp0,
               "Ann": AnnSamp0,
               "Moller": MollerSamp0,
               "Bhabha": BhabhaSamp0}

diff_xsections={"PairProd" : dsigma_pairprod_dimensionless,
                "Comp"     : dsigma_compton_dCT,    
                "Brem"     : dsigma_brem_dimensionless,
                "Ann"      : dsigma_annihilation_dCT,
                "Moller"   : dsigma_moller_dCT,
                "Bhabha"   : dsigma_bhabha_dCT }

FF_dict =      {"PairProd" : g2_elastic,
                "Comp"     : unity,
                "Brem"     : g2_elastic,
                "Ann"      : unity,
                "Moller"   : unity,
                "Bhabha"   : unity }

QSq_functions={"PairProd" : pair_production_q_sq_dimensionless, "Brem"  : brem_q_sq_dimensionless, "Comp": dummy, "Ann": dummy, "Moller":dummy, "Bhabha":dummy }

neval0 = 300
n_trials = 100

xSec_dict={}
samp_dict ={}

for process_key in Process_Files.keys():
    process_file=Process_Files[process_key]
    diff_xsec  =diff_xsections[process_key]
    QSq        =QSq_functions[process_key]
    xSec_dict[process_key]={}
    samp_dict[process_key]=[]
    FF_func = FF_dict[process_key]

    for tm in TargetMaterials:
        xSec_dict[process_key][tm]=[]

    counter=0
    for ki in tqdm(range(len(process_file))):
        counter=counter+1
        
        E_inc, adaptive_map = process_file[ki]
        integrand = vg.Integrator(map=adaptive_map, nstrat=nstrat_options[process_key])
        save_copy = copy.deepcopy(adaptive_map)
        
        pts = []
        max_wgtTimesF=0

        
    
        xSec={}
        for tm in TargetMaterials:
            xSec[tm]=0.0

        integrand.set(max_nhcube=1, neval=neval0)
        for trial_number in range(n_trials):
            for x, wgt in integrand.random():

                Z_H=1
                
                EvtInfo={'E_inc': E_inc, 'm_e': m_electron, 'Z_T': Z_H, 'alpha_FS': alpha_em, 'mV': 0, 'Eg_min':0.001}
                MM_H = wgt*diff_xsec(EvtInfo, x)
                if MM_H > max_wgtTimesF:
                    max_F=MM_H
                    max_x = np.asarray(x)
                    max_wgt= wgt 

                FF_H = FF_func(EvtInfo, QSq(x, EvtInfo))

                for tm in TargetMaterials:
                    ZT = Z[tm]
                    EvtInfoTM={'E_inc': E_inc, 'm_e': m_electron, 'Z_T': Z[tm], 'alpha_FS': alpha_em, 'mV': 0, 'Eg_min':0.001}
                    FF= FF_func(EvtInfoTM, QSq(x, EvtInfoTM) )
                    xSec[tm] += MM_H*FF/FF_H/n_trials


        samp_dict[process_key].append([E_inc, \
                                      {"neval":neval0, "max_F": max_F, "Eg_min":Egamma_min,"adaptive_map": save_copy}])
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
