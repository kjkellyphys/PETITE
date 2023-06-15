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
PickDir = Dir0 + "/../data/VEGAS_backend/"
SvDir  = Dir0 + "/../data/VEGAS_dictionaries/"
#Brem Samples were generated with Egamma_min = 0.001 GeV = 1 MeV
#Egamma_min = 0.001
Egamma_min = 0.010
CompSamp0 = np.load(PickDir+"SM/Compton_AdaptiveMaps.npy", allow_pickle=True)
AnnSamp0 = np.load(PickDir+"SM/Annihilation_AdaptiveMaps.npy", allow_pickle=True)

darkbrem_adaptivemaps = pickle.load(open(PickDir+"DarkV/darkbrem_adaptivemaps.p", "rb"))
vector_masses = list(darkbrem_adaptivemaps.keys())

TargetMaterials = ['graphite','lead']
Z = {'graphite':6.0, 'lead':82.0}
A = {'graphite':12.0, 'lead':207.0}

Process_Files={"ExactBrem":darkbrem_adaptivemaps,
               "Comp": CompSamp0,
               "Ann": AnnSamp0}

diff_xsections={"PairProd" : dsigma_pairprod_dimensionless,
                "Comp"     : dsigma_compton_dCT,    
                "Brem"     : dsigma_brem_dimensionless,
                "ExactBrem": dsig_etl_helper,
                "Ann"      : dsigma_annihilation_dCT }

FF_dict =      {"PairProd" : g2_elastic,
                "Comp"     : unity,
                "Brem"     : g2_elastic,
                "ExactBrem": Gelastic_inelastic,
                "Ann"      : unity }

QSq_functions={"PairProd" : pair_production_q_sq_dimensionless, "Brem"  : brem_q_sq_dimensionless, "ExactBrem":exactbrem_qsq, "Comp": dummy, "Ann": dummy }

neval0 = 300
n_trials = 100

Z_H=1
A_H=1

xSec_dict = {}
xSec_dict_0 = {}
MF_dict_0 = {}
samp_dict = {}
for mVi, mV in enumerate(vector_masses):
    print(mV)
    xSec_dict[mV]={}
    xSec_dict_0[mV]={}
    MF_dict_0[mV]={}
    samp_dict[mV]={}

    for process_key in Process_Files.keys():
        process_file=Process_Files[process_key]
        if process_key == "ExactBrem":
            process_file = process_file[mV]

        diff_xsec  =diff_xsections[process_key]
        QSq        =QSq_functions[process_key]
        xSec_dict[mV][process_key]={}
        xSec_dict_0[mV][process_key]={}
        MF_dict_0[mV][process_key]={}
        samp_dict[mV][process_key]=[]
        FF_func = FF_dict[process_key]

        for tm in TargetMaterials:
            xSec_dict[mV][process_key][tm]=[]
            xSec_dict_0[mV][process_key][tm]=[]
            MF_dict_0[mV][process_key][tm]=[]

        counter=0
        for ki in tqdm(range(len(process_file))):
            counter=counter+1
            
            E_inc, adaptive_map = process_file[ki]
            integrand = vg.Integrator(map=adaptive_map, nstrat=nstrat_options[process_key])
            save_copy = copy.deepcopy(adaptive_map)

            EvtInfo={'E_inc': E_inc, 'm_e': m_electron, 'Z_T': Z_H, 'A_T': A_H, 'mT':A_H, 'alpha_FS': alpha_em, 'mV': mV, 'Eg_min':Egamma_min}
            
            max_F=0
            xSec={}
            max_F_TM={}
            for tm in TargetMaterials:
                xSec[tm]=0.0
                max_F_TM[tm] = 0.0

            integrand.set(max_nhcube=1, neval=neval0)
            for trial_number in range(n_trials):
                for x, wgt in integrand.random():
                    MM_H = wgt*diff_xsec(EvtInfo, x)
                    if MM_H > max_F:
                        max_F=MM_H
                        max_x = np.asarray(x)
                        max_wgt= wgt 

                    FF_H = FF_func(EvtInfo, QSq(x, EvtInfo))

                    for tm in TargetMaterials:
                        EvtInfoTM={'E_inc': E_inc, 'm_e': m_electron, 'Z_T': Z[tm], 'A_T': A[tm], 'mT':A[tm], 'alpha_FS': alpha_em, 'mV': mV, 'Eg_min':Egamma_min}
                        FF= FF_func(EvtInfoTM, QSq(x, EvtInfoTM) )
                        MM_TM = wgt*diff_xsec(EvtInfoTM, x)
                        xSec[tm] += MM_TM/n_trials
                        if MM_TM > max_F_TM[tm]:
                            max_F_TM[tm] = MM_TM

            for tm in TargetMaterials:
                xSec_dict_0[mV][process_key][tm].append([E_inc, xSec[tm] ] ) 
                MF_dict_0[mV][process_key][tm].append(max_F_TM[tm])

            samp_dict[mV][process_key].append([E_inc, \
                                        {"neval":neval0, "max_F": {tm:max_F_TM[tm] for tm in TargetMaterials},\
                                         "Eg_min":Egamma_min,\
                                         "integrator": save_copy}])

        for tm in TargetMaterials:        
            xSec_dict[mV][process_key][tm]= np.asarray(xSec_dict_0[mV][process_key][tm])

f_xSecs = open(SvDir + "dark_xSec_Dicts_New.pkl","wb")
f_samps = open(SvDir + "dark_samp_dicts_New.pkl","wb")

pickle.dump(xSec_dict,f_xSecs)
pickle.dump(samp_dict,f_samps)

f_xSecs.close()
f_samps.close()


print("Run Time of Script:  ", datetime.now() - startTime)
