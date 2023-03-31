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
PickDir = Dir0 + "/NBP/"
SvDir  = Dir0 + "/RyanDicts/"
PPSamp0 = np.load("/Users/kjkelly/Dropbox/ResearchProjects/DarkShowers/LOCAL_Dark_Showers/raw_integrators/PairProd_TMP/PairProdIntegrators_Dimensionless.npy", allow_pickle=True)
BremSamp0 = np.load("/Users/kjkelly/Dropbox/ResearchProjects/DarkShowers/LOCAL_Dark_Showers/raw_integrators/Brem_TMP/BremIntegrators_Dimensionless.npy", allow_pickle=True)
#Brem Samples were generated with Egamma_min = 0.001 GeV = 1 MeV
Egamma_min = 0.001

CompSamp0 = np.load(PickDir+"ComptonPickles.npy", allow_pickle=True)
AnnSamp0 = np.load(PickDir+"AnnihilationPickles.npy", allow_pickle=True)


TargetMaterials = ['graphite','lead']
Z = {'graphite':6.0, 'lead':82.0}

Process_Files={"PairProd" : PPSamp0,
               "Comp": CompSamp0,
               "Brem" : BremSamp0,
               "Ann": AnnSamp0}

diff_xsections={"PairProd" : dsigma_pairprod_dimensionless,
                "Comp"     : dsigma_compton_dCT,    
                "Brem"     : dsigma_brem_dimensionless,
                "Ann"      : dsigma_annihilation_dCT }

FF_dict =      {"PairProd" : g2_elastic,
                "Comp"     : unity,
                "Brem"     : g2_elastic,
                "Ann"      : unity }

QSq_functions={"PairProd" : pair_production_q_sq_dimensionless, "Brem"  : brem_q_sq_dimensionless, "Comp": dummy, "Ann": dummy }

neval0 = 300
n_trials = 100

UnWS, XSecPP = [], []

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
        
        E_inc, integrand = process_file[ki]
        save_copy_integrand=copy.deepcopy(integrand)
        
        pts = []
        max_wgtTimesF=0

        
    
        xSec={}
        for tm in TargetMaterials:
            xSec[tm]=0.0

        integrand.set(max_nhcube=1, neval=neval0)
        for trial_number in range(n_trials):
            for x, wgt in integrand.random():

                Z_H=1
                
                EvtInfo={'E_inc': E_inc, 'm_e': m_electron, 'Z_T': Z_H, 'alpha_FS': alpha_em, 'm_V': 0, 'Eg_min':0.001}
                MM_H = wgt*diff_xsec(EvtInfo, x)
                if MM_H > max_wgtTimesF:
                    max_F=MM_H
                    max_x = np.asarray(x)
                    max_wgt= wgt 

                FF_H = FF_func(1.0, m_electron, QSq(x, EvtInfo))

                for tm in TargetMaterials:
                    ZT = Z[tm]
                    FF= FF_func(ZT, m_electron, QSq(x, EvtInfo) )
                    xSec[tm] += MM_H*FF/FF_H/n_trials


        samp_dict[process_key].append([E_inc, \
                                      {"neval":neval0, "max_F": max_F, "max_X": max_x, "max_wgt": max_wgt, "Eg_min":Egamma_min,\
                                       "integrator": save_copy_integrand}])
        for tm in TargetMaterials:
            xSec_dict[process_key][tm].append([E_inc, xSec[tm] ] ) 
            
    xSec_dict[process_key][tm]= np.asarray(xSec_dict[process_key][tm] ) 
    samp_dict[process_key]    = samp_dict[process_key]

f_xSecs = open(SvDir + "Mar24_xSec_Dicts_neval.pkl","wb")
f_samps = open(SvDir + "Mar24_samp_Dicts_neval.pkl","wb")

pickle.dump(xSec_dict,f_xSecs)
pickle.dump(samp_dict,f_samps)

f_xSecs.close()
f_samps.close()


print("Run Time of Script:  ", datetime.now() - startTime)
