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
import os, sys
import argparse
import random as rnd
from datetime import datetime
startTime = datetime.now()


#Dictionary of proceses with corresponing x-secs, form factors and Q**2 functions
process_info ={'PairProd' : {'diff_xsection': dsigma_pairprod_dP_T,   'form_factor': g2_elastic, 'QSq_func': pair_production_q_sq},
               'Comp'     : {'diff_xsection': dsigma_compton_dCT,     'form_factor': unity,      'QSq_func': dummy},
               'Brem'     : {'diff_xsection': dsigma_brem_dP_T,       'form_factor': g2_elastic, 'QSq_func': brem_q_sq},
               'Ann'      : {'diff_xsection': dsigma_annihilation_dCT,'form_factor': unity,      'QSq_func': dummy}}


#List of command line arguments
parser = argparse.ArgumentParser(description='Process VEGAS integrators, find maxes etc', formatter_class = argparse.ArgumentDefaultsHelpFormatter)    
# mandatory parameters    
parser.add_argument('-import_file', type=str, help='file to import and find max (path relative to main PETITE directory)', required=True)
# optional parameters
parser.add_argument('-A', type=float, default=12, help='atomic mass number')
parser.add_argument('-Z', type=float, default=6, help='atomic number')
parser.add_argument('-mT', type=float, default=11.178, help='nuclear target mass in GeV')
parser.add_argument('-process', type=str, default='DarkBrem', help='processes to be run, if mV non-zero only DarkBrem \
    (choose from "PairProd", "Brem", "DarkBrem", "Comp", "Ann")')

parser.add_argument('-save_location', type=str, default='cooked_integrators', help='directory to save integrators in (path relative to main PETITE directory)')
parser.add_argument('-verbosity', type=bool, default=False, help='verbosity mode')

args = parser.parse_args()

#Set up process to run
process_file = np.load("../" + args.import_file, allow_pickle=True)
diff_xsec    = process_info[args.process]['diff_xsection']
FF_func      = process_info[args.process]['form_factor']
QSq_func     = process_info[args.process]['QSq_func']
xSec_dict = {}
samp_dict = []
UnWS, XSecPP = [], []
print(process_file[0])
# run over the energies, integrators in imported file
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
    max_wgtTimesF = 0
    xSec = {}

    for x, wgt in integrand.random(): #scan over integrand

        # with a proton as target
        Z_H=1
            
        EvtInfo={'E_inc': E_inc, 'm_e': m_electron, 'Z_T': Z_H, 'alpha_FS': alpha_em, 'm_V': 0}
        MM_H_0 = wgt*diff_xsec(EvtInfo, x)
        MM_H= MM_H_0   #max(MM_H_0, MM_H_higher, MM_H_lower)
        if MM_H > max_wgtTimesF:
            max_F=MM_H
            max_x = np.asarray(x)
            max_wgt= wgt 

        # with nucleus as target
        form_factor = FF_func(args.Z, m_electron, QSq_func(x, m_electron, E_inc))
        xSec += MM_H_0*form_factor


    samp_dict.append([E_inc, {"max_F": max_F, "max_X": max_x, "max_wgt": max_wgt, "integrator": save_copy_integrand}])
    xSec_dict.append([E_inc, xSec]) 
            
    

f_xSecs = open(args.save_location + args.process + "/xSec_Dicts.pkl","wb")
f_samps = open(args.save_location + args.process + "/samp_Dicts.pkl","wb")

pickle.dump(xSec_dict,f_xSecs)
pickle.dump(samp_dict,f_samps)

f_xSecs.close()
f_samps.close()


print("Run Time of Script:  ", datetime.now() - startTime)
