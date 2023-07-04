""" Generate samples of Standard Model EM shower events and save them.

    Uses saved VEGAS integrators to generate e+/- pair production and annihilation, 
    bremstrahlung and compton events for a range of initial particle energies 
    and target materials.
    The events are unweighted and saved for use in constructing a realistic shower.

    Typical usage:

    python Gen_Samples.py
"""

def get_file_names(path):
    all_files = os.listdir(path)
    print(all_files)
    pickle_files = [file for file in all_files if file.endswith(".p")]
    if 'readme.txt' in all_files:
        readme_file = 'readme.txt'
    else:
        readme_file = 0
    return(pickle_files, readme_file)

# do the find max work on an individual file
def do_find_max_work(params, process_file):

    [diff_xsec, FF_func, QSq_func] = [params['diff_xsec'], params['FF_func'], params['QSq_func']]
    E_inc, integrand = process_file
    save_copy_integrand=copy.deepcopy(integrand)

    max_wgtTimesF = 0
    xSec = 0

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
        form_factor = FF_func(params['Z_T'], m_electron, QSq_func(x, m_electron, E_inc))
        xSec += MM_H_0*form_factor

    samp_dict = [E_inc, {"max_F": max_F, "max_X": max_x, "max_wgt": max_wgt, "integrator": save_copy_integrand}]
    xSec_dict = [E_inc, xSec]
    return(samp_dict, xSec_dict)

def main(params):
    #Dictionary of proceses with corresponding x-secs, form factors and Q**2 functions
    process_info ={'PairProd' : {'diff_xsection': dsigma_pairprod_dimensionless,   'form_factor': g2_elastic, 'QSq_func': pair_production_q_sq},
                   'Comp'     : {'diff_xsection': dsigma_compton_dCT,     'form_factor': unity,      'QSq_func': dummy},
                    'Brem'     : {'diff_xsection': dsigma_brem_dimensionless,       'form_factor': g2_elastic, 'QSq_func': brem_q_sq},
                    'Ann'      : {'diff_xsection': dsigma_annihilation_dCT,'form_factor': unity,      'QSq_func': dummy}}
    #Set up process to run
    path = "../" + params['import_directory'] + "/"
    file_list, readme_file = get_file_names(path)
    #print readme for info purposes
    if not(readme_file==0):
        print("Files to process")
        with open(path + readme_file, 'r') as file_temp:
            print(file_temp.read())

    diff_xsec    = process_info[params['process']]['diff_xsection']
    FF_func      = process_info[params['process']]['form_factor']
    QSq_func     = process_info[params['process']]['QSq_func']
    process_params = params
    process_params['diff_xsec'] = diff_xsec
    process_params['FF_func'] = FF_func
    process_params['QSq_func'] = QSq_func
    samp_dict = []
    xSec_dict = []

    for file in file_list:
        process_file = np.load(path + file, allow_pickle=True)
        samp_dict_TEMP, xSec_dict_TEMP = do_find_max_work(process_params, process_file)
        samp_dict.append(samp_dict_TEMP)
        xSec_dict.append(xSec_dict_TEMP)

                
    save_path = "../" + params['save_location'] + "/" + params['process']    
    if os.path.exists(save_path) == False:
            os.system("mkdir -p " + save_path)
    f_xSecs = open(save_path + "/sm_xsecs.pkl","wb")
    f_samps = open(save_path + "/sm_maps.pkl","wb")

    pickle.dump(np.array(xSec_dict),f_xSecs)
    pickle.dump(samp_dict,f_samps)

    f_xSecs.close()
    f_samps.close()
    return()


###################################################################
import os, sys
path = os.getcwd()
path = os.path.join(path,"../PETITE")
sys.path.insert(0,path)

from all_processes import *
import pickle
import copy
import numpy as np
import argparse
import random as rnd
from datetime import datetime
startTime = datetime.now()




if __name__ == "__main__":
    #List of command line arguments
    parser = argparse.ArgumentParser(description='Process VEGAS integrators, find maxes etc', formatter_class = argparse.ArgumentDefaultsHelpFormatter)    
    # mandatory parameters    
    parser.add_argument('-import_directory', type=str, help='directory to import files from (path relative to main PETITE directory)', required=True)
    # optional parameters
    parser.add_argument('-A', type=float, default=12, help='atomic mass number')
    parser.add_argument('-Z', type=float, default=6, help='atomic number')
    parser.add_argument('-mT', type=float, default=11.178, help='nuclear target mass in GeV')
    parser.add_argument('-process', type=str, default='DarkBrem', help='processes to be run, if mV non-zero only DarkBrem \
        (choose from "PairProd", "Brem", "DarkBrem", "Comp", "Ann")')

    parser.add_argument('-save_location', type=str, default='cooked_integrators', help='directory to save integrators in (path relative to main PETITE directory)')
    parser.add_argument('-verbosity', type=bool, default=False, help='verbosity mode')

    args = parser.parse_args()
    print(args)
    params = {'A': args.A, 'Z_T': args.Z, 'mT': args.mT, 'process': args.process, 
              'import_directory': args.import_directory, 'save_location': args.save_location, 'verbosity_mode': args.verbsoity}
    main(params)

    print("Run Time of Script:  ", datetime.now() - startTime)
