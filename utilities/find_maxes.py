""" Generate samples of Standard Model EM shower events and save them.

    Uses saved VEGAS integrators to generate e+/- pair production and annihilation, 
    bremstrahlung and compton events for a range of initial particle energies 
    and target materials.
    The events are unweighted and saved for use in constructing a realistic shower.

    Typical usage:

    python Gen_Samples.py
"""
import os, sys
path = os.getcwd()
path = os.path.join(path,"../PETITE")
sys.path.insert(0,path)

from PETITE.all_processes import *
import pickle
import copy
import numpy as np
import argparse
import random as rnd
from datetime import datetime
import vegas
from tqdm import tqdm

#Dictionary of proceses with corresponding x-secs, form factors and Q**2 functions
process_info ={'PairProd' : {'diff_xsection': dsigma_pairprod_dimensionless,   'form_factor': g2_elastic, 'QSq_func': pair_production_q_sq_dimensionless},
                   'Comp'     : {'diff_xsection': dsigma_compton_dCT,     'form_factor': unity,      'QSq_func': dummy},
                    'Brem'     : {'diff_xsection': dsigma_brem_dimensionless,       'form_factor': g2_elastic, 'QSq_func': brem_q_sq_dimensionless},
                    'Ann'      : {'diff_xsection': dsigma_annihilation_dCT,'form_factor': unity,      'QSq_func': dummy},
                    'Moller'   : {'diff_xsection': dsigma_moller_dCT, 'form_factor': unity,      'QSq_func': dummy},
                    'Bhabha'   : {'diff_xsection': dsigma_bhabha_dCT, 'form_factor': unity,      'QSq_func': dummy}}

def get_file_names(path):
    all_files = os.listdir(path)
    print(all_files)
    #pickle_files = [file for file in all_files if file.endswith(".p")]
    pickle_files = [file for file in all_files if file.endswith("_v2.npy")]
    if 'readme.txt' in all_files:
        readme_file = 'readme.txt'
    else:
        readme_file = 0
    return(pickle_files, readme_file)

# do the find max work on an individual file
def do_find_max_work(params, process_file):

    [diff_xsec, FF_func, QSq_func] = [params['diff_xsec'], params['FF_func'], params['QSq_func']]
    event_info, integrand_or_map = process_file
    if type(event_info) == np.float64:
        event_info = {'E_inc':event_info, 'process':params['process']}

    if type(integrand_or_map) == vegas._vegas.Integrator:
        integrand = integrand
        save_copy = copy.deepcopy(integrand.map)

    elif type(integrand_or_map) == vegas._vegas.AdaptiveMap:
        integrand = vegas.Integrator(map=integrand_or_map, nstrat=nstrat_options[params['process']])
        save_copy = copy.deepcopy(integrand_or_map)

    event_info_H = event_info
    event_info_H['Z_T'] = 1.0  #Take training information, make event_info for hydrogen target

    max_F = 0
    xSec = {}
    for ZT in params['Z_T']:
        xSec[ZT] = 0.0

    integrand.set(max_nhcube=1, neval=params['neval'])
    for trial_number in range(params['n_trials']):
        for x, wgt in integrand.random(): #scan over integrand
                
            #EvtInfo={'E_inc': E_inc, 'm_e': m_electron, 'Z_T': Z_H, 'alpha_FS': alpha_em, 'm_V': 0}
            MM_H = wgt*diff_xsec(event_info_H, x)
            if MM_H > max_F:
                max_F=MM_H

            form_factor_hydrogen = FF_func(event_info_H, QSq_func(x, event_info_H))

            # with nucleus as target
            for ZT in params['Z_T']:
                event_info_target = event_info
                event_info_target['Z_T'] = ZT #Take training information, make event_info for desired target
                form_factor_target = FF_func(event_info_target, QSq_func(x, event_info_target))
                xSec[ZT] += MM_H*form_factor_target/form_factor_hydrogen/params['n_trials']

    samp_dict_info = {"neval":params['neval'], "max_F": max_F, "adaptive_map": save_copy}
    if "Eg_min" in event_info.keys():
        samp_dict_info['Eg_min'] = event_info['Eg_min']
    if 'Ee_min' in event_info.keys():
        samp_dict_info['Ee_min'] = event_info['Ee_min']
    samp_dict = [event_info['E_inc'], samp_dict_info]
    #xSec_dict = [event_info['E_inc'], xSec]
    return(samp_dict, xSec, event_info['E_inc'])

def main(params):
    print(params['process'])
    #Set up process to run
    path = "../" + params['import_directory'] + "/"
    file_list, readme_file = get_file_names(path)
    #print readme for info purposes
    if not(readme_file==0):
        print("Files to process")
        with open(path + readme_file, 'r') as file_temp:
            print(file_temp.read())

    samp_dict = {}
    xSec_dict = {}
    for process in params['process']:
        samp_dict[process] = []
        xSec_dict[process] = {}

        for ZT in params['Z_T']:
            xSec_dict[process][ZT] = []

    for file in file_list:
        print(file)
        process_file_array = np.load(path + file, allow_pickle=True)

        for process_file in tqdm(process_file_array):
            for process in params['process']:
                if process_file[0]['process'] == process:
                    process_params = copy.deepcopy(params)
                    process_params['process'] = process
                    process_params['diff_xsec'] = process_info[process]['diff_xsection']
                    process_params['FF_func'] = process_info[process]['form_factor']
                    process_params['QSq_func'] = process_info[process]['QSq_func']

                    samp_dict_TEMP, xSec_dict_TEMP, energy_TEMP = do_find_max_work(process_params, process_file)
                    samp_dict[process].append(samp_dict_TEMP)
                    for ZT in params['Z_T']:
                        xSec_dict[process][ZT].append([energy_TEMP, xSec_dict_TEMP[ZT]])

                
    save_path = "../" + params['save_location']  
    if os.path.exists(save_path) == False:
            os.system("mkdir -p " + save_path)
    f_xSecs = open(save_path + "/sm_xsecs.pkl","rb")
    f_samps = open(save_path + "/sm_maps.pkl","rb")
    try:
        xSec_dict_existing = pickle.load(f_xSecs)
    except:
        xSec_dict_existing = {}
    try:
        samp_dict_existing = pickle.load(f_samps)
    except:
        samp_dict_existing = {}

    for process in params['process']:
        if process in xSec_dict_existing.keys():
            print("Overwriting existing cross section for process " + process)
        xSec_dict_existing[process] = xSec_dict[process]
        if process in samp_dict_existing.keys():
            print("Overwriting existing samples for process " + process)
        samp_dict_existing[process] = np.array(samp_dict[process])

    f_xSecs_save = open(save_path + "/sm_xsecs.pkl","wb")
    f_samps_save = open(save_path + "/sm_maps.pkl","wb")
    pickle.dump(xSec_dict_existing,f_xSecs_save)
    pickle.dump(samp_dict_existing,f_samps_save)

    f_xSecs.close()
    f_samps.close()
    return()


###################################################################

startTime = datetime.now()




if __name__ == "__main__":
    #List of command line arguments
    parser = argparse.ArgumentParser(description='Process VEGAS integrators, find maxes etc', formatter_class = argparse.ArgumentDefaultsHelpFormatter)    
    # mandatory parameters    
    parser.add_argument('-import_directory', type=str, help='directory to import files from (path relative to main PETITE directory)', required=True)
    # optional parameters
    parser.add_argument('-A', type=float, default=12, help='atomic mass number')
    parser.add_argument('-Z', type=float, action='append', default=[6.0], help='atomic number of targets to save', required=True)
    #parser.add_argument('-Z', type=list, default=[6], help='atomic number of targets to save')
    parser.add_argument('-mT', type=float, default=11.178, help='nuclear target mass in GeV')
    parser.add_argument('-process', type=str, default='DarkBrem', help='processes to be run, if mV non-zero only DarkBrem \
        (choose from "PairProd", "Brem", "DarkBrem", "Comp", "Ann")')

    parser.add_argument('-save_location', type=str, default='cooked_integrators', help='directory to save integrators in (path relative to main PETITE directory)')
    parser.add_argument('-verbosity', type=bool, default=False, help='verbosity mode')
    parser.add_argument('-neval', type=int, default=300, help='neval value to provide to VEGAS for making integrator objects')
    parser.add_argument('-n_trials', type=int, default=100, help='number of evaluations to perform for estimating cross-section')

    args = parser.parse_args()
    print(args)
    if "all" in args.process:
        processes_to_do = ['PairProd', 'Comp', 'Ann', 'Brem', 'Moller', 'Bhabha']
    else:
        processes_to_do = [args.process]
    params = {'A': args.A, 'Z_T': np.unique(args.Z), 'mT': args.mT, 'process':processes_to_do, 
              'import_directory': args.import_directory, 'save_location': args.save_location, 'verbosity_mode': args.verbosity,
              'neval':args.neval, 'n_trials':args.n_trials}
    main(params)

    print("Run Time of Script:  ", datetime.now() - startTime)
