""" Generate samples of Standard Model EM and dark shower events and save them.

    Uses saved VEGAS adaptive maps to generate the final adaptive maps and cross sections given processes for a range of initial particle energies 
    and target materials.

    Typical usage:

    python find_maxes.py -A=12 -Z=6 -mT=12 -process='Comp' -import_directory='/Users/johndoe/PETITE/data/Comp' 
"""
import os

from PETITE.physical_constants import target_information#, m_electron, m_muon
import pickle
import copy
import numpy as np
import argparse
import random as rnd
from datetime import datetime
import vegas
from tqdm import tqdm
import importlib
from generate_integrators import generate_mass_string

def get_file_names(path):
    ''' Get the names of all the files in a directory.
    Input:
        path: path to the directory
    Output:
        pickle_files: list of names of all the .p files in the directory (VEGAS integrator adaptive maps for given process and given energy)
        readme_file: name of the readme file in the directory (contains information about the process and the energy)
    '''
    all_files = os.listdir(path)
    # print('files:', all_files)
    pickle_files = [file for file in all_files if file.endswith(".p")]
    if 'readme.txt' in all_files:
        readme_file = 'readme.txt'
    else:
        readme_file = 0
    return(pickle_files, readme_file)

# do the find max work on an individual file
def do_find_max_work(params, model_name, process_file):
    """ Find the maximum value of the integrand for a given process_file.
    Input:
        params: dictionary of parameters for the process
        process_file: array of event_info and integrand which was read from a file (0.p, 1.p, ...)
    Output:
        samp_dict: dictionary of information about the sampling, containing:
            'neval': number of evaluations used in VEGAS (neval)
            'max_F': maximum value of the integrand (max_F)
            'Eg_min': minimum energy of the outgoing photon (Eg_min) if it was specified in the event_info
            'Te_min': minimum kinetic energy of the outgoing electron (Te_min) if it was specified in the event_info
            'adaptive_map': adaptive map used in the sampling (adaptive_map)
        xSec: dictionary of cross sections for each target material
        event_info['T_inc']: kinetic energy of the incoming particle
    """

    if model_name == "sm":
        process_info = importlib.import_module('PETITE.sm.process_information').sm_process_info
        vg_options = importlib.import_module('PETITE.sm.sm_processes').vegas_integrator_options
    else:
        process_info = importlib.import_module(f'PETITE.{model_name}.bsm_process_information').bsm_process_info
        vg_options = importlib.import_module(f'PETITE.{model_name}.bsm_processes').bsm_vegas_integrator_options
    
    event_info, integrand_or_map = process_file
    process = event_info['process']
    diff_xsec = process_info[process]['differential_cross_section']
    
    integrand = vegas.Integrator(map=integrand_or_map, **vg_options[process])
    save_copy = copy.deepcopy(integrand_or_map)
    integrand.set(max_nhcube=1, neval=params["neval"])

    max_F_TM = {}
    xSec = {}
    for tm in params['process_targets']:
        xSec[tm] = 0.0
        max_F_TM[tm] = 0.0

    for tm in params['process_targets']:
        event_info_target = copy.deepcopy(event_info)
        event_info_target["Z_T"] = target_information[tm]["Z_T"]
        event_info_target["A_T"] = target_information[tm]["A_T"]
        event_info_target["mT"] = target_information[tm]["mT"]
        ndim = process_info[process]['dimensionalities']
        batch_f = diff_xsec(event_info_target, ndim, batch_mode=True)

        for trial_number in range(params["n_trials"]):
            for x, wgt in integrand.random_batch():  # scan over integrand
                MM = wgt * batch_f(x)
                if np.max(MM) > max_F_TM[tm]:
                    max_F_TM[tm] = np.max(MM)
                xSec[tm] += np.sum(MM) / params["n_trials"]

    samp_dict_info = {"neval":params['neval'], "max_F": {tm:max_F_TM[tm] for tm in params['process_targets']}, "adaptive_map": save_copy}
    if "Eg_min" in event_info.keys():
        samp_dict_info['Eg_min'] = event_info['Eg_min']
    if 'Te_min' in event_info.keys():
        samp_dict_info['Te_min'] = event_info['Te_min']
    samp_dict = [event_info['T_inc'], samp_dict_info]
    #xSec_dict = [event_info['E_inc'], xSec]
    return(samp_dict, xSec, event_info['T_inc'])

def main(params):
    """ Find the maximum value of the integrand for a given process by looking into the directory containing the adaptive maps.
    Input:
        params: dictionary of parameters for the process containing the following keys:
            'process': list of processes to be run
            'Z_T': list of target materials to be run
            'neval': number of evaluations used in VEGAS
            'save_location': path to the mother directory whose subdirs contain the adaptive maps
            # add all needed parameters here!
    """
    '''
    Pseudoalgorithm:
    Loops over all processes in params['process']. 
    Access each 'process' directory containing the 'process_AdaptiveMaps.npy', and load these files.
    Runs do_find_max_work function on each of these files for each energy. 
    All the results are saved in a single file as a dictionary and pickled (sm_maps.pkl).
    *Later I will create the xsec dictionary and save it in a separate file (sm_xsec.pkl).
    '''

    # params['process'] should be a list of all processes to be run
    # available processes are ['PairProd', 'Comp', 'Ann', 'Brem', 'Moller', 'Bhabha']
    # If a single process is given, convert single process to a list if necessary
    if isinstance(params['process'], str):
        params['process'] = [params['process']]
    # Convert single target to a list if necessary
    if isinstance(params['process_targets'], str):
        params['process_targets'] = [params['process_targets']]
    print('List of processes: ', params['process'])
    print('List of target materials: ', params['process_targets'])

    # Initialise dictionaries to store samples and cross sections
    final_sampling_dict = {}
    final_xsec_dict = {}

    process_info = importlib.import_module('PETITE.sm.process_information').sm_process_info

    # Before starting, check for invalid processes
    for process in params['process']:
        # if process is not in list of processes, raise an error
        if process not in process_info.keys():
            raise ValueError('Process \'', process ,'\' not in list of available processes for standard model showers.')

    # Loop over all processes
    for process in params['process']:
        # Get the path to the directory containing the adaptive maps
        path = params['save_location'] + '/auxiliary/' + process + '/'
        # Get adaptive map main file (created by stitch_integrators)
        adaptive_maps_file = path + process + '_AdaptiveMaps.npy'
        # Load adaptive map file. Format: list of [params, adaptive_map]
        try:
            adaptive_maps = np.load(adaptive_maps_file, allow_pickle=True)
        except Exception as e:
            print(e)
            print(
                f"Tried to load adaptive maps {adaptive_maps_file} but could not find them.",
            )
            path = params["save_location"] + '/' + process + "/"
            adaptive_maps_file = path + process + "_AdaptiveMaps.npy"
            print(f"Trying again at {adaptive_maps_file}")
            adaptive_maps = np.load(adaptive_maps_file, allow_pickle=True)
        final_sampling_dict[process] = []
        final_xsec_dict[process] = {}
        # Initialise dictionary to store cross sections for each target material
        for tm in params['process_targets']:
            final_xsec_dict[process][tm] = []
        print('File being processed: ', adaptive_maps_file)
        # Add relevant info to params
        params['process']   = process
        #params['FF_func']   = process_info[process]['form_factor']
        #params['QSq_func']  = process_info[process]['QSq_func']

        ####=====------ FIND MAX ------=====####
        # loop over all adaptive_map in adaptive_maps
        for adaptive_map in adaptive_maps:
            sampling, cross_section, incoming_energy = do_find_max_work(params, "sm", adaptive_map)
            # Append sampling dictionary to final sampling dictionary
            final_sampling_dict[process].append(sampling)
            # Append incoming energy and cross section dictionary to final cross section dictionary for each target material
            for tm in params['process_targets']:
                final_xsec_dict[process][tm].append([incoming_energy, cross_section[tm]])
        ####=====------ FIND MAX ------=====####

    # Save sampling and cross section dictionaries to mother directory
    with open(params['save_location'] + '/sm_maps.pkl', 'wb') as f:
        pickle.dump(final_sampling_dict, f)
    with open(params['save_location'] + '/sm_xsec.pkl', 'wb') as f:
        pickle.dump(final_xsec_dict, f)
    print("Saved cross sections to " + params['save_location'] + "/sm_xsecs.pkl")
    print("Saved samples to " + params['save_location'] + "/sm_maps.pkl")
    return

def main_bsm(params, model_name="dark_vector"):
    """ Find the maximum value of the integrand for a given process by looking into the directory containing the adaptive maps.
    Input:
        params: dictionary of parameters for the process containing the following keys:
            'process': list of processes to be run
            'Z_T': list of target materials to be run
            'neval': number of evaluations used in VEGAS
            'save_location': path to the mother directory whose subdirs contain the adaptive maps
            # add all needed parameters here!
    """
    '''
    Pseudoalgorithm:
    Loops over all processes in params['process']. 
    Access each 'process' directory containing the 'process_AdaptiveMaps.npy', and load these files.
    Runs do_find_max_work function on each of these files for each energy. 
    All the results are saved in a single file as a dictionary and pickled (sm_maps.pkl).
    *Later I will create the xsec dictionary and save it in a separate file (sm_xsec.pkl).
    '''

    # params['process'] should be a list of all processes to be run
    # available processes are ['PairProd', 'Comp', 'Ann', 'Brem', 'Moller', 'Bhabha']
    # If a single process is given, convert single process to a list if necessary
    if isinstance(params['process'], str):
        params['process'] = [params['process']]
    if isinstance(params['process'], type({}.keys())):
        params['process'] = list(params['process'])
    # Convert single target to a list if necessary
    if isinstance(params['process_targets'], str):
        params['process_targets'] = [params['process_targets']]
    if isinstance(params['organizational_list'], float):
        params['organizational_list'] = [params['organizational_list']]

    print('List of processes: ', params['process'])
    print('List of target materials: ', params['process_targets'])
    print('List of organizational parameters: ', params['organizational_list'])
    
    org_key = importlib.import_module(f'PETITE.{model_name}.bsm_process_information').organizational_key
    process_info = importlib.import_module(f'PETITE.{model_name}.bsm_process_information').bsm_process_info
    
    # Initialise dictionaries to store samples and cross sections
    final_sampling_dict = {}
    final_xsec_dict = {}
    
    #Loop over all masses
    for MM in params['organizational_list']:
        final_sampling_dict[MM] = {}
        final_xsec_dict[MM] = {}

        # Before starting, check for invalid processes
        for process in params['process']:
            # if process is not in list of processes, raise an error
            if process not in process_info.keys():
                raise ValueError('Process \'', process ,'\' not in list of available processes for specified model, ', model_name, '.')

        # Loop over all processes
        for process in params['process']:
            # Get the path to the directory containing the adaptive maps
            mass_str = generate_mass_string(MM, mode=params['mode'])
            path1 = params['save_location'] + '/auxiliary/' + model_name + '/' + process + '/' + org_key + '_' + mass_str + '/' + process + '_AdaptiveMaps.npy'
            path2 = params['save_location'] + '/' + model_name + '/' + process + '/' + org_key + '_' + mass_str + '/' + process + '_AdaptiveMaps.npy'
            adaptive_maps_file = path1
            # Load adaptive map file. Format: list of [params, adaptive_map]
            try:
                adaptive_maps = np.load(adaptive_maps_file, allow_pickle=True)
            except Exception as e:
                adaptive_maps_file = path2# + process + "_AdaptiveMaps.npy"
                adaptive_maps = np.load(adaptive_maps_file, allow_pickle=True)
            final_sampling_dict[MM][process] = []
            final_xsec_dict[MM][process] = {}
            # Initialise dictionary to store cross sections for each target material
            for tm in params['process_targets']:
                final_xsec_dict[MM][process][tm] = []
            print('File being processed: ', adaptive_maps_file)
            # Add relevant info to params
            fm_params = copy.deepcopy(params)
            fm_params['process']   = process
            fm_params[org_key]        = MM
            #fm_params['FF_func']   = process_info[process]['form_factor']
            #fm_params['QSq_func']  = process_info[process]['QSq_func']

            ####=====------ FIND MAX ------=====####
            # loop over all adaptive_map in adaptive_maps
            for adaptive_map in adaptive_maps:
                sampling, cross_section, incoming_energy = do_find_max_work(fm_params, model_name, adaptive_map)
                # Append sampling dictionary to final sampling dictionary
                final_sampling_dict[MM][process].append(sampling)
                # Append incoming energy and cross section dictionary to final cross section dictionary for each target material
                for tm in params['process_targets']:
                    final_xsec_dict[MM][process][tm].append([incoming_energy, cross_section[tm]])
            ####=====------ FIND MAX ------=====####

    # Save sampling and cross section dictionaries to mother directory
    with open(params['save_location'] + '/' + model_name + '_maps.pkl', 'wb') as f:
        pickle.dump(final_sampling_dict, f)
    with open(params['save_location'] + '/' + model_name + '_xsec.pkl', 'wb') as f:
        pickle.dump(final_xsec_dict, f)
    print("Saved cross sections to " + params['save_location'] + "/"+model_name+"_xsec.pkl")
    print("Saved samples to " + params['save_location'] + "/"+model_name+"_maps.pkl")
    return



startTime = datetime.now()



'''
if __name__ == "__main__":
    #List of command line arguments
    parser = argparse.ArgumentParser(description='Process VEGAS integrators, find maxes etc', formatter_class = argparse.ArgumentDefaultsHelpFormatter)    
    # mandatory parameters    
    parser.add_argument('-import_directory', type=str, help='directory to import files from (path relative to main PETITE directory)', required=True)
    # optional parameters
    parser.add_argument('-A', type=float, default=12, help='atomic mass number')
    parser.add_argument('-Z', type=float, action='append', default=[6.0], help='atomic number of targets to save')
    #parser.add_argument('-Z', type=list, default=[6], help='atomic number of targets to save')
    parser.add_argument('-mT', type=float, default=11.178, help='nuclear target mass in GeV')
    parser.add_argument('-process', type=str, default='DarkBrem', help='processes to be run, if mV non-zero only DarkBrem \
        (choose from "PairProd", "Brem", "DarkBrem", "Comp", "Ann")')

    parser.add_argument('-save_location', type=str, default='cooked_integrators', help='directory to save integrators in (path relative to main PETITE directory)')
    parser.add_argument('-verbosity', type=bool, default=False, help='verbosity mode')
    parser.add_argument('-neval', type=int, default=300, help='neval value to provide to VEGAS for making integrator objects', required=True)
    parser.add_argument('-n_trials', type=int, default=100, help='number of evaluations to perform for estimating cross-section', required=True)

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
'''