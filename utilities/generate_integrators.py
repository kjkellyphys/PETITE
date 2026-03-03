""" This script generates VEGAS integrators for various production processes and stitch them together for different energies.
    It can be called to generate the default set of integrators by using "script_generateintegrators.py" in the /utilities/ directory.
"""

import numpy as np
import os
import importlib

from PETITE.physical_constants import target_information
from glob import glob

from multiprocessing import Pool
import pickle
from functools import partial
import datetime

# helper function to turn float to string
def generate_mass_string(mass, mode="MeV"):
    if mode == "MeV":
        return str(int(np.floor(mass*1000.)))+"MeV"
    elif mode == "keV":
        return str(int(np.floor(mass*1000000.)))+"keV"
    else:
        raise ValueError("Mode must be either 'MeV' or 'keV'")

def make_readme(params, process, process_directory):
    """Creates a readme file to accompany the integrators with supplementary information on the parameters used to generate the integrators (Z, A, mV, etc.)
    Input: 
        params: dictionary of parameters used in generating integrators
        process: string of process name
        file_info: directory where files should be saved
    """
    readme_file = open(process_directory + process + "_readme.txt", 'w')
    readme_file.write("Integrators for " + process)
    readme_file.write("\n\nKinetic Energy/GeV |  Filename\n\n")
    for index, kinetic_energy in enumerate(params['initial_kinetic_energy_list']):
        line = "{kinen:9.3f}  |  {proc}_{indx}.p\n".format(kinen = kinetic_energy, indx = index, proc=process)
        readme_file.write(line)
    readme_file.write(f'\nIntegrators made on {datetime.datetime.now()}')
    readme_file.close()
    return()

def run_vegas_in_parallel(params, process, verbosity_mode, model_name, process_directory, energy_index, overwrite=False):
    '''Run VEGAS in parallel for a given energy index and process, and save the integrator adaptive map 
    and relevant parameters to a pickle file.
    Input:
        params: dictionary of parameters containing
            mV : dark vector mass in GeV
            A : target atomic mass number
            Z : target atomic number
            mT : target mass in GeV
        process: string of process name
        verbosity_mode: boolean
        process_directory: directory where files should be saved
        energy_index: index of kinetic energy in initial_kinetic_energy_list
    '''
    params['T_inc'] = params['initial_kinetic_energy_list'][energy_index]
    file_name = process_directory + process + '_' + str(energy_index) + ".p"
    if os.path.exists(file_name) and not overwrite:
        print("Already generated integrator for this point\n")
    else:
        if params["verbosity"]:
            print("Starting VEGAS for energy index ", energy_index)

        if model_name == "sm":
            vegas_integration_function = importlib.import_module('PETITE.sm.sm_processes').vegas_integration
        else:
            vegas_integration_function = importlib.import_module(f'PETITE.{model_name}.bsm_processes').bsm_vegas_integration
        VEGAS_integrator = vegas_integration_function(params, process, verbose=verbosity_mode, mode="Pickle")

        # VEGAS_integrator = 0
        if params["verbosity"]:
            print("Done VEGAS for energy index ", energy_index)

        # Objects to be saved. Should include all important parameters (in params) and the VEGAS integrator adaptive map.
        params['process'] = process
        object_to_save = [params, VEGAS_integrator.map]
        pickle.dump(object_to_save, open(file_name, "wb"))
        if params["verbosity"]:
            print("File created: " + file_name)
    return()


def make_integrators(params, model_name, process, paralellize=True, overwrite=False):
    """
    Generate vegas integrator pickles for a given process and set of parameters.
    Input:
        params: dictionary of parameters containing
            mV : dark vector mass in GeV
            A : target atomic mass number
            Z : target atomic number
            mT : target mass in GeV
        process: string of process name
        paralellize: boolean to run in parallel or not
    """

    if model_name == "sm":
        process_directory = params['save_location'] + '/sm/' + process + '/'
    else:
        #determine the name of the organizational parameter and its value for this process
        organizational_key = importlib.import_module(f'PETITE.{model_name}.bsm_process_information').organizational_key
        organizational_parameter = params[organizational_key]
        process_directory = params['save_location'] + '/' + model_name + '/' + process + '/' + organizational_key + '_' + generate_mass_string(organizational_parameter, mode=params['mode']) + '/'

    verbosity_mode = params['verbosity']
    initial_kinetic_energy_list = params['initial_kinetic_energy_list'] 
    kinetic_energy_index_list = range(len(initial_kinetic_energy_list))

    if verbosity_mode:
        print("Doing process: ", process)
        if model_name != "sm":
            print("Mass: ", organizational_parameter)

    if not(os.path.exists(process_directory)):
        os.system("mkdir -p " + process_directory)

    if paralellize:
        # pool parallelizes the generation of integrators
        pool = Pool()
        _ = pool.map(
            partial(
                run_vegas_in_parallel,
                params,
                process,
                verbosity_mode,
                model_name,
                process_directory,
                overwrite=overwrite,
            ),
            kinetic_energy_index_list,
        )
    else:
        for kinetic_energy_index in kinetic_energy_index_list:
            run_vegas_in_parallel(
                params,
                process,
                verbosity_mode,
                model_name,
                process_directory,
                kinetic_energy_index,
                overwrite=overwrite,
            )
    # make the human readable file contining info on params of run and put in directory
    make_readme(params, process, process_directory)
    if params["verbosity"]:
        print(
            "make_integrators is complete, readme files created in "
            + process_directory
            + " for convenience"
        )

    return ()

# Set up parameters for and then run find_maxes
def call_find_maxes(params, model_name, list_of_processes, verbose=False): 
    """Call find_maxes.py to find the maximum of the integrand for each energy.
    Note that find_maxes.py will save the maximum of the integrand and the corresponding energy to a pickle file.
    Input:
        params: dictionary of parameters used in generating integrators
        list_of_processes: vector containing all processes to be run
    """
    import find_maxes
    if verbose:
        print("Now running find_maxes....please wait")
    find_maxes_params = params
    # add process key to find_maxes_params
    find_maxes_params['process'] = list_of_processes
    # import directory is the directory where the integrators are saved
    # find_maxes_params['import_directory'] = params['save_location'] + '/' + list_of_processes + '/'
    # if params['neval'] is not present, default to 300
    if 'neval' not in find_maxes_params:
        find_maxes_params['neval'] = 300
    # if params['n_trials'] is not present, default to 100
    if 'n_trials' not in find_maxes_params:
        find_maxes_params['n_trials'] = 100
    if verbose:
        print("Parameters used in find_maxes: ", find_maxes_params)

    if model_name == "sm":     
        find_maxes.main(find_maxes_params)
    else:
        find_maxes.main_bsm(find_maxes_params, model_name)
    return

def stitch_integrators(dir):
    """
    Stich together integrators adaptive maps for different energies
    Input:
        dir: directory for each process where integrator adaptive maps are saved
        file_name: name of file to save stiched integrator to
    """
    # get all files in directory
    files = glob(dir + '/*.p')
    # sort files by energy
    files.sort(key=lambda x: int(x.split('_')[-1].split('.')[0]))
    # get process name from first file
    process = files[0].split('/')[-1].split('_')[0]
    # stitch them together in a vector
    to_save = [pickle.load(open(file_name, 'rb')) for file_name in files]  
    # save stitched integrator as numpy arrays
    file_name = dir + '/' + process + '_AdaptiveMaps.npy'
    np.save(file_name, to_save)
    print('Stitched integrator saved as ' + file_name)
    return

def cleanup(dir):
    """
    Cleanup all the unnecessary files (<process>_0.p, <process>_1.p etc) that 
    are now saved in <process>_AdaptiveMaps.npy 
    Input:
        dir: directory where cleanup to be done
    """
    print("Cleaning up files in " + dir)
    os.system("rm " + dir + "*.p")
    return

def organize_directories_final(dir):
    '''
    Moves all folders in mother directory dir to auxiliary directory dir/auxiliary, leaving behind the .pkl files
    '''
    # get all directories in mother directory
    directories = glob(dir + '/*/')
    # create auxiliary directory if it doesn't exist
    if not(os.path.exists(dir + '/auxiliary/')):
        os.system("mkdir -p " + dir + "/auxiliary/")
    # move all directories to auxiliary directory
    for directory in directories:
        os.system("mv " + directory + " " + dir + "/auxiliary/")
    return