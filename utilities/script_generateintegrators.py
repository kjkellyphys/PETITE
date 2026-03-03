import numpy as np
import importlib
import os

import PETITE.physical_constants as const
import generate_integrators as genint

from PETITE.sm.process_information import sm_process_info
SM_processes = sm_process_info.keys()
#By default, load in all possible SM processes. If you prefer a subset, specify that here:
#processes_to_do = []
sm_processes_to_generate = SM_processes

DEFAULT_PATH = os.getcwd()
DEFAULT_PATH = os.path.split(DEFAULT_PATH)[0]
#print("Default path:", DEFAULT_PATH)

def mainSM(Te_min=0.0015, Te_max=100.0, Npoints=100, path=DEFAULT_PATH):
    ##################################
    ##         SM processes         ##
    ##################################
    # List of incoming particle energies at which to calculate integrators for each process
    # Necessary parameters for processing the integrators to determine cross sections
    training_params = {"save_location": path}
    processing_params = {'process_targets':['graphite','lead','iron','aluminum','molybdenum'], 'save_location':path}
    #args = training_params.update(processing_params)
    # List of processes to do
    # Loop over processes, carrying out each step of the calculation, they can also be called in one command generate_integrators

    for process in sm_processes_to_generate:
        path1 = training_params['save_location'] + '/auxiliary/' + process + "/" + process + "_AdaptiveMaps.npy"
        path2 = training_params['save_location'] + '/' + process + '/' + process + "_AdaptiveMaps.npy"

        initial_kinetic_energy_list = np.geomspace(Te_min, Te_max, Npoints)
        # Necessary parameters for generating the integrators, note save_location should be altered as preferred
        training_params = {'verbosity':True, 'initial_kinetic_energy_list':initial_kinetic_energy_list,
                        'save_location':path,
                        'run_find_maxes':True}

        if os.path.exists(path1) or os.path.exists(path2):
            print("Already finished this whole process, skipping")
            continue
        else:
            genint.make_integrators(training_params, process)
            genint.stitch_integrators(training_params['save_location'] + '/' + process + '/')
            genint.cleanup(training_params['save_location'] + "/" + process + "/")
    # List of processes to run find_maxes on, need not be the same as list above
    find_maxes_processes_to_do = sm_processes_to_generate
    genint.call_find_maxes(processing_params, "sm", find_maxes_processes_to_do)

default_mass_list = [0.003, 0.010, 0.030, 0.100, 0.300, 1.000]
def mainBSM(model_name="dark_vector", model_keys={}, bsm_processes="all", Te_max=100.0, Npoints=100, path=DEFAULT_PATH, organizational_list=default_mass_list, mode="MeV", verbosity=True):
    save_location = path + '/'
    training_params = {'verbosity':verbosity, 'initial_kinetic_energy_list':None,
                       'save_location':save_location,
                       'run_find_maxes':True, 'training_target':'hydrogen', 'mT':200.0, "mode":mode}
    bsm_process_information = importlib.import_module(f'PETITE.{model_name}.bsm_process_information')
    required_keys = bsm_process_information.model_required_keys
    for key in required_keys:
        if key not in model_keys.keys():
            print(f"Missing required key {key} for model {model_name}, using default value {bsm_process_information.default_key_values[key]}")
            model_keys[key] = bsm_process_information.default_key_values[key]
    training_params.update(model_keys)

    for mass in organizational_list:
        organizational_parameter_key = bsm_process_information.organizational_key
        training_params.update({organizational_parameter_key: mass})
        if bsm_processes == "all":
            processes_to_do = bsm_process_information.bsm_process_info.keys()
        else:
            processes_to_do = bsm_processes

        for process in processes_to_do:
            bsm_function_class = bsm_process_information.bsm_process_info[process]['differential_cross_section']
            bsm_function = bsm_function_class(event_info={organizational_parameter_key: mass}, ndim=bsm_process_information.bsm_process_info[process]['dimensionalities'])
            kinetic_energy_list = bsm_function.kinetic_energy_list(Te_max, Npoints)
            training_params.update({"initial_kinetic_energy_list": kinetic_energy_list})

            path1 = save_location + "/auxiliary/" + model_name + "/" + process + "/" + organizational_parameter_key + "_" + genint.generate_mass_string(mass, mode=mode) + "/" + process + "_AdaptiveMaps.npy"
            path2 = save_location + "/" + model_name + "/" + process + "/" + organizational_parameter_key + "_" + genint.generate_mass_string(mass, mode=mode) + "/" + process + "_AdaptiveMaps.npy"
            directory_name = save_location + "/" + model_name + "/" + process + "/" + organizational_parameter_key + "_" + genint.generate_mass_string(mass, mode=mode) + "/"
        
            if os.path.exists(path1) or os.path.exists(path2):
                print("Already finished this whole process, skipping")
                continue
            else:
                genint.make_integrators(training_params, model_name=model_name, process=process)
                genint.stitch_integrators(directory_name)
                genint.cleanup(directory_name)
                genint.organize_directories_final(directory_name)

    processing_params = {
        "process_targets": ["graphite", "lead", "iron", "aluminum","molybdenum"],
        "save_location": save_location,
        "organizational_list": organizational_list,
        "mode": mode,
    }
    genint.call_find_maxes(processing_params, model_name, processes_to_do)

#fine list of masses to use for a more detailed study
# Running this many masses takes a long time.
# Integrators for this scan available upon request: kjkelly@tamu.edu
mV_list_fine = np.geomspace(0.00120398, 0.200, 29)

if __name__ == "__main__":
    mainSM(path=DEFAULT_PATH+"/data_Te/")
    mainBSM(model_name="dark_vector", bsm_processes="all", Npoints=100, path=DEFAULT_PATH+"/data_Te/", organizational_list=default_mass_list, mode="MeV")
    mainBSM(model_name="mcp", bsm_processes="all", Npoints=100, path=DEFAULT_PATH+"/data_Te/", organizational_list=[0.003, 0.010, 0.030, 0.100], mode="MeV")