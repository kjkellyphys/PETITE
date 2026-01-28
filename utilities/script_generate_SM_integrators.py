import numpy as np
import importlib
import os

import PETITE.physical_constants as const
import generate_integrators as genint

DEFAULT_PATH = os.getcwd()
DEFAULT_PATH = os.path.split(DEFAULT_PATH)[0]
print("Default path:", DEFAULT_PATH)

def main(Te_min=0.0015, Te_max=100.0, Npoints=100, path=DEFAULT_PATH):
    ##################################
    ##         SM processes         ##
    ##################################
    # List of incoming particle energies at which to calculate integrators for each process
    # Necessary parameters for processing the integrators to determine cross sections
    training_params = {"save_location": path}
    processing_params = {'process_targets':['graphite','lead','iron','aluminum','molybdenum'], 'save_location':path}

    # List of processes to do
    processes_to_do = ['Comp', 'Ann', 'Moller', 'Bhabha', 'Brem', 'PairProd','MuonBrem', 'MuonE']

    for process in processes_to_do:
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
    find_maxes_processes_to_do = ['MuonBrem', 'MuonE','Brem', 'PairProd', 'Comp', 'Ann', 'Moller', 'Bhabha']
    genint.call_find_maxes(processing_params, find_maxes_processes_to_do)

#fine list of masses to use for a more detailed study
# Running this many masses takes a long time.
# Integrators for this scan available upon request: kjkelly@tamu.edu
mV_list_fine = np.geomspace(0.00120398, 0.200, 29)

if __name__ == "__main__":
    main(path=DEFAULT_PATH+"/data_Te/")
