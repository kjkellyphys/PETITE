import numpy as np
import importlib
generate_integrators = importlib.import_module("generate_integrators")
import sys, os
path = os.getcwd()
path = os.path.split(path)[0]
print(path)

def main():
    ##################################
    ##         SM processes         ##
    ##################################
    # List of incoming particle energies at which to calculate integrators for each process
    initial_energy_list = np.logspace(-2.0, 2.0, 100)
    # Necessary parameters for generating the integrators, note save_location should be altered as preferred
    training_params = {"verbosity":True,
                    "initial_energy_list":initial_energy_list,
                    "save_location":path + '/user_test',}

    # Necessary parameters for processing the integrators to determine cross sections
    processing_params = {'process_targets':['graphite','lead'], 'save_location':"/Users/kjkelly/Dropbox/GitHub/PETITE/macpro_test"}
    # List of processes to do
    processes_to_do = ['Brem', 'PairProd', 'Comp', 'Ann', 'Moller', 'Bhabha']
    # Loop over processes, carrying out each step of the calculation, they can also be called in one command generate_integrators
    for process in processes_to_do:
        generate_integrators.make_integrators(training_params, process)
        generate_integrators.stitch_integrators(training_params['save_location'] + '/' + process + '/')
        generate_integrators.cleanup(training_params['save_location'] + '/' + process + '/')

    # List of processes to run find_maxes on, need not be the same as list above
    find_maxes_processes_to_do = ['Brem', 'PairProd', 'Comp', 'Ann', 'Moller', 'Bhabha']
    generate_integrators.call_find_maxes(processing_params, find_maxes_processes_to_do)

if __name__ == "__main__":
    main()