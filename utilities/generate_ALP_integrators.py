import numpy as np
import importlib
generate_integrators = importlib.import_module("generate_integrators")
from PETITE.physical_constants import *
import sys, os
path = os.getcwd()
path = os.path.split(path)[0]
print(path)

def main(doSM=True, doDark=True):
    '''
    ##################################
    ##         SM processes         ##
    ##################################
    # List of incoming particle energies at which to calculate integrators for each process
    initial_energy_list = np.logspace(np.log10(0.0016), np.log10(100), 100)
    # Necessary parameters for generating the integrators, note save_location should be altered as preferred
    training_params = {'verbosity':True, 'initial_energy_list':initial_energy_list,
                    'save_location':path + '/data',
                    'run_find_maxes':True}
    # Necessary parameters for processing the integrators to determine cross sections
    processing_params = {'process_targets':['graphite','lead','iron','aluminum'], 'save_location':path + '/data'}
    #args = training_params.update(processing_params)
    # List of processes to do
    processes_to_do = ['Comp', 'Ann', 'Moller', 'Bhabha', 'Brem', 'PairProd']
    # Loop over processes, carrying out each step of the calculation, they can also be called in one command generate_integrators
    if doSM:
        for process in processes_to_do:
            if os.path.exists(training_params['save_location'] + '/auxiliary/' + process + "/" + process + "_AdaptiveMaps.npy"):
                print("Already finished this whole process, skipping")
                continue
            else:
                generate_integrators.make_integrators(training_params, process)
                generate_integrators.stitch_integrators(training_params['save_location'] + '/' + process + '/')
                generate_integrators.cleanup(training_params['save_location'] + "/" + process + "/")
        # List of processes to run find_maxes on, need not be the same as list above
        find_maxes_processes_to_do = ['Brem', 'PairProd', 'Comp', 'Ann', 'Moller', 'Bhabha']
        generate_integrators.call_find_maxes(processing_params, find_maxes_processes_to_do)
    '''

    ##################################
    ##         Dark processes       ##
    ##################################
    initial_energy_list_general = np.logspace(np.log10(0.0016), np.log10(100), 100)
    # Dark vector masses in GeV
    mA_list = [0.003, 0.010, 0.030, 0.100, 0.300, 1.000]
    #mV_list = np.logspace(np.log10(0.003), np.log10(0.200), 24)
    save_location = path + '/ALPdata/'
    training_params = {'verbosity':True, 'initial_energy_list':initial_energy_list_general,
                    'save_location':save_location,
                    'run_find_maxes':True, 'mV_list':mA_list, 'training_target':'hydrogen', 'mT':200.0}
    processes_to_do = ['DarkALPBrem']

    if doDark:
        for mV in mA_list:
            for process in processes_to_do:
                energy_list = initial_energy_list_general
                if process == 'DarkALPBrem':
                    #if mV > energy_list[0]:
                    energy_list = np.logspace(np.log10(1.25*mV), np.log10(energy_list[-1]), len(energy_list))
                training_params.update({'initial_energy_list':energy_list})
                training_params.update({"mV":mV})
                if os.path.exists(training_params['save_location'] + '/auxiliary/' + process + '/mA_' + str(int(np.floor(mV*1000.))) + "MeV/" + process + "_AdaptiveMaps.npy"):
                    print("Already finished this whole process, skipping")
                    continue
                else:
                    generate_integrators.make_integrators(training_params, process)
                    generate_integrators.stitch_integrators(training_params['save_location'] + process + '/mA_' + str(int(np.floor(mV*1000.))) + "MeV/")
                    generate_integrators.cleanup(training_params['save_location'] + process + '/mA_' + str(int(np.floor(mV*1000.))) + "MeV/")
                    generate_integrators.organize_directories_final(training_params['save_location'] + process + '/mA_' + str(int(np.floor(mV*1000.))) + "MeV/")
                    #generate_integrators.stitch_integrators(training_params['save_location'] + process + '/mV_' + str(int(np.floor(mV*1000000.))) + "keV/")
                    #generate_integrators.cleanup(training_params['save_location'] + process + '/mV_' + str(int(np.floor(mV*1000000.))) + "keV/")

        processing_params = {'process_targets':['graphite','lead','iron','aluminum'], 'save_location':save_location, 'mV_list':mA_list}
        generate_integrators.call_find_maxes(processing_params, processes_to_do)

if __name__ == "__main__":
    main(doSM=False, doDark=True)
