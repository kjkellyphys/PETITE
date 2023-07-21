import numpy as np
import importlib
generate_integrators = importlib.import_module("generate_integrators")
import sys, os
path = os.getcwd()
path = os.path.split(path)[0]
print(path)

def main(doSM=True, doDark=True):
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
    if doSM:
        for process in processes_to_do:
            generate_integrators.make_integrators(training_params, process)
            generate_integrators.stitch_integrators(training_params['save_location'] + '/' + process + '/')
            generate_integrators.cleanup(training_params['save_location'] + '/' + process + '/')

        # List of processes to run find_maxes on, need not be the same as list above
        find_maxes_processes_to_do = ['Brem', 'PairProd', 'Comp', 'Ann', 'Moller', 'Bhabha']
        generate_integrators.call_find_maxes(processing_params, find_maxes_processes_to_do)

    ##################################
    ##         Dark processes       ##
    ##################################
    initial_energy_list = np.logspace(np.log10(0.01), np.log10(100), 100)
    # Dark vector masses in GeV
    #mV_list = [0.001, 0.003, 0.010, 0.030, 0.100, 0.300, 1.00]
    mV_list = [0.001, 0.010, 0.100]
    save_location = path + '/macpro_test/'
    training_params = {'verbosity':True, 'initial_energy_list':initial_energy_list,
                    'save_location':save_location,
                    'run_find_maxes':True, 'mV_list':mV_list, 'training_target':'hydrogen', 'mT':200.0}
    processes_to_do = ['DarkBrem', 'DarkAnn']

    if doDark:
        for mV in mV_list:
            for process in processes_to_do:
            #    process = 'DarkBrem'
                if mV > initial_energy_list[0]:
                    initial_energy_list = np.logspace(np.log10(1.01*mV), np.log10(initial_energy_list[-1]), len(initial_energy_list))
                    training_params.update({'initial_energy_list':initial_energy_list})
                training_params.update({"mV":mV})
                generate_integrators.make_integrators(training_params, process)
                generate_integrators.stitch_integrators(training_params['save_location'] + process + '/mV_' + str(int(np.floor(mV*1000.))) + "MeV/")
                generate_integrators.cleanup(training_params['save_location'] + process + '/mV_' + str(int(np.floor(mV*1000.))) + "MeV/")

        processing_params = {'process_targets':['graphite','lead'], 'save_location':save_location}
        generate_integrators.call_find_maxes(processing_params, processes_to_do)

if __name__ == "__main__":
    main(doSM=False, doDark=True)
