import numpy as np
import importlib
generate_integrators = importlib.import_module("generate_integrators")
from PETITE.physical_constants import *
import sys, os
path = os.getcwd()
path = os.path.split(path)[0]
print(path)

def main(doSM=True, doDark=True):
    ##################################
    ##         SM processes         ##
    ##################################
    # List of incoming particle energies at which to calculate integrators for each process
    initial_energy_list = np.logspace(np.log10(0.0016), np.log10(100), 100)
    # Necessary parameters for generating the integrators, note save_location should be altered as preferred
    training_params = {'verbosity':True, 'initial_energy_list':initial_energy_list,
                    'save_location':path + '/macpro_test',
                    'run_find_maxes':True}
    # Necessary parameters for processing the integrators to determine cross sections
    processing_params = {'process_targets':['graphite','lead','iron'], 'save_location':path + '/macpro_test'}
    #args = training_params.update(processing_params)
    # List of processes to do
    processes_to_do = ['Comp', 'Ann', 'Moller', 'Bhabha', 'Brem', 'PairProd']
    # Loop over processes, carrying out each step of the calculation, they can also be called in one command generate_integrators
    if doSM:
        #for process in processes_to_do:
        #    generate_integrators.make_integrators(training_params, process)
        #    generate_integrators.stitch_integrators(training_params['save_location'] + '/' + process + '/')
        #    generate_integrators.cleanup(training_params['save_location'] + "/" + process + "/")
        # List of processes to run find_maxes on, need not be the same as list above
        find_maxes_processes_to_do = ['Brem', 'PairProd', 'Comp', 'Ann', 'Moller', 'Bhabha']
        generate_integrators.call_find_maxes(processing_params, find_maxes_processes_to_do)

    ##################################
    ##         Dark processes       ##
    ##################################
    initial_energy_list_general = np.logspace(np.log10(0.0016), np.log10(100), 100)
    # Dark vector masses in GeV
    mV_list = [0.003, 0.010, 0.030, 0.100, 0.300, 1.000]
    save_location = path + '/macpro2_test/'
    training_params = {'verbosity':True, 'initial_energy_list':initial_energy_list_general,
                    'save_location':save_location,
                    'run_find_maxes':True, 'mV_list':mV_list, 'training_target':'hydrogen', 'mT':200.0}
    processes_to_do = ['DarkAnn', 'DarkComp', 'DarkBrem']

    if doDark:
        for mV in mV_list:
            for process in processes_to_do:
                energy_list = initial_energy_list_general
                if process == 'DarkBrem':
                    if mV > energy_list[0]:
                        energy_list = np.logspace(np.log10(1.2*mV), np.log10(energy_list[-1]), len(energy_list))
                if process == 'DarkAnn':
                    ER0 = ((mV**2 - 2*m_electron**2)/(2*m_electron))
                    Emax = np.max([100.0, 100*ER0])
                    energy_list = ER0*(1 + np.logspace(-4, np.log10((Emax - ER0)/ER0), len(energy_list)))
                    #Ee0 = (1 + 1e-3)*((mV**2 - 2*m_electron**2)/(2*m_electron))
                    #if 100*Ee0 > initial_energy_list[-1]:
                    #    Emax = 100*Ee0
                    #else:
                    #    Emax = initial_energy_list[-1]
                    #energy_list = np.logspace(np.log10(Ee0), np.log10(Emax), len(energy_list))
                    #if mV**2 > 2*m_electron*(energy_list[0] + m_electron):
                    #    Ee0 = 1.05*((mV**2 - 2*m_electron**2)/(2*m_electron))
                    #    energy_list = np.logspace(np.log10(Ee0), np.log10(energy_list[-1]), len(energy_list))
                if process == 'DarkComp':
                    if energy_list[0] < mV*(1 + mV/(2*m_electron)):
                        Eg0 = 1.05*(mV*(1 + mV/(2*m_electron)))
                        energy_list = np.logspace(np.log10(Eg0), np.log10(energy_list[-1]), len(energy_list))
                training_params.update({'initial_energy_list':energy_list})
                training_params.update({"mV":mV})
                if os.path.exists(training_params['save_location'] + process + '/mV_' + str(int(np.floor(mV*1000.))) + "MeV/" + process + "_AdaptiveMaps.npy"):
                    print("Already finished this whole process, skipping")
                    continue
                else:
                    generate_integrators.make_integrators(training_params, process)
                    generate_integrators.stitch_integrators(training_params['save_location'] + process + '/mV_' + str(int(np.floor(mV*1000.))) + "MeV/")
                    generate_integrators.cleanup(training_params['save_location'] + process + '/mV_' + str(int(np.floor(mV*1000.))) + "MeV/")

        processing_params = {'process_targets':['graphite','lead','iron'], 'save_location':save_location, 'mV_list':mV_list}
        generate_integrators.call_find_maxes(processing_params, processes_to_do)

if __name__ == "__main__":
    main(doSM=False, doDark=True)
