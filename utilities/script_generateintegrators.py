import numpy as np
import importlib
import os

import PETITE.physical_constants as const
import generate_integrators as genint

DEFAULT_PATH = os.getcwd()
DEFAULT_PATH = os.path.split(DEFAULT_PATH)[0]
print("Default path:", DEFAULT_PATH)

default_mV_list = [0.003, 0.010, 0.030, 0.100, 0.300, 1.000]
def main(doSM=True, doDark=True, Ee_min=0.0016, Ee_max=100.0, Npoints=100, path=DEFAULT_PATH, mode="MeV", mV_list=default_mV_list):
    ##################################
    ##         SM processes         ##
    ##################################
    # List of incoming particle energies at which to calculate integrators for each process
    initial_energy_list = np.geomspace(Ee_min, Ee_max, Npoints)
    # Necessary parameters for generating the integrators, note save_location should be altered as preferred
    training_params = {'verbosity':True, 'initial_energy_list':initial_energy_list,
                    'save_location':path,
                    'run_find_maxes':True}
    # Necessary parameters for processing the integrators to determine cross sections
    processing_params = {'process_targets':['graphite','lead','iron','aluminum','molybdenum'], 'save_location':path}
    #args = training_params.update(processing_params)
    # List of processes to do
    processes_to_do = ['MuonBrem', 'MuonE','Comp', 'Ann', 'Moller', 'Bhabha', 'Brem', 'PairProd']
    # Loop over processes, carrying out each step of the calculation, they can also be called in one command generate_integrators
    if doSM:
        for process in processes_to_do:
            path1 = training_params['save_location'] + '/auxiliary/' + process + "/" + process + "_AdaptiveMaps.npy"
            path2 = training_params['save_location'] + '/' + process + '/' + process + "_AdaptiveMaps.npy"
            if os.path.exists(path1) or os.path.exists(path2):
                print("Already finished this whole process, skipping")
                continue
            else:
                if process == "MuonBrem" or process == "MuonE":
                    energy_list_new = np.geomspace(0.200, Ee_max, Npoints)
                    training_params.update({"initial_energy_list": energy_list_new})
                genint.make_integrators(training_params, process)
                genint.stitch_integrators(training_params['save_location'] + '/' + process + '/')
                genint.cleanup(training_params['save_location'] + "/" + process + "/")
        # List of processes to run find_maxes on, need not be the same as list above
        find_maxes_processes_to_do = ['MuonBrem', 'MuonE','Brem', 'PairProd', 'Comp', 'Ann', 'Moller', 'Bhabha']
        genint.call_find_maxes(processing_params, find_maxes_processes_to_do)

    ##################################
    ##         Dark processes       ##
    ##################################
    initial_energy_list_general = np.geomspace(Ee_min, Ee_max, Npoints)
    save_location = path
    training_params = {'verbosity':True, 'initial_energy_list':initial_energy_list_general,
                    'save_location':save_location,
                    'run_find_maxes':True, 'mV_list':mV_list, 'training_target':'hydrogen', 'mT':200.0}
    processes_to_do = ['DarkAnn', 'DarkComp', 'DarkBrem', 'DarkMuonBrem']
    if doDark:
        for mV in mV_list:
            for process in processes_to_do:
                energy_list = initial_energy_list_general
                if process == "DarkBrem":
                    # if mV > energy_list[0]:
                    energy_list = np.geomspace(
                        1.25 * mV, energy_list[-1], len(energy_list)
                    )
                if process == "DarkAnn":
                    ER0 = (mV**2 - 2 * const.m_electron**2) / (2 * const.m_electron)
                    Emax = np.max([Ee_max, 100 * ER0])
                    energy_list = ER0 * (
                        1 + np.geomspace(1e-4, (Emax - ER0) / ER0, len(energy_list))
                    )
                if process == "DarkComp":
                    Eg0 = mV * (1 + mV / (2 * const.m_electron))
                    Emax = np.max([Ee_max, 100 * Eg0])
                    energy_list = Eg0 * (
                        1 + np.geomspace(1e-4, (Emax - Eg0) / Eg0, len(energy_list))
                    )
                if process == "DarkMuonBrem":
                    energy_list = np.geomspace(
                        1.25 * (mV + const.m_muon), energy_list[-1], len(energy_list)
                    )
                training_params.update({"initial_energy_list": energy_list})
                training_params.update({"mV": mV})
                if mode == "MeV":
                    path1 = training_params["save_location"] + "/auxiliary/" + process + "/mV_" + str(int(np.floor(mV * 1000.0))) + "MeV/" + process + "_AdaptiveMaps.npy"
                    path2 = training_params["save_location"] + "/" + process + "/mV_" + str(int(np.floor(mV * 1000.0))) + "MeV/" + process + "_AdaptiveMaps.npy"
                    directory_name = training_params["save_location"] + process + "/mV_" + str(int(np.floor(mV * 1000.0))) + "MeV/"
                    training_params.update({"mode": "MeV"})
                elif mode == "keV":
                    path1 = training_params["save_location"] + "/auxiliary/" + process + "/mV_" + str(int(np.floor(mV * 1000000.0))) + "keV/" + process + "_AdaptiveMaps.npy"
                    path2 = training_params["save_location"] + "/" + process + "/mV_" + str(int(np.floor(mV * 1000000.0))) + "keV/" + process + "_AdaptiveMaps.npy"
                    directory_name = training_params["save_location"] + process + "/mV_" + str(int(np.floor(mV * 1000000.0))) + "keV/"
                    training_params.update({"mode": "keV"})
                else:
                    raise ValueError("Mode must be either 'MeV' or 'keV'")
                if os.path.exists(path1) or os.path.exists(path2):
                    print("Already finished this whole process, skipping")
                    continue
                else:
                    genint.make_integrators(training_params, process)
                    genint.stitch_integrators(directory_name)
                    genint.cleanup(directory_name)
                    genint.organize_directories_final(directory_name)

        processing_params = {
            "process_targets": ["graphite", "lead", "iron", "aluminum","molybdenum"],
            "save_location": save_location,
            "mV_list": mV_list,
            "mode": mode,
        }
        genint.call_find_maxes(processing_params, processes_to_do)

#fine list of masses to use for a more detailed study
# Running this many masses takes a long time.
# Integrators for this scan available upon request: kjkelly@tamu.edu
mV_list_fine = np.geomspace(0.00120398, 0.200, 29)

if __name__ == "__main__":
    main(doSM=True, doDark=True, path=DEFAULT_PATH+"/data/")
    main(doSM=True, doDark=True, path=DEFAULT_PATH+"/data_400GeV/", Ee_max=400.0, Npoints=150)
    #main(doSM=True, doDark=True, path=DEFAULT_PATH+"/data_400GeV_manymasses/", Ee_max=400.0, Npoints=150, mode='keV', mV_list=mV_list_fine)
