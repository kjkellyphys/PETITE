import numpy as np
import importlib
generate_integrators = importlib.import_module("generate_integrators")

def main():
    initial_energy_list = np.logspace(-2.0, 2.0, 100)
    training_params = {"verbosity":True,
                    "initial_energy_list":initial_energy_list,
                    "save_location":"/Users/kjkelly/Dropbox/GitHub/PETITE/macpro_test"}

    processes_to_do = ['Brem', 'PairProd', 'Comp', 'Ann', 'Moller', 'Bhabha']
    for process in processes_to_do:
        generate_integrators.make_integrators(training_params, process)
        generate_integrators.stitch_integrators(training_params['save_location'] + '/' + process + '/')

    processing_params = {'process_targets':['graphite','lead'], 'save_location':"/Users/kjkelly/Dropbox/GitHub/PETITE/macpro_test"}
    generate_integrators.call_find_maxes(processing_params, processes_to_do)

if __name__ == "__main__":
    main()