""" This script generates VEGAS integrators for various production processes and stitch them together for different energies.
"""

import numpy as np
import sys, os
path = os.getcwd()
path = os.path.join(path,"../PETITE")
sys.path.insert(0,path)
from PETITE.all_processes import *
from PETITE.physical_constants import *
from glob import glob


from copy import deepcopy
from multiprocessing import Pool
import pickle
from functools import partial
import argparse
import datetime
import vegas

# helper function to turn float to string
def generate_vector_mass_string(mV):
    return str(int(np.floor(mV*1000.)))+"MeV"

def make_readme(params, process, process_directory):
    """Creates a readme file to accompany the integrators with supplementary information on the parameters used to generate the integrators (Z, A, mV, etc.)
    Input: 
        params: dictionary of parameters used in generating integrators
        process: string of process name
        file_info: directory where files should be saved
    """
    readme_file = open(process_directory + process + "_readme.txt", 'w')
    readme_file.write("Integrators for " + process)
    if process == 'ExactBrem':
        line = "\nTarget has (Z, A, mass) = ({atomic_Z}, {atomic_A}, {atomic_mass})\n".\
            format(atomic_Z=params['Z_T'], atomic_A=params['A_T'], atomic_mass=params['mT'])
        readme_file.write(line)
    readme_file.write("\n\nEnergy/GeV |  Filename\n\n")
    for index, energy in enumerate(params['initial_energy_list']):
        line = "{en:9.3f}  |  {proc}_{indx}.p\n".format(en = energy, indx = index, proc=process)
        readme_file.write(line)
    readme_file.write(f'\nIntegrators made on {datetime.datetime.now()}')
    readme_file.close()
    return()


def run_vegas_in_parallel(params, process, verbosity_mode, process_directory, energy_index):
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
        energy_index: index of energy in initial_energy_list
    '''
    params['E_inc'] = params['initial_energy_list'][energy_index]
    file_name = process_directory + process + '_' + str(energy_index) + ".p"
    if os.path.exists(file_name):
        print("Already generated integrator for this point\n")
    else:
        print('Starting VEGAS for energy index ',energy_index)
        VEGAS_integrator = vegas_integration(params, process, verbose=verbosity_mode, mode='Pickle') 
        #VEGAS_integrator = 0
        print('Done VEGAS for energy index ',energy_index)
        # Objects to be saved. Should include all important parameters (in params) and the VEGAS integrator adaptive map.
        params['process'] = process
        object_to_save = [params, VEGAS_integrator.map]
        pickle.dump(object_to_save, open(file_name, "wb"))
        print('File created: ' + file_name)
    return()



def make_integrators(params, process):
    """
    Generate vegas integrator pickles for a given process and set of parameters.
    Input:
        params: dictionary of parameters containing
            mV : dark vector mass in GeV
            A : target atomic mass number
            Z : target atomic number
            mT : target mass in GeV
        process: string of process name
    """
    mV = params['mV']
    atomic_A = params['A_T']
    atomic_Z = params['Z_T']
    mT = params['mT'] 
    verbosity_mode = params['verbosity']
    params['m_e'] = m_electron
    params['alpha_FS'] = alpha_em

    initial_energy_list = params['initial_energy_list']
    
    vec_mass_string = generate_vector_mass_string(mV)
    energy_index_list = range(len(initial_energy_list))

    print("Parameters:")
    print(params)
    print('Doing process: ', process)
    
    # energy_index_list = range(len(initial_energy_list))
    # vec_mass_string = generate_vector_mass_string(params['mV'])

    # Create process specific directory in mother directory for saving VEGAS adaptive maps
    process_directory = params['save_location'] + '/' + process + '/'
    
    # If directory does not exist, create it
    if not(os.path.exists(process_directory)):
        os.system("mkdir -p " + process_directory)
    # pool parallelizes the generation of integrators    
    pool = Pool()
    res = pool.map(partial(run_vegas_in_parallel, params, process, verbosity_mode, process_directory), energy_index_list)
    # make the human readable file contining info on params of run and put in directory 
    make_readme(params, process, process_directory)
    print('make_integrators is complete, readme files created in ' + process_directory + ' for convenience')


    return()

# Set up parameters for and then run find_maxes
def call_find_maxes(params, list_of_processes): 
    """Call find_maxes.py to find the maximum of the integrand for each energy.
    Note that find_maxes.py will save the maximum of the integrand and the corresponding energy to a pickle file.
    Input:
        params: dictionary of parameters used in generating integrators
        list_of_processes: vector containing all processes to be run
    """
    import find_maxes
    if (params['run_find_maxes']):
        # find_maxes_params['process'] = process
        # find_maxes_params['import_directory'] = params['save_location'] + "/" + process
        # find_maxes_params['save_location'] = params['find_maxes_save_location']
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
        print('Parameters used in find_maxes: ', find_maxes_params)
        find_maxes.main(find_maxes_params)
    else:
        print('Not running find_maxes')
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


# def make_readme(args): #FIXME: add right path, discuss and implement what exatly goes here.
#     """Writes a short readme file with the details of the pickles generated"""
#     f = open('README.md', 'w')
#     f.write('Parameters used for generating integrators on ', datetime.datetime.now(),' :')
#     f.write(args)
#     f.close()
#     return

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Produce VEGAS integrators for various production processes', formatter_class = argparse.ArgumentDefaultsHelpFormatter)    
    # mandatory parameters
    #None
    

    # optional parameters
    parser.add_argument('-A', type=float, default=12, help='atomic mass number')
    parser.add_argument('-Z', type=float, action='append', default=[6.0], help='atomic number of targets to save')
    parser.add_argument('-mT', type=float, default=11.178, help='nuclear target mass in GeV')


    parser.add_argument('-save_location', type=str, default='data/VEGAS_backend/SM/', help='directory to save integrators in (path relative to main PETITE directory)')
    parser.add_argument('-process', nargs='+', type=str, default=['ExactBrem'], help='list of processes to be run "all" does whole list, if mV non-zero only DarkBrem \
        (choose from "PairProd", "Brem", "ExactBrem", "Comp", "Ann", "Moller", "Bhabha")')
    parser.add_argument('-mV', nargs='+', type=float, default=[0.05], help='dark vector mass in GeV (can be a space-separated list)')
    parser.add_argument('-min_energy', type=float, default=0.01, help='minimum initial energy (in GeV) to evaluate (must be larger than mV)')
    parser.add_argument('-max_energy', type=float, default=100., help='maximum initial energy (in GeV) to evaluate')
    parser.add_argument('-num_energy_pts', type=int, default=100, help='number of initial energy values to evaluate, scan is done in log space')
    parser.add_argument('-run_find_maxes', type=bool, default=True,  help='run Find_Maxes.py after done')
    parser.add_argument('-verbosity', type=bool, default=False, help='verbosity mode')
    # stich integrators
    parser.add_argument('-stitch', type=bool, default=True, help='stitch integrators for different energies together')

    args = parser.parse_args()

    print('**** Arguments passed to generate_integrators ****')
    print(args)

    params = {'A_T': args.A, 'Z_T': np.unique(args.Z), 'mT': args.mT, 'save_location': args.save_location, 'run_find_maxes':args.run_find_maxes}
    verbosity_mode = args.verbosity
    params['verbosity'] = verbosity_mode
    if (args.mV == 0 or not(args.process == ['ExactBrem']) ):# doing SM processes
        if  "all" in args.process:
            process_list_to_do = ['Brem','PairProd','Comp','Ann','Moller','Bhabha']
        else:#make sure DarkBrem not accidentally in list
            try:
                process_list_to_do = args.process.remove('ExactBrem')
            except:
                process_list_to_do = args.process
                pass
        for process in process_list_to_do:
            initial_energy_list = np.logspace(np.log10(args.min_energy), np.log10(args.max_energy), args.num_energy_pts)
            params.update({'mV' : 0})
            params.update({'initial_energy_list': initial_energy_list})
            make_integrators(params, process)
            # stitch integrators for different energies together
            stitch_integrators('../' + params['save_location'] + "/" + process, '../' + params['save_location'] + "/" + process + '.npy')
        call_find_maxes(params, process_list_to_do)
    else:# doing DarkBrem
        for mV in args.mV:
            process = 'ExactBrem'
            print("Working on mV = ", mV)
            min_energy = max(args.min_energy, 1.01 * mV)
            initial_energy_list = np.logspace(np.log10(min_energy), np.log10(args.max_energy), args.num_energy_pts)
            params.update({'mV' : mV})
            params.update({'initial_energy_list': initial_energy_list})
            make_integrators(params, process)
            # stitch integrators for different energies together (add mV to the name of output file)
            stitch_integrators('../' + params['save_location'] + "/" + process, '../' + params['save_location'] + "/" + process + '_mV_' + str(int(np.floor(mV*1000.))) + 'MeV.npy')
            call_find_maxes(params, process)
            
    print("Goodbye!")


    

        
