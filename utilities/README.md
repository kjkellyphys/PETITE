## Utilities manual
The utilities directory contains several helpful scripts that are used to generate the necessary integrators and look-up dictionaries used by PETITE for shower evolution. 
There is a jupyter notebooks in the utilities directory that contains some examples of how to use these scripts.
"using_utilities.ipynb" generates integrators and look-up dictionaries for standard model processes, as well as dark sector processes. 
These dictionaries are needed for shower generation.
The user can choose the process or processes, the grid of incoming particle energies to use, and the dark vector mass.

In the following, we will describe the two main scripts in the utilities directory, generate_integrators.py and find_maxes.py, and how to use them.

# generate_integrators.py  
This script generates the VEGAS integrators for all processes involved in a shower, over a range of energies. 
It takes both command line options, a full list can be seen by "python generate_integrators.py -h". 
For example,

python generate_integrators.py -A=12 -Z=6 -mT=12 -process 'Comp' 'Ann' -mV 0.05 1 -num_energy_pts=10 -min_energy=0.1 -max_energy=100

Most of the functions in generator_integrators.py get `params` as an input, which is a dictionary containing all the relevant parameters for the simulation: the atomic number, the atomic mass, the dark vector mass, the mass of the target, etc.

Relevant functions:
 - make_readme: creates a readme file for the integrators with information on the process, the grid of energies, and the dark vector mass.
 - run_vegas_in_parallel: runs vegas in parallel for a given process, and `params`. It is called by make_integrator. Saves the integrators (adaptive maps only) as pkl files in the directory specified by `params['import_directory']`.
 - make_integrator: calls run_vegas_in_parallel and make_readme for a given process, and `params`.
 - call_find_maxes: calls find_maxes.py for a given process, and `params`. This creates preliminary look-up dictionary used by PETITE.
 - stitch_integrators: stitches together the integrators created by find_maxes for a given process, and `params`. This creates the final look-up dictionary used by PETITE.
 - cleanup: removes the preliminary adaptive maps created by find_maxes for a given process, and `params`.
 - main: the main function that is called when generate_integrators.py is run. It loops over all processes and calls make_integrator, call_find_maxes, stitch_integrators, and cleanup for each process.


# Find_Maxes.py
This processes the previously generated integrators into a format that is most useful for PETITE, it is automatically called by generate_integrators unless generate_integrators is called with flag "-run_find_maxes=False".

It can also be run independently as e.g.
python find_maxes.py -A=12 -Z=6 -mT=12 -process='Comp' -import_directory='raw_integrators/Comp'