The utilities directory contains several helpful scripts that are used to generate the necessary integrators and look-up dictionaries used by PETITE for shower evolution. 


# generate_integrators.py  
This script generates the VEGAS integrators for all processes that can produce a dark vector in a shower, over a range of energies. 
It takes both command line options, a full list can be seen by "python generate_integrators.py -h" and can read a run card for various VEGAS options.
e.g.

python generate_integrators.py -A=12 -Z=6 -mT=12 -process='DarkBrem' -mV=0.050 -num_energy_pts=10 -min_energy=0.1 -max_energy=100

# Find_Maxes.py
This processes the previously generated integrators into a format that is most useful for PETITE, it is automatically called by generate_integrators unless 