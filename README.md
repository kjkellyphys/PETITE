# PETITE
PETITE: Package for Electromagnetic Transitions In Thick-target Environments
Monte Carlo generator for production of dark sector objects in thick-target experiments.
PETITE generates electromagnetic showers for incoming electrons, positrons, or photons propagating through a dense medium, and includes the possibility of dark sector particle production.

## Installation
To install, from the top directory run
 > pip install .

### Dependencies
PETITE, its tutorials and tools require the following packages: numpy, vegas (>= 5.4.2), scipy, matplotlib, tqdm. Using `pip install .` should install all requirements, but if needed you can manually install them with
 > pip install <package_name>

## Running PETITE
*The notebook `./examples/tutorial.ipynb` offers an alternative to this readme.*

PETITE comes with pre-generated VEGAS integrator files stored in the `./data_Te/` directory.
Key classes (in `./src/PETITE/shower.py` and `./src/PETITE/bsm_shower.py`):
- `Shower`: imports the VEGAS integrators and other infrastructure for SM electromagnetic processes in the relevant material.
- `BSMShower`: extends `Shower` to reprocess SM shower particles and assign production weights for dark sector particles.
- `generate_shower`: generates an SM shower for a given initial `Particle`.
- `generate_bsm_shower`: reprocesses an SM shower to generate dark sector particles.

### Generating a full SM shower
(1) Define the initial particle that seeds the shower, e.g. an electron with 4-momentum [E0, px, py, pz] entering at position [0, 0, 0]:
 > incoming_electron = Particle([E0, px, py, pz], [0, 0, 0], {"PID": 11})

(2) Set up the shower object with `Shower(<data directory>, <material>, <min kinetic energy in GeV>)`, e.g.
 > sGraphite = Shower("./data_Te/", "graphite", 0.010)

(3) Generate the shower
 > standard_shower = sGraphite.generate_shower(incoming_electron)

The output of `generate_shower` is a list of `Particle` objects produced during shower development.

### Generating a BSM shower
(1) As for the SM shower, define the initial particle.

(2) Set up the BSM shower object with `BSMShower(<data directory>, <material>, <min energy in GeV>, model_name=<model>, model_keys=<mass parameters>)`, e.g. for a dark vector with mass 50 MeV:
 > sGraphite = BSMShower("./data_Te/", "graphite", 0.010, model_name="dark_vector", model_keys={"mV": 0.050})

Available models are `"dark_vector"` and `"mcp"`. The `model_keys` dictionary specifies the BSM particle mass: `{"mV": <mass in GeV>}` for dark vectors, `{"m_chi": <mass in GeV>}` for MCPs.

(3) Generate the BSM shower (this also generates the underlying SM shower internally)
 > sm_shower, bsm_shower = sGraphite.generate_bsm_shower(SParams=incoming_electron)

The first return value is the SM shower; the second is the list of BSM `Particle` objects with associated production weights.

We can plot event displays for the shower with
 > event_display(shower_object)

### The `Particle` object
Particles are stored in PETITE as `Particle` objects (see `./src/PETITE/particle.py`). The `Particle` class has the following attributes:
- `p0`: 4-momentum of the particle at the start of the shower in the convention `[E, px, py, pz]` (GeV). If given a scalar, it is assumed to be the energy of a particle propagating along the z-axis.
- `r0`: 3-position of the particle at the start of the shower.
- `id_dictionary`: dictionary of particle properties, including
    - `PID`: PDG particle ID number
    - `ID`: ID for shower development (unique for each particle)
    - `parent_ID`: shower ID of parent particle
    - `generation_number`: number of splittings from the initial particle
    - `mass`: particle mass in GeV
    - `generation_process`: process that generated the particle
    - `weight`: particle weight (used for BSM shower reweighting)
    - `stability`: whether the particle is `"stable"`, `"short-lived"`, or `"long-lived"`

These attributes can be used for analyzing the shower; see `./examples/tutorial.ipynb` for examples.

*This concludes the minimum information needed to run PETITE. The following sections are intended for advanced users.*

# Advanced usage
The following is intended for advanced users who wish to generate their own VEGAS integrators for a given material and/or dark sector model, to understand better the VEGAS sampling procedure, or to understand the file structure of the integrators.

## Generating new VEGAS integrators
It is possible to generate new files for both SM and dark processes using tools in the `./utilities/` directory. Jupyter notebook examples of how to run PETITE using pre-generated files are given in the `./examples/` directory.

## Structure of pre-generated VEGAS adaptive maps
All pre-generated data files live in `./data_Te/`.

### Standard model files
`sm_maps.pkl` is a dictionary of dictionaries. The first layer of keys are the SM processes:
`PairProd`, `Comp`, `Ann`, `Brem`, `Bhabha`, `Moller`, `MuonBrem`, `MuonE`.
For each process there are several entries corresponding to different incoming kinetic energies. Each entry is a pair `(T_inc, dict)` where the dictionary contains the VEGAS parameters and adaptive map: `{'neval', 'max_F', 'adaptive_map'}`.

Cross section tables used to select the interaction process at each shower step are stored in `sm_xsec.pkl`. It is a dictionary keyed first by process name, then by material, with values being arrays of `(kinetic energy, cross section)` pairs.

### BSM model files
Each BSM model has its own maps and cross section files:
- Dark vector: `dark_vector_maps.pkl`, `dark_vector_xsec.pkl`
- MCP: `mcp_maps.pkl`, `mcp_xsec.pkl`

The structure mirrors the SM files, with an additional top-level key for the BSM particle mass (e.g. `mV = 0.050` GeV). The BSM processes are `DarkBrem`, `DarkComp`, `DarkAnn`, `DarkMuonBrem` (dark vector) and `mcpAnnihilation` (MCP).
