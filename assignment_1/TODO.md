# Overview

# Read Me:

## Use:
### Preparation
1. Set up environment using requirements.txt and environment.yml files
2. Verify that the correct modules and dependencies are installed
3. Verify the correct environment setup (python version, hardware and OS compatibility)
4. Activate the environment
### Running the Simulation
1. Navigate to ----
2. Select or input the correct run parameters
3. Run ---- in the console
   1. This may take a while
   2. Progress should be shown
   3. The program will save progress intermediately
   4. get a coffee
4. Once done the console will show a run summary
### Analyzing the Simulation Result
1. Verify that the data has been saved properly
2. Run desired plotting functions
3. Look at the plots
   1. Cool plots right?

---
---

## Authors & Copyleft




## To Dos:
### ESSENTIAL
https://en.wikibooks.org/wiki/Molecular_Simulation/Radial_Distribution_Functions
- [ ] Make requirements.txt file (https://mariashaukat.medium.com/a-guide-to-conda-environments-in-jupyter-pycharm-and-github-5ba3833d859a)
- [ ] Make environment.yml file
- [ ]
- [ ]
- [ ]

### Simulation backend:
- [ ] Implement equipartition, the normalization is (partially) wrong (v, acc, force?)
- [ ] Use computed force/potential for particle interaction i, j to set force/potential for BOTH particles
- [ ] Implement conservation of energy (or constant temperature? Not sure, I think first one is better) 
  -> scale velocities or accelerations with it
- [ ] Have a look at https://pycallgraph.readthedocs.io/en/master/ - not active, look for alternative
- [ ] Track and plot stat variables (temperature, energy etc) (maybe also for you if you have the time)
- [ ] Try 3D 
    1) write initial position functions 
    2) write plotter (use plotly for 3D plots)
    3) write mpl plots for projections (should be generalized for any dim, at least 2D and 3D)
    4) set n_dim to 3 and hit run()
    
- [ ] Implement variable time stepping (probably use adaptive RK45) scale with minimum timestep 
  (this sounds horribly slow) - Symplectic integrator 
- [ ] Define cut-off distances after which the interactions are not computed
- [ ] Define interpolant/lookup for LJ force/potential to speed up code- [ ]
- [ ] Energy plotting functions
- [ ] Stats plotting functions
- [ ] numba all of this (not sure if this works with OOP)
- [ ] multithread (numba) all of this (not sure if this works with OOP)

### Expand Scope
- [ ] Implement other particle species
- [ ] Implement other forces/interactions

### Prettyfication: (most important)
- [ ] Implement live plotting using PyQtGraph
- [ ] Write nice progress bar functions