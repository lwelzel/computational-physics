# Overview
This project simulates Argon Molecular Dynamics

# Read Me:
## Use:
TLDR:
1. Do the normal stuff you would do for a project, .yml files are in the same folder as the readme
2. I recommend to run md_simulation_func.py for a test run.
   1. Plotting might throw some errors depending if you have run enough simulations to extract statistical information
   2. But then you can always run many_sim_wrapper.py if you want to melt some cpus and simulate many runs


### Preparation
1. Set up environment using requirements.txt and environment.yml files. This might be multiple files.
2. Verify that the correct modules and dependencies are installed
3. Verify the correct environment setup (python version, hardware and OS compatibility)
4. Activate the environment
### Running the Simulation
1. Navigate to \computational-physics\assignment_1\
2. Select file to run or input the correct run parameters
   1. many_sim_wrapper.py can be used to run many simulations simultaneously. 
      1. Be careful with this as it will utilize 90% of your cores
      2. Beware that this might cause flashing lights due to progress bars depending on your terminal and OS
      3. At the moment progress print statements are disabled, so it might be that the program is still working when there are no outputs
      4. Generally, the demons will try to finish their work (at least save at next savepoint) before allowing you to stop
   2. md_simulation_func.py can be used to run a single simulation. Enable progress print statements in the class MolDyn to see what the simulation is doing
3. Run chosen file or method in the console
   1. This may take a while. On my PC:
      1. 100 particles will take seconds to minutes depending on the initial state and number of steps
      2. 1000 particles will take minutes. The program scales between O(n) to O(n(n/2)) interactions per timestep for n particles
      3. running 36 * 3 simulations with 108 particles in parallel for 2500 steps (5. time units) took 10ish minutes. Most of that was spent on relaxing gases
   2. Progress should be shown
   3. The program will save progress intermediately, especially for long runs
      1. The program will not save un-relaxed states
   4. get a coffee
4. Once done the console will show a run summary if that option is enabled
   1. This output will indicate if the simulation was shut down nominally
   2. It will also indicate if any time-overruns were detected.
### Analyzing the Simulation Result
1. Verify that the data has been saved properly
2. Run desired plotting functions
3. Look at the plots
   1. Cool plots right?

---
---

## Authors & Copyleft
See project overview for authors. Feel free to use this but I am already sorry if you need to.

## To Dos:
### ESSENTIAL
https://en.wikibooks.org/wiki/Molecular_Simulation/Radial_Distribution_Functions
- [x] Make requirements.txt file (https://mariashaukat.medium.com/a-guide-to-conda-environments-in-jupyter-pycharm-and-github-5ba3833d859a)
- [x] Make environment.yml file


### Simulation backend:
- [x] Implement equipartition, the normalization is (partially) wrong (v, acc, force?)
- [x] Use computed force/potential for particle interaction i, j to set force/potential for BOTH particles
- [x] Implement conservation of energy (or constant temperature? Not sure, I think first one is better) 
  -> scale velocities or accelerations with it
- [ ] Have a look at https://pycallgraph.readthedocs.io/en/master/ - not active, look for alternative
- [ ] Track and plot stat variables (temperature, energy etc) (maybe also for you if you have the time)
- [x] Try 3D 
    1) write initial position functions 
    2) write plotter (use plotly for 3D plots)
    3) write mpl plots for projections (should be generalized for any dim, at least 2D and 3D)
    4) set n_dim to 3 and hit run()
    
- [ ] Implement variable time stepping (probably use adaptive RK45) scale with minimum timestep 
  (this sounds horribly slow) - Symplectic integrator 
- [x] Define cut-off distances after which the interactions are not computed
- [x] Define interpolant/lookup for LJ force/potential to speed up code- [ ]
- [x] Energy plotting functions
- [x] Stats plotting functions
- [ ] numba all of this (not sure if this works with OOP)
- [x] multithread (numba) all of this (not sure if this works with OOP)

### Expand Scope
- [ ] Implement other particle species
- [ ] Implement other forces/interactions

### Prettyfication: (most important)
- [ ] Implement live plotting using PyQtGraph
- [ ] Write nice progress bar functions