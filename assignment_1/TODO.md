# To Dos:

### Simulation backend:
- [ ] Implement equipartition, the normalization is (partially) wrong (v, acc, force?)
- [ ] Use computed force/potential for particle interaction i, j to set force/potential for BOTH particles
- [ ] Implement conservation of energy (or constant temperature? Not sure, I think first one is better) 
  -> scale velocities or accelerations with it
- [ ] Have a look at https://pycallgraph.readthedocs.io/en/master/ - not active, look for alternative
- [ ] Try 3D 
    1) write initial position functions 
    2) write plotter (use plotly for 3D plots)
    3) write mpl plots for projections (should be generalized for any dim, at least 2D and 3D)
    4) set n_dim to 3 and hit run()
    
- [ ] Implement variable time stepping (probably use adaptive RK45) scale with minimum timestep 
  (this sounds horribly slow)
- [ ] Define cut-off distances after which the interactions are not computed
- [ ] Define interpolant/lookup for LJ force/potential to speed up code
- [ ] numba all of this (not sure if this works with OOP)
- [ ] multithread (numba) all of this (not sure if this works with OOP)

### Expand Scope
- [ ] Implement other particle species
- [ ] Implement other forces/interactions

### Prettyfication: (most important)
- [ ] Implement live plotting using PyQtGraph
- [ ] Write nice progress bar functions