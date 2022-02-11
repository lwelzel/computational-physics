import numpy as np
import pandas as pd
from particle_class import Particle
from constants_class import Constants

def set_up_particles(n_particles=10, n_dim=2, n_steps=1000, particle_mass=1):
    '''

    :param n_particles:
    :param n_dim:
    :param n_steps:
    :return:
    '''
    # TODO: mass might be array
    # TODO: setup should accept external initial_pos, vel, acc

    # general
    shape = (n_particles, n_dim)

    # setup rng
    rng = np.random.default_rng()
    initial_particle_position = rng.uniform(low=0, high=1, size=shape)
    initial_particle_velociety = np.zeros(shape=shape) # rng.uniform(low=-1, high=1, size=(n_particles, n_dim))
    initial_particle_acceleration = np.zeros(shape=shape) # rng.uniform(low=-1, high=1, size=(n_particles, n_dim))

    # particle attrs: n_steps, n_dim, initial_pos, initial_vel, initial_acc, mass, name="",
    particles = np.array([Particle(n_steps=n_steps,
                                   n_dim=n_dim,
                                   initial_pos=initial_pos,
                                   initial_vel=inital_vel,
                                   initial_acc=initial_acc,
                                   mass=particle_mass)
                          for initial_pos, inital_vel, initial_acc
                          in zip(initial_particle_position,
                                 initial_particle_velociety,
                                 initial_particle_acceleration)
                          ], dtype=object)

    return particles

def save_particle_past():
    """
    Save objects? This seems really bad
    Save np arrays as zips: yes
        Save as (particle, step, parameter, dims)
        Save particle properties separately as (particle, props?)
    :return:
    """
    return

def normalize_md_problem(space, velocity, time,
                         energy, force, pressure, density, temperature,
                         sigma, eta):
    # TODO: think if normalization makes sense and what needs to be normalized
    # return _space, _velocity, _time,\
    #        _energy, _force, _pressure, _density, _temperature
    return



