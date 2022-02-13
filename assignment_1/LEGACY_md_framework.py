import numpy as np
import pandas as pd
from pathlib import Path
from time import time, strftime, gmtime
from tempfile import mkdtemp
from argon_class import Argon
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
    initial_particle_velocity = np.zeros(shape=shape) # rng.uniform(low=-1, high=1, size=(n_particles, n_dim))
    initial_particle_force = np.zeros(shape=shape) # rng.uniform(low=-1, high=1, size=(n_particles, n_dim))

    # particle attrs: n_steps, n_dim, initial_pos, initial_vel, initial_force, **kwargs
    [Argon(n_steps=n_steps,
           n_dim=n_dim,
           initial_pos=initial_pos,
           initial_vel=initial_vel,
           initial_force=initial_force)
     for initial_pos, initial_vel, initial_force
     in zip(initial_particle_position,
            initial_particle_velocity,
            initial_particle_force)
     ]
    assert Argon.__n_active__ == n_particles, f"The number of particles that were generated is wrong.\n" \
                                              f"Particles to be generated:      {n_particles:.0f}\n" \
                                              f"Particles that were generated:  {Argon.__n_active__:.0f}\n" \
                                              f"Oops..."


def save_particle_past(object_array, loc="", name="MD_simulation" + strftime("%Y-%m-%d %H:%M:%S", gmtime())):
    """
    Store particle positions like:
        (particle, step, dims)

    Save objects? This seems really bad
    Save np arrays as zips: yes
        Save as (particle, step, parameter, dims)
        Save particle properties separately as (particle, props?)
    :return:
    """
    loc = Path(loc)

    # make memory mapped file to reduce extra memory cost during saving
    n_particles = len(object_array)
    n_particle_shape = object_array[0].pos.shape
    mmapfilename = Path(mkdtemp()).parent.joinpath('tempmmapfile.dat')
    storage_array = np.memmap(mmapfilename, dtype='float64', mode='w+', shape=(n_particles, *n_particle_shape))

    # using non mmapped arrays
    # n_particles = len(object_array)
    # n_particle_shape = object_array[0].pos.shape
    # storage_array = np.zeros(shape=(n_particles, *n_particle_shape))

    for i, particle in enumerate(object_array):
        storage_array[i, :] = particle.pos[:]

    np.save(loc / name, particle_positions=storage_array)



