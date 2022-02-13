import numpy as np
import pandas as pd
from pathlib import Path
from time import time, strftime, gmtime
from tempfile import mkdtemp
from tqdm import tqdm
from argon_class import Argon
from constants_class import Constants

def set_up_simulation(n_particles, n_dim, n_steps,
                      timestep, box_length):
    '''

    :param n_particles:
    :param n_dim:
    :param n_steps:
    :return:
    '''
    # TODO: mass might be array
    # TODO: setup should accept external initial_pos, vel, acc

    # META
    # we take half box_length since our box extends +/- box_length / 2
    Argon.update_meta_class(__instances__=True,
                            __occupation__=True,
                            max_particles=n_particles,
                            n_dim=n_dim,
                            timestep=timestep,
                            box_length=box_length)
    # Argon.setup_mic()  # redundant when .update_meta_class is called with box_length

    # general
    shape = (n_particles, n_dim, 1)

    # setup rng
    rng = np.random.default_rng()
    initial_particle_position = rng.uniform(low=-1, high=1, size=shape) * box_length / 2
    initial_particle_position = np.array([
        [[0.55],
         [0.5]],
        [[-0.5],
         [0.49]],
        [[-0.51],
         [-0.51]],
        [[0.502],
         [-0.25]]
    ]) * box_length / 2

    initial_particle_velocity = np.zeros(shape=shape)  # rng.uniform(low=-1, high=1, size=(n_particles, n_dim))
    initial_particle_force = np.zeros(shape=shape)  # rng.uniform(low=-1, high=1, size=(n_particles, n_dim))

    # particle attrs: n_steps, n_dim, initial_pos, initial_vel, initial_force, **kwargs
    for i, (initial_pos, initial_vel, initial_force) in enumerate(tqdm(zip(initial_particle_position,
                                                                           initial_particle_velocity,
                                                                           initial_particle_force),
                                                                       desc="Setting up particles"
                                                                       )):
        ________ = Argon(n_steps=n_steps,
                         n_dim=n_dim,
                         initial_pos=initial_pos,
                         initial_vel=initial_vel,
                         initial_force=initial_force)
    # normalize simulation problem of particle
    Argon.normalize_problem()

    assert Argon.__n_active__ == n_particles, f"The number of particles that were generated is wrong.\n" \
                                              f"Particles to be generated:      {n_particles:.0f}\n" \
                                              f"Particles that were generated:  {Argon.__n_active__:.0f}\n" \
                                              f"Oops..."


def run_md_simulation():
    # last step does not need to be simulated
    for step in tqdm(np.arange(Argon.__instances__[0].n_steps - 1),
                     desc="MD Simulation Progress"):
        for particle in Argon.__instances__:
            particle.propagate()
        Argon.tick()

        # print("pos: ")
        # print(Argon.__instances__[0].pos[Argon.__instances__[0].__class__.current_step])
        # print("vel: ")
        # print(Argon.__instances__[0].vel[Argon.__instances__[0].__class__.current_step])
        # print("force: ")
        # print(Argon.__instances__[0].force[Argon.__instances__[0].__class__.current_step])
        # print("dpos: ")
        # print(Argon.__instances__[0].dpos)
        # print()


def save_particle_past(object_array, loc="", name="MD_simulation"):
    """
    Store particle positions like:
        (particle, step, dims, 1)

    if runs should be preserved use
        name="MD_simulation_" + strftime("%Y-%m-%d_%H-%M-%S", gmtime())

    Save objects? This seems really bad
    Save np arrays as zips: yes
        Save as (particle, step, parameter, dims, 1)
        Save particle properties separately as (particle, props?)
    :return:
    """
    loc = Path(loc)

    # make memory mapped file to reduce extra memory cost during saving
    n_particles = len(object_array)
    n_particle_shape = object_array[0].pos.shape
    shape = (n_particles, *n_particle_shape)
    print(f"\tSaving {np.product(shape)} data points (shape: {shape})\n"
          f"\tthis may take a while...")
    mmapfilename = Path(mkdtemp()).parent.joinpath('tempmmapfile.dat')
    storage_array = np.memmap(mmapfilename, dtype='float64', mode='w+', shape=shape)

    # using non mmapped arrays
    # n_particles = len(object_array)
    # n_particle_shape = object_array[0].pos.shape
    # storage_array = np.zeros(shape=(n_particles, *n_particle_shape))

    for i, particle in enumerate(tqdm(object_array,
                                      desc="\tSaving")):
        storage_array[i, :] = particle.pos[:]

    np.save(loc / name, storage_array)

def main(n_particles=4, n_dim=2, n_steps=5000,
         timestep=1e-21, box_length=3.405e-8):
    """
    Main wrapper for the MD simulation
    :param n_particles:
    :param n_dim:
    :param n_steps:
    :return:
    """
    intro()
    print(f"\nSetting up MD simulation and particles...")
    set_up_simulation(n_particles, n_dim, n_steps,
                      timestep, box_length)
    print(f"\nRunning MD simulation...")
    run_md_simulation()
    print(f"\nSaving MD simulation Data...")
    save_particle_past(Argon.__instances__)
    print(f"\nDone.")
    return

def intro():
    print(
        """
        MD SIM\n
        by: L. Welzel, C. Slaughter
        for Comp. Phys. - LU
        """
    )

if __name__ == "__main__":
    main()
    # for development only
    import static_plotting
    static_plotting.main()
