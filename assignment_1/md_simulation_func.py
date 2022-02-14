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
    initial_particle_position = rng.uniform(low=-1, high=1, size=shape) * box_length / 2  # random particle positions
    # TODO: np.mgrid is more efficient
    # TODO: implement rough sphere packing for even distribution - eazypeazy NP-hard
    # shitty approach for square/cube numbers of particles below - OF COURSE ONLY WORKS FOR SQUARES
    initial_particle_position, step = np.linspace(-1, 1, int(np.sqrt(n_particles)), endpoint=False, retstep=True)
    initial_particle_position += step/2
    initial_particle_position = (np.array(np.meshgrid(initial_particle_position,
                                                     initial_particle_position)).T * box_length / 2).reshape(shape)
    initial_particle_position += rng.normal(loc=0, scale=5e-2, size=shape) * box_length / 2
    # initial_particle_position = np.array([
    #     [[0.55],
    #      [0.45]],
    #     [[-0.55],
    #      [0.45]],
    #     [[-0.55],
    #      [-0.45]],
    #     [[0.35],
    #      [-0.45]],
    # ]) * box_length / 2

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

    if False:
        plot_lj()


def run_md_simulation():
    # last step does not need to be simulated
    for step in tqdm(np.arange(Argon.__instances__[0].n_steps - 1),
                     desc="MD Simulation Progress"):
        for particle in Argon.__instances__:
            particle.propagate()
        Argon.tick()


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

def main(n_particles=16, n_dim=2, n_steps=1000,
         timestep=1e-22, box_length=3.405e-8):
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

def plot_lj():
    import matplotlib.pyplot as plt
    fig, (ax, ax1) = plt.subplots(nrows=2, ncols=1,
                                  constrained_layout=True,
                                  sharex=True,
                                  figsize=(6, 12))
    x_min = 0.8
    x = np.linspace(x_min, 3, 500)
    ax.plot(x, Argon.potential_lennard_jones(Argon.__instances__[0], x),
            color="k")

    # AXIS TICK LABELS
    ax.set_ylim(-2, 2)
    ax.set_xlim(x_min, None)
    ax.set_ylabel(r'$\Phi ~ (E_{LJ}/\epsilon)$ [-]')

    ax1.plot(x, Argon.force_lennard_jones(Argon.__instances__[0], x),
             color="k")

    # AXIS TICK LABELS
    ax1.set_ylim(-2, 2)
    ax1.set_xlim(x_min, None)
    ax1.set_xlabel(r'$\frac{R}{\sigma}$ [-]')
    ax1.set_ylabel(r'$\nabla\Phi ~ (F\sigma/\epsilon)$ [-]')

    # GLOBAL
    ax.axhline(0, ls="dotted", color="gray")
    ax.axhline(-1, ls="dotted", color="gray")
    ax.axvline(1, ls="dotted", color="gray")
    ax1.axhline(0, ls="dotted", color="gray")
    ax1.axvline(1, ls="dotted", color="gray")

    ax.set_title(
        r'$by~L. Welzel~and~C. Slaughter$'
        f'\n\nArgon: LJ-Potential',
        fontsize=11)
    ax1.set_title(
        f'Argon: LJ-Force',
        fontsize=11)
    fig.suptitle(f'LJ-Potential & Force', fontsize=20, weight="bold")
    plt.show()

if __name__ == "__main__":
    main()
    # for development only
    import static_plotting
    static_plotting.main()
