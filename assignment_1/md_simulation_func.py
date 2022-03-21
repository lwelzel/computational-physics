import numpy as np
from pathlib import Path
from md_simulation_class import MolDyn
from argon_class import Argon


def set_up_simulation(n_particles=2 ** 3 * 4, n_dim=3, n_steps=1000,
                      time_total=5.e-12, initial_timestep=1e-16,
                      max_steps=1e6, max_real_time=3 * 60,
                      temperature=0.5, density=1.2):
    # TODO: mass might be array
    # TODO: setup should accept external initial_pos, vel, acc

    # TODO:

    # META
    rng = np.random.default_rng()
    MolDyn(n_particles=n_particles, n_dim=n_dim, n_steps=n_steps,
           time_total=time_total, initial_timestep=initial_timestep,
           max_steps=max_steps, max_real_time=max_real_time,
           temperature=temperature, density=density,
           file_location=Path(""),
           name="MD_simulation")  # box_length=box_length,

    # general
    shape = (n_particles, n_dim, 1)

    n_sets = int(np.round((n_particles / 4) ** (1 / 3)))

    start_pos, step = np.linspace(-MolDyn.sim.box_length / 2, MolDyn.sim.box_length / 2, n_sets, endpoint=False, retstep=True)
    start_pos = start_pos + step / 2  # shifts off box edge


    initial_particle_position = np.zeros(shape=shape)
    initial_particle_velocity = rng.normal(0., 1., size=shape)
    initial_particle_acc = np.zeros(shape=shape)

    fcc_unit = np.array([
        [[0.], [0.], [0.]],
        [[0.], [1.], [1.]],
        [[1.], [0.], [1.]],
        [[1.], [1.], [0.]]
    ])

    fcc_grid_x, fcc_grid_y, fcc_grid_y = np.meshgrid(*np.tile(start_pos, (n_dim, 1)))

    print(fcc_grid.shape)
    import plotly.graph_objects as go
    fig = go.Figure(data=[go.Scatter3d(x=x, y=y, z=z,
                                       mode='markers')])
    fig.show()

    idx_count = 0
    for idxx, first_x in enumerate(start_pos):
        for idxy, first_y in enumerate(start_pos):
            for idxz, first_z in enumerate(start_pos):
                set_positions = np.zeros(shape=(4, n_dim, 1))
                set_positions[0] = np.array([first_x, first_y, first_z]).reshape(3, 1)
                set_positions[1] = np.array([first_x, first_y + pos_change, first_z + pos_change]).reshape(3, 1)
                set_positions[2] = np.array([first_x + pos_change, first_y + pos_change, first_z]).reshape(3, 1)
                set_positions[3] = np.array([first_x + pos_change, first_y, first_z + pos_change]).reshape(3, 1)
                curr_idx = 4 * (idx_count)
                end_idx = curr_idx + 4
                initial_particle_position[curr_idx:end_idx, :, :] = set_positions

                idx_count += 1

    show_3d_init_pos(initial_positions=initial_particle_position)

    #     #Old Particle Position Setup
    #     # setup rng
    #     rng = np.random.default_rng()
    #     initial_particle_position = rng.uniform(low=-1, high=1, size=shape) * MolDyn.sim.box_length  # random particle positions
    #     # TODO: np.mgrid is more efficient
    #     # TODO: implement rough sphere packing for even distribution - eazypeazy NP-hard
    #     # shitty approach for square/cube numbers of particles below - OF COURSE ONLY WORKS FOR SQUARES
    #     initial_particle_position, step = np.linspace(-1, 1, int(np.around((n_particles)**(1/n_dim))), endpoint=False, retstep=True)
    #     initial_particle_position += step/2
    #
    #     dim_list=[]
    #     for dim in range(n_dim):
    #     	dim_list.append(initial_particle_position)
    #
    #
    #     initial_particle_position = (np.array(np.meshgrid(*dim_list)).T * MolDyn.sim.box_length / 2).reshape(shape)
    #     initial_particle_position += rng.normal(loc=0, scale=5e-2, size=shape) * MolDyn.sim.box_length / 2
    #
    #     initial_particle_velocity = np.zeros(shape=shape)  # rng.uniform(low=-1, high=1, size=(n_particles, n_dim))
    #     initial_particle_acc = np.zeros(shape=shape)  # rng.uniform(low=-1, high=1, size=(n_particles, n_dim))

    argon = (Argon, n_particles, initial_particle_position, initial_particle_velocity, initial_particle_acc)
    print('Initialization complete.')

    MolDyn.sim.set_up_simulation(argon)
    print('Relaxation complete.\n')

    if False:
        plot_lj()


def run_md_simulation():
    print("Running MD simulation.")
    MolDyn.sim.run()


def main():
    """
    Main wrapper for the MD simulation
    :param n_particles:
    :param n_dim:
    :param n_steps:
    :return:
    """
    intro()
    set_up_simulation()
    run_md_simulation()
    return


def intro():
    print(
        "\n"
        "================================\n\n"
        "         MD SIM\n"
        "  by: L. Welzel, C. Slaughter\n"
        "     for Comp. Phys. - LU\n\n"
        "================================\n"
        )


def plot_lj():
    import matplotlib.pyplot as plt
    fig, (ax, ax1) = plt.subplots(nrows=2, ncols=1,
                                  constrained_layout=True,
                                  sharex=True,
                                  figsize=(6, 12))
    x_min = 0.8
    x = np.linspace(x_min, 3, 500)
    ax.plot(x, Argon.potential_lennard_jones(MolDyn.sim.instances[0], x),
            color="k")

    # AXIS TICK LABELS
    ax.set_ylim(-2, 2)
    ax.set_xlim(x_min, None)
    ax.set_ylabel(r'$\Phi ~ (E_{LJ}/\epsilon)$ [-]')

    ax1.plot(x, Argon.force_lennard_jones(MolDyn.sim.instances[0], x),
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

def show_3d_init_pos(initial_positions):
    import plotly.graph_objects as go

    initial_positions = np.squeeze(initial_positions)
    x, y, z = initial_positions[:, 0], initial_positions[:, 1], initial_positions[:, 2]

    print(x.shape, y.shape, z.shape)

    fig = go.Figure(data=[go.Scatter3d(x=x, y=y, z=z,
                                       mode='markers')])
    fig.show()


if __name__ == "__main__":
    main()
    # for development only
    import static_plotting

    static_plotting.main()
