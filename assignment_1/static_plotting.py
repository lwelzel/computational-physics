import numpy as np
import h5py
import plotly.graph_objects as go
import matplotlib as mpl
from matplotlib import cm
import matplotlib.pyplot as plt
from pathlib import Path
import pandas as pd
from itertools import combinations
from static_utils import SimBox
import pathos.multiprocessing as mp
from astropy.stats import histogram
from astropy.visualization import hist
from scipy.interpolate import griddata

# These are various functions to plot and read data
# They do what it says on the box


def read_npy_data(loc, ):
    """
    Should read data that is like:
        (particles, steps, n_dims, 1)
    :param loc:
    :return:
    """
    # TODO: implement proper sampling and slicing
    first_step = 0
    samples_shape = np.array([100, 100, 3, 1])

    samples = np.load(loc, mmap_mode="r", allow_pickle=True)
    assert len(samples_shape) == samples.ndim, "The sampled and memory mapped arrays do not have the same number" \
                                               " of dimensions. You done goofed!"
    mmap_samples_raw_shape = samples.shape

    # hard coded sampling
    samples_shape = np.array([np.min([sample, mmap_sample])
                              for sample, mmap_sample
                              in zip(samples_shape, mmap_samples_raw_shape)])

    # TODO: Doesnt work -> make work, see:
    #  https://stackoverflow.com/questions/50685409/select-n-evenly-spaced-out-elements-in-array-including-first-and-last
    #  https://stackoverflow.com/questions/42573568/numpy-multidimensional-indexing-and-the-function-take
    # idxs_flat = tuple([np.round(np.linspace(first_step, idx_max - 1, n_samples)).astype(int)
    #                    for idx_max, n_samples
    #                    in zip(mmap_samples_raw_shape, samples_shape)])
    # idxs = np.array(np.meshgrid(*idxs_flat))
    # print(idxs.shape)
    # print(np.rollaxis(idxs, -1, 0).shape)
    # samples = np.take(samples, np.ravel_multi_index(np.rollaxis(idxs, -1, 0), samples.shape))
    #
    # print(samples)
    # print(samples.shape)

    return samples


def read_h5_data(loc, ):
    """
    Should read data that is like:
        (steps, particles, n_dims)
    :param loc:
    :return: data like: (particles, steps, n_dims)
    """
    # TODO: implement proper sampling and slicing
    first_step = 0
    samples_shape = np.array([100, 100, 3, 1])

    with h5py.File(loc, "r") as file:
        # # print("Keys: %s" % f.keys())
        # a_group_key = list(file.keys())[0]
        #
        # # Get the data
        # samples = list(file[a_group_key])
        pos = np.array(file["position"])
        vel = np.array(file["velocity"])
        pressure = np.array(file["pressure"])
        potential_energy = np.array(file["potential_energy"])
        kinetic_energy = np.array(file["kinetic_energy"])
        header = dict(file.attrs)

    pos = np.swapaxes(pos, 0, 1)
    vel = np.swapaxes(vel, 0, 1)

    return header, pos, vel, pressure, potential_energy, kinetic_energy


def plotly_3d_static(pos, vel, header):
    length = header["box_length"]
    half_len = length / 2

    # add bounding box:
    edge1 = np.linspace(-half_len, half_len, 50)
    edge2 = np.linspace(-half_len, half_len, 50)
    edge1, edge2 = np.meshgrid(edge1, edge2)

    plane = half_len * np.ones(edge1.shape)

    plane_color = [[0.0, 'rgb(200,200,200)'], [1.0, 'rgb(200,200,200)']]
    opacity = .25

    zpos = dict(type='surface', x=edge1, y=edge2, z=plane, opacity=opacity, colorscale=plane_color, showscale=False)
    zneg = dict(type='surface', x=edge1, y=edge2, z=-plane, opacity=opacity, colorscale=plane_color, showscale=False)
    xpos = dict(type='surface', x=plane, y=edge2, z=edge1, opacity=opacity, colorscale=plane_color, showscale=False)
    xneg = dict(type='surface', x=-plane, y=edge2, z=edge1, opacity=opacity, colorscale=plane_color, showscale=False)
    ypos = dict(type='surface', x=edge1, y=plane, z=edge2, opacity=opacity, colorscale=plane_color, showscale=False)
    yneg = dict(type='surface', x=edge1, y=-plane, z=edge2, opacity=opacity, colorscale=plane_color, showscale=False)

    fig = go.Figure(data=[xpos, xneg, ypos, yneg, zpos, zneg])

    size = np.ones(len(pos[0]))
    size[-1] = 5
    size *= 2

    for idx in range(pos.shape[0]):
        df = pd.DataFrame(pos[idx])
        df.columns = ['x', 'y', 'z']
        vdf = pd.DataFrame(vel[idx])
        df['velocity'] = np.sqrt(vdf[0] ** 2 + vdf[1] ** 2 + vdf[2] ** 2)

        curr_atom = go.Scatter3d(x=df['x'], y=df['y'], z=df['z'], mode='markers',
                                 marker=dict(size=size, color=df['velocity'], cmin=np.min(df['velocity']),
                                             cmax=np.max(df['velocity']), colorscale='Viridis', line=dict(width=0)),
                                 hoverinfo='name')
        fig.add_trace(curr_atom)

    colorbar_trace = go.Scatter3d(x=[None], y=[None], z=[None], mode='markers',
                                  marker=dict(colorscale='Viridis', cmin=np.min(df['velocity']),
                                              cmax=np.max(df['velocity']),
                                              colorbar=dict(thickness=20, title='Speed'), ), hoverinfo='none')
    fig.add_trace(colorbar_trace)

    fig.update_layout(showlegend=False)
    fig.show()

    return


def mpl_strict_2d_static(pos, header):
    assert isinstance(pos, np.ndarray), 'Do not accept anything but np.ndarrays'
    assert pos.shape[2] <= 3, "This method is strictly for 2D or 3D problems at this point"
    assert pos.shape[2] > 1, "This method is strictly for 2D or 3D problems at this point"

    if pos.shape[2] == 2:
        n_rows = 1
        n_cols = 1
        n_plots = [0]
    elif pos.shape[2] == 3:
        n_rows = 2
        n_cols = 2
        n_plots = [0, 2, 3]

    fig, axes = plt.subplots(nrows=n_rows, ncols=n_cols,
                             constrained_layout=True, subplot_kw={'aspect': 1},
                             # sharex=True, sharey=True,
                             figsize=(12, 12))
    # structure  is (particles, steps, n_dims, 1)

    if pos.shape[2] == 3:
        axes[0, 1].axis('off')
        axes = axes.flatten()

    # sizes of markers, last position is fat
    size = np.ones(len(pos[0]))
    size[-1] = 15
    size *= .5
    # alpha
    alpha = np.ones(len(pos[0])) * 0.025
    alpha[-1] = 1

    # colors of markers, colors sample of plt colormaps
    norm = mpl.colors.Normalize(vmin=0, vmax=1)
    cmap = mpl.cm.get_cmap('hsv')
    colors = np.zeros(shape=(*pos.shape[:2], 4))
    # TODO: this should be also done in the other loop
    cpos = np.linspace(0, 1, len(pos) + 1, endpoint=True)
    for i, c_start in enumerate(cpos[:-1]):
        colors[i] = cmap(np.linspace(c_start, cpos[i + 1], len(pos[0])))

    # for Plot labels
    label_format = '{:.0f}'
    n_ticks = 5

    # For MCI box
    length = header["box_length"]
    half_len = length / 2
    max = length * 1.05

    for plt_idx in n_plots:
        if pos.shape[2] == 3:
            ax = axes[plt_idx]
        else:
            ax = axes
        if plt_idx == 0:  # XY Plane
            for i, (particle, color) in enumerate(zip(pos, colors)):
                ax.scatter(*np.reshape(particle[:, [0, 1]], newshape=(len(particle), -1)).T, s=size, c=color,
                           alpha=alpha)
                if len(n_plots) > 1:
                    ax.set_title('Projection onto XY Plane')
                    ax.set_xlabel(r'$\frac{r_{x}}{\sigma}$ [-]')
                    ax.set_ylabel(r'$\frac{r_{y}}{\sigma}$ [-]')

        if plt_idx == 2:  # XZ Plane
            for i, (particle, color) in enumerate(zip(pos, colors)):
                ax.scatter(*np.reshape(particle[:, [0, 2]], newshape=(len(particle), -1)).T, s=size, c=color,
                           alpha=alpha)
                ax.set_title('Projection onto XZ Plane')
                ax.set_xlabel(r'$\frac{r_{x}}{\sigma}$ [-]')
                ax.set_ylabel(r'$\frac{r_{z}}{\sigma}$ [-]')

        if plt_idx == 3:  # YZ Plane
            for i, (particle, color) in enumerate(zip(pos, colors)):
                ax.scatter(*np.reshape(particle[:, [1, 2]], newshape=(len(particle), -1)).T, s=size, c=color,
                           alpha=alpha)
                ax.set_title('Projection onto YZ Plane')
                ax.set_xlabel(r'$\frac{r_{y}}{\sigma}$ [-]')
                ax.set_ylabel(r'$\frac{r_{z}}{\sigma}$ [-]')

        # AXIS TICK LABELS
        # max = np.power(10, np.ceil(np.log10(np.max(pos)))) * 1.05  # for adding box margin
        ax.set_xlim(-max, max)
        ax.set_ylim(-max, max)

        rect = mpl.patches.Rectangle((-length / 2, -length / 2), length, length, linewidth=1, edgecolor='k',
                                     facecolor='none', label="Bounding MCI Box")

        # Add the patch to the Axes
        ax.add_patch(rect)

        # ANNOTATIONS
        ax.legend()

    fig.suptitle(f'Argon: LJ-Potential\n' r'$by~L. Welzel~and~C. Slaughter$', fontsize=20, weight="bold")

    plt.savefig("MD_Argon_strict_2D.png", dpi=300, format="png", metadata=None,
                facecolor='auto', edgecolor='auto')
    plt.show()
    return


def get_pressure(path="simulation_data\long_sims", string="den=3e-01_temp=3e+00"):
    fig, ax = plt.subplots(nrows=1, ncols=1,
                           constrained_layout=True,
                           figsize=(7, 5))
    for string, name in zip(["den=3e-01_temp=3e+00", "den=8e-01_temp=1e+00", "den=1e+00_temp=5e-01"],
                            ["Gas", "Liquid", "Solid"]):
        files = Path(path).rglob(f"*{string}*.h5")
        files = np.array([path for path in files]).flatten()
        n_part = 108

        header, pos, vel, pressure, potential_energy, kinetic_energy = read_h5_data(files[0])

        corr_array = np.zeros((len(files), len(pressure)))

        box_length = header["box_length"]
        n_dim = header["n_dim"]
        n_particles = header["n_particles"]
        T = header["temperature"]
        rho = header["density"]
        volume = box_length ** n_dim
        box = SimBox(box_length, n_dim)

        n_cores = int(mp.cpu_count() * 0.8)
        pool = mp.Pool(n_cores)



        for i, file in enumerate(files):
            header, pos, vel, pressure, potential_energy, kinetic_energy = read_h5_data(file)
            pressure_terms = 0.5 * pressure
            avg_pressure = T * rho - 1 / (3 * n_particles * T) * np.mean(pressure_terms)
            pressures = T * rho - rho / (3 * n_particles * T) * (pressure_terms)
            # ax.plot(header["timestep"] * np.arange(len(pressures)), pressures)
            corr_array[i] = pressures

        corr_array_mean = np.mean(corr_array, axis=0)
        errors = np.std(corr_array, axis=0)
        ax.plot(header["timestep"] * np.arange(len(corr_array_mean)), corr_array_mean,
                label=name)

        ax.fill_between(header["timestep"] * np.arange(len(corr_array_mean)),
                        corr_array_mean + errors * 6,
                        corr_array_mean - errors * 6,
                        label=r"$6\sigma$ CI " + name,
                        alpha=0.3)

    ax.set_xlabel(r'$time$ [-]')
    ax.set_ylabel(r'$\frac{P}{k_B T \rho}$ [-]')
    ax.legend()
    ax.set_title('Pressure vs. Time')
    plt.show()
    return

def get_corr(path="simulation_data\long_sims", string="den=1e+00_temp=5e-01"):
    files = Path(path).rglob(f"*{string}*.h5")
    files = np.array([path for path in files]).flatten()
    n_part = 108

    header, pos, vel, pressure, potential_energy, kinetic_energy = read_h5_data(files[0])

    box_length = header["box_length"]
    n_dim = header["n_dim"]
    n_particles = header["n_particles"]
    volume = box_length ** n_dim
    box = SimBox(box_length, n_dim)

    bin_width = 0.01
    bins = np.arange(0.5, header["box_length"], bin_width)

    corr_array = np.zeros((len(files), int(len(bins)-1)))

    n_cores = int(mp.cpu_count() * 0.8)
    pool = mp.Pool(n_cores)

    for i, file in enumerate(files):
        header, pos, vel, pressure, potential_energy, kinetic_energy = read_h5_data(file)
        pos = np.squeeze(pos[:108, -1])
        # reference: https://stackoverflow.com/questions/16003217/n-d-version-of-itertools-combinations-in-numpy
        comb_idx = np.transpose(np.triu_indices(len(pos), 1))
        pos_pairs = pos[comb_idx]
        pos_pairs = np.reshape(pos_pairs, (*pos_pairs.shape, 1))

        dist = np.squeeze(np.array(pool.map(box.get_distance_absoluteA1, pos_pairs)))

        hist, bin_edges = histogram(dist, bins=bins)

        corr_array[i] = hist

    corr_array = np.mean(corr_array, axis=0).flatten()

    x = bin_edges[:-1] + (bin_edges[1:] - bin_edges[:-1]) / 2

    hist = 2 * volume / (n_particles * (n_particles - 1)) \
           * 1 / (4 * np.pi * np.square(bin_edges[:-1]) * (bin_edges[1:] - bin_edges[:-1])) \
           * corr_array

    fig, ax = plt.subplots(nrows=1, ncols=1,
                           constrained_layout=True,
                           figsize=(7, 5))

    # # solid expected positions:
    for i in np.arange(8):
        ax.axvline(np.sqrt(i * 1 / (1-0.1)), alpha=0.5, c="gray", linewidth=0.75)
    # ax.axhline(1., c="red", ls="dashed")

    ax.bar(x, hist, width=bin_width, align="center",
           edgecolor="k", facecolor="none", alpha=1)

    ax.set_xlabel(r'$r / \sigma$ [-]')
    ax.set_ylabel(r'$g(r)$ [-]')
    ax.set_xlim(0.5, np.ceil(header["box_length"] / 2))
    fig.suptitle(f'Radial Distribution Function')
    ax.set_title(f'Density: {header["density"]} | Temperature: {header["temperature"]}\n'
                 f'Runs: {len(files)} | Total Time: {len(files) * header["time_total"]} | Particles: {header["n_particles"]}',
                 fontsize=10)
    plt.show()
    return


def mpl_plot_pair_corr(header, pos):
    box_length = header["box_length"]
    n_dim = header["n_dim"]
    n_particles = header["n_particles"]
    volume = box_length ** n_dim
    box = SimBox(box_length, n_dim)
    # (particles, steps, n_dims)
    pos = np.squeeze(pos[:, -1::])
    # reference: https://stackoverflow.com/questions/16003217/n-d-version-of-itertools-combinations-in-numpy
    comb_idx = np.transpose(np.triu_indices(len(pos), 1))
    pos_pairs = pos[comb_idx]
    pos_pairs = np.reshape(pos_pairs, (*pos_pairs.shape, 1))

    n_cores = int(mp.cpu_count() * 0.8)
    pool = mp.Pool(n_cores)
    dist = pool.map(box.get_distance_absoluteA1, pos_pairs)
    pool.close()
    pool.join()
    dist = np.array(dist)

    corr_array = dist.flatten()

    bin_width = 0.01
    bins = np.arange(0.5, header["box_length"], bin_width)

    hist, bin_edges = histogram(corr_array, bins=bins)
    x = bin_edges[:-1] + (bin_edges[1:] - bin_edges[:-1]) / 2

    hist = 2 * volume / (n_particles * (n_particles - 1)) \
           * 1 / (4 * np.pi * np.square(bin_edges[:-1]) * (bin_edges[1:] - bin_edges[:-1])) \
           * hist

    fig, ax = plt.subplots(nrows=1, ncols=1,
                           constrained_layout=True,
                           figsize=(7, 5))

    # solid expected positions:
    for i in np.arange(10):
        ax.axvline(np.sqrt(i), alpha=0.5, c="gray", linewidth=0.75)

    ax.bar(x, hist, width=bin_width, align="center",
           edgecolor="k", facecolor="none", alpha=1)

    ax.set_xlabel(r'$r / \sigma$')
    ax.set_ylabel(r'$g(r)$')
    ax.set_xlim(0.5, np.ceil(header["box_length"] / 2))
    ax.set_title(f'Radial Distribution Function')
    plt.show()
    return


def mpl_plot_pressure(pressure_terms, header):
    pressure_terms = 0.5 * pressure_terms
    fig, ax = plt.subplots(nrows=1, ncols=1, constrained_layout=True, figsize=(7, 5))
    T = header["temperature"]
    rho = header["density"]
    n_particles = header["n_particles"]

    avg_pressure = T * rho * (1 - 1 / (3 * n_particles * T) * np.mean(pressure_terms))
    pressures = T * rho * (1 - 1 / (3 * n_particles * T) * (pressure_terms))

    ax.plot(pressures, c="black")
    ax.set_xlabel(r'$step$ [-]')
    ax.set_ylabel(r'$P$ [-]')
    ax.set_ylim(0, 2)
    ax.set_title('Pressure vs. Time (Average Pressure = {:.3f})'.format(avg_pressure))
    plt.show()

    return


def mpl_plot_energy(header, kinetic_energy):
    fig, ax = plt.subplots(nrows=1, ncols=1,
                           constrained_layout=True,
                           figsize=(7, 5))

    ax.plot(kinetic_energy[np.nonzero(kinetic_energy)], c="black")
    ax.axhline(header["target_kinetic_energy"])
    ax.set_xlabel(r'$step$ [-]')
    ax.set_ylabel(r'$E_{kin}$ [-]')
    ax.set_title(f'Kinetic Energy')
    plt.show()
    return


def mpl_plot_energy_cons(header, kinetic_energy, potential_energy):
    tot_energy = kinetic_energy[np.nonzero(kinetic_energy)] + potential_energy[np.nonzero(kinetic_energy)]
    fig, ax = plt.subplots(nrows=1, ncols=1,
                           constrained_layout=True,
                           figsize=(7, 5))
    ax.plot(np.abs(potential_energy[np.nonzero(kinetic_energy)] / potential_energy[np.nonzero(kinetic_energy)][0]),
            label=r'|$E_{pot}/E_{pot}(0)|$ [-]', ls="dotted")
    ax.plot(kinetic_energy[np.nonzero(kinetic_energy)] / header["target_kinetic_energy"],
            label=r'$E_{kin} / E_{kin, ~target}$ [-]', ls="dashed")
    ax.plot(np.abs(tot_energy / tot_energy[0]), label=r'$|E_{tot}/E_{pot}(0)|$ [-]', ls="solid")

    ax.set_xlabel(r'$step$ [-]')
    ax.set_ylabel(r'Energy [-]')
    ax.set_title(f'Simulation Energy vs Time')
    # ax.set_yscale("log")
    ax.legend()
    plt.show()

    return

def energy_all(path="simulation_data\long_sims", string="den=3e-01_temp=3e+00"):
    fig, ax = plt.subplots(nrows=1, ncols=1,
                           constrained_layout=True,
                           figsize=(11, 7))

    # ax.set_yscale("symlog")
    for string, name, color in zip(["den=3e-01_temp=3e+00", "den=8e-01_temp=1e+00", "den=1e+00_temp=5e-01"],
                            ["Gas", "Liquid", "Solid"],
                                   ["b", "orange", "green"]):
        files = Path(path).rglob(f"*{string}*.h5")
        files = np.array([path for path in files]).flatten()
        n_part = 108

        header, pos, vel, pressure, potential_energy, kinetic_energy = read_h5_data(files[0])

        corr_array = np.zeros((len(files), len(kinetic_energy[np.nonzero(kinetic_energy)])))

        box_length = header["box_length"]
        n_dim = header["n_dim"]
        n_particles = header["n_particles"]
        T = header["temperature"]
        rho = header["density"]
        volume = box_length ** n_dim
        box = SimBox(box_length, n_dim)

        n_cores = int(mp.cpu_count() * 0.8)
        pool = mp.Pool(n_cores)



        for i, file in enumerate(files):
            header, pos, vel, pressure, potential_energy, kinetic_energy = read_h5_data(file)

            # tot_energy = kinetic_energy[np.nonzero(kinetic_energy)] + potential_energy[np.nonzero(kinetic_energy)]
            # ax.plot(
            #     np.abs(potential_energy[np.nonzero(kinetic_energy)] / potential_energy[np.nonzero(kinetic_energy)][0]),
            #     label=r'|$E_{pot}/E_{pot}(0)|$ [-]', ls="dotted")
            # ax.plot(kinetic_energy[np.nonzero(kinetic_energy)] / header["target_kinetic_energy"],  ls="dashed")
            # ax.plot(np.abs(tot_energy / tot_energy[0]), label=r'$|E_{tot}/E_{pot}(0)|$ [-]', ls="solid")

            corr_array[i] = kinetic_energy[np.nonzero(kinetic_energy)]

        corr_array_mean = np.mean(corr_array, axis=0)
        errors = np.std(corr_array, axis=0)
        ax.plot(header["timestep"] * np.arange(len(corr_array_mean)), corr_array_mean,
                label=name)

        ax.fill_between(header["timestep"] * np.arange(len(corr_array_mean)),
                        corr_array_mean + errors * 6,
                        corr_array_mean - errors * 6,
                        label=r"$6\sigma$ CI " + name,
                        alpha=0.2)

        ax.axhline(header["target_kinetic_energy"], color=color, ls="dashed", label=r"$E_{kin, ~ target}$")

    ax.set_xlabel(r'$time$ [-]')
    ax.set_ylabel(r'$E_{kin}$ [-]')
    ax.legend()
    fig.suptitle(f'Kinetic Energy Conservation')
    ax.set_title(f'Runs: {len(files)} | Total Time: {len(files) * header["time_total"]} | Particles: {header["n_particles"]}',
                 fontsize=10)

    plt.show()
    return

def energy_all_total(path="simulation_data\long_sims", string="den=3e-01_temp=3e+00"):
    fig, ax = plt.subplots(nrows=1, ncols=1,
                           constrained_layout=True,
                           figsize=(11, 7))
    ax.axhline(1, color="black", ls="dotted", label=r"$E_{total, ~ initial}$", linewidth=0.75)

    # ax.set_yscale("symlog")
    ci = 2
    for string, name, color in zip(["den=3e-01_temp=3e+00", "den=8e-01_temp=1e+00", "den=1e+00_temp=5e-01"],
                            ["Gas", "Liquid", "Solid"],
                                   ["b", "orange", "green"]):
        files = Path(path).rglob(f"*{string}*.h5")
        files = np.array([path for path in files]).flatten()
        n_part = 108

        header, pos, vel, pressure, potential_energy, kinetic_energy = read_h5_data(files[0])

        corr_array = np.zeros((len(files), len(kinetic_energy[np.nonzero(kinetic_energy)])))

        box_length = header["box_length"]
        n_dim = header["n_dim"]
        n_particles = header["n_particles"]
        T = header["temperature"]
        rho = header["density"]
        volume = box_length ** n_dim
        box = SimBox(box_length, n_dim)

        n_cores = int(mp.cpu_count() * 0.8)
        pool = mp.Pool(n_cores)



        for i, file in enumerate(files):
            header, pos, vel, pressure, potential_energy, kinetic_energy = read_h5_data(file)

            tot_energy = kinetic_energy[np.nonzero(kinetic_energy)] + potential_energy[np.nonzero(kinetic_energy)]


            corr_array[i] = tot_energy / tot_energy[0]

        corr_array_mean = np.mean(corr_array, axis=0)
        errors = np.std(corr_array, axis=0)
        ax.plot(header["timestep"] * np.arange(len(corr_array_mean)), corr_array_mean,
                label=name)


        ax.fill_between(header["timestep"] * np.arange(len(corr_array_mean)),
                        corr_array_mean + errors * ci,
                        corr_array_mean - errors * ci,
                        label=f"{ci}"r"$\sigma$ CI " + name,
                        alpha=0.2)
        ci *= 3


    ax.set_xlabel(r'$time$ [-]')
    ax.set_ylabel(r'$E_{total}$ [-]')
    ax.legend()
    fig.suptitle(f'Total Energy Conservation')
    ax.set_title(f'Runs: {len(files)} | Total Time: {len(files) * header["time_total"]} | Particles: {header["n_particles"]}',
                 fontsize=10)

    plt.show()
    return

def plot_lj():
    import matplotlib.pyplot as plt
    fig, (ax, ax1) = plt.subplots(nrows=2, ncols=1,
                                  constrained_layout=True,
                                  sharex=True,
                                  figsize=(6, 12))
    x_min = 0.8
    x = np.linspace(x_min, 3, 500)
    r = x  # 1. / x
    ax.plot(x, 4 * (np.power(r, -12) - np.power(r, -6)),
            color="k")

    # AXIS TICK LABELS
    ax.set_ylim(-2, 2)
    ax.set_xlim(x_min, None)
    ax.set_ylabel(r'$\Phi ~ (E_{LJ}/\epsilon)$ [-]')

    ax1.plot(x, - 24 * (np.power(r, 6) - 2) / (np.power(r, 13)),
             color="k")

    # AXIS TICK LABELS
    ax1.set_ylim(-3, 2)
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

def massive_plot(path="simulation_data\massive_para_sweep"):
    files = Path(path).rglob(f"*.h5")
    files = np.array([path for path in files]).flatten()
    n_part = 108

    temperatures = np.zeros(len(files))
    densities = np.zeros(len(files))

    median_nonzeros = np.zeros(len(files))


    n_cores = int(mp.cpu_count() * 0.8)
    pool = mp.Pool(n_cores)

    # for i, file in enumerate(files):
    #     # print(i, " of ", len(files))
    #     header, pos, vel, pressure, potential_energy, kinetic_energy = read_h5_data(file)
    #     temperatures[i] = header["temperature"]
    #     densities[i] = header["density"]
    #
    #     box_length = header["box_length"]
    #     n_dim = header["n_dim"]
    #     n_particles = header["n_particles"]
    #     volume = box_length ** n_dim
    #     box = SimBox(box_length, n_dim)
    #
    #     pos = np.squeeze(pos[:, -1::])
    #     comb_idx = np.transpose(np.triu_indices(len(pos), 1))
    #     pos_pairs = pos[comb_idx]
    #     pos_pairs = np.reshape(pos_pairs, (*pos_pairs.shape, 1))
    #
    #     dist = np.array([box.get_distance_absoluteA1(pos_pair) for pos_pair in pos_pairs])
    #     pool.close()
    #     pool.join()
    #     dist = np.array(dist)
    #
    #     corr_array = dist.flatten()
    #
    #     bin_width = 0.01
    #     bins = np.arange(0.5, header["box_length"], bin_width)
    #
    #     hist, bin_edges = histogram(corr_array, bins=bins)
    #
    #     hist = 2 * volume / (n_particles * (n_particles - 1)) \
    #            * 1 / (4 * np.pi * np.square(bin_edges[:-1]) * (bin_edges[1:] - bin_edges[:-1])) \
    #            * hist
    #
    #     median_nonzeros[i] = 1 / np.abs(1 - np.median(hist[np.nonzero(hist)]))

    def get_gassyness(file):
        header, pos, vel, pressure, potential_energy, kinetic_energy = read_h5_data(file)
        temperature = header["temperature"]
        density = header["density"]

        box_length = header["box_length"]
        n_dim = header["n_dim"]
        n_particles = header["n_particles"]
        volume = box_length ** n_dim
        box = SimBox(box_length, n_dim)

        pos = np.squeeze(pos[:, -1::])
        comb_idx = np.transpose(np.triu_indices(len(pos), 1))
        pos_pairs = pos[comb_idx]
        pos_pairs = np.reshape(pos_pairs, (*pos_pairs.shape, 1))

        dist = np.array([box.get_distance_absoluteA1(pos_pair) for pos_pair in pos_pairs])
        dist = np.array(dist)

        corr_array = dist.flatten()

        bin_width = 0.01
        bins = np.arange(0.5, header["box_length"], bin_width)

        hist, bin_edges = histogram(corr_array, bins=bins)

        hist = 2 * volume / (n_particles * (n_particles - 1)) \
               * 1 / (4 * np.pi * np.square(bin_edges[:-1]) * (bin_edges[1:] - bin_edges[:-1])) \
               * hist

        median_nonzero = 1 / np.median(hist[np.nonzero(hist)])

        return temperature, density, median_nonzero, temperature * density * (1 - 1 / (3 * n_particles * temperature) * np.mean(0.5 * pressure))

    res = pool.map(get_gassyness, files)
    res = np.array(res)

    temperatures, densities, median_nonzeros, pressure = res.T

    fig, ax = plt.subplots(nrows=1, ncols=1,
                           constrained_layout=True,
                           figsize=(9, 9))

    ti = np.linspace(temperatures.min(), temperatures.max(), 1000)
    di = np.linspace(densities.min(), densities.max(), 1000)

    zi = griddata((temperatures, densities), median_nonzeros, (ti[None,:], di[:,None]), method='cubic')


    CS = plt.contourf(ti, di, zi, 15, cmap=plt.cm.viridis,
                      vmax=None, vmin=None)
    plt.colorbar(CS, label="Gassy-ness [-]")

    ax.set_xlabel(r'$T$ [-]')
    ax.set_ylabel(r'$\rho$ [-]')
    ax.set_title(f'Where do we find gas?')
    plt.show()




    fig, ax = plt.subplots(nrows=1, ncols=1,
                           constrained_layout=True,
                           figsize=(9, 9))

    ti = np.linspace(temperatures.min(), temperatures.max(), 1000)
    di = np.linspace(densities.min(), densities.max(), 1000)

    zi = griddata((temperatures, densities), pressure, (ti[None, :], di[:, None]), method='cubic')

    CS = plt.contourf(ti, di, zi, 15, cmap=plt.cm.viridis,
                      vmax=None, vmin=None)
    plt.colorbar(CS, label=r'$\frac{P}{k_B T \rho}$ [-]')

    ax.set_xlabel(r'$T$ [-]')
    ax.set_ylabel(r'$\rho$ [-]')
    ax.set_title(f'Where do we find gas?')
    plt.show()

    pool.close()
    pool.join()




def main():
    try:
        plot_lj()

        # ["den=3e-01_temp=3e+00", "den=8e-01_temp=1e+00", "den=1e+00_temp=5e-01"]
        files = Path("simulation_data").rglob(f"*.h5")
        file = np.array([path for path in files]).flatten()[0]
        print(f"Showing example plots for {file}")
        header, pos, vel, pressure, potential_energy, kinetic_energy = read_h5_data(file)
        plotly_3d_static(pos, vel, header)
        mpl_plot_energy_cons(header, kinetic_energy, potential_energy)

        get_pressure()
        get_corr()
        energy_all()
        energy_all_total()

        massive_plot()

    except BaseException:
        print("I dont make any other plots right now.\n"
              "This might be because I cant find any saved runs.")




if __name__ == "__main__":
    main()
