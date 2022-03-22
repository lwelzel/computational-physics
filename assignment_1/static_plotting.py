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
        vdf=pd.DataFrame(vel[idx])
        df['velocity'] = np.sqrt(vdf[0]**2+vdf[1]**2+vdf[2]**2)
        
        curr_atom=go.Scatter3d(x=df['x'], y=df['y'], z=df['z'], mode='markers', marker=dict(size=size, color = df['velocity'], cmin=np.min(df['velocity']), cmax=np.max(df['velocity']), colorscale='Viridis', line=dict(width=0)), hoverinfo='name')
        fig.add_trace(curr_atom)
        
    colorbar_trace = go.Scatter3d(x=[None], y=[None], z=[None], mode='markers', marker=dict(colorscale='Viridis', cmin=np.min(df['velocity']), cmax=np.max(df['velocity']), colorbar=dict(thickness=20, title = 'Speed'),), hoverinfo='none')
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


def mpl_plot_pair_corr(header, pos):
    box_length = header["box_length"]
    n_dim = header["n_dim"]
    n_particles = header["n_particles"]
    volume = box_length ** n_dim
    box = SimBox(box_length, n_dim)
    # (particles, steps, n_dims)
    pos = np.squeeze(pos[:, -1: :])
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

    bins = np.arange(0.5, header["box_length"], 0.01)

    hist, bin_edges = histogram(dist, bins=bins)

    hist = 2 * volume / (n_particles * (n_particles-1)) \
           * 1 / (4 * np.pi * np.square(bin_edges[:-1]) * (bin_edges[1:] - bin_edges[:-1])) \
           * hist

    fig, ax = plt.subplots(nrows=1, ncols=1,
                           constrained_layout=True,
                           figsize=(7, 5))

    # solid expected positions:
    for i in np.arange(25):
        ax.axvline(np.sqrt(i), alpha=0.5, c="gray", linewidth=0.75)

    ax.bar(bin_edges[:-1], hist)
    ax.set_xlabel(r'$r / \sigma$')
    ax.set_ylabel(r'$g(r)$')
    ax.set_title(f'Radial Distribution Function')
    plt.show()
    return


def mpl_plot_pressure(sim):
    return

def mpl_plot_energy(header, kinetic_energy):

    fig, ax = plt.subplots(nrows=1, ncols=1,
                           constrained_layout=True,
                           figsize=(7, 5))

    ax.plot(kinetic_energy[:-2], c="black")
    ax.axhline(header["target_kinetic_energy"])
    ax.set_xlabel(r'$step$ [-]')
    ax.set_ylabel(r'$E_{kin}$ [-]')
    ax.set_title(f'Kinetic Energy')
    plt.show()

    return


def main():
    header, pos, vel, pressure, potential_energy, kinetic_energy = read_h5_data(Path("MD_simulation.h5"))
    plotly_3d_static(pos, vel, header)
    # mpl_plot_pair_corr(header, pos)
    # mpl_plot_energy(header, kinetic_energy)

if __name__ == "__main__":
    main()
