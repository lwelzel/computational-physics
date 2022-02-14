import numpy as np
import plotly.graph_objects as go
import matplotlib as mpl
from matplotlib import cm
import matplotlib.pyplot as plt
from pathlib import Path

def read_npy_data(loc,):
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

def plotly_3d_static(pos):

    return

def mpl_strict_2d_static(pos):
    assert isinstance(pos, np.ndarray), 'Do not accept anything but np.ndarrays'
    assert pos.shape[2] == 2, "This method is strictly for 2D problems at this point"

    fig, ax = plt.subplots(nrows=1, ncols=1,
                           constrained_layout=True, subplot_kw={'aspect': 1},
                           # sharex=True, sharey=True,
                           figsize=(10, 10))
    # structure  is (particles, steps, n_dims, 1)

    # sizes of markers, last position is fat
    size = np.ones(len(pos[0]))
    size[-1] = 15
    size *= 1
    # alpha
    alpha = np.ones(len(pos[0])) * 0.025
    alpha[-1] = 1

    # colors of markers, colors sample of plt colormaps
    norm = mpl.colors.Normalize(vmin=0, vmax=1)
    cmap = mpl.cm.get_cmap('hsv')
    colors = np.zeros(shape=(*pos.shape[:2], 4))
    # TODO: this should be also done in the other loop
    cpos = np.linspace(0, 1, len(pos)+1, endpoint=True)
    for i, c_start in enumerate(cpos[:-1]):
        colors[i] = cmap(np.linspace(c_start, cpos[i+1], len(pos[0])))

    for i, (particle, color) in enumerate(zip(pos, colors)):
        ax.scatter(*np.reshape(particle,
                               newshape=(len(particle), -1)).T,
                   s=size,
                   c=color,
                   alpha=alpha,
                   # label=f"P{i:03}"
                   )



    # AXIS TICK LABELS
    max = np.power(10, np.ceil(np.log10(np.max(pos)))) * 1.05  # for adding box margin
    ax.set_xlim(-max, max)
    ax.set_ylim(-max, max)
    label_format = '{:.0f}'
    n_ticks = 5
    ax.set_xlabel(r'$\frac{r_{x}}{\sigma}$ [-]')
    ax.set_ylabel(r'$\frac{r_{y}}{\sigma}$ [-]')

    # x_labels = np.linspace(-800, 800, n_ticks)
    # x_ticks = (x_labels / au_pscale) + 100
    # x_labels = [label_format.format(x) for x in x_labels]
    # ax.set_xticks(x_ticks)
    # ax.set_xticklabels(x_labels)
    # ax.set_yticks(x_ticks)
    # ax.set_yticklabels(x_labels)

    rect = mpl.patches.Rectangle((-50, -50), 100, 100, linewidth=1,
                                 edgecolor='k', facecolor='none',
                                 label="Bounding MCI Box")

    # Add the patch to the Axes
    ax.add_patch(rect)

    # ANNOTATIONS
    ax.legend()

    # GLOBAL
    ax.set_title(
        f'Argon: LJ-Potential\n'
        r'$by~L. Welzel~and~C. Slaughter$',
        fontsize=11)
    fig.suptitle(f'MD Simulation', fontsize=20, weight="bold")
    plt.savefig("MD_Argon_strict_2D.png", dpi=300, format="png", metadata=None,
        facecolor='auto', edgecolor='auto')
    plt.show()
    return

def main():
    pos = read_npy_data(Path("MD_simulation.npy"))
    mpl_strict_2d_static(pos)
    # plotly_3d_static(pos)

if __name__ == "__main__":
    main()
