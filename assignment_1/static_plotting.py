import numpy as np
import h5py
# import plotly.graph_objects as go
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

def read_h5_data(loc,):
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
        samples = np.array(file["position"])

    samples = np.swapaxes(samples, 0, 1)

    return samples

def plotly_3d_static(pos):
    return

def mpl_strict_2d_static(pos):
    assert isinstance(pos, np.ndarray), 'Do not accept anything but np.ndarrays'
    assert pos.shape[2] <= 3, "This method is strictly for 2D or 3D problems at this point"
    assert pos.shape[2] > 1, "This method is strictly for 2D or 3D problems at this point"
    
    if pos.shape[2] == 2:
    	n_rows=1
    	n_cols=1
    	n_plots=[0]
    elif pos.shape[2] == 3:
    	n_rows=2
    	n_cols=2
    	n_plots=[0,2,3]
	
    fig, axs = plt.subplots(nrows=n_rows, ncols=n_cols,
                           constrained_layout=True, subplot_kw={'aspect': 1},
                           # sharex=True, sharey=True,
                           figsize=(7, 7))
    # structure  is (particles, steps, n_dims, 1)
    
    
    axs[0,1].axis('off')
    axes=axs.flatten()

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
    
    #for Plot labels
    label_format = '{:.0f}'
    n_ticks = 5
    
    for plt_idx in n_plots:
    	ax=axes[plt_idx]
    	if plt_idx == 0: #XY Plane
    		for i, (particle, color) in enumerate(zip(pos, colors)):
    			ax.scatter(*np.reshape(particle[:,[0,1]],newshape=(len(particle), -1)).T, s=size, c=color, alpha=alpha)
    			if len(n_plots) > 1:
    				ax.set_title('Projection onto XY Plane')
    				ax.set_xlabel(r'$\frac{r_{x}}{\sigma}$ [-]')
    				ax.set_ylabel(r'$\frac{r_{y}}{\sigma}$ [-]')
    	
    	if plt_idx == 2: #XZ Plane
        	for i, (particle, color) in enumerate(zip(pos, colors)):
        		ax.scatter(*np.reshape(particle[:,[0,2]],newshape=(len(particle), -1)).T, s=size, c=color, alpha=alpha)
        		ax.set_title('Projection onto XZ Plane')
        		ax.set_xlabel(r'$\frac{r_{x}}{\sigma}$ [-]')
        		ax.set_ylabel(r'$\frac{r_{z}}{\sigma}$ [-]')
    	
    	if plt_idx == 3: #YZ Plane
    		for i, (particle, color) in enumerate(zip(pos, colors)):
    			ax.scatter(*np.reshape(particle[:,[1,2]],newshape=(len(particle), -1)).T, s=size, c=color, alpha=alpha)
    			ax.set_title('Projection onto YZ Plane')
    			ax.set_xlabel(r'$\frac{r_{y}}{\sigma}$ [-]')
    			ax.set_ylabel(r'$\frac{r_{z}}{\sigma}$ [-]')
    			
    	# AXIS TICK LABELS
    	max = np.power(10, np.ceil(np.log10(np.max(pos)))) * 1.05  # for adding box margin
    	ax.set_xlim(-max, max)
    	ax.set_ylim(-max, max)
    	
    	rect = mpl.patches.Rectangle((-50, -50), 100, 100, linewidth=1, edgecolor='k', facecolor='none', label="Bounding MCI Box")
    	
    	#Add the patch to the Axes
    	ax.add_patch(rect)
    	
    	#ANNOTATIONS
    	ax.legend()

        
    	
    	
    	# for i, (particle, color) in enumerate(zip(pos, colors)):
#         	ax.scatter(*np.reshape(particle,
#                                newshape=(len(particle), -1)).T,
#                    s=size,
#                    c=color,
#                    alpha=alpha,
#                    # label=f"P{i:03}"
#                    )


    #fig.suptitle(f'MD Simulation', fontsize=20, weight="bold")
    fig.suptitle(f'Argon: LJ-Potential\n' r'$by~L. Welzel~and~C. Slaughter$', fontsize=20, weight="bold")
    
    plt.savefig("MD_Argon_strict_2D.png", dpi=300, format="png", metadata=None,
        facecolor='auto', edgecolor='auto')
    plt.show()
    return

def main():
    pos = read_h5_data(Path("MD_simulation.h5"))
    mpl_strict_2d_static(pos)
    # plotly_3d_static(pos)

if __name__ == "__main__":
    main()
