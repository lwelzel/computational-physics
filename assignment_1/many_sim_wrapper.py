import numpy as np
import pathos.multiprocessing as mp
from md_simulation_func import main

# """
# conda activate compphys
# cd C:\Users\lukas\Documents\Git\computational-physics\assignment_1
# python many_sim_wrapper.py
# """

### IF YOU RUN THIS WITHOUT CARE IT WILL MELT YOUR CPU


def gen_many_runs():
    rng = np.random.default_rng()
    n_cpus = int(mp.cpu_count() * 0.9)
    pool = mp.Pool(n_cpus)
    print(f"CPUs active: {n_cpus}")

    n_runs_per_it = int(n_cpus / 3.1)
    n_runs = n_runs_per_it * 3  # must be multiple of 3

    print(f"Parallel runs: {n_runs}")

    n_particles = 3 ** 3 * 4 * np.ones(n_runs).astype(int)
    n_dim = 3 * np.ones(n_runs).astype(int)
    n_steps = 2510 * np.ones(n_runs).astype(int)
    time_total = 5.e0 * np.ones(n_runs)
    initial_timestep = 2.e-3 * np.ones(n_runs)
    max_steps = 1e6 * np.ones(n_runs).astype(int)
    max_real_time = 3 * 60 * np.ones(n_runs)
    density = np.repeat([1.2, 0.8, 0.3], n_runs_per_it)
    temperature = np.repeat([0.5, 1., 3.], n_runs_per_it)
    id = np.arange(0, n_runs)

    i = 0
    while True:
        __ = pool.starmap(main, zip(n_particles,
                                    n_dim,
                                    n_steps,
                                    time_total,
                                    initial_timestep,
                                    max_steps,
                                    max_real_time,
                                    density,
                                    temperature,
                                    id))

        print(f"---=== Outer Run {i} | Total Runs {n_runs * i} ===---")
        i += 1
        if i > 10:
            break

    pool.close()
    pool.join()

    return


if __name__ == "__main__":
    gen_many_runs()
