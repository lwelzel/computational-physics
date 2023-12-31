import numpy as np
import h5py
from constants_class import Constants
# from scipy.spatial.distance import cdist, euclidean  # slow af
from sys import version_info, stdout
from tqdm import tqdm
from pathlib import Path
from time import perf_counter, strftime, gmtime
from datetime import timedelta
from itertools import count
from scipy.optimize import curve_fit
from scipy.signal import argrelextrema
from scipy.interpolate import interp1d
import matplotlib.pyplot as plt

# hacky shit for numba and fast runtimes
lj_cut_off = 2.
__force = lambda r: - 24 * (np.power(r, 6) - 2) / (np.power(r, 13))
__potential = lambda r: 4 * (np.power(r, -12) - np.power(r, -6))
lj_force_offset = __force(lj_cut_off)
lj_pot_offset = __potential(lj_cut_off)

# np.can_cast(int, float, casting='safe')
class MolDyn(object):
    """
    CLass for holding simulation objects
    """
    # class holds its own instance
    sim = None

    if version_info < (3, 9):
        raise EnvironmentError("Please update to Python 3.9 or later (code written for Python 3.9.x).\n"
                               "This is required for the proper function of MetaClasses\n"
                               "which use the double  @classmethod @property decorator.\n"
                               "We use this, pythonic approach, to keep track of our particles.")

    def __init__(self,
                 n_particles: int, n_dim: int, n_steps: int,
                 time_total, initial_timestep=0,
                 max_steps: int = int(1e6), max_real_time=3*60,
                 temperature=0.5, density=1.2,
                 file_location=Path(""),
                 name: str = ".\simulation_data",
                 id=0.,
                 **kwargs):

        super(MolDyn, self).__init__()

        print(f"Running MDS with -> Density: {density}\tTemperature: {temperature}\t#Particles: {n_particles}")

        # META PROPERTIES
        # give instance to class
        self.__class__.sim = self
        self.name = name
        self.rng = np.random.default_rng()

        # META PLOTTING
        # [Rescaling, other, other]
        self.plot_which = np.array([True, True, True])
        self.plot_init = np.array([True, True, True])

        # SIMULATION PARAMETERS
        self.n_species = int(1)
        self.n_particles = int(n_particles)
        self.n_dim = int(n_dim)
        self.n_steps = int(n_steps)
        self.max_steps = int(max_steps)
        self.time_total = time_total
        self.max_real_time = max_real_time
        self.time_total = time_total
        self.init_density = density
        # TODO: remove mass scaling factor when passing normalized positions
        self.box_length = (self.n_particles/ self.init_density)**(1/3)

        # this is the hacky-est stuff ever, but it does work
        self.lj_cut_off = lj_cut_off
        self.force = lambda r: - 24 * (np.power(r, 6) - 2) / (np.power(r, 13))
        self.potential = lambda r: 4 * (np.power(r, -12) - np.power(r, -6))
        self.lj_force_offset = self.force(self.lj_cut_off)
        self.lj_pot_offset = self.potential(self.lj_cut_off)
        self.r_samp = np.linspace(0.01, self.lj_cut_off, 1000)
        self.force = lambda r: - 24 * (np.power(r, 6) - 2) / (np.power(r, 13)) - self.lj_force_offset
        self.potential = lambda r: 4 * (np.power(r, -12) - np.power(r, -6)) - self.lj_pot_offset
        self.force = interp1d(self.r_samp, self.force(self.r_samp),
                              kind="cubic", bounds_error=False, fill_value=0., assume_sorted=True)
        self.potential = interp1d(self.r_samp, self.potential(self.r_samp),
                                  kind="cubic", bounds_error=False, fill_value=0., assume_sorted=True)

        # SAVE PARAMETERS
        self.dir_location = Path(file_location)
        self.name = f"run={strftime('%Y-%m-%d-%H-%M-%S', gmtime())}_den={density:.0e}_temp={temperature:.0e}_part={n_particles}_id={id}.h5"
        Path(self.dir_location).mkdir(parents=True, exist_ok=True)
        self.file_location = Path(file_location) / self.name
        print(f"Run will be saved to: {self.file_location}")
        self.file = h5py.File(self.file_location, "w")
        self.__h5_position_name__ = "position"
        self.__h5_velocity_name__ = "velocity"
        self.file.create_dataset(self.__h5_position_name__,
                                 compression="gzip",
                                 shape=(self.n_steps, self.n_particles, self.n_dim),
                                 chunks=(self.n_steps, self.n_particles, self.n_dim),
                                 maxshape=(self.max_steps, self.n_particles, self.n_dim),
                                 dtype=np.float64)
        self.file.create_dataset("velocity",
                                 compression="gzip",
                                 shape=(self.n_steps, self.n_particles, self.n_dim),
                                 chunks=(self.n_steps, self.n_particles, self.n_dim),
                                 maxshape=(self.max_steps, self.n_particles, self.n_dim),
                                 dtype=np.float64)
        self.__h5_stat_data_names__ = np.array(["time_steps",
                                                "scale_velocity",
                                                "temperature",
                                                "kinetic_energy",
                                                "potential_energy",
                                                "pressure",
                                                "density",
                                                "surface_tension"])
        for stat_name in self.__h5_stat_data_names__:
            self.file.create_dataset(stat_name,
                                     compression="gzip",
                                     shape=self.n_steps,
                                     chunks=self.n_steps,
                                     maxshape=self.max_steps,
                                     dtype=np.float64)

        # TODO: maybe this is a bad idea but Id like to avoid passing around objects all the time.
        #  So what I suggest is that we store the instances in the class itself (at least as references).
        #  This is probably really stupid and is gonna bite us in the butt later but for now I dont see how it could.
        #  Except if there is a lot of particles of course but then we are screwed anyway :)
        #  Has the same(-ish) problem to the one above, but is nicer

        self.__instances__ = np.full(self.n_particles, fill_value=None, dtype=object)
        self.__species__ = np.full(self.n_species, fill_value=None, dtype=object)
        # this is really bad, I should just be able to check in __n_active__ using __instances__ but its funky
        # also is the mask that is used to filter currently active particle
        self.__occupation__ = np.zeros(self.n_particles)
        self.__global_mask__ = np.tril(np.zeros((self.n_particles, self.n_particles)) == 0, k=0)
        self.idservice = count(1)

        # SIMULATION PROPERTIES
        # MIC implementation is based on discussion of MIC Boundary Conditions for C/C++
        #   U.K. Deiters, 2013: "Efficient Coding of the Minimum Image Convention". Z. Phys. Chem.
        # define box extents
        self.box = np.array([])
        self.box_r = np.array([])
        self.box2 = np.array([])
        self.box2_r = np.array([])

        # stepping
        self.sim_running = False
        self.current_step = 0  # needs to be increased each step by calling .tick() on the class
        self.current_step_total = 0
        self.current_timestep = initial_timestep
        self.time = np.float64(0.)
        self.time_steps = np.zeros(self.n_steps)
        self.start_system_time = perf_counter()

        # CONSTANTS
        self.bk = Constants.bk

        # STATISTICAL PROPERTIES
        self.scale_velocity = np.ones(self.n_steps)
        self.temperature = np.zeros(self.n_steps)
        self.kinetic_energy = np.zeros(self.n_steps)
        self.virial = np.zeros(self.n_steps)
        self.potential_energy = np.zeros(self.n_steps)
        self.pressure = np.zeros(self.n_steps)
        self.density = np.zeros(self.n_steps)
        self.surface_tension = np.zeros(self.n_steps)

        self.av_particle_mass = 1
        self.av_particle_sigma = 1
        self.av_particle_epsilon = 1

        # CONTROL PROPERTIES
        self.target_temperature = temperature
        self.target_kinetic_energy = (self.n_dim - 1) * 3 / 2 * 1 * self.target_temperature
        # relaxation
        self.state = "gaseous" if (self.target_temperature > 1. and self.init_density < 1.) else "not gaseous"
        #print(f"Conservative state of matter assumption for relaxation: Matter is {self.sim.state}.")
        self.scale = 1.
        self.scale_tracker = np.ones(3)
        self.i_scale = 0
        # hot and sparse media need much longer to relax:
        self.relaxation_steps = int(np.minimum(100 * self.target_temperature * (1 / self.init_density),
                                               self.n_steps * 0.9))

        # if the number of relaxation steps is small the relaxation_threshold should be small
        # 1% error per 1 T is acceptable if the first relaxation dip is within relaxation_steps
        self.relaxation_threshold = 0.01 * self.target_temperature  # * self.target_temperature
        self.relaxation_curve = np.zeros_like(self.kinetic_energy)
        self.relaxed = True

    def __repr__(self):
        return f"\nMolecular Dynamics Simulation\n" \
               f"\tsaved at {self.file_location}\n" \
               f"\tat time {timedelta(seconds=self.time)} ({self.time / self.time_total:.1f}% completed)\n" \
               f"\twith species: [{'NOT IMPLEMENTED'} for spec in self.__species__]\n" \
               f"\twith a total of {self.n_particles} particles\n"

    def set_up_simulation(self,
                          *args):
        """
        Sets up the molecular dynamics simulation
        :param args: n tuples of: (
        ParticleSpecies     - class
        n_particles         - int
        initial_position    - np.ndarray OR None
        initial_velocity    - np.ndarray OR None
        initial_acc         - np.ndarray OR None
        **kwargs            - dict of kwargs to pass to ParticleSpecies constructor
        :return:
        """

        self.n_species = len(args)
        self.__species__ = np.full(self.n_species, fill_value=None, dtype=object)
        
        self.setup_mic()

        # set initial properties
        for i, element in enumerate(args):
            species, n_particles, initial_positions, initial_velocities, initial_acc = element
            self.__species__[i] = species
            if initial_positions is None:
                initial_positions = np.zeros((n_particles, self.n_dim, 1))
            if initial_velocities is None:
                # including first relaxation step
                rnd_vel = self.rng.normal(0., 1., size=(n_particles, self.n_dim, 1))
                # total momentum should be zero in each axis, all particles have the same mass
                mean_vel = np.mean([vel
                                    for vel in rnd_vel], axis=0)
                initial_velocities = rnd_vel - mean_vel

                ekin = 0.5 * np.sum([np.square(np.linalg.norm(vel))
                                     for vel in initial_velocities])
                scale = np.sqrt(self.target_kinetic_energy / ekin)
                initial_velocities = initial_velocities * scale


            if initial_acc is None:
                initial_acc = np.zeros((n_particles, self.n_dim, 1))

            for initial_position, initial_velocity, initial_acc in zip(initial_positions,
                                                                       initial_velocities,
                                                                       initial_acc):
                species(self, initial_position, initial_velocity, initial_acc)

        self.setup_mic()
        self.normalize_problem()
        print('Normalization complete.')

        # rescale the problem until relaxed
        rescale = np.inf

        print(f"Rescaling problem...\n"
              f"\tAcceptable relative error in kinetic energies: {self.relaxation_threshold:.2e}\n"
              f"\tTaking {self.relaxation_steps} steps for each relaxation run.")
        i_r = 0
        while not np.allclose(rescale, 1., rtol=0., atol=self.relaxation_threshold):
            for __ in np.arange(self.relaxation_steps):
                self.tick()

            self.set_global_mask()

            rescale = self.rescale_velocity()  # resets particle history
            self.time = 0.
            self.lap_sim(scaling_lap=True, scale=rescale)
            i_r += 1
            if i_r > 20:
                fig = plt.figure("relaxation")
                ax = fig.axes[0]
                ax.set_yscale("log")
                ax.set_xlabel(r'step [-]')
                ax.set_ylabel(r'$E_{kin}$ [-]')
                ax.set_title(f'Kinetic Energy')
                ax.legend()
                ax.set_yscale("log")
                plt.savefig(fname=(self.dir_location / "rescaling_images") / f"relaxation{self.name[:-3]}.png",
                            dpi=300, format="png")

                self.relaxed = False
                break

        print(f"\tActual relative error in kinetic energies: {np.abs(rescale - 1.):.2e}")

        ### save simulation data to h5 file
        meta_dict = {"n_particles": self.n_particles,
                     "n_dim": self.n_dim,
                     "time_total": self.time_total,
                     "box_length": self.box_length,
                     "timestep": self.current_timestep,
                     "target_kinetic_energy": self.target_kinetic_energy,
                     "density": self.init_density,
                     "temperature": self.target_temperature,
                     "relaxed": self.relaxed
                     }

        # Store metadata in hdf5 file
        for k in meta_dict.keys():
            self.file.attrs[k] = meta_dict[k]


    def run(self):
        """
        Runs the simulation
        :return:
        """
        # reasoning: checking if statements is slow and we dont care if we overrun the time by a little as long as we
        # dont add unreasonable amounts of compute time hence we check only every n_steps
        # (the number of steps until data gets saved if any of our constraints are overrun)
        # if our simulation is complete, then just save the data
        # each block will be n_steps long so dont set this unreasonably high
        # this should significantly speed up our program
        pbar1 = tqdm(total=100,
                     leave=False)
        self.sim_running = True
        while True:
            for __ in np.arange(self.n_steps - 1):
                self.tick()
                pbar1.total = np.max([pbar1.total, int(self.time/self.time_total * 100)])
                pbar1.update(int(self.time/self.time_total * 100) - pbar1.n)

            # try statements are much faster than if else
            try:
                assert self.time <= self.time_total, \
                    f"\nSimulation completed.\n" \
                    f"\tSimulation will exit nominally.\n" \
                    f"\t\t Total simulated time:   {self.get_real_time(self.time):.2e} s.\n" \
                    f"\t\t      -  Time overrun:   {(self.time/self.time_total - 1) * 100:.2f}% \n" \
                    f"\t\t      - Overrun saved:   {True}\n" \
                    f"\t\t Total steps completed:  {self.current_step_total + 1:.1e}" \
                    f" ({((self.current_step_total + 1)/self.max_steps) * 100:.2f}% of max)\n" \
                    f"\t\t Total energy error:     {'Not Implemented'} J ({'Not Implemented'}%)\n" \
                    f"\t\t        - Potential:     {'Not Implemented'} J \n" \
                    f"\t\t        -   Kinetic:     {'Not Implemented'} J \n" \
                    f"\t\t Temperature error:      {'Not Implemented'} K ({'Not Implemented'}%) \n"
                assert self.start_system_time - perf_counter() <= self.max_real_time, \
                    f"\nMaximum allowed simulation time was exceeded.\n" \
                    f"\tSimulation will save current progress and exit.\n" \
                    f"\t\tCurrent simulation time: {timedelta(seconds=self.get_real_time(self.time))}\n" \
                    f"\t\tWanted simulation time: {timedelta(seconds=self.get_real_time(self.time_total))}\n" \
                    f"\t\tRemaining simulation time: {timedelta(seconds=self.get_real_time(self.time_total - self.time))} " \
                    f"({self.time / self.time_total:.1f}% completed)\n"
                assert self.current_step_total <= self.max_steps,\
                    "\nMaximum allowed steps were exceeded.\n " \
                    f"\tSimulation will save current progress and exit.\n" \
                    f"\t\tCurrent simulation time: {timedelta(seconds=self.get_real_time(self.time))}\n" \
                    f"\t\tWanted simulation time: {timedelta(seconds=self.get_real_time(self.time_total))}\n" \
                    f"\t\tRemaining simulation time: {timedelta(seconds=self.get_real_time(self.time_total - self.time))} " \
                    f"({self.time / self.time_total:.1f}% completed)\n"
            except AssertionError as e:
                print(e)
                self.sim_running = False
                break
            finally:
                self.save_block()
                self.lap_sim()


        self.file.close()
        plt.close('all')
        print(f"Done.\n\t Total runtime: {timedelta(seconds=perf_counter() - self.start_system_time)}")
    
    def tick(self):
        """
        Moves the simulation forward by one step
        :return:
        """
        # redo the SOI every ten steps
        if self.current_step % 10 == 0:
            self.set_global_mask()

        # TODO: should only iterate over n/2 of the particles and set equal but opposite force
        for particle in self.instances:
            particle.propagate_pos()

        for particle in self.instances:
            particle.propagate_vel()

        # UPDATE STATISTICAL PROPERTIES
        self.kinetic_energy[self.current_step] = self.get_ekin()

        self.time += self.current_timestep
        self.current_step += 1
        self.current_step_total += 1

    def save_block(self):
        """
        Save simulation block (length n_steps)
        :return:
        """
        # TODO: progress bars and stuff, see enlighten
        #  https://stackoverflow.com/questions/23113494/double-progress-bar-in-python
        # shape     = (max_steps,   n_particles,    n_dim)
        # chunks    = (n_steps,     n_particles,    n_dim)
        # save particle positions
        # print("Saving simulation progress (particles)")
        for i, particle in enumerate(tqdm(self.instances,
                                          leave=False)):
            self.file[self.__h5_position_name__][-self.n_steps:, i] = particle.pos.reshape((self.n_steps, self.n_dim))
            self.file[self.__h5_velocity_name__][-self.n_steps:, i] = particle.vel.reshape((self.n_steps, self.n_dim))

        # prepare potential for virial for pressure
        # multiply by 0.5^2 because we iterate over all pairs (i.e. all pairs get counted twice)
        #self.potential_energy = self.potential_energy * 0.5 ** 2
        # pressure in dimensionless units
        # assume that we preserve the target temperature during the simulation
        self.pressure = 1 - 1 / (3 * self.n_dim * 1 * self.target_temperature) * self.potential_energy

        # save statistical properties
        # print("Saving simulation progress (statistics)")
        for i, stat in enumerate(tqdm([self.time_steps,
                                       self.scale_velocity,
                                       self.temperature,
                                       self.kinetic_energy,
                                       self.potential_energy,
                                       self.pressure,
                                       self.density,
                                       self.surface_tension],
                                      leave=False)):
            self.file[self.__h5_stat_data_names__[i]][-self.n_steps:] = stat
            n = i

        if self.sim_running:
            self.file[self.__h5_position_name__].resize((self.file[self.__h5_position_name__].shape[0] + self.n_steps), axis=0)
            self.file[self.__h5_velocity_name__].resize((self.file[self.__h5_velocity_name__].shape[0] + self.n_steps), axis=0)
            for i in np.arange(n):
                self.file[self.__h5_stat_data_names__[i]].resize(
                    (self.file[self.__h5_stat_data_names__[i]].shape[0] + self.n_steps), axis=0)

    def setup_mic(self):
        """
        Sets up the bounding box of the simulation and constructs MCI helpers/enforcers
        needs to be re-called when adjusting box_length
        :return:
        """
        self.box = np.ones(shape=(self.n_dim, 1)) * self.box_length
        self.box_r = 1 / self.box
        self.box2 = 0.5 * self.box
        self.box2_r = 1 / self.box2

    @property
    def n_active(self):
        """
        Method to find the number of currently active instance.
        When creating a new instance the method is used to set the position of the instance in the instance class array
        :return: Number of active instances
        """
        return np.count_nonzero(self.occupation)

    @property
    def instances(self):
        return self.__instances__

    @property
    def occupation(self):
        return self.__occupation__

    def get_real_time(self, normalized_time):
        return normalized_time / np.sqrt(self.av_particle_epsilon /
                                         (self.av_particle_mass * np.power(self.av_particle_sigma, 2)))

    def normalize_problem(self):
        """
        Normalize parameters of simulation based on
        https://en.wikipedia.org/wiki/Lennard-Jones_potential#Dimensionless_(reduced_units)
        :sets: normalized parameters
        """
        for particle in self.instances:
            particle.normalize_particle()

        # TODO: this should be in a class method, like this redundant class
        for species in self.__species__:
            species.mass = 1.
            species.sigma = 1.
            species.internal_energy = 1.
            species.bk = 1.

        self.setup_mic()
        for particle in self.instances:
            particle.set_resulting_force()

    def get_ekin(self):
        """
        This function calculates kinetic energy of the system at the current step
        :return:
        """
        # TODO: Use einsteinsum notation for speeeed
        # np.einsum('ij,ij->j',
        #           particle.vel[self.current_step],
        #           particle.vel[self.current_step])
        return 0.5 * np.sum([particle.mass * np.sum(np.square((particle.vel[self.current_step])))
                             for particle in self.__instances__])

    @staticmethod
    def sine_curve(x, a, b, c, d):
        """
        Arbitrary sine function wrapper for fitting
        """
        return a * np.sin(b * x + c) + d

    def rescale_velocity(self, rescale=None):
        """
        Rescales the velocity of the system and estimates how good the relaxations state is.
        It relies on extrapolating the systems kinetic energy to get the bounds
        within which the final equilibrium kinetic energy is
        :param rescale:
        :return:
        """
        try:
            step_last_relaxation_dip = argrelextrema(self.kinetic_energy[:self.current_step], np.less)[0][-1]
            step_last_relaxation_peak = argrelextrema(self.kinetic_energy[:self.current_step], np.greater)[0][-1]
            step_diff = np.abs(step_last_relaxation_peak - step_last_relaxation_dip)

            start, stop = np.sort([step_last_relaxation_dip, step_last_relaxation_peak])

            start, stop = start - step_diff * 0.2, stop + step_diff * 0.2
            start, stop = int(np.maximum(start, 0)), int(np.minimum(stop, self.current_step))


        except IndexError:
            start, stop = 0, self.current_step

        p0 = [(np.max(self.kinetic_energy[start:stop])
               - np.min(self.kinetic_energy[start:stop])) / 2,
              1 / np.abs(np.argmin(self.kinetic_energy[start:stop])
                         - np.argmax(self.kinetic_energy[start:stop])),
              np.argmax(np.abs(self.kinetic_energy[start:stop])) * 2.,
              np.mean(self.kinetic_energy[start:stop])]

        try:
            para, __ = curve_fit(self.sine_curve,
                                 np.arange(len(self.kinetic_energy[start:stop]), dtype=float),
                                 self.kinetic_energy[start:stop],
                                 p0=p0)
            ekin = para[-1]

        except (RuntimeError, TypeError):
            ekin = np.mean(self.kinetic_energy[start:stop])
            para = [1./1000, 1./1000, 0, ekin]

        if ekin < 0:
            ekin = np.mean(self.kinetic_energy[start:stop])

        scale = np.sqrt(self.target_kinetic_energy / ekin)

        np.put(self.scale_tracker, ind=self.i_scale, v=scale, mode="wrap")


        if ((((np.argmin(self.scale_tracker)) == 1 or (np.argmax(self.scale_tracker) == 1))
                and (self.i_scale >= 2))
                and (self.state != "gaseous")):
            self.scale *= np.mean(self.scale_tracker[1:])
            print(f"\tDetecting oscillation in rescaling, converge to mean. (i = {self.i_scale})")
        else:
            self.scale *= scale

        self.i_scale += 1

        # some plotting for the relaxation
        if self.plot_which[0]:
            if self.plot_init[0]:
                fig, ax = plt.subplots(nrows=1, ncols=1,
                                       constrained_layout=True,
                                       figsize=(7, 5),
                                       num="relaxation")
                self.plot_init[0] = False

            alpha = np.clip(1. / (1. + np.abs(self.target_kinetic_energy - ekin)), a_min=0.1, a_max=1.)

            overshoot = 100
            estimated = self.sine_curve(np.arange(len(self.kinetic_energy[start:stop]) + overshoot),
                                        *para)

            fig = plt.figure("relaxation")
            ax = fig.axes[0]
            ax.set_yscale("log")

            if np.allclose(scale, 1., rtol=0., atol=self.relaxation_threshold):
                ax.plot(self.kinetic_energy[1:self.current_step], c="black", label="Actual", alpha=alpha)
                ax.plot(np.arange(start, stop + overshoot), estimated, c="red", label="Estimated", ls="dashed", alpha=alpha)
                ax.axhline(self.target_kinetic_energy, ls="solid", c="green", label="Target", alpha=alpha)
                ax.axhline(para[-1], ls="dashed", c="gray", label="Rescaling Energy", alpha=alpha)
                ax.axvline(start, ls="dotted", c="magenta", label="Fit Start", alpha=alpha)
                ax.axvline(stop, ls="dashed", c="pink", label="Fit Stop", alpha=alpha)

                ax.set_xlabel(r'step [-]')
                ax.set_ylabel(r'$E_{kin}$ [-]')
                ax.set_title(f'Kinetic Energy')
                ax.legend()
                ax.set_yscale("log")
                plt.savefig(fname=(self.dir_location / "rescaling_images") / f"relaxation{self.name[:-3]}.png", dpi=300, format="png")
            else:
                ax.plot(self.kinetic_energy[1:self.current_step], c="black", alpha=alpha)
                ax.plot(np.arange(start, stop + overshoot), estimated, c="red", ls="dashed", alpha=alpha)
                ax.axhline(para[-1], ls="dashed", c="gray", alpha=alpha)
                ax.axvline(start, ls="dotted", c="magenta", alpha=alpha)
                ax.axvline(stop, ls="dashed", c="pink", alpha=alpha)

        print(f"\t\tRescaling:\tlambda: {self.scale :.5e}\ttarget Ekin: {self.target_kinetic_energy:.2e}\tEkin: {ekin:.2e}")
        return scale

    def lap_sim(self, scaling_lap=False, scale=np.nan):
        """
        Resets the system in preperation for doing another lap.
        This makes it possible to run very large simulations with little RAM
        since it save the progress to the disk every so often
        :param scaling_lap: True if the simulation proper hasnt started yet
        :param scale: The current scaling that has been applied
        :return:
        """
        if scaling_lap:
            for particle in self.instances:
                particle.reset_scaling_lap()
                if self.state == "gaseous":
                    particle.vel *= scale
                else:
                    particle.vel *= self.scale
        else:
            for particle in self.instances:
                particle.reset_lap()


        self.scale_velocity = np.ones(self.n_steps)
        self.temperature = np.zeros(self.n_steps)
        self.kinetic_energy = np.zeros(self.n_steps)
        self.virial = np.zeros(self.n_steps)
        self.potential_energy = np.zeros(self.n_steps)
        self.pressure = np.zeros(self.n_steps)
        self.density = np.zeros(self.n_steps)
        self.surface_tension = np.zeros(self.n_steps)

        self.current_step = 0
        for particle in self.instances:
            particle.set_resulting_force()

    def set_global_mask(self):
        """
        I love this one
        Sets the global mask for the particle system that is used for the pairwise interactions.
        This is done by calculating the absolute distances between all particles and setting the mask
        to 1 if the absolute distance is greater than the set cut-off.
        With the mask, we can exclude interactions between particles that are
        too far away from each other to interact in a significant way.
        Note that this is only done for the upper triangle of the global mask matrix.
        This is because the force and potential is symmetric
        :return:
        """
        self.__global_mask__ = (self.lj_cut_off + 1.) * np.ones((self.n_particles, self.n_particles))
        for i, part1 in enumerate(self.instances):
            for j, part2 in enumerate(self.instances):
                self.__global_mask__[i, j], __ = part1.get_distance_absoluteA1(part2)

        self.__global_mask__ = np.triu(self.__global_mask__)
        self.__global_mask__[np.logical_and(self.__global_mask__ > self.lj_cut_off, self.__global_mask__ != 0)] = 0
        self.__global_mask__ = self.__global_mask__ == 0
        # number of interactions saved
        # print(np.sum(self.__global_mask__) - np.sum(np.triu(np.ones_like(self.__global_mask__))))

    # TODO: NUMBA THIS
    # TODO: Do this: https://stackoverflow.com/questions/41769100/how-do-i-use-numba-on-a-member-function-of-a-class
    @staticmethod
    def force_lennard_jones(r):
        return - 24 * (np.power(r, 6) - 2) / (np.power(r, 13)) - lj_force_offset

    @staticmethod
    def potential_lennard_jones(r):
        return 4 * (np.power(r, -12) - np.power(r, -6)) - lj_pot_offset

    @staticmethod
    def get_distance_absoluteA1(pos1, pos2, box2_r, box):
        dpos = pos1 - pos2
        k = int(dpos * box2_r)  # this casts the result to int, finding in where the closest box is
        dpos = dpos - k * box  # casting back to float must be explicit

        # using the properties of the Einstein summation convention implementation in numpy, which is very fast
        # if numba doesnt have this then we can use something else as well
        return np.sqrt(np.einsum('ij,ij->j', dpos, dpos)), dpos

    @staticmethod
    def get_force_potential(pos1, pos2, box2_r, box):
        """
        Calculates the Lennard-Jones potential and force between two particles in a 3 tourus. This can be jitted

        Parameters:
        pos1 (ndarray): an array of shape (N,3) containing the coordinates of the first particle
        pos2 (ndarray): an array of shape (M,3) containing the coordinates of the second particle
        box2_r (ndarray): an array of shape (M,3) containing the reciprocal of the half box dimensions
        box (ndarray): an array of shape (3,) containing the box dimensions

        Returns:
        force (ndarray): an array of shape (N,M,3) containing the forces on the first particles, due to their interaction with the other particles
        potential (ndarray): an array of shape (N,M) containing the potential energy of the first particles, due to their interaction with the other particles
        """
        dpos = pos1 - pos2
        k = (dpos * box2_r).astype(int)  # this casts the result to int, finding in where the closest box is
        dpos = dpos - k * box  # casting back to float must be explicit
        dist = np.sqrt(np.einsum('ij,ij->j', dpos, dpos))

        force = - 24 * (np.power(dist, 6) - 2) / (np.power(dist, 13)) - lj_force_offset
        potential = 4 * (np.power(dist, -12) - np.power(dist, -6)) - lj_pot_offset

        return force * dpos / dist, potential, force * dist





