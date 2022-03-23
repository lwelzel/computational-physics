import numpy as np
import h5py
from constants_class import Constants
# from scipy.spatial.distance import cdist, euclidean  # slow af
from sys import version_info, stdout
from tqdm import tqdm
from pathlib import Path
from time import perf_counter
from datetime import timedelta
from itertools import count
from scipy.optimize import curve_fit
from scipy.signal import argrelextrema
import matplotlib.pyplot as plt

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
                 name: str = "MD_simulation",
                 **kwargs):

        super(MolDyn, self).__init__()

        # META PROPERTIES
        # give instance to class
        self.__class__.sim = self
        self.name = name

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

        # SAVE PARAMETERS
        self.file_location = Path(file_location) / (name + ".h5")
        self.file_location = self.file_location
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
        # 0 when unoccupied, int when occupied
        self.__occupation__ = np.zeros(self.n_particles)
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
        # TODO: put simulation stuff in its own class - but *think* first!
        #  issues:
        #  - keeping track of instances
        #  - instances referring to simulation para

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
        # TODO: implement way to update (some)
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
        self.scale = 1.
        self.relaxation_steps = 100
        # if the number of relaxation steps is small the relaxation_threshold should be small
        # 1% error is acceptable if the first relaxation dip is within relaxation_steps
        self.relaxation_threshold = 0.01
        self.relaxation_curve = np.zeros_like(self.kinetic_energy)

    def __repr__(self):
        return f"\nMolecular Dynamics Simulation\n" \
               f"\tsaved at {self.file_location}\n" \
               f"\tat time {timedelta(seconds=self.time)} ({self.time / self.time_total:.1f}% completed)\n" \
               f"\twith species: {[print(f'{spec.name}, ') for spec in self.__species__]}\n" \
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

        for i, element in enumerate(args):
            species, n_particles, initial_positions, initial_velocities, initial_acc = element
            self.__species__[i] = species
            if initial_positions is None:
                initial_positions = np.zeros((n_particles, self.n_dim, 1))
            if initial_velocities is None:
                initial_velocities = np.zeros((n_particles, self.n_dim, 1))
            if initial_acc is None:
                initial_acc = np.zeros((n_particles, self.n_dim, 1))

            for initial_position, initial_velocity, initial_acc in zip(initial_positions,
                                                                       initial_velocities,
                                                                       initial_acc):
                species(self, initial_position, initial_velocity, initial_acc)

        # check if both self.__species__ and self.__instances__ are full (no None)
        # TODO: check if needed and stable
        self.setup_mic()
        self.normalize_problem()
        print('Normalization complete.')

        # rescale the problem until relaxed
        rescale = np.inf

        print(f"Rescaling problem...\n"
              f"\tAcceptable relative error in kinetic energies: {self.relaxation_threshold:.2e}")
        while not np.allclose(rescale, 1., rtol=0., atol=self.relaxation_threshold):
            for __ in np.arange(self.relaxation_steps):
                self.tick()

            rescale = self.rescale_velocity()  # resets particle history
            self.current_step = 0



        print(f"\tActual relative error in kinetic energies: {np.abs(rescale - 1.):.2e}")

        ### save simulation data to h5 file
        meta_dict = {"n_particles": self.n_particles,
                     "n_dim": self.n_dim,
                     "time_total": self.time_total,
                     "box_length": self.box_length,
                     "timestep": self.current_timestep,
                     "target_kinetic_energy": self.target_kinetic_energy
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
                # TODO: if below block is commented out and in reset_lap() the initial values are reset
                #  we get a measure for the stability of the program
                for particle in self.instances:
                    particle.reset_lap()
                self.current_step = 0
                # TODO: set up new iteration using last vales as initial values

        print(f"Done.\n\t Total runtime: {timedelta(seconds=perf_counter() - self.start_system_time)}")

    def tick(self):
        """
        Moves the simulation forward by one step
        :return:
        """
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
        # self.potential_energy = self.potential_energy * 0.5 ** 2
        # pressure in dimensionless units
        # assume that we preserve the target temperature during the simulation
        self.pressure = 1 - 1 / (3 * self.n_dim * 1 * self.target_temperature) * self.potential_energy * 0.5 ** 2

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
        Bad name
        Normalize parameters of simulation based on
                                https://en.wikipedia.org/wiki/Lennard-Jones_potential#Dimensionless_(reduced_units)
        :sets: normalized parameters
        """
        
        
        self.av_particle_mass = np.mean([particle.particle_mass for particle in self.instances])
        self.av_particle_sigma = np.mean([particle.sigma for particle in self.instances])
        self.av_particle_epsilon = np.mean([particle.internal_energy for particle in self.instances])

        # bk = Constants.bk
        # TODO: remove when passing normalized box length
        # self.box_length *= 1 / self.av_particle_sigma
        
        # self.current_timestep *= np.sqrt(self.av_particle_epsilon /
        #                                  (self.av_particle_mass * np.power(self.av_particle_sigma, 2)))  # average particle mass
        #
        # self.time_total *= np.sqrt(self.av_particle_epsilon /
        #                            (self.av_particle_mass * np.power(self.av_particle_sigma, 2)))

        self.temperature *= Constants.bk / self.av_particle_epsilon
        self.potential_energy *= 1 / self.av_particle_epsilon
        self.kinetic_energy *= 1 / self.av_particle_epsilon
        self.pressure *= np.power(self.av_particle_sigma, 3) / self.av_particle_epsilon
        self.density *= np.power(self.av_particle_sigma, self.n_dim)
        self.surface_tension *= np.power(self.av_particle_sigma, 2) / self.av_particle_epsilon

        # iterate over particles and normalize values
        # for type (what a shitty name) in species:

        for spec in self.__species__:
            spec.normalize_class()


        for particle in self.instances:
            particle.normalize_particle()



        # TODO: this should be in a class method, like this redundant class
        for species in self.__species__:
            species.mass = 1.
            species.sigma = 1.
            species.internal_energy = 1.
            species.bk = 1.

        self.setup_mic()

    def track_properties(self):
        # TODO: track temperature
        # TODO: track energy
        return

    def get_ekin(self):
        # np.einsum('ij,ij->j',
        #           particle.vel[self.current_step],
        #           particle.vel[self.current_step])
        l = np.array([particle.mass * np.square(np.linalg.norm(particle.vel[self.current_step]))
                             for particle in self.__instances__])
        # print()
        # print(f"\t\t\t\tmean ekin: {np.mean(l):.2e} \t median ekin: {np.median(l):.2e}\n"
        #       f"\t\t\t\tmin ekin: {np.min(l):.2e} \t max ekin: {np.max(l):.2e}")
        # print()
        return 0.5 * np.sum([particle.mass * np.square(np.linalg.norm(particle.vel[self.current_step]))
                             for particle in self.__instances__])

    @staticmethod
    def sine_curve(x, a, b, c, d):
        return a * np.sin(b * x + c) + d

    def rescale_velocity(self):
        try:
            step_relaxation_dip = np.minimum(int(argrelextrema(self.kinetic_energy[:self.current_step], np.less)[0][0] * 1.1),
                                             self.current_step)
        except IndexError:
            step_relaxation_dip = self.current_step

        p0 = [(np.max(self.kinetic_energy[:step_relaxation_dip])
               - np.min(self.kinetic_energy[:step_relaxation_dip])) / 2,
              1 / np.abs(np.argmin(self.kinetic_energy[:step_relaxation_dip])
                         - np.argmax(self.kinetic_energy[:step_relaxation_dip])),
              np.argmax(np.abs(self.kinetic_energy[:step_relaxation_dip])) * 2.,
              np.mean(self.kinetic_energy[:step_relaxation_dip])]

        para, __ = curve_fit(self.sine_curve,
                             np.arange(len(self.kinetic_energy[:step_relaxation_dip]), dtype=float),
                             self.kinetic_energy[:step_relaxation_dip],
                             p0=p0)


        ekin = -1. # para[-1]  # self.get_ekin()
        if ekin < 0:
            ekin = np.mean(self.kinetic_energy[:self.current_step])

        scale = np.sqrt(self.target_kinetic_energy / ekin)
        self.scale *= scale

        if self.plot_which[0]:
            if self.plot_init[0]:
                fig, ax = plt.subplots(nrows=1, ncols=1,
                                       constrained_layout=True,
                                       figsize=(7, 5),
                                       num="relaxation")
                self.plot_init[0] = False

            alpha = np.clip(1. / (1. + 10 * np.abs(self.target_kinetic_energy - ekin)) ** 2, a_min=0.1, a_max=1.)

            estimated = self.sine_curve(np.arange(len(self.kinetic_energy[:self.current_step]) + 100),
                                        *para)

            fig = plt.figure("relaxation")
            ax = fig.axes[0]
            ax.set_yscale("log")

            if np.allclose(scale, 1., rtol=0., atol=self.relaxation_threshold):
                ax.plot(self.kinetic_energy[:self.current_step], c="black", label="Actual", alpha=alpha)
                ax.plot(estimated, c="red", label="Estimated", ls="dashed", alpha=alpha)
                ax.axhline(self.target_kinetic_energy, ls="solid", c="green", label="Target", alpha=alpha)
                ax.axhline(para[-1], ls="dashed", c="gray", label="Rescaling Energy", alpha=alpha)
                ax.axvline(step_relaxation_dip, ls="dotted", c="magenta", label="Relaxation Dip", alpha=alpha)

                ax.set_xlabel(r'step [-]')
                ax.set_ylabel(r'$E_{kin}$ [-]')
                ax.set_title(f'Kinetic Energy')
                ax.legend()
                ax.set_yscale("log")
                plt.show()
            else:
                ax.plot(self.kinetic_energy[:self.current_step], c="black", alpha=alpha)
                ax.plot(estimated, c="red", ls="dashed", alpha=alpha)
                ax.axhline(para[-1], ls="dashed", c="gray", alpha=alpha)
                ax.axvline(step_relaxation_dip, ls="dotted", c="magenta", alpha=alpha)

            # ax.set_xlabel(r'step [-]')
            # ax.set_ylabel(r'$E_{kin}$ [-]')
            # ax.set_title(f'Kinetic Energy')
            # ax.legend()
            # ax.set_yscale("log")
            # plt.show()

        print(f"\t\tRescaling:\tlambda: {self.scale :.5e}\ttarget Ekin: {self.target_kinetic_energy:.2e}\tEkin: {ekin:.2e}")
        for particle in self.__instances__:
            particle.reset_scaling_lap()
            particle.vel *= self.scale
        return scale

