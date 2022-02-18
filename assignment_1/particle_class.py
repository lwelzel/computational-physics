import numpy as np
from weakref import ref, proxy
# from scipy.spatial.distance import cdist, euclidean  # slow af
from sys import version_info
# from md_simulation_class import MolDyn

class Particle(object):
    """
    Particle Class to handle instance tracking and self-knowledge of the Particle class
    """

    # PARTICLE PROPERTIES
    particle_name = "Base_Particle"
    particle_mass = 1
    particle_internal_energy = 1
    particle_sigma = 1

    # normalization
    mass = particle_mass
    internal_energy = particle_internal_energy
    sigma = particle_sigma

    def __init__(self, sim,
                 initial_pos, initial_vel, initial_acc,
                 **kwargs):
        """

        :param n_steps:
        :param n_dim:
        :param initial_pos: shape = (3, 1)
        :param initial_vel: shape = (3, 1)
        :param initial_acc: shape = (3, 1)
        :param kwargs:
        """
        # setup
        super(Particle, self).__init__()

        # meta
        self.sim = sim
        self.__id__ = next(self.sim.idservice)
        self.__huid__ = f"{self.particle_name}{self.__id__:05}"

        # define location of the instance in the instance class array
        self.__instance_loc__ = self.sim.n_active
        self.sim.occupation[self.__instance_loc__] = self.__id__
        # save instance to simulation instance array
        # since particles have a reference to MolDyn (sim) and a MolDyn.sim has references to particles
        # we should use *weak* references to avoid memory leaks the issues should be fixed in Python >3.7 but still
        self.sim.instances[self.__instance_loc__] = self
        # mask out the particle itself so it is not double counted
        # TODO: this can be better implemented using a diagonal square bool matrix (false on diagonal)
        # TODO: MD sim class should have this and then check np.nonzero(occupation - self.id)
        self.mask = self.sim.occupation - self.__id__ == 0

        # general backend
        shape = (self.sim.n_steps, self.sim.n_dim, 1)

        # TODO: hash ID for each particle

        # each vector (position, velocity, acc)
        # is pre-initialized with n steps in n_dimensions
        # TODO: maybe we should use floats with higher precision to avoid explosions of absolute errors?
        #  np.float128 -> memory size issues ofc also not that easy to make sure we dont lose precision
        #  my machine does not provide anything over double precision
        self.pos = np.zeros(shape=shape)
        self.vel = np.zeros(shape=shape)
        self.acc = np.zeros(shape=shape)
        self.force = np.zeros(shape=shape)

        # initial values
        self.pos[0] = initial_pos
        self.vel[0] = initial_vel
        self.acc[0] = initial_acc

        # other particle properties
        # TODO: those are just placeholders at this point.
        #  It probably will eventually make sense to also have those be np.arrays that track over the simulation.
        #  Also required for smart error tracking etc.

        # parameters for positions
        # k keeps track of particles "outside" the box, if particles bound very quickly
        # (more than 127 boxes in a single tick) we are f-ed
        # or we can just replace it with a larger int (np.int128 is max I think)
        self.k = np.zeros(shape=(self.sim.n_dim, 1), dtype=np.int64)
        # convenience difference vector for positions
        self.dpos = np.zeros(shape=(self.sim.n_dim, 1))
        # convenience velocity vector
        self.v_vel = np.zeros(shape=(self.sim.n_dim, 1))

    def __repr__(self):
        return f"\nParticle Class __repr__ not implemented.\n"


    def wrap_d_vector(self):
        self.k[:] = self.dpos[:] \
                    * self.sim.box2_r  # this casts the result to int, finding in where the closest box is
        self.dpos[:] = self.dpos[:] - self.k * self.sim.box  # casting back to float must be explicit


    def get_distance_vectorA1(self, other):
        """
        Get the distance between two particles
        Class A1 algorithm - all positions are stored as absolutes
        :param other: other instance of Particle (sub-) class
        :return: distance
        """
        self.dpos[:] = other.pos[self.sim.current_step] - self.pos[self.sim.current_step]
        self.wrap_d_vector()

    def get_distance_absoluteA1(self, other):
        self.get_distance_vectorA1(other)
        # using the properties of the Einstein summation convention implementation in numpy, which is very fast
        return np.sqrt(np.einsum('ij,ij->j', self.dpos, self.dpos)), self.dpos

    def get_distance_vectorB1(self, other):
        """
        Get the distance between two particles
        Class B1 algorithm - positions and box are relative to one particle
        :param other: other instance of Particle (sub-) class
        :param other:
        :return:
        """
        # TODO: implement and check performance
        return

    def set_resulting_force(self):
        # TODO: can this masked array be defined before and not new at every step?
        #  If np.ma.array(arr, mask) stores a reference to arr instead of the thing itself that should work
        # TODO: make an interpolant for this function so it does not need to be computed at every step
        # TODO: equal an opposite reaction: compute forces just once
        for other in np.ma.array(self.sim.instances, mask=self.mask).compressed():
            force, potential = self.get_force_potential(other)
            # no idea why 0.5, maybe because of two particles?
            # Does not matter since we are only interested in the relative quantities at the moment
            self.sim.potential_energy[self.sim.current_step] += 0.5 * potential
            self.force[self.sim.current_step] = self.force[self.sim.current_step] + force

    def get_force_potential(self, other):
        dist, vector = self.get_distance_absoluteA1(other)
        return - self.force_lennard_jones(dist) * vector / dist, self.potential_lennard_jones(dist)

    def force_lennard_jones(self, r):
        sigma_r_ratio = self.__class__.sigma / r
        return - 24.0 * self.__class__.internal_energy * np.power(sigma_r_ratio, 2) \
               * (2.0 * np.power(sigma_r_ratio, 12) - np.power(sigma_r_ratio, 6))

    def potential_lennard_jones(self, r):
        sigma_r_ratio = self.__class__.sigma / r
        return 4 * self.__class__.internal_energy * (np.power(sigma_r_ratio, 12) - np.power(sigma_r_ratio, 6))

    def next_position(self):
        self.dpos[:] = self.vel[self.sim.current_step] * self.sim.current_timestep
        return

    def propagate(self):
        self.dpos[:] = self.pos[self.sim.current_step] \
                       + self.vel[self.sim.current_step] * self.sim.current_timestep
        self.wrap_d_vector()

        self.pos[self.sim.current_step + 1] = self.dpos
        self.set_resulting_force()
        self.vel[self.sim.current_step + 1] = self.vel[self.sim.current_step] \
                                                    + 1 / self.__class__.particle_mass \
                                                    * self.force[self.sim.current_step] * self.sim.current_timestep
        return

    def normalize_particle(self):
        """
        Bad name
        Normalize parameters of simulation based on
                                https://en.wikipedia.org/wiki/Lennard-Jones_potential#Dimensionless_(reduced_units)
        :sets: normalized parameters
        """
        # TODO: make this a class method which then iterates over all particles of a species
        # iterate over particles and normalize values
        self.pos *= 1 / self.__class__.sigma
        # TODO: implement equipartition, the normalization is wrong here!
        self.vel *= 1 / self.__class__.sigma * \
                    1 / np.sqrt(self.__class__.internal_energy /
                                (self.__class__.particle_mass * np.power(self.__class__.sigma, 2)))
        self.acc *= 1 / self.__class__.sigma * \
                    1 /(self.__class__.internal_energy /
                        (self.__class__.particle_mass * np.power(self.__class__.sigma, 2)))

        self.force *= self.__class__.sigma / self.__class__.internal_energy

    def reset_lap(self):
        self.pos[0] = self.pos[-1]
        self.vel[0] = self.vel[-1]
        self.force[0] = self.force[-1]









