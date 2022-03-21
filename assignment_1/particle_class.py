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

        # INTEGRATOR
        self.propagate = self.propagate_Verlet

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
        self.initial_pos = initial_pos
        self.initial_vel = initial_vel
        self.initial_acc = initial_acc

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
        self.dpos[:] = (self.dpos + self.sim.box2) % self.sim.box - self.sim.box2

    def get_distance_vectorA1(self, other, future_step=0):
        """
        Get the distance between two particles
        Class A1 algorithm - all positions are stored as absolutes
        :param other: other instance of Particle (sub-) class
        :return: distance
        """
        self.dpos[:] = other.pos[self.sim.current_step + future_step] - self.pos[self.sim.current_step + future_step]
        self.wrap_d_vector()

    def get_distance_absoluteA1(self, other, future_step=0):
        self.get_distance_vectorA1(other, future_step)
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

    def set_resulting_force(self, future_step=0):
        # TODO: can this masked array be defined before and not new at every step?
        #  If np.ma.array(arr, mask) stores a reference to arr instead of the thing itself that should work
        # TODO: make an interpolant for this function so it does not need to be computed at every step
        # TODO: equal an opposite reaction: compute forces just once
        # print(self.__id__)
        # print(self.force[self.sim.current_step + future_step])
        for other in np.ma.array(self.sim.instances, mask=self.mask).compressed():
            force, potential = self.get_force_potential(other, future_step)
            # TODO: half it here because it evaluates it twice for every particle pair
            self.sim.potential_energy[self.sim.current_step + future_step] += potential
            self.force[self.sim.current_step + future_step] = self.force[self.sim.current_step + future_step] + force
        #     if np.sum(np.abs(force)) > 0.:
        #         print(self.__id__, other.__id__)
        #         print(f"Force:     {np.sqrt(np.sum(np.square(force))).astype(float):.3e}")
        #         print(f"Potential: {potential[0]:.3e}")
        #         print(f"Distance:  {np.sqrt(np.sum(np.square(self.dpos))).astype(float):.3e}")
        #
        # # print(self.force[self.sim.current_step + future_step])
        # print()
        # print()

    def get_force_potential(self, other, future_step=0):
        dist, vector = self.get_distance_absoluteA1(other, future_step)
        return - self.force_lennard_jones(dist) * vector / dist, self.potential_lennard_jones(dist)

    def force_lennard_jones(self, r):
        sigma_r_ratio = self.__class__.sigma / r
        return 24.0 * self.__class__.internal_energy * np.power(sigma_r_ratio, 2) \
               * (2.0 * np.power(sigma_r_ratio, 12) - np.power(sigma_r_ratio, 6))

    def potential_lennard_jones(self, r):
        sigma_r_ratio = self.__class__.sigma / r
        return 4 * self.__class__.internal_energy * (np.power(sigma_r_ratio, 12) - np.power(sigma_r_ratio, 6))

    def propagate_explicit_euler(self):
        """"
        I am actually not sure if this is still correct.
        """
        self.dpos[:] = self.pos[self.sim.current_step] \
                       + self.vel[self.sim.current_step] * self.sim.current_timestep
        self.wrap_d_vector()
        self.set_resulting_force()
        self.pos[self.sim.current_step + 1] = self.dpos
        self.vel[self.sim.current_step + 1] = self.vel[self.sim.current_step] \
                                              + self.sim.current_timestep / (2 * self.__class__.mass) \
                                              * (self.force[self.sim.current_step])
        return

    def propagate_SABA(self):
        """
        Adapted from:
        H. Rein, D. Tamayo: 'JANUS: A bit-wise reversible integrator for N-body dynamics'. (2017)
        and
        H. Rein, D. Tamayo, G. Brown:
        'High order symplectic integrators for planetary dynamics and their implementation in REBOUND'. (2019)
        - formally and practically symplectic (integar operations are bijective)
        - satisfies Liouville’s theorem
        - exactly time-reversible
        :return:
        """
        pass
        return

    def propagate_WHCKL(self):
        """
        :return:
        """

        pass
        return

    def propagate_CC4A(self):
        """
        Adapted from:
        .S. Chin, C. Chen: Forward Symplectic Integrators for Solving Gravitational Few-Body Problems'. (2003)
        :return:
        """
        f, p = None, None
        vn13 = self.vel[self.sim.current_step] + self.sim.current_timestep / 6 * 1
        return

    def propagate_Verlet(self):
        """

        :return:
        """

        self.dpos[:] = self.pos[self.sim.current_step] \
                       + self.vel[self.sim.current_step] * self.sim.current_timestep \
                       + self.force[self.sim.current_step] * np.square(self.sim.current_timestep) \
                       / (2 * self.__class__.mass)

        self.wrap_d_vector()
        self.pos[self.sim.current_step + 1] = self.dpos

        self.set_resulting_force(future_step=1)

        self.vel[self.sim.current_step + 1] = self.vel[self.sim.current_step] \
                                              + self.sim.current_timestep / (2 * self.__class__.mass) \
                                              * (self.force[self.sim.current_step]
                                                 + self.force[self.sim.current_step + 1])

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
        # TODO: remove when passing normalized box length/ positions
        # self.pos *= 1 / self.__class__.sigma

        # self.vel *= 1 / np.sqrt(self.__class__.internal_energy /
        #                         (self.__class__.particle_mass))
        self.acc *= 1 / self.__class__.sigma * \
                    1 / (self.__class__.internal_energy /
                         (self.__class__.particle_mass * np.power(self.__class__.sigma, 2)))

        self.force *= self.__class__.sigma / self.__class__.internal_energy

        # self.set_resulting_force()


    def reset_lap(self):
        self.pos[0] = self.pos[self.sim.current_step]
        self.vel[0] = self.vel[self.sim.current_step]
        self.force[0] = self.force[self.sim.current_step]
        self.acc[0] = self.acc[self.sim.current_step]

        zeros = np.zeros_like(self.pos[1:])
        self.pos[1:] = zeros
        self.vel[1:] = zeros
        self.force[1:] = zeros
        self.acc[1:] = zeros

    def reset_scaling_lap(self):
        self.pos[0] = self.pos[self.sim.current_step, :]  # self.initial_pos
        self.vel[0] = self.vel[self.sim.current_step, :]

        zeros = np.zeros_like(self.pos)
        self.pos[1:] = zeros[1:]
        self.vel[1:] = zeros[1:]
        self.force = zeros
        self.acc = zeros

    @classmethod
    def normalize_class(cls):
        cls.mass = 1.
        cls.internal_energy = 1.
        cls.sigma = 1.

    @property
    def position(self):
        return (self.pos[self.sim.current_step])

