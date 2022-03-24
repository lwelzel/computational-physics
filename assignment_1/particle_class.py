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
        # mask out the particle itself, so it is not double counted
        self.mask = np.ma.array(self.sim.instances, mask=self.sim.__global_mask__[self.__id__-1]).compressed()

        # INTEGRATOR
        self.propagate_pos = self.propagate_Verlet_pos
        self.propagate_vel = self.propagate_Verlet_vel

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
        self.k[:] = self.dpos[:] \
                    * self.sim.box2_r  # this casts the result to int, finding in where the closest box is
        self.dpos[:] = self.dpos[:] - self.k * self.sim.box  # casting back to float must be explicit

    def get_distance_vectorA1(self, other, future_step=0):
        """
        Get the distance between two particles
        Class A1 algorithm - all positions are stored as absolutes
        :param other: other instance of Particle (sub-) class
        :return: distance
        """
        self.dpos[:] = self.pos[self.sim.current_step + future_step] - other.pos[self.sim.current_step + future_step]
        self.wrap_d_vector()

    def get_distance_absoluteA1(self, other, future_step=0):
        self.get_distance_vectorA1(other, future_step)
        # using the properties of the Einstein summation convention implementation in numpy, which is very fast
        return np.sqrt(np.einsum('ij,ij->j', self.dpos, self.dpos)), self.dpos

    def set_resulting_force(self, future_step=0):
        for other in np.ma.array(self.sim.instances, mask=self.sim.__global_mask__[self.__id__-1]).compressed():
            force, potential, pressure_term = self.sim.get_force_potential(self.pos[self.sim.current_step + future_step],
                                                                           other.pos[self.sim.current_step + future_step],
                                                                           self.sim.box2_r, self.sim.box)

            self.sim.potential_energy[self.sim.current_step + future_step] += potential
            self.sim.pressure[self.sim.current_step] += pressure_term
            self.force[self.sim.current_step + future_step] = self.force[self.sim.current_step + future_step] + force
            other.force[self.sim.current_step + future_step] = other.force[self.sim.current_step + future_step] - force



    def get_force_potential(self, other, future_step=0):
        dist, vector = self.get_distance_absoluteA1(other, future_step)
        return self.force_lennard_jones(dist) * vector / dist, self.potential_lennard_jones(dist), self.force_lennard_jones(dist) * dist

    def force_lennard_jones(self, r):
        return - 24*(np.power(r,6)-2)/(np.power(r,13))

    def potential_lennard_jones(self, r):
        return 4 * (np.power(r, -12) - np.power(r,-6))

    def propagate_explicit_euler(self):
        self.dpos[:] = self.pos[self.sim.current_step] \
                       + self.vel[self.sim.current_step] * self.sim.current_timestep
        self.wrap_d_vector()
        self.set_resulting_force()
        self.pos[self.sim.current_step + 1] = self.dpos
        self.vel[self.sim.current_step + 1] = self.vel[self.sim.current_step] \
                                              + self.sim.current_timestep / (2 * self.__class__.mass) \
                                              * (self.force[self.sim.current_step])
        return

    def propagate_Verlet_pos(self):
        self.dpos[:] = self.pos[self.sim.current_step] \
                       + self.vel[self.sim.current_step] * self.sim.current_timestep \
                       + self.force[self.sim.current_step] * np.square(self.sim.current_timestep) \
                       / (2 * self.__class__.mass)

        self.wrap_d_vector()
        self.pos[self.sim.current_step + 1] = self.dpos

    def propagate_Verlet_vel(self):
        self.set_resulting_force(future_step=1)

        self.vel[self.sim.current_step + 1] = self.vel[self.sim.current_step] \
                                              + self.sim.current_timestep / (2 * self.__class__.mass) \
                                              * (self.force[self.sim.current_step]
                                                 + self.force[self.sim.current_step + 1])

        return

    def normalize_particle(self):
        self.mask = np.ma.array(self.sim.instances, mask=self.sim.__global_mask__[self.__id__ - 1]).compressed()



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
        if self.sim.state == "gaseous":
            self.pos[0] = self.pos[self.sim.current_step, :]  # self.pos[self.sim.current_step, :]  # self.initial_pos
            self.vel[0] = self.vel[self.sim.current_step, :]  # self.vel[self.sim.current_step, :]  # self.initial_vel
        else:
            self.pos[0] = self.initial_pos  # self.pos[self.sim.current_step, :]  # self.initial_pos
            self.vel[0] = self.initial_vel  # self.vel[self.sim.current_step, :]  # self.initial_vel

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

