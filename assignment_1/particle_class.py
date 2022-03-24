import numpy as np
from weakref import ref, proxy
# from scipy.spatial.distance import cdist, euclidean  # slow af
from sys import version_info


# from md_simulation_class import MolDyn

class Particle(object):
    """
    Meta Particle Class to handle instance tracking and self-knowledge of the Particle class
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

        :param sim: MolDyn simulation instance
        :param initial_pos: initial positions, array
        :param initial_vel: initial velocities, array
        :param initial_acc: initial accelerations, array
        :param kwargs: other kwargs that might be added in the future
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
        """
        This function takes the difference vector of a particle pair, dpos and converts it to a
        coordinate difference vector taking into account periodic boundary conditions.
        :return: na
        """
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
        """
        :param other: The other particle.
        :param future_step:The future time step to predict the absolute distance between the two particles.
        :return: absolute distance between this particle and another particle as well as its vector
        """
        self.get_distance_vectorA1(other, future_step)
        # using the properties of the Einstein summation convention implementation in numpy, which is very fast
        return np.sqrt(np.einsum('ij,ij->j', self.dpos, self.dpos)), self.dpos

    def set_resulting_force(self, future_step=0):
        """
        This function, as part of the Particle class, sets the resulting force
        of each particle in the simulation and itself, after getting the force and potential by calling the
        get_force_potential function from the Simulation class.
        It also contributes to some statistical simulation parameters.
        The generator for the iteration takes into account only particles withing the current SOI
        :param future_step:
        :return:
        Notes
        -----
        The return value is None, but the resulting force of the particle is set.
        """
        for other in np.ma.array(self.sim.instances, mask=self.sim.__global_mask__[self.__id__-1]).compressed():
            force, potential, pressure_term = self.sim.get_force_potential(self.pos[self.sim.current_step + future_step],
                                                                           other.pos[self.sim.current_step + future_step],
                                                                           self.sim.box2_r, self.sim.box)

            self.sim.potential_energy[self.sim.current_step + future_step] += potential
            self.sim.pressure[self.sim.current_step] += pressure_term
            self.force[self.sim.current_step + future_step] = self.force[self.sim.current_step + future_step] + force
            other.force[self.sim.current_step + future_step] = other.force[self.sim.current_step + future_step] - force



    def get_force_potential(self, other, future_step=0):
        """
        :param other: The other particle
        :param future_step: The time step in the future to check for
        :return: tuple consisting of the force vector, potential, and force magnitude.
        """
        dist, vector = self.get_distance_absoluteA1(other, future_step)
        return self.force_lennard_jones(dist) * vector / dist, self.potential_lennard_jones(dist), self.force_lennard_jones(dist) * dist

    def force_lennard_jones(self, r):
        """
        Define the force on each particle due to the Lennard-Jones potential.


        :param r: The separation of the particle pair, array-like
        :return f: The force on each particle, array-like
        """
        return - 24*(np.power(r,6)-2)/(np.power(r,13))

    def potential_lennard_jones(self, r):
        """
        This function returns the potential energy of two particles with
        the Lennard-Jones force law. The particles will experience a repulsive
        force for small distances and an attractive force for large distances.
        :param r: The distance between the two particles.
        :return: The potential energy of the two particles.
        """
        return 4 * (np.power(r, -12) - np.power(r,-6))

    def propagate_explicit_euler(self):
        """
        Propagates the velocity and position of the particle using an
        explicit Euler method.

        1. The particle's future position is set to its current position plus
           its velocity multiplied by the current time step.
        2. The particle's future velocity is set to its current velocity plus
           half the time step multiplied by the resulting force.
        :return:
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

    def propagate_Verlet_pos(self):
        """
        This function increments the particle's position for the next time step.
        The particle position for the next step is calculated using the Verlet
        integration method.

        The Verlet integration method is a modification of the Euler method, which is (semi) symplectic.
        :return:
        """
        self.dpos[:] = self.pos[self.sim.current_step] \
                       + self.vel[self.sim.current_step] * self.sim.current_timestep \
                       + self.force[self.sim.current_step] * np.square(self.sim.current_timestep) \
                       / (2 * self.__class__.mass)

        self.wrap_d_vector()
        self.pos[self.sim.current_step + 1] = self.dpos

    def propagate_Verlet_vel(self):
        """
        Calculate the velocity at the next time step for a particle using the Verlet velocity algorithm.
        The Verlet velocity algorithm is simply the velocity at the next time step calculated using the average of the
        particle's current velocity and the velocity halfway through the time step.
        :return:
        """
        self.set_resulting_force(future_step=1)

        self.vel[self.sim.current_step + 1] = self.vel[self.sim.current_step] \
                                              + self.sim.current_timestep / (2 * self.__class__.mass) \
                                              * (self.force[self.sim.current_step]
                                                 + self.force[self.sim.current_step + 1])

        return

    def normalize_particle(self):
        """
        The name is mainly legacy before the inputs were changed to be normalized.
        Now the function sets the mask of the particle pairs (its own row from the pair matrix)

        This is done by creating a mask for each particle that is based on
        the global mask and the particle's instance number.

        The particle's mask is then used to compress the particle's array,
        which removes any invalid values.
        :return:
        """
        self.mask = np.ma.array(self.sim.instances, mask=self.sim.__global_mask__[self.__id__ - 1]).compressed()



    def reset_lap(self):
        """
        Resets the position, velocity, force, and acceleration
        of all particles to their respective values at the
        start of the current set.
        :return:
        """
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
        """
        Resets the position, velocity, force, and acceleration
        of all particles to their respective values at the
        start of the current step.
        Some funkyness for gasses to speed up relaxation.
        :return:
        """
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
        """
        Does what it says
        :return:
        """
        cls.mass = 1.
        cls.internal_energy = 1.
        cls.sigma = 1.

    @property
    def position(self):
        """
        Does what it says
        :return:
        """
        return (self.pos[self.sim.current_step])

