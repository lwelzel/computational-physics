import numpy as np
from constants_class import Constants
# from scipy.spatial.distance import cdist, euclidean  # slow af
from sys import version_info
from itertools import count

# np.can_cast(int, float, casting='safe')

class MetaClassParticle(object):
    """
    Meta Class for the Particle Class to handle instance tracking and self-knowledge of the Particle class
    """
    if version_info < (3, 9):
        raise EnvironmentError("Please update to Python 3.9 or later (code written for Python 3.9.x).\n"
                               "This is required for the proper function of MetaClasses\n"
                               "which use the double  @classmethod @property decorator.\n"
                               "We use this, pythonic approach, to keep track of our particles.")

    # TODO: maybe this is a bad idea but Id like to avoid passing around objects all the time.
    #  So what I suggest is that we store the instances in the class itself (at least as references).
    #  This is probably really stupid and is gonna bite us in the butt later but for now I dont see how it could.
    #  Except if there is a lot of particles of course but then we are screwed anyway :)
    #  Has the same(-ish) problem to the one above, but is nicer

    # META PARAMETERS
    # ok stupid solution first, max number of particles = 1000
    # this also makes mp really impossible, oh well...
    # TODO: should be able to modify via class method
    max_particles = 1000  # max number of particles that can be created per class
    """
    i.e. a value of 10 means that a maximum of 10 instances of Particle sub-class e.g. Argon can be created
    Intended function: defined per sub-sub class to this Meta Class, 
    setting __max_particles = N means:
        - unlimited instances of MetaClassParticle
        - unlimited instances of (semi meta class) Particle
        - max N instances of Particle sub-class (e.g. Argon)
    """

    __instances__ = np.full(max_particles, fill_value=None, dtype=object)
    # this is really bad, I should just be able to check in __n_active__ using __instances__ but its funky
    # also is the mask that is used to filter currently active particle
    # 0 when unoccupied, int when occupied
    __occupation__ = np.full(max_particles, fill_value=0)

    # PARTICLE PARAMETERS
    particle_name = "Meta_Particle"
    # normalization
    bk = Constants.bk

    # SIMULATION PROPERTIES
    # MIC implementation is based on discussion of MIC Boundary Conditions for C/C++
    #   U.K. Deiters, 2013: "Efficient Coding of the Minimum Image Convention". Z. Phys. Chem.
    n_dim = 0
    timestep = 1 # s
    # define box extents
    box_length = 1
    box = np.array([])
    box_r = np.array([])
    box2 = np.array([])
    box2_r = np.array([])
    # stepping
    # TODO: put simulation stuff in its own class - but *think* first!
    #  issues:
    #  - keeping track of instances
    #  - instances referring to simulation para
    current_step = 0  # needs to be increased each step by calling .tick() on the class

    # META INSTANCE ATTRIBUTES
    idservice = count(1)

    def __init__(self, n_dim, **kwargs):
        super(MetaClassParticle, self).__init__(**kwargs)

        # meta
        self.__id__ = next(self.__class__.idservice)
        self.__huid__ = f"{self.__class__.particle_name}{self.__id__:05}"

        # define location of the instance in the instance class array
        self.__instance_loc__ = self.__n_active__
        self.__class__.__occupation__[self.__instance_loc__] = self.__id__
        # mask out the particle itself so it is not double counted
        # TODO: this can be better implemented using a diagonal square bool matrix (false on diagonal)
        self.mask = self.__occupation__ - self.__id__ == 0
        # save instance to instance class array
        self.__class__.__instances__[self.__instance_loc__] = self

        # general setup
        self.__class__.n_dim = n_dim
        self.__class__.current_step = 0


    @classmethod
    def setup_mic(cls):
        """
        Sets up the bounding box of the simulation and constructs MCI helpers/enforcers
        needs to be re-called when adjusting box_length
        :return:
        """
        cls.box = np.ones(shape=(cls.n_dim, 1)) * cls.box_length
        cls.box_r = 1 / cls.box
        cls.box2 = 0.5 * cls.box
        cls.box2_r = 1 / cls.box2

    @classmethod
    @property
    def __n_active__(cls):
        """
        Method to find the number of currently active instance.
        When creating a new instance the method is used to set the position of the instance in the instance class array
        :return: Number of active instances
        """
        return np.count_nonzero(cls.__occupation__)

    @classmethod
    @property
    def get__instances(cls):
        return cls.__instances__

    def __delete_self__(self):
        """
        Delete an instance from the class instance array
        aka thanos snap the instance
        :param which_loc: index of the instance to be deleted in the class instance array
        :return:
        """
        self.__class__.__instances__[self.__instance_loc__:] = self.__class__.__instances__[self.__instance_loc__ + 1:]
        self.__class__.__occupation__[self.__instance_loc__:] = self.__class__.__occupation__[self.__instance_loc__ + 1:]
        self.__class__.__instances__[-1] = None
        self.__class__.__occupation__[-1] = 0
        for ins in self.__class__.__instances__[self.__instance_loc__:]:
            ins.__instance_loc__ -= 1
            # I am not sure if below is a good idea, it will surely bite us later
            ins.__id__ -= 1

    @classmethod
    def update_meta_class(cls,
                          __instances__=False,
                          __occupation__=False,
                          max_particles=None,
                          particle_name=None,
                          n_dim=None,
                          timestep=None,
                          box_length=None):
        """
        Updates attributes of the meta class(es)
        Warning: calling this after set-up will delete data
        :param __instances__: True if it should be updated, False otherwise
        :param __occupation__: True if it should be updated, False otherwise
        :param max_particles: variable if it should be updated, None otherwise
        :param particle_name: variable if it should be updated, None otherwise
        :param n_dim: variable if it should be updated, None otherwise
        :param timestep: variable if it should be updated, None otherwise
        :return:
        """
        if max_particles is not None:
            cls.max_particles = max_particles
        if particle_name is not None:
            cls.particle_name = particle_name
        if n_dim is not None:
            cls.n_dim = n_dim
        if timestep is not None:
            cls.timestep = timestep
        if box_length is not None:
            cls.box_length = box_length
            cls.setup_mic()


        # these are the important ones to update
        if __instances__:
            cls.__instances__ = np.full(cls.max_particles, fill_value=None, dtype=object)
        if __occupation__:
            cls.__occupation__ = np.full(cls.max_particles, fill_value=0)

    @classmethod
    def tick(cls):
        cls.current_step += 1


class Particle(MetaClassParticle):
    """
    Class to keep track of particles
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

    # MEDIUM PROPERTIES
    # TODO: implement way to update
    temperature = 0
    energy = 0
    pressure = 0
    density = 0
    surface_tension = 0

    def __init__(self, n_steps, n_dim,
                 initial_pos, initial_vel, initial_force,
                 **kwargs):
        """

        :param n_steps:
        :param n_dim:
        :param initial_pos: shape = (3, 1)
        :param initial_vel: shape = (3, 1)
        :param initial_force: shape = (3, 1)
        :param kwargs:
        """
        # setup
        super(Particle, self).__init__(n_dim, **kwargs)

        # general backend
        self.n_steps = n_steps
        shape = (self.n_steps, self.n_dim, 1)

        # TODO: hash ID for each particle

        # each vector (position, velocity, force)
        # is pre-initialized with n steps in n_dimensions
        self.pos = np.full(shape=shape, fill_value=0.)
        self.vel = np.full(shape=shape, fill_value=0.)
        self.force = np.full(shape=shape, fill_value=0.)

        # initial values
        self.pos[0] = initial_pos
        self.vel[0] = initial_vel
        self.force[0] = initial_force

        # other particle properties
        # TODO: those are just placeholders at this point.
        #  It probably will eventually make sense to also have those be np.arrays that track over the simulation.
        #  Also required for smart error tracking etc.

        # parameters for positions
        # k keeps track of particles "outside" the box, if particles bound very quickly
        # (more than 127 boxes in a single tick) we are f-ed
        # or we can just replace it with a larger int (np.int128 is max I think)
        self.k = np.full(shape=(self.__class__.n_dim, 1), fill_value=0, dtype=np.int64)
        # convenience difference vector for positions
        self.dpos = np.full(shape=(self.__class__.n_dim, 1), fill_value=0.)


    def wrap_d_vector(self):
        self.k[:] = self.dpos[:] \
                    * self.__class__.box2_r  # this casts the result to int, finding in where the closest box is
        self.dpos[:] = self.dpos[:] - self.k * self.__class__.box  # casting back to float must be explicit


    def get_distance_vectorA1(self, other):
        """
        Get the distance between two particles
        Class A1 algorithm - all positions are stored as absolutes
        :param other: other instance of Particle (sub-) class
        :return: distance
        """
        self.dpos[:] = other.pos[self.__class__.current_step] - self.pos[self.__class__.current_step]
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
        for other in np.ma.array(self.__class__.__instances__, mask=self.mask).compressed():
            self.force[self.__class__.current_step] = self.force[self.__class__.current_step] + self.get_force(other)

    def get_force(self, other):
        dist, vector = self.get_distance_absoluteA1(other)
        potential = self.potential_lennard_jones(dist)
        return - potential * vector / dist

    def force_lennard_jones(self, r):
        sigma_r_ratio = self.__class__.sigma / r
        return 24.0 * self.__class__.internal_energy * np.power(sigma_r_ratio, 2) \
               * (2.0 * np.power(sigma_r_ratio, 12) - np.power(sigma_r_ratio, 6))

    def potential_lennard_jones(self, r):
        sigma_r_ratio = self.__class__.sigma / r
        return - 4 * self.__class__.internal_energy * (np.power(sigma_r_ratio, 12) - np.power(sigma_r_ratio, 6))

    def next_position(self):
        self.dpos[:] = self.vel[self.__class__.current_step] * self.__class__.timestep
        return


    def propagate(self):
        self.dpos[:] = self.pos[self.__class__.current_step] \
                       + self.vel[self.__class__.current_step] * self.__class__.timestep
        self.wrap_d_vector()
        self.pos[self.__class__.current_step + 1] = self.dpos
        self.set_resulting_force()
        self.vel[self.__class__.current_step + 1] = self.vel[self.__class__.current_step] \
                                                    + 1 / self.__class__.particle_mass \
                                                    * self.force[self.__class__.current_step] * self.__class__.timestep
        return

    @classmethod
    def normalize_problem(cls):
        """
        Normalize parameters of simulation based on
                                https://en.wikipedia.org/wiki/Lennard-Jones_potential#Dimensionless_(reduced_units)
        :sets: normalized parameters
        """
        cls.box_length *= 1 / cls.sigma

        cls.timestep *= np.sqrt(cls.internal_energy /
                                (cls.particle_mass * np.power(cls.sigma, 2)))


        cls.temperature *= Constants.bk / cls.internal_energy
        cls.energy *= 1 / cls.internal_energy
        cls.pressure *= np.power(cls.sigma, 3) / cls.internal_energy
        cls.density *= np.power(cls.sigma, 3)
        cls.surface_tension *= np.power(cls.sigma, 2) / cls.internal_energy

        # iterate over particles and normalize values
        for particle in cls.__instances__:
            particle.pos *= 1 / cls.sigma
            particle.vel *= np.sqrt(cls.internal_energy /
                                    (cls.particle_mass * np.power(cls.sigma, 2)))\
                            / cls.sigma
            particle.force *= cls.sigma / cls.internal_energy

        cls.update_meta_class(timestep=cls.timestep,
                              box_length=cls.box_length)

        cls.mass = 1
        cls.sigma = 1
        cls.internal_energy = 1
        cls.bk = 1








