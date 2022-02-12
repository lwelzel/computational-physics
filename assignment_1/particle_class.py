import numpy as np
# from scipy.spatial.distance import cdist, euclidean  # slow af
from sys import version_info

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

    # ok stupid solution first, max number of particles = 10000
    # this also makes mp really impossible, oh well...
    # TODO: should be able to modify via class method
    __max_particles = 10000 # max number of particles that can be created per class
    """
    i.e. a value of 10 means that a maximum of 10 instances of Particle sub-class e.g. Argon can be created
    Intended function: defined per sub-sub class to this Meta Class, 
    setting __max_particles = N means:
        - unlimited instances of MetaClassParticle
        - unlimited instances of (semi meta class) Particle
        - max N instances of Particle sub-class (e.g. Argon)
    """

    __instances__ = np.full(__max_particles, fill_value=np.nan, dtype=object)

    # PARTICLE PROPERTIES
    particle_name = "Base_Particle"
    particle_mass = 1

    # SIMULATION PROPERTIES
    # MIC implementation is based on discussion of MIC Boundary Conditions for C/C++
    #   U.K. Deiters, 2013: "Efficient Coding of the Minimum Image Convention". Z. Phys. Chem.
    n_dim = 0
    # define box extents
    box = np.array([])
    box_r = np.array([])
    box2 = np.array([])
    box2_r = np.array([])
    current_step = 0

    def __init__(self, n_dim):
        super(MetaClassParticle, self).__init__()
        self.__instance_loc__ = self.__n_active__
        self.__class__.__instances__[self.__instance_loc__] = self

        self.__class__.n_dim = n_dim
        self.__class__.current_step = 0


    @classmethod
    def setup_mic(cls):
        cls.box = np.ones(shape=cls.n_dim)
        cls.box_r = 1 / cls.box
        cls.box2 = 0.5 * cls.box
        cls.box2_r = 1 / cls.box2

    @classmethod
    @property
    def __n_active__(cls):
        """
        Method to find the number of currently active instance.
        When creating a new instance the method is used to set the position of the instance in the instance array
        :return: Number of active instances
        """
        nans = np.isnan(cls.__instances__)
        return np.size(nans) - np.count_nonzero(nans)

    @classmethod
    @property
    def get__instances(cls):
        return cls.__instances__

    def __delete_self__(self):
        """
        Delete an instance from the class instance array
        aka thanos snap the instance
        :param which_loc: index of the instance to be delted in the class instance array
        :return:
        """
        self.__class__.__instances__[self.__instance_loc__:] = self.__class__.__instances__[self.__instance_loc__ + 1:]
        self.__class__.__instances__[-1] = np.nan
        for ins in self.__class__.__instances__[self.__instance_loc__:]:
            ins.__instance_loc__ -= 1
        return


class Particle(MetaClassParticle):
    """
    Class to keep track of particles
    """
    def __init__(self, n_steps, n_dim,
                 initial_pos, initial_vel, initial_acc,
                 **kwargs):
        """

        :param n_steps:
        :param n_dim:
        :param initial_pos:
        :param initial_vel:
        :param initial_acc:
        :param kwargs:
        """
        # setup
        super(Particle, self).__init__(n_dim)

        # general backend
        self.n_steps = n_steps
        shape = (self.n_steps, self.n_dim)

        # TODO: hash ID for each particle

        # each vector (position, velocity, acceleration)
        # is pre-initialized with n steps in n_dimensions
        self.pos = np.full(shape=shape, fill_value=np.nan)
        self.vel = np.full(shape=shape, fill_value=np.nan)
        self.acc = np.full(shape=shape, fill_value=np.nan)

        # initial values
        self.pos[0] = initial_pos
        self.vel[0] = initial_vel
        self.acc[0] = initial_acc

        # parameters for positions
        # k keeps track of particles "outside" the box, if particles bound very quickly
        # (more than 127 boxes in a single tick) we are f-ed
        # or we can just replace it with a larger int (np.int128 is max I think)
        self.k = np.full(shape=self.__class__.n_dim, fill_value=np.nan, dtype=np.int8)

        # below is if we use a set (add particles separately)
        # self.__class__.__instances__.add(self)

    def get_distance_vectorA1(self, other):
        '''
        Get the distance between two particles
        Class A1 algorithm
        :param other: other instance of Particle (sub-) class
        :return: distance
        '''
        dpos = other.pos[self.__class__.current_step] - self.pos[self.__class__.current_step]
        self.k[:] = dpos * self.__class__.box2_r  # this casts the result to int, finding in where the closest box is
        dpos -= self.k * self.__class__.box
        return dpos

    def get_distance_absoluteA1(self, other):
        # np.linalg.norm(self.get_distance_vectorA1(other), ord=None)  # 2-norm is the eucl. dist. between 2 points
        # ^ this is slower
        dpos = self.get_distance_vectorA1(other)
        # using the properties of the Einstein summation convention implementation in numpy, which is very fast
        # but the code will need some rewriting for that
        # np.sqrt(np.einsum('ij->i', dpos, dpos))
        return np.linalg.norm(self.get_distance_vectorA1(other), ord=None)




