import numpy as np

class Particle(object, n_particles=None):
    """
    Class to keep track of particles
    """
    # TODO: should the class keep and updated version of the particle positions for each step so we dont need to
    #  access the individual particles each tick? Below implementation requires all particles to be initialized

    # global_position_matrix = []

    # TODO: maybe this is a bad idea but Id like to avoid passing around objects all the time.
    #  So what I suggest is that we store the instances in the class itself (at least as references).
    #  This is probably really stupid and is gonna bite us in the butt later but for now I dont see how it could.
    #  Except if there is a lot of particles of course but then we are screwed anyway :)
    #  Has the same(-ish) problem to the one above, but is nicer

    # ok stupid solution first, max number of particles = 10000
    # this also makes mp really impossible, oh well...
    __max_particles = 10000 # should be passed to class
    __instances__ = np.full(__max_particles, fill_value=np.nan, dtype=object)

    def __init__(self, n_steps, n_dim,
                 initial_pos, initial_vel, initial_acc,
                 mass, name="",
                 **kwargs):
        """

        :param n_steps:
        :param pos:
        :param vel:
        :param acc:
        :param mass:
        :param kwargs:
        """
        # setup
        super(Particle, self).__init__()

        # general backend
        self.n_steps = n_steps
        self.n_dim = n_dim
        shape = (self.n_steps, self.n_dim)

        # particle properties
        self.name = name
        self.mass = mass
        # TODO: hash ID for each particle

        # each vector (position, velocity, acceleration)
        # is pre-initialized with n steps in n_dimensions
        pos = np.full(shape=shape, fill_value=np.nan)
        vel = np.full(shape=shape, fill_value=np.nan)
        acc = np.full(shape=shape, fill_value=np.nan)

        # below is if we use a set (add particles separately)
        # self.__class__.__instances__.add(self)

    def propagate(self):
        return


    @classmethod
    def get_instances(cls):
        return cls.__instances__

    @classmethod
    def get_n_active_particles(cls):
        nans = np.isnan(cls.__instances__)
        return np.size(nans) - np.count_nonzero(nans)

    @staticmethod
    def get_shortest_distance(pos1, pos2):
        return




