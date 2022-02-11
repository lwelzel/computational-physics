import numpy as np

class Particle(object):
    """
    Class to keep track of particles
    """
    def __init__(self, n_steps, n_dim, pos, vel, acc,
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