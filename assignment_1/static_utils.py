import numpy as np

class SimBox(object):
    """
    Class to handle simulation boxes (MCI) outside simulations
    """
    def __init__(self, box_length, n_dim, **kwargs):
        # setup
        super(SimBox, self).__init__()

        self.n_dim = n_dim
        self.box_length = box_length

        self.box = np.ones(shape=(self.n_dim, 1)) * self.box_length
        self.box_r = 1 / self.box
        self.box2 = 0.5 * self.box
        self.box2_r = 1 / self.box2

        self.k = np.zeros(shape=(self.n_dim, 1), dtype=np.int64)
        self.dpos = np.zeros(shape=(self.n_dim, 1))


    def wrap_d_vector(self):
        self.dpos[:] = (self.dpos + self.box2) % self.box - self.box2

    def get_distance_vectorA1(self, pos1, pos2):
        """
        Get the distance between two particles
        Class A1 algorithm - all positions are stored as absolutes
        :param other: other instance of Particle (sub-) class
        :return: distance
        """
        self.dpos[:] = pos1 - pos2
        self.wrap_d_vector()

    def get_distance_absoluteA1(self, pos):
        self.get_distance_vectorA1(*pos)
        # using the properties of the Einstein summation convention implementation in numpy, which is very fast
        return np.sqrt(np.einsum('ij,ij->j', self.dpos, self.dpos))