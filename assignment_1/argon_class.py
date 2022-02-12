import numpy as np
from particle_class import Particle

class Argon(Particle, n_particles=None):
    """
    Class to keep track of particles
    """
    # TODO: maybe this is a bad idea but Id like to avoid passing around objects all the time.
    #  So what I suggest is that we store the instances in the class itself (at least as references).
    #  This is probably really stupid and is gonna bite us in the butt later but for now I dont see how it could.
    #  Except if there is a lot of particles of course but then we are screwed anyway :)
    #  Has the same(-ish) problem to the one above, but is nicer

    # ok stupid solution first, max number of particles = 10000
    # this also makes mp really impossible, oh well...

    # particle properties
    particle_name = "Argon_Particle"
    particle_mass = 1 # kg # Argon Mass

    def __init__(self, n_steps, n_dim,
                 initial_pos, initial_vel, initial_acc,
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
        super(Argon, self).__init__(n_steps, n_dim, initial_pos, initial_vel, initial_acc)