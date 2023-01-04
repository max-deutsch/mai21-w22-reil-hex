import numpy as np
from ModelBase import ModelBase


class RandomModel(ModelBase):
    def __init__(self, board_size, random_state=555):
        self.rng = np.random.default_rng(random_state)
        super().__init__(board_size)

    def determine_prior_probability(self, state):
        return list([self.rng.uniform(-1, 1) for x in range(self.board_size)])

    def determine_state_value(self, state):
        return self.rng.uniform(-1, 1)
