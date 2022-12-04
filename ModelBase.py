from abc import ABC, abstractmethod


class ModelBase(ABC):

    def __init__(self, board_size):
        self.board_size = board_size

    @abstractmethod
    def determine_prior_probability(self, state):
        pass

    @abstractmethod
    def determine_state_value(self, state):
        pass
