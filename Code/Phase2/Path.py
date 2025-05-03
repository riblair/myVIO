from abc import ABC, abstractmethod
import numpy as np


class Path(ABC):

    def __init__(self, timeframe=10):
        self.t_f = timeframe
        self.current_time = 0

    @abstractmethod
    def get_position(self, time) -> np.ndarray:
        "Given a time between [0, t_f], compute X(t)"
        raise NotImplementedError("Implement me :)")

class StraightLine(Path):

    def __init__(self, point_a: np.ndarray, point_b: np.ndarray, timeframe=10):
        super().__init__(timeframe)
        self.x_i = point_a
        self.direction_vector = (point_b - point_a) / timeframe

    def get_position(self, time):
        return self.direction_vector * time/(self.t_f)
    
class Sinusoid(Path):

    def __init__(self, point_a: np.ndarray, point_b: np.ndarray, timeframe=10):
        super().__init__(timeframe)
        self.x_i = point_a
        self.direction_vector = (point_b - point_a) / timeframe

    def get_position(self, time):
        return self.direction_vector * time/(self.t_f)
    