from abc import ABC, abstractmethod
import numpy as np
import math
import utils as util
import yaml

R_H = np.array([0, 0, 1])
Q_H = np.array([0, 0, 0, 1])

# See Bottom of file for ALL EXAMPLE PATH

class Path(ABC):

    def __init__(self, name, timeframe=10):
        self.t_f = timeframe
        self.current_time = 0
        self.name = name

    @abstractmethod
    def get_position(self, time) -> np.ndarray:
        "Given a time between [0, t_f], compute X(t)"
        raise NotImplementedError("Implement me :)")
    
    @abstractmethod
    def export_hyperparameters(self, filepath):
        raise NotImplementedError("Implement me :)")
    
    def get_orientation(self, time) -> np.ndarray:
        # Get the angle from the direction vector
        if time == 0:
            pose_1 = self.get_position(0)
            pose_2 = self.get_position(util.IMU_DT)
        else:
            pose_1 = self.get_position(time-util.IMU_DT)
            pose_2 = self.get_position(time)
        direction_vector = pose_2-pose_1
        dir_norm = np.linalg.norm(direction_vector)
        if dir_norm > 1e-10 : dir_normalized = direction_vector / dir_norm
        else: return Q_H
        # find quaternion from "hover" angle
        q_v_h = util.from_two_vectors(dir_normalized, R_H)
        # Multiply this by a scalar to get the final quat...
        interpolated_q = util.slerp(Q_H, q_v_h, 0.2)     
        return interpolated_q

class StraightLine(Path):
    "Drone moves in a straight line between two points"
    def __init__(self, point_a: np.ndarray, point_b: np.ndarray, timeframe=10):
        super().__init__("Straight_Line", timeframe)
        self.x_i = point_a
        self.direction_vector = (point_b - point_a) / timeframe

    def get_position(self, time) -> np.ndarray:
        if(time > self.t_f):
            print("ERROR: Time out of bounds")
            return np.zeros(3)
        return self.direction_vector * time + self.x_i
    
    def export_hyperparameters(self, filepath):
        data = {
            'name' : str(self.name),
            'x_i' : self.x_i.tolist(),
            'direction' : self.direction_vector.tolist()
        }
        with open(filepath, 'w') as yaml_file:
            yaml.dump(data, yaml_file)
    
class Sinusoid(Path):
    "Moves from point a to point b with with sinusiodally variations in each axis"
    def __init__(self, point_a: np.ndarray, point_b: np.ndarray, x_params, y_params, z_params, timeframe=10):
        " x,y,z params are a set of [Amplitude, Frequency, Offset] that vary the position in that specific axis"
        super().__init__("Sinusoid", timeframe)
        self.x_i = point_a
        self.direction_vector = (point_b - point_a) / timeframe
        self.x_params = x_params
        self.y_params = y_params
        self.z_params = z_params

    def get_position(self, time):
        position = self.direction_vector * time + self.x_i
        position[0] += self.x_params[0] * math.sin(self.x_params[1]*time+self.x_params[2])
        position[1] += self.y_params[0] * math.sin(self.y_params[1]*time+self.y_params[2])
        position[2] += self.z_params[0] * math.sin(self.z_params[1]*time+self.z_params[2])
        return position
    
    def export_hyperparameters(self, filepath):
        data = {
            'name' : str(self.name),
            'x_i' : self.x_i.tolist(),
            'direction' : self.direction_vector.tolist(),
            'x_params' : list(self.x_params),
            'y_params' : list(self.y_params),
            'z_params' : list(self.z_params)
        }
        with open(filepath, 'w') as yaml_file:
            yaml.dump(data, yaml_file)

    
class Circle(Path):
    def __init__(self, center: np.ndarray, diameter: float, timeframe=10):
        super().__init__("Circle", timeframe)
        self.center = center
        self.radius = diameter / 2

    def get_position(self, time) -> np.ndarray:
        if time > self.t_f:
            print("ERROR: Time out of bounds")
            return np.zeros(3)
        theta = 2 * np.pi * (time / self.t_f)  # full circle
        x = self.center[0] + self.radius * np.cos(theta)
        y = self.center[1] + self.radius * np.sin(theta)
        z = self.center[2]  # constant height
        return np.array([x, y, z])
    
    def export_hyperparameters(self, filepath):
        data = {
            'name' : str(self.name),
            'center' : self.center.tolist(),
            'radius' : self.radius
        }
        with open(filepath, 'w') as yaml_file:
            yaml.dump(data, yaml_file)

class FigureEight(Path):
    def __init__(self, x_i: np.ndarray, major_axis: float, minor_axis: float, timeframe=10):
        super().__init__("Figure_Eight", timeframe)
        self.origin = x_i
        self.a = major_axis / 2
        self.b = minor_axis / 2

    def get_position(self, time) -> np.ndarray:
        if time > self.t_f:
            print("ERROR: Time out of bounds")
            return np.zeros(3)
        t = 2 * np.pi * (time / self.t_f)
        x = self.origin[0] + self.a * np.sin(t)
        y = self.origin[1] + self.b * np.sin(t) * np.cos(t)
        z = self.origin[2]
        return np.array([x, y, z])
    
    def export_hyperparameters(self, filepath):
        data = {
            'name' : str(self.name),
            'origin' : self.origin.tolist(),
            'a' : self.a,
            'b' : self.b
        }
        with open(filepath, 'w') as yaml_file:
            yaml.dump(data, yaml_file)

class HyperbolicParaboloid(Path):
    def __init__(self, x_i: np.ndarray, x_width: float, y_width: float, z_width: float, timeframe=10):
        super().__init__("Hyperbolic_Paraboloid", timeframe)
        self.origin = x_i
        self.r_x = x_width / 2
        self.r_y = y_width / 2
        self.z_scale = z_width

    def get_position(self, time) -> np.ndarray:
        if time > self.t_f:
            print("ERROR: Time out of bounds")
            return np.zeros(3)

        theta = 2 * np.pi * (time / self.t_f)
        x = self.r_x * np.cos(theta)
        y = self.r_y * np.sin(theta)
        z = self.z_scale * (x**2 - y**2)

        return self.origin + np.array([x, y, z])
    
    def export_hyperparameters(self, filepath):
        data = {
            'name' : str(self.name),
            'origin' : self.origin.tolist(),
            'r_x' : self.r_x,
            'r_y' : self.r_y,
            'z_scale' : self.z_scale
        }
        with open(filepath, 'w') as yaml_file:
            yaml.dump(data, yaml_file)

ACCEPTABLE_PATHS = ["Straight_Line", "Circle", "Sinusoid", "Figure_Eight", "Hyperbolic_Paraboloid"]

start_x_train = np.array([-25,-30,50])
end_x_train = np.array([19, 28.5, 35])
start_x_test = np.array([25,-30,30])
end_x_test = np.array([45, 28.5, 35])
start_x_val = np.array([-25,-30,50])
end_x_val = np.array([-40, -40, 60])

STRAIGHT_LINE_TRAIN = StraightLine(start_x_train, end_x_train)
STRAIGHT_LINE_TEST = StraightLine(start_x_test, end_x_test)
STRAIGHT_LINE_VAL = StraightLine(start_x_val, end_x_val)

SINUSOID_TRAIN = Sinusoid(start_x_train, end_x_train, x_params=(6, 1, 1), y_params=(4, 0.35, 3), z_params=(6, 0.35, 2))
SINUSOID_TEST = Sinusoid(start_x_test, end_x_train,  x_params=(9, 2, 7), y_params=(12, 0.45, 5), z_params=(5, 0.5, 0))
SINUSOID_VAL = Sinusoid(start_x_val, end_x_train,   x_params=(10, 1.5, 4), y_params=(8, 0.55, 2), z_params=(4, 0.45, 1.4))

CIRCLE_TRAIN = Circle(np.array([2,15,50]), 40)
CIRCLE_TEST = Circle(np.array([-4,5,34]), 38)
CIRCLE_VAL = Circle(np.array([-25,0,75]), 36)

FIGURE_EIGHT_TRAIN = FigureEight(np.array([-10,0,50]), 50, 30)
FIGURE_EIGHT_TEST = FigureEight(np.array([0,5,60]), 40, 40)
FIGURE_EIGHT_VAL = FigureEight(np.array([0,0,55]), 30, 50)

HYPERBOLIC_PARABOLOID_TRAIN = HyperbolicParaboloid(np.array([10,0,50]), 38, 35, 0.01)
HYPERBOLIC_PARABOLOID_TEST = HyperbolicParaboloid(np.array([0,3,57]), 53, 27, 0.02)
HYPERBOLIC_PARABOLOID_VAL = HyperbolicParaboloid(np.array([-2,0,45]), 60, 30, 0.005)