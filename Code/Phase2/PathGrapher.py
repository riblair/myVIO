import matplotlib.pyplot as plt
import Path

class PathGrapher():

    def __init__(self, gt:Path, estim_file: str = None):
        # Figure out a good way to pass in the path variables...
        self.ground_truth = gt
        self.estim = estim_file



    def generate_xy_plot(self):
        # plot the X Y g.t vs estim values
        pass

    def generate_xz_plot(self):
        # plot the X Z g.t vs estim values
        pass

    def generate_orientation_error_plots(self):
        # 3 subgraphs, roll, pitch, yaw vs time.
        pass