import numpy as np
import matplotlib.pyplot as plt
from Path import Path, StraightLine, Sinusoid, Circle, FigureEight, HyperbolicParaboloid
import utils as util

class PathGrapher:
    def __init__(self, gt: Path, estim_file: str = None):
        self.ground_truth = gt
        self.estim = self._load_estimated_path(estim_file)

    def _load_estimated_path(self, estim_file):
        if estim_file is None: return None
        times = np.linspace(0, self.ground_truth.t_f, num=int(self.ground_truth.t_f/util.IMU_DT))
        ...
        return ...

    def _generate_ground_truth(self, o_type: str = "quat"):
        times = np.linspace(0, self.ground_truth.t_f, num=int(self.ground_truth.t_f/util.IMU_DT))
        positions = np.array([self.ground_truth.get_position(t) for t in times])
        orientations = np.array([self.ground_truth.get_orientation(t) for t in times])
        if o_type.lower() == "quat":
            f_orientations = orientations
        elif o_type.lower() == "euler":
            f_orientations = np.array([util.euler_from_quat(orientation) for orientation in orientations])
        else:
            print(f"[ERROR] Wrong type given for _generate_ground_truth. Expected ['quat', 'euler'], given {o_type}")
            exit(1)
        return times, positions, f_orientations

    def generate_xy_plot(self, show_now: bool = False):
        times, gt_positions, orientations = self._generate_ground_truth()
        if self.estim is not None: est_positions = np.array([self.estim[t][:3] for t in times])

        plt.figure()
        plt.plot(gt_positions[:, 0], gt_positions[:, 1], label='Ground Truth')
        if self.estim is not None: plt.plot(est_positions[:, 0], est_positions[:, 1], label='Estimated', linestyle='--')
        plt.xlabel('X')
        plt.xlim([-10,10])
        plt.ylabel('Y')
        plt.ylim([-10,10])
        plt.title(f'[{self.ground_truth.name}] XY Trajectory')
        plt.legend()
        plt.grid(True)
        if show_now: plt.show()

    def generate_xz_plot(self, show_now: bool = False):
        times, gt_positions, orientations = self._generate_ground_truth()
        if self.estim is not None: est_positions = np.array([self.estim[t][:3] for t in times])

        plt.figure()
        plt.plot(gt_positions[:, 0], gt_positions[:, 2], label='Ground Truth')
        if self.estim is not None: plt.plot(est_positions[:, 0], est_positions[:, 2], label='Estimated', linestyle='--')
        plt.xlabel('X')
        plt.xlim([-10,10])
        plt.ylabel('Z')
        plt.ylim([-10,10])
        plt.title(f'[{self.ground_truth.name}] XZ Trajectory')
        plt.legend()
        plt.grid(True)
        if show_now: plt.show()

    def generate_3d_plot(self, show_now: bool = False):
        times, gt_positions, orientations = self._generate_ground_truth()
        if self.estim is not None: est_positions = np.array([self.estim[t][:3] for t in times])

        fig = plt.figure()
        ax = fig.add_subplot(projection='3d')
        ax.plot(gt_positions[:, 0], gt_positions[:, 1], gt_positions[:, 2], label='Ground Truth')
        if self.estim is not None: ax.plot(est_positions[:, 0], gt_positions[:, 1], est_positions[:, 2], label='Estimated', linestyle='--')
        ax.set_xlabel('X')
        ax.set_xlim([-10,10])
        ax.set_ylabel('Y')
        ax.set_ylim([-10,10])
        ax.set_zlabel('Z')
        ax.set_zlim([-10,10])
        ax.set_title(f'[{self.ground_truth.name}] 3d Trajectory')
        ax.legend()
        ax.grid(True)
        if show_now: plt.show()

    def generate_orientation_plots(self, show_now: bool = False):
        times, gt_positions, orientations = self._generate_ground_truth('euler')
        if self.estim is not None: est_positions = np.array([self.estim[t][:3] for t in times])
        
        fig, axs = plt.subplots(3, 1, figsize=(10, 8), sharex=True)
        labels = ['Roll', 'Pitch', 'Yaw']
        
        for i in range(3):
            axs[i].plot(times, orientations[:, i], label=f'GT {labels[i]}')
            axs[i].set_ylabel(f'{labels[i]} (rad)')
            axs[i].legend()
            axs[i].grid(True)

        axs[2].set_xlabel('Time (s)')
        fig.suptitle('Ground Truth Orientation (Euler Angles) vs Time')
        plt.tight_layout()
        if show_now: plt.show()

    def generate_plots(self):
        self.generate_xy_plot()
        self.generate_xz_plot()
        self.generate_3d_plot(False)
        self.generate_orientation_plots(True)


if __name__ == '__main__':
    path_list = []
    path_list.append(PathGrapher(StraightLine(np.array([-2,-5,0]), np.array([9,8.5,5]))))
    path_list.append(PathGrapher(Sinusoid(np.array([-7,-2,3]), np.array([8,9,9]), x_params=(0.25, 5, 0), y_params=(0.25, 2, 1), z_params=(0.25, 3, 2))))
    path_list.append(PathGrapher(Circle(np.zeros(3), 15)))
    path_list.append(PathGrapher(FigureEight(np.zeros(3), 15, 15)))
    path_list.append(PathGrapher(HyperbolicParaboloid(np.zeros(3), 18, 14, 0.05)))
    
    for pg in path_list:
        pg.generate_plots()
    plt.show()