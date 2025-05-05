import numpy as np
import matplotlib.pyplot as plt
from Path import Path, SINUSOID_TRAIN, SINUSOID_TEST
import utils as util

class PathGrapher:
    def __init__(self, gt: Path, estim_file: str = None):
        self.ground_truth = gt
        self.estim = estim_file

    def _load_estimated_path(self):
        positions = []
        orientations = []
        times = []

        with open(self.estim, 'r') as f:
            lines = f.readlines()
            for i, line in enumerate(lines):
                parts = line.strip().split(',')
                if len(parts) != 7:
                    print(f"[WARNING] Skipping malformed line {i}: {line.strip()}")
                    continue

                # Parse values
                x, y, z, qx, qy, qz, qw = map(float, parts)
                positions.append([x, y, z])
                orientations.append([qx, qy, qz, qw])
        times = np.linspace(0, self.ground_truth.t_f, num=(int(self.ground_truth.t_f/util.CAM_DT)-1))
        return times, np.array(positions), np.array(orientations)

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
        if self.estim is not None: 
            ___, est_positions, ___ = self._load_estimated_path()

        plt.figure()
        plt.plot(gt_positions[:, 0], gt_positions[:, 1], label='Ground Truth')
        if self.estim is not None: plt.plot(est_positions[:, 0], est_positions[:, 1], label='Estimated', linestyle='--')
        plt.xlabel('X')
        plt.xlim([-50,50])
        plt.ylabel('Y')
        plt.ylim([-50,50])
        plt.title(f'[{self.ground_truth.name}] XY Trajectory')
        plt.legend()
        plt.grid(True)
        if show_now: plt.show()

    def generate_xz_plot(self, show_now: bool = False):
        times, gt_positions, orientations = self._generate_ground_truth()
        if self.estim is not None: 
            ___, est_positions, ___ = self._load_estimated_path()

        plt.figure()
        plt.plot(gt_positions[:, 0], gt_positions[:, 2], label='Ground Truth')
        if self.estim is not None: plt.plot(est_positions[:, 0], est_positions[:, 2], label='Estimated', linestyle='--')
        plt.xlabel('X')
        plt.xlim([-50,50])
        plt.ylabel('Z')
        plt.ylim([0,75])
        plt.title(f'[{self.ground_truth.name}] XZ Trajectory')
        plt.legend()
        plt.grid(True)
        if show_now: plt.show()

    def generate_3d_plot(self, show_now: bool = False):
        times, gt_positions, orientations = self._generate_ground_truth()
        if self.estim is not None: 
            e_times, est_positions, ___ = self._load_estimated_path()

        fig = plt.figure()
        ax = fig.add_subplot(projection='3d')
        ax.plot(gt_positions[:, 0], gt_positions[:, 1], gt_positions[:, 2], label='Ground Truth')
        if self.estim is not None: ax.plot(est_positions[:, 0], est_positions[:, 1], est_positions[:, 2], label='Estimated', linestyle='--')
        ax.set_xlabel('X')
        ax.set_xlim([-50,50])
        ax.set_ylabel('Y')
        ax.set_ylim([-50,50])
        ax.set_zlabel('Z')
        ax.set_zlim([0,75])
        ax.set_title(f'[{self.ground_truth.name}] 3d Trajectory')
        ax.legend()
        ax.grid(True)
        if show_now: plt.show()

    def generate_orientation_plots(self, show_now: bool = False):
        times, gt_positions, orientations = self._generate_ground_truth('euler')
        if self.estim is not None: 
            ___, est_positions, est_orientations = self._load_estimated_path()
        est_orientations = np.array([util.euler_from_quat(orientation) for orientation in est_orientations])
        
        fig, axs = plt.subplots(3, 1, figsize=(10, 8), sharex=True)
        labels = ['Roll', 'Pitch', 'Yaw']
        
        for i in range(3):
            axs[i].plot(times, orientations[:, i], label=f'GT {labels[i]}')
            axs[i].plot(times, est_orientations[:, i], label=f'Estim {labels[i]}')
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
        self.generate_3d_plot(True)
        # self.generate_orientation_plots(True)


if __name__ == '__main__':
    # path_list = []
    # # path_list.append(PathGrapher(STRAIGHT_LINE))
    # # path_list.append(PathGrapher(CIRCLE))
    # # path_list.append(PathGrapher(SINUSOID_TRAIN))
    # # path_list.append(PathGrapher(FIGURE_EIGHT))
    # # path_list.append(PathGrapher(HYPERBOLIC_PARABOLOID))

    pg = PathGrapher(SINUSOID_TEST, "final_trajectory.csv")

    pg.generate_plots()
    
    # for pg in path_list:
    #     pg.generate_plots()
    # plt.show()