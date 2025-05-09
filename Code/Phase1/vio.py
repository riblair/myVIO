
from queue import Queue
from threading import Thread
from utils import *
import numpy as np

from config import ConfigEuRoC
from image import ImageProcessor
from msckf import MSCKF

class VIO(object):
    def __init__(self, config, img_queue, imu_queue, viewer=None):
        self.config = config
        self.viewer = viewer

        self.img_queue = img_queue
        self.imu_queue = imu_queue
        self.feature_queue = Queue()

        self.image_processor = ImageProcessor(config)
        self.msckf = MSCKF(config)

        self.img_thread = Thread(target=self.process_img)
        self.imu_thread = Thread(target=self.process_imu)
        self.vio_thread = Thread(target=self.process_feature)
        self.img_thread.start()
        self.imu_thread.start()
        self.vio_thread.start()

    def process_img(self):
        while True:
            img_msg = self.img_queue.get()
            if img_msg is None:
                self.feature_queue.put(None)
                return
            # print('img_msg', img_msg.timestamp)

            if self.viewer is not None:
                self.viewer.update_image(img_msg.cam0_image)

            feature_msg = self.image_processor.stareo_callback(img_msg)

            if feature_msg is not None:
                self.feature_queue.put(feature_msg)

    def process_imu(self):
        while True:
            imu_msg = self.imu_queue.get()
            if imu_msg is None:
                return
            # print('imu_msg', imu_msg.timestamp)

            self.image_processor.imu_callback(imu_msg)
            self.msckf.imu_callback(imu_msg)

    def process_feature(self):
        step_count = 0
        # measured_position_array = np.zeros((3,1))
        measured_data_array = np.zeros((1,8))
        measured_data = np.zeros(8)
        while True:
            feature_msg = self.feature_queue.get()
            if feature_msg is None:
                # graph_positional_error(measured_position_array)
                csv_writer(measured_data_array[1:, :])
                return
            print('feature_msg', feature_msg.timestamp)
            result = self.msckf.feature_callback(feature_msg)

            if result is not None:
                measured_data[0] = feature_msg.timestamp
                measured_data[1:4] = self.msckf.state_server.imu_state.position.flatten()
                measured_data[4] = self.msckf.state_server.imu_state.orientation[3]
                measured_data[5:] = self.msckf.state_server.imu_state.orientation[:3].flatten()

                measured_data_array = np.vstack((measured_data_array, measured_data[np.newaxis, :]))
                # measured_position_array = np.hstack((measured_position_array, measured_position))

            step_count += 1
            if result is not None and self.viewer is not None:
                self.viewer.update_pose(result.cam0_pose)
        


if __name__ == '__main__':
    import time
    import argparse

    from dataset import EuRoCDataset, DataPublisher
    from viewer import Viewer

    parser = argparse.ArgumentParser()
    parser.add_argument('--path', type=str, default='data', 
        help='Path of EuRoC MAV dataset.')
    parser.add_argument('--view', action='store_true', help='Show trajectory.')
    args = parser.parse_args()

    if args.view:
        viewer = Viewer()
        # pass
    else:
        viewer = None

    dataset = EuRoCDataset(args.path)
    dataset.set_starttime(offset=40.)   # start from static state
    dataset.starttime

    img_queue = Queue()
    imu_queue = Queue()
    # gt_queue = Queue()

    config = ConfigEuRoC()
    msckf_vio = VIO(config, img_queue, imu_queue, viewer=viewer)


    duration = float('inf')
    ratio = 0.4  # make it smaller if image processing and MSCKF computation is slow
    imu_publisher = DataPublisher(
        dataset.imu, imu_queue, duration, ratio)
    img_publisher = DataPublisher(
        dataset.stereo, img_queue, duration, ratio)

    now = time.time()
    imu_publisher.start(now)
    img_publisher.start(now)
