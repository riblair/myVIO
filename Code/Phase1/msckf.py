import numpy as np
from scipy.stats import chi2
from scipy.spatial.transform import Rotation
from scipy.sparse import csr_matrix
from scipy.linalg import cho_factor, cho_solve

from utils import *
from feature import Feature

import time
from collections import namedtuple
import math



class IMUState(object):
    # id for next IMU state
    next_id = 0

    # Gravity vector in the world frame
    gravity = np.array([0., 0., -9.81]).reshape((3,1))

    # Transformation offset from the IMU frame to the body frame. 
    # The transformation takes a vector from the IMU frame to the 
    # body frame. The z axis of the body frame should point upwards.
    # Normally, this transform should be identity.
    T_imu_body = Isometry3d(np.identity(3), np.zeros(3))

    def __init__(self, new_id=None):
        # An unique identifier for the IMU state.
        self.id = new_id
        # Time when the state is recorded
        self.timestamp = None

        # Orientation
        # Take a vector from the world frame to the IMU (body) frame.
        self.orientation = np.array([0., 0., 0., 1.])

        # Position of the IMU (body) frame in the world frame.
        self.position = np.zeros(3)
        # Velocity of the IMU (body) frame in the world frame.
        self.velocity = np.zeros(3)

        # Bias for measured angular velocity and acceleration.
        self.gyro_bias = np.zeros(3)
        self.acc_bias = np.zeros(3)

        # These three variables should have the same physical
        # interpretation with `orientation`, `position`, and
        # `velocity`. There three variables are used to modify
        # the transition matrices to make the observability matrix
        # have proper null space.
        self.orientation_null = np.array([0., 0., 0., 1.])
        self.position_null = np.zeros((3,1))
        self.velocity_null = np.zeros((3,1))

        # Transformation between the IMU and the left camera (cam0)
        self.R_imu_cam0 = np.identity(3)
        self.t_cam0_imu = np.zeros(3)


class CAMState(object):
    # Takes a vector from the cam0 frame to the cam1 frame.
    R_cam0_cam1 = None
    t_cam0_cam1 = None

    def __init__(self, new_id=None):
        # An unique identifier for the CAM state.
        self.id = new_id
        # Time when the state is recorded
        self.timestamp = None

        # Orientation
        # Take a vector from the world frame to the camera frame.
        self.orientation = np.array([0., 0., 0., 1.])

        # Position of the camera frame in the world frame.
        self.position = np.zeros(3)

        # These two variables should have the same physical
        # interpretation with `orientation` and `position`.
        # There two variables are used to modify the measurement
        # Jacobian matrices to make the observability matrix
        # have proper null space.
        self.orientation_null = np.array([0., 0., 0., 1.])
        self.position_null = np.zeros(3)

        
class StateServer(object):
    """
    Store one IMU states and several camera states for constructing 
    measurement model.
    """
    def __init__(self):
        self.imu_state = IMUState()
        self.cam_states = dict()   # <CAMStateID, CAMState>, ordered dict

        # State covariance matrix
        self.state_cov = np.zeros((21, 21))
        self.continuous_noise_cov = np.zeros((12, 12))



class MSCKF(object):
    def __init__(self, config):
        self.config = config
        self.optimization_config = config.optimization_config

        # IMU data buffer
        # This is buffer is used to handle the unsynchronization or
        # transfer delay between IMU and Image messages.
        self.imu_msg_buffer = []

        # State vector
        self.state_server = StateServer()
        # Features used
        self.map_server = dict()   # <FeatureID, Feature>

        # Chi squared test table.
        # Initialize the chi squared test table with confidence level 0.95.
        self.chi_squared_test_table = dict()
        for i in range(1, 100):
            self.chi_squared_test_table[i] = chi2.ppf(0.05, i)

        # Set the initial IMU state.
        # The intial orientation and position will be set to the origin implicitly.
        # But the initial velocity and bias can be set by parameters.
        # TODO: is it reasonable to set the initial bias to 0?
        self.state_server.imu_state.velocity = config.velocity
        self.reset_state_cov()

        continuous_noise_cov = np.identity(12)
        continuous_noise_cov[:3, :3] *= self.config.gyro_noise
        continuous_noise_cov[3:6, 3:6] *= self.config.gyro_bias_noise
        continuous_noise_cov[6:9, 6:9] *= self.config.acc_noise
        continuous_noise_cov[9:, 9:] *= self.config.acc_bias_noise
        self.state_server.continuous_noise_cov = continuous_noise_cov

        # Gravity vector in the world frame
        IMUState.gravity = config.gravity

        # Transformation between the IMU and the left camera (cam0)
        T_cam0_imu = np.linalg.inv(config.T_imu_cam0)
        self.state_server.imu_state.R_imu_cam0 = T_cam0_imu[:3, :3].T
        self.state_server.imu_state.t_cam0_imu = T_cam0_imu[:3, 3]

        # Extrinsic parameters of camera and IMU.
        T_cam0_cam1 = config.T_cn_cnm1
        CAMState.R_cam0_cam1 = T_cam0_cam1[:3, :3]
        CAMState.t_cam0_cam1 = T_cam0_cam1[:3, 3]
        Feature.R_cam0_cam1 = CAMState.R_cam0_cam1
        Feature.t_cam0_cam1 = CAMState.t_cam0_cam1
        IMUState.T_imu_body = Isometry3d(
            config.T_imu_body[:3, :3],
            config.T_imu_body[:3, 3])

        # Tracking rate.
        self.tracking_rate = None

        # Indicate if the gravity vector is set.
        self.is_gravity_set = False
        # Indicate if the received image is the first one. The system will 
        # start after receiving the first image.
        self.is_first_img = True

    def imu_callback(self, imu_msg):
        """
        Callback function for the imu message.
        """
        # IMU msgs are pushed backed into a buffer instead of being processed 
        # immediately. The IMU msgs are processed when the next image is  
        # available, in which way, we can easily handle the transfer delay.
        self.imu_msg_buffer.append(imu_msg)

        if not self.is_gravity_set:
            if len(self.imu_msg_buffer) >= 200:
                self.initialize_gravity_and_bias()
                self.is_gravity_set = True

    def feature_callback(self, feature_msg):
        """
        Callback function for feature measurements.
        """
        if not self.is_gravity_set:
            return
        start = time.time()

        # Start the system if the first image is received.
        # The frame where the first image is received will be the origin.
        if self.is_first_img:
            self.is_first_img = False
            self.state_server.imu_state.timestamp = feature_msg.timestamp

        t = time.time()

        # Propogate the IMU state.
        # that are received before the image msg.
        self.batch_imu_processing(feature_msg.timestamp)

        print('---batch_imu_processing    ', time.time() - t)
        t = time.time()

        # Augment the state vector.
        self.state_augmentation(feature_msg.timestamp)

        print('---state_augmentation      ', time.time() - t)
        t = time.time()

        # Add new observations for existing features or new features 
        # in the map server.
        self.add_feature_observations(feature_msg)

        print('---add_feature_observations', time.time() - t)
        t = time.time()

        # Perform measurement update if necessary.
        # And prune features and camera states.
        self.remove_lost_features()

        print('---remove_lost_features    ', time.time() - t)
        t = time.time()

        self.prune_cam_state_buffer()

        print('---prune_cam_state_buffer  ', time.time() - t)
        print('---msckf elapsed:          ', time.time() - start, f'({feature_msg.timestamp})')

        try:
            # Publish the odometry.
            return self.publish(feature_msg.timestamp)
        finally:
            # Reset the system if necessary.
            self.online_reset()

    def initialize_gravity_and_bias(self):
        """
        IMPLEMENT THIS!!!!!
        """
        """
        Initialize the IMU bias and initial orientation based on the 
        first few IMU readings.
        """
        # Initialize the gyro_bias given the current angular and linear velocity

        gyro_sum = np.zeros(3)
        acc_sum = np.zeros(3)
        for imu_msg in self.imu_msg_buffer:
            gyro_sum += imu_msg.angular_velocity
            acc_sum += imu_msg.linear_acceleration

        gyro_bias = gyro_sum / len(self.imu_msg_buffer)
        acc_avg = acc_sum / len(self.imu_msg_buffer)

        self.state_server.imu_state.gyro_bias = gyro_bias
        # self.acc_bias = lin_avg

        # Find the gravity in the IMU frame.
        
        # Normalize the gravity and save to IMUState   
        gravity_norm = np.linalg.norm(acc_avg)
        IMUState.gravity = np.array([0., 0., -gravity_norm]).reshape((3,1))

        # Initialize the initial orientation, so that the estimation
        # is consistent with the inertial frame.

        q0_i_w = from_two_vectors(acc_avg, -IMUState.gravity.flatten())
        R0_i_w = to_rotation(q0_i_w)
        self.state_server.imu_state.orientation = to_quaternion(R0_i_w.T)
        # self.state_server.imu_state.orientation = np.array([0,0,0,1])
        return
        # self.state_server.imu_state.orientation = to_quaternion(to_rotation().T)

    # Filter related functions
    # (batch_imu_processing, process_model, predict_new_state)
    def batch_imu_processing(self, time_bound):
        """
        IMPLEMENT THIS!!!!!
        """
        """
        Process the imu message given the time bound
        """
        used_imu_msg_cntr = 0
        # Process the imu messages in the imu_msg_buffer 
        for msg in self.imu_msg_buffer:
            # Repeat until the time_bound is reached
            if msg.timestamp < self.state_server.imu_state.timestamp:
                used_imu_msg_cntr += 1
                continue
            elif msg.timestamp > time_bound:
                break
            
            # Execute process model.
            self.process_model(msg.timestamp, msg.angular_velocity, msg.linear_acceleration)
            used_imu_msg_cntr += 1
            
        # Set the current imu id to be the IMUState.next_id
        self.state_server.imu_state.id = IMUState.next_id
        
        # IMUState.next_id increments
        IMUState.next_id+=1

        # Remove all used IMU msgs.
        self.imu_msg_buffer = self.imu_msg_buffer[used_imu_msg_cntr:]

    def process_model(self, time, m_gyro, m_acc):
        """
        IMPLEMENT THIS!!!!!
        """
        """
        Section III.A: The dynamics of the error IMU state following equation (2) in the "MSCKF" paper.
        """
        # Get the error IMU state
        imu_state = self.state_server.imu_state
        gyro = m_gyro - imu_state.gyro_bias
        acc = m_acc - imu_state.acc_bias
        # test = acc @ to_rotation(self.state_server.imu_state.orientation)
        dtime = time - imu_state.timestamp
        # dtime = 0.005

        # Compute discrete transition F, Q matrices in Appendix A in "MSCKF" paper
        F = np.zeros((21, 21))
        G = np.zeros((21, 12))
        
        F[0:3, 0:3] = -skew(gyro)
        F[0:3, 3:6] = -np.eye(3) # from 3:6 3:6
        F[6:9, 0:3] = -np.transpose(to_rotation(imu_state.orientation)) @ skew(acc)
        F[6:9, 9:12] = -np.transpose(to_rotation(imu_state.orientation))
        F[12:15, 6:9] = np.eye(3)
        
        G[0:3, 0:3] = -np.eye(3)
        G[3:6, 3:6] = np.eye(3)
        G[6:9, 6:9] = -np.transpose(to_rotation(imu_state.orientation))
        G[9:12, 9:12] = np.eye(3)
        
        # Approximate matrix exponential to the 3rd order, which can be 
        # considered to be accurate enough assuming dt is within 0.01s.
        Fdt = F * dtime
        Fdt_square = Fdt @ Fdt
        Fdt_cube = Fdt_square @ Fdt
        Phi = np.eye(21) + Fdt + 0.5*Fdt_square + (1.0/6.0)*Fdt_cube

        # Propogate the state using 4th order Runge-Kutta
        self.predict_new_state(dtime, gyro, acc)

        # Modify the transition matrix
        R_kk_1 = to_rotation(imu_state.orientation_null)
        Phi[0:3, 0:3] = to_rotation(imu_state.orientation) @ np.transpose(R_kk_1)
        
        u = R_kk_1 @ IMUState.gravity
        u = np.reshape(u, (3,1)) 
        s = np.linalg.inv(np.transpose(u) @ u) @ np.transpose(u) # this causes crashes due to matmul / singular matrix errors
        A1 = Phi[6:9, 0:3]
        w1 = skew((imu_state.velocity_null - imu_state.velocity).flatten()) @ IMUState.gravity
        Phi[6:9, 0:3] = A1 - (A1@u-w1)@s
        
        A2 = Phi[12:15, 0:3]
        w2 = skew((dtime*imu_state.velocity_null+imu_state.position_null-imu_state.position).flatten()) @ IMUState.gravity
        Phi[12:15, 0:3] = A2 - (A2@u-w2)@s
        
        # Propogate the state covariance matrix.
        Q = Phi @ (G @ self.state_server.continuous_noise_cov @ G.T) @ Phi.T * dtime
        self.state_server.state_cov[0:21, 0:21] = Phi @ self.state_server.state_cov[0:21, 0:21] @ Phi.T + Q

        if len(self.state_server.cam_states) > 0:
            self.state_server.state_cov[0:21, 21:self.state_server.state_cov.shape[1]] = (
                Phi @ self.state_server.state_cov[0:21, 21:self.state_server.state_cov.shape[1]])
            self.state_server.state_cov[21:self.state_server.state_cov.shape[0], 0:21] = (
                self.state_server.state_cov[21:self.state_server.state_cov.shape[0], 0:21] @ Phi.T)

        
        # Fix the covariance to be symmetric
        state_cov_fixed = (self.state_server.state_cov+self.state_server.state_cov.T) / 2.0
        self.state_server.state_cov = state_cov_fixed
        
        # Update the state correspondes to null space.
        imu_state.orientation_null = imu_state.orientation
        imu_state.position_null = imu_state.position
        imu_state.velocity_null = imu_state.velocity
        imu_state.timestamp = time

        self.state_server.imu_state = imu_state
        # self.state_server.imu_state.timestamp = time
        return
        

    def predict_new_state(self, dt, gyro, acc):
        """
        IMPLEMENT THIS!!!!!
        """
        """Propogate the state using 4th order Runge-Kutta for equstion (1) in "MSCKF" paper"""
        # compute norm of gyro
        
        gyro_norm = np.linalg.norm(gyro)
        
        # Get the Omega matrix, the equation above equation (2) in "MSCKF" paper
        
        omega_mat = np.zeros((4,4))
        omega_mat[0:3, 0:3] = -skew(gyro)
        omega_mat[0:3, 3] = gyro
        omega_mat[3, 0:3] = -gyro
        
        # Get the orientation, velocity, position
        curr_q = self.state_server.imu_state.orientation
        curr_v = np.reshape(self.state_server.imu_state.velocity, (3,1))
        curr_p = np.reshape(self.state_server.imu_state.position, (3,1))
        acc = np.reshape(acc, (3,1))
        
        # Compute the dq_dt, dq_dt2 in equation (1) in "MSCKF" paper
        dq_dt = (math.cos(gyro_norm*dt*0.5)*np.identity(4) + 1/gyro_norm*math.sin(gyro_norm*dt*0.5)*omega_mat) @ curr_q
        dq_dt2 = (math.cos(gyro_norm*dt*0.25)*np.identity(4) + 1/gyro_norm*math.sin(gyro_norm*dt*0.25)*omega_mat) @ curr_q

        dR_dt_transpose = to_rotation(dq_dt).T
        dR_dt2_transpose = to_rotation(dq_dt2).T
        
        # Apply 4th order Runge-Kutta 
        # k1 = f(tn, yn)
        k1_v_dot = to_rotation(curr_q).T @ acc + IMUState.gravity
        k1_p_dot = curr_v

        # k2 = f(tn+dt/2, yn+k1*dt/2)
        k1_v = curr_v + k1_v_dot * dt / 2  # 3x3
        k2_v_dot =  dR_dt2_transpose @ acc + IMUState.gravity
        k2_p_dot = k1_v
        
        # k3 = f(tn+dt/2, yn+k2*dt/2)
        k2_v = curr_v + k2_v_dot * dt / 2
        k3_v_dot = dR_dt2_transpose @ acc + IMUState.gravity
        k3_p_dot = k2_v

        # k4 = f(tn+dt, yn+k3*dt)
        k3_v = curr_v + k3_v_dot * dt
        k4_v_dot = dR_dt_transpose @ acc + IMUState.gravity
        k4_p_dot = k3_v

        # yn+1 = yn + dt/6*(k1+2*k2+2*k3+k4)
        new_q = dq_dt
        new_v = curr_v + dt/6*(k1_v_dot + 2*k2_v_dot + 2*k3_v_dot + k4_v_dot)
        new_p = curr_p + dt/6*(k1_p_dot + 2*k2_p_dot + 2*k3_p_dot + k4_p_dot)

        # update the imu state
        self.state_server.imu_state.orientation = quaternion_normalize(new_q)
        self.state_server.imu_state.velocity = new_v 
        self.state_server.imu_state.position = new_p

    
    def state_augmentation(self, time):
        """
        IMPLEMENT THIS!!!!!
        """
        """
        Compute the state covariance matrix in equation (3) in the "MSCKF" paper.
        """
        # Get the imu_state, rotation from imu to cam0, and translation from cam0 to imu
        r_i_c = self.state_server.imu_state.R_imu_cam0
        t_c_i = np.reshape(self.state_server.imu_state.t_cam0_imu, (3,1))

        # Add a new camera state to the state server.
        r_w_i = to_rotation(self.state_server.imu_state.orientation)
        r_w_c = r_i_c @ r_w_i
        t_c_w = self.state_server.imu_state.position + r_w_i.T @ t_c_i

        new_cam_state = CAMState(self.state_server.imu_state.id)

        new_cam_state.timestamp = time
        new_cam_state.orientation = to_quaternion(r_w_c)
        new_cam_state.position = t_c_w.flatten()

        new_cam_state.orientation_null = new_cam_state.orientation
        new_cam_state.position_null = new_cam_state.position

        self.state_server.cam_states[self.state_server.imu_state.id] = new_cam_state


        # Update the covariance matrix of the state.
        # To simplify computation, the matrix J below is the nontrivial block
        # Appendix B of "MSCKF" paper.
        J_mat = np.zeros((6,21))
        J_mat[0:3, 0:3] = r_i_c
        J_mat[0:3, 15:18] = np.identity(3)

        J_mat[3:6, 0:3] = skew((r_w_i.T @ t_c_i).flatten())
        J_mat[3:6, 12:15] = np.identity(3)
        J_mat[3:6, 18:21] = r_w_i.T

        # Resize the state covariance matrix.
        old_rows = self.state_server.state_cov.shape[0]
        old_cols = self.state_server.state_cov.shape[1]

        new_state_cov = np.zeros((old_rows+6, old_cols+6))
        new_state_cov[0:old_rows, 0:old_cols] = self.state_server.state_cov

        p_11 = new_state_cov[0:21, 0:21]
        p_12 = new_state_cov[0:21, 21:old_cols] # could be wrong...

        # Fill in the augmented state covariance.
        bottom_row = np.hstack((J_mat @ p_11, J_mat @ p_12))

        new_state_cov[old_rows:, :old_cols] = bottom_row
        new_state_cov[:old_rows, old_cols:] = bottom_row.T
        new_state_cov[old_rows:, old_cols:] = J_mat @ p_11 @ J_mat.T

        # Fix the covariance to be symmetric
        self.state_server.state_cov = (new_state_cov + new_state_cov.T) / 2

    def add_feature_observations(self, feature_msg):
        """
        IMPLEMENT THIS!!!!!
        """
        # get the current imu state id and number of current features
        state_id = self.state_server.imu_state.id
        curr_feature_num = len(self.map_server)
        tracked_feature_num = 0
        
        # add all features in the feature_msg to self.map_server
        for feature in feature_msg.features:
            # if self.map_server.get(feature.id) == len(self.map_server) - 1:  # equivalent to C++ end()
            if feature.id not in self.map_server:
                self.map_server[feature.id] = Feature(feature.id, self.optimization_config)
                self.map_server[feature.id].observations[state_id] = np.array([feature.u0, feature.v0, feature.u1, feature.v1]).reshape((4,1))
            else:
                self.map_server[feature.id].observations[state_id] = np.array([feature.u0, feature.v0, feature.u1, feature.v1]).reshape((4,1))
                tracked_feature_num += 1

        # update the tracking rate
        if curr_feature_num != 0:
            self.tracking_rate = tracked_feature_num / curr_feature_num
        return

    def measurement_jacobian(self, cam_state_id, feature_id):
        """
        This function is used to compute the measurement Jacobian
        for a single feature observed at a single camera frame.
        """
        # Prepare all the required data.
        cam_state = self.state_server.cam_states[cam_state_id]
        feature = self.map_server[feature_id]

        # Cam0 pose.
        R_w_c0 = to_rotation(cam_state.orientation)
        t_c0_w = cam_state.position

        # Cam1 pose.
        R_w_c1 = CAMState.R_cam0_cam1 @ R_w_c0
        t_c1_w = t_c0_w - R_w_c1.T @ CAMState.t_cam0_cam1

        # 3d feature position in the world frame.
        # And its observation with the stereo cameras.
        p_w = feature.position
        z = feature.observations[cam_state_id]

        # Convert the feature position from the world frame to
        # the cam0 and cam1 frame.
        p_c0 = R_w_c0 @ (p_w - t_c0_w)
        p_c1 = R_w_c1 @ (p_w - t_c1_w)

        # Compute the Jacobians.
        dz_dpc0 = np.zeros((4, 3))
        dz_dpc0[0, 0] = 1 / p_c0[2]
        dz_dpc0[1, 1] = 1 / p_c0[2]
        dz_dpc0[0, 2] = -p_c0[0] / (p_c0[2] * p_c0[2])
        dz_dpc0[1, 2] = -p_c0[1] / (p_c0[2] * p_c0[2])

        dz_dpc1 = np.zeros((4, 3))
        dz_dpc1[2, 0] = 1 / p_c1[2]
        dz_dpc1[3, 1] = 1 / p_c1[2]
        dz_dpc1[2, 2] = -p_c1[0] / (p_c1[2] * p_c1[2])
        dz_dpc1[3, 2] = -p_c1[1] / (p_c1[2] * p_c1[2])

        dpc0_dxc = np.zeros((3, 6))
        dpc0_dxc[:, :3] = skew(p_c0)
        dpc0_dxc[:, 3:] = -R_w_c0

        dpc1_dxc = np.zeros((3, 6))
        dpc1_dxc[:, :3] = CAMState.R_cam0_cam1 @ skew(p_c0)
        dpc1_dxc[:, 3:] = -R_w_c1

        dpc0_dpg = R_w_c0
        dpc1_dpg = R_w_c1

        H_x = dz_dpc0 @ dpc0_dxc + dz_dpc1 @ dpc1_dxc   # shape: (4, 6)
        H_f = dz_dpc0 @ dpc0_dpg + dz_dpc1 @ dpc1_dpg   # shape: (4, 3)

        # Modifty the measurement Jacobian to ensure observability constrain.
        A = H_x   # shape: (4, 6)
        u = np.zeros(6)
        u[:3] = (to_rotation(cam_state.orientation_null) @ IMUState.gravity).flatten()
        u[3:] = (skew(p_w - cam_state.position_null) @ IMUState.gravity).flatten()

        H_x = A - (A @ u)[:, None] * u / (u @ u)
        H_f = -H_x[:4, 3:6]

        # Compute the residual.
        r = z.flatten() - np.array([*p_c0[:2]/p_c0[2], *p_c1[:2]/p_c1[2]])

        # H_x: shape (4, 6)
        # H_f: shape (4, 3)
        # r  : shape (4,)
        return H_x, H_f, r

    def feature_jacobian(self, feature_id, cam_state_ids):
        """
        This function computes the Jacobian of all measurements viewed 
        in the given camera states of this feature.
        """
        feature = self.map_server[feature_id]

        # Check how many camera states in the provided camera id 
        # camera has actually seen this feature.
        valid_cam_state_ids = []
        for cam_id in cam_state_ids:
            if cam_id in feature.observations:
                valid_cam_state_ids.append(cam_id)

        jacobian_row_size = 4 * len(valid_cam_state_ids)

        cam_states = self.state_server.cam_states
        H_xj = np.zeros((jacobian_row_size, 
            21+len(self.state_server.cam_states)*6))
        H_fj = np.zeros((jacobian_row_size, 3))
        r_j = np.zeros(jacobian_row_size)

        stack_count = 0
        for cam_id in valid_cam_state_ids:
            H_xi, H_fi, r_i = self.measurement_jacobian(cam_id, feature.id)

            # Stack the Jacobians.
            idx = list(self.state_server.cam_states.keys()).index(cam_id)
            H_xj[stack_count:stack_count+4, 21+6*idx:21+6*(idx+1)] = H_xi
            H_fj[stack_count:stack_count+4, :3] = H_fi
            r_j[stack_count:stack_count+4] = r_i
            stack_count += 4

        # Project the residual and Jacobians onto the nullspace of H_fj.
        # svd of H_fj
        U, _, _ = np.linalg.svd(H_fj)
        A = U[:, 3:]

        H_x = A.T @ H_xj
        r = A.T @ r_j

        return H_x, r

    def measurement_update(self, H, r):
        """
        IMPLEMENT THIS!!!!!
        """
        """
        Section III.B: by stacking multiple observations, we can compute the residuals in equation (6) in "MSCKF" paper 
        """
        # Check if H and r are empty
        if H.size == 0 or r.size == 0:  # Could also check for size, but C++ checks for zero. Kept this for consistency
            return

        # Decompose the final Jacobian matrix to reduce computational
        # complexity.
        # H_thin = None  # C++ assigns them as just empty matricies. They get values later in code, but this should be a temp fix.
        # r_thin = None
        
        if H.shape[0] > H.shape[1]:
            H_sparse = csr_matrix(H).toarray()
            Q, R = np.linalg.qr(H_sparse)
            
            H_temp = Q.T @ H  # shape: (k, n)
            r_temp = Q.T @ r  # shape: (k,)

            # Compute the number of rows to keep
            num_rows = 21 + 6 * len(self.state_server.cam_states)

            # Take the top rows as in Eigen's .topRows() and .head()
            H_thin = H_temp[:num_rows, :]
            r_thin = r_temp[:num_rows]
            
        else:
            H_thin = H
            r_thin = r

        # Compute the Kalman gain.
        P = self.state_server.state_cov
        S = H_thin @ P @ H_thin.T + self.config.observation_noise * np.eye(H_thin.shape[0])
        c, lower = cho_factor(S)
        K_transpose = cho_solve((c, lower), H_thin @ P)
        K = K_transpose.T

        # Compute the error of the state.
        delta_x = (K @ r_thin[:, np.newaxis]).flatten() # hopefully this is always 40,1 
        
        # Update the IMU state.
        delta_x_imu = delta_x[:21]
        
        if (np.linalg.norm(delta_x_imu[6:9]) > 0.5 or np.linalg.norm(delta_x_imu[12:15]) > 1.0):
            print("delta velocity: %f\n", np.linalg.norm(delta_x_imu.delta_x_imu[6:9]))
            print("delta position: %f\n", np.linalg.norm(delta_x_imu.delta_x_imu[12:15]))

        dq_imu = small_angle_quaternion(delta_x_imu[0:3])
        self.state_server.imu_state.orientation = quaternion_multiplication(dq_imu, self.state_server.imu_state.orientation)
        self.state_server.imu_state.gyro_bias += delta_x_imu[3:6]
        self.state_server.imu_state.velocity += (delta_x_imu[6:9]).reshape((3,1))
        self.state_server.imu_state.acc_bias += (delta_x_imu[9:12])
        print(f"state_server imu_acc bias: {self.state_server.imu_state.acc_bias}\n")
        self.state_server.imu_state.position += (delta_x_imu[12:15]).reshape((3,1))
        
        dq_extrinsic = small_angle_quaternion(delta_x_imu[15:18]) # was 0:3 instead of 15:18?
        self.state_server.imu_state.R_imu_cam0 = to_rotation(dq_extrinsic) @ self.state_server.imu_state.R_imu_cam0
        self.state_server.imu_state.t_cam0_imu += delta_x_imu[18:21]
        
        # Update the camera states.
        # cam_state_iter = iter(self.state_server.cam_states)
        # for i in range(len(self.state_server.cam_states)):
        #     delta_x_cam = delta_x[21+i*6: 21+i+6 + 6]
        #     dq_cam = small_angle_quaternion(delta_x_cam[0:3])
        #     cam_state_iter[1].orientation = quaternion_multiplication(
        #         dq_cam, cam_state_iter[1].orientation
        #     )
        #     cam_state_iter[1].position += delta_x_cam[len(delta_x_cam)-4:]
        
        for i, (cam_id, cam_state) in enumerate(self.state_server.cam_states.items()):
            delta_x_cam = delta_x[21 + i*6 : 21 + (i+1)*6]        # Equivalent to segment<6>(21+i*6)
            dq_cam = small_angle_quaternion(delta_x_cam[:3])        # head<3>()
            cam_state.orientation = quaternion_multiplication(dq_cam, cam_state.orientation)
            cam_state.position += delta_x_cam[3:]  

        # Update state covariance.
        I_KH = np.eye(K.shape[0], H_thin.shape[1]) - K@H_thin
        self.state_server.state_cov = I_KH @ self.state_server.state_cov

        # Fix the covariance to be symmetric
        state_cov_fixed = (self.state_server.state_cov + self.state_server.state_cov.T) / 2.0
        self.state_server.state_cov = state_cov_fixed

    def gating_test(self, H, r, dof):
        P1 = H @ self.state_server.state_cov @ H.T
        P2 = self.config.observation_noise * np.identity(len(H))
        gamma = r @ np.linalg.solve(P1+P2, r)

        if(gamma < self.chi_squared_test_table[dof]):
            return True
        else:
            return False

    def remove_lost_features(self):
        # Remove the features that lost track.
        # BTW, find the size the final Jacobian matrix and residual vector.
        jacobian_row_size = 0
        invalid_feature_ids = []
        processed_feature_ids = []

        for feature in self.map_server.values():
            # Pass the features that are still being tracked.
            if self.state_server.imu_state.id in feature.observations:
                continue
            if len(feature.observations) < 3:
                invalid_feature_ids.append(feature.id)
                continue

            # Check if the feature can be initialized if it has not been.
            if not feature.is_initialized:
                # Ensure there is enough translation to triangulate the feature
                if not feature.check_motion(self.state_server.cam_states):
                    invalid_feature_ids.append(feature.id)
                    continue

                # Intialize the feature position based on all current available 
                # measurements.
                ret = feature.initialize_position(self.state_server.cam_states)
                if ret is False:
                    invalid_feature_ids.append(feature.id)
                    continue

            jacobian_row_size += (4 * len(feature.observations) - 3)
            processed_feature_ids.append(feature.id)

        # Remove the features that do not have enough measurements.
        for feature_id in invalid_feature_ids:
            del self.map_server[feature_id]

        # Return if there is no lost feature to be processed.
        if len(processed_feature_ids) == 0:
            return

        H_x = np.zeros((jacobian_row_size, 
            21+6*len(self.state_server.cam_states)))
        r = np.zeros(jacobian_row_size)
        stack_count = 0

        # Process the features which lose track.
        for feature_id in processed_feature_ids:
            feature = self.map_server[feature_id]

            cam_state_ids = []
            for cam_id, measurement in feature.observations.items():
                cam_state_ids.append(cam_id)

            H_xj, r_j = self.feature_jacobian(feature.id, cam_state_ids)

            if self.gating_test(H_xj, r_j, len(cam_state_ids)-1):
                H_x[stack_count:stack_count+H_xj.shape[0], :H_xj.shape[1]] = H_xj
                r[stack_count:stack_count+len(r_j)] = r_j
                stack_count += H_xj.shape[0]

            # Put an upper bound on the row size of measurement Jacobian,
            # which helps guarantee the executation time.
            if stack_count > 1500:
                break

        H_x = H_x[:stack_count]
        r = r[:stack_count]

        # Perform the measurement update step.
        self.measurement_update(H_x, r)

        # Remove all processed features from the map.
        for feature_id in processed_feature_ids:
            del self.map_server[feature_id]

    def find_redundant_cam_states(self):
        # Move the iterator to the key position.
        cam_state_pairs = list(self.state_server.cam_states.items())

        key_cam_state_idx = len(cam_state_pairs) - 4
        cam_state_idx = key_cam_state_idx + 1
        first_cam_state_idx = 0

        # Pose of the key camera state.
        key_position = cam_state_pairs[key_cam_state_idx][1].position
        key_rotation = to_rotation(
            cam_state_pairs[key_cam_state_idx][1].orientation)

        rm_cam_state_ids = []

        # Mark the camera states to be removed based on the
        # motion between states.
        for i in range(2):
            position = cam_state_pairs[cam_state_idx][1].position
            rotation = to_rotation(
                cam_state_pairs[cam_state_idx][1].orientation)
            
            distance = np.linalg.norm(position - key_position)
            angle = 2 * np.arccos(to_quaternion(
                rotation @ key_rotation.T)[-1])

            if angle < 0.2618 and distance < 0.4 and self.tracking_rate > 0.5:
                rm_cam_state_ids.append(cam_state_pairs[cam_state_idx][0])
                cam_state_idx += 1
            else:
                rm_cam_state_ids.append(cam_state_pairs[first_cam_state_idx][0])
                first_cam_state_idx += 1
                cam_state_idx += 1

        # Sort the elements in the output list.
        rm_cam_state_ids = sorted(rm_cam_state_ids)
        return rm_cam_state_ids


    def prune_cam_state_buffer(self):
        if len(self.state_server.cam_states) < self.config.max_cam_state_size:
            return

        # Find two camera states to be removed.
        rm_cam_state_ids = self.find_redundant_cam_states()

        # Find the size of the Jacobian matrix.
        jacobian_row_size = 0
        for feature in self.map_server.values():
            # Check how many camera states to be removed are associated
            # with this feature.
            involved_cam_state_ids = []
            for cam_id in rm_cam_state_ids:
                if cam_id in feature.observations:
                    involved_cam_state_ids.append(cam_id)

            if len(involved_cam_state_ids) == 0:
                continue
            if len(involved_cam_state_ids) == 1:
                del feature.observations[involved_cam_state_ids[0]]
                continue

            if not feature.is_initialized:
                # Check if the feature can be initialize.
                if not feature.check_motion(self.state_server.cam_states):
                    # If the feature cannot be initialized, just remove
                    # the observations associated with the camera states
                    # to be removed.
                    for cam_id in involved_cam_state_ids:
                        del feature.observations[cam_id]
                    continue

                ret = feature.initialize_position(self.state_server.cam_states)
                if ret is False:
                    for cam_id in involved_cam_state_ids:
                        del feature.observations[cam_id]
                    continue

            jacobian_row_size += 4*len(involved_cam_state_ids) - 3

        # Compute the Jacobian and residual.
        H_x = np.zeros((jacobian_row_size, 21+6*len(self.state_server.cam_states)))
        r = np.zeros(jacobian_row_size)

        stack_count = 0
        for feature in self.map_server.values():
            # Check how many camera states to be removed are associated
            # with this feature.
            involved_cam_state_ids = []
            for cam_id in rm_cam_state_ids:
                if cam_id in feature.observations:
                    involved_cam_state_ids.append(cam_id)

            if len(involved_cam_state_ids) == 0:
                continue

            H_xj, r_j = self.feature_jacobian(feature.id, involved_cam_state_ids)

            if self.gating_test(H_xj, r_j, len(involved_cam_state_ids)):
                H_x[stack_count:stack_count+H_xj.shape[0], :H_xj.shape[1]] = H_xj
                r[stack_count:stack_count+len(r_j)] = r_j
                stack_count += H_xj.shape[0]

            for cam_id in involved_cam_state_ids:
                del feature.observations[cam_id]

        H_x = H_x[:stack_count]
        r = r[:stack_count]

        # Perform measurement update.
        self.measurement_update(H_x, r)

        for cam_id in rm_cam_state_ids:
            idx = list(self.state_server.cam_states.keys()).index(cam_id)
            cam_state_start = 21 + 6*idx
            cam_state_end = cam_state_start + 6

            # Remove the corresponding rows and columns in the state
            # covariance matrix.
            state_cov = self.state_server.state_cov.copy()
            if cam_state_end < state_cov.shape[0]:
                size = state_cov.shape[0]
                state_cov[cam_state_start:-6, :] = state_cov[cam_state_end:, :]
                state_cov[:, cam_state_start:-6] = state_cov[:, cam_state_end:]
            self.state_server.state_cov = state_cov[:-6, :-6]

            # Remove this camera state in the state vector.
            del self.state_server.cam_states[cam_id]

    def reset_state_cov(self):
        """
        Reset the state covariance.
        """
        state_cov = np.zeros((21, 21))
        state_cov[ 3: 6,  3: 6] = self.config.gyro_bias_cov * np.identity(3)
        state_cov[ 6: 9,  6: 9] = self.config.velocity_cov * np.identity(3)
        state_cov[ 9:12,  9:12] = self.config.acc_bias_cov * np.identity(3)
        state_cov[15:18, 15:18] = self.config.extrinsic_rotation_cov * np.identity(3)
        state_cov[18:21, 18:21] = self.config.extrinsic_translation_cov * np.identity(3)
        self.state_server.state_cov = state_cov

    def reset(self):
        """
        Reset the VIO to initial status.
        """
        # Reset the IMU state.
        imu_state = IMUState()
        imu_state.id = self.state_server.imu_state.id
        imu_state.R_imu_cam0 = self.state_server.imu_state.R_imu_cam0
        imu_state.t_cam0_imu = self.state_server.imu_state.t_cam0_imu
        self.state_server.imu_state = imu_state

        # Remove all existing camera states.
        self.state_server.cam_states.clear()

        # Reset the state covariance.
        self.reset_state_cov()

        # Clear all exsiting features in the map.
        self.map_server.clear()

        # Clear the IMU msg buffer.
        self.imu_msg_buffer.clear()

        # Reset the starting flags.
        self.is_gravity_set = False
        self.is_first_img = True

    def online_reset(self):
        """
        Reset the system online if the uncertainty is too large.
        """
        # Never perform online reset if position std threshold is non-positive.
        if self.config.position_std_threshold <= 0:
            return

        # Check the uncertainty of positions to determine if 
        # the system can be reset.
        position_x_std = np.sqrt(self.state_server.state_cov[12, 12])
        position_y_std = np.sqrt(self.state_server.state_cov[13, 13])
        position_z_std = np.sqrt(self.state_server.state_cov[14, 14])

        if max(position_x_std, position_y_std, position_z_std 
            ) < self.config.position_std_threshold:
            return

        print('Start online reset...')

        # Remove all existing camera states.
        self.state_server.cam_states.clear()

        # Clear all exsiting features in the map.
        self.map_server.clear()

        # Reset the state covariance.
        self.reset_state_cov()

    def publish(self, time):
        imu_state = self.state_server.imu_state
        print('+++publish:')
        print('   timestamp:', imu_state.timestamp)
        print('   orientation:', imu_state.orientation)
        print('   position:', imu_state.position)
        print('   velocity:', imu_state.velocity)
        print()
        
        T_i_w = Isometry3d(
            to_rotation(imu_state.orientation).T,
            imu_state.position)
        T_b_w = IMUState.T_imu_body * T_i_w * IMUState.T_imu_body.inverse()
        body_velocity = IMUState.T_imu_body.R @ imu_state.velocity

        R_w_c = imu_state.R_imu_cam0 @ T_i_w.R.T
        t_c_w = (imu_state.position + T_i_w.R @ np.reshape(imu_state.t_cam0_imu, (3,1))).flatten()
        T_c_w = Isometry3d(R_w_c.T, t_c_w)
        if t_c_w.size == 9:
            pass

        return namedtuple('vio_result', ['timestamp', 'pose', 'velocity', 'cam0_pose'])(
            time, T_b_w, body_velocity, T_c_w)
