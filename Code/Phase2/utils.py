import numpy as np
import csv
import matplotlib
import math
import ImuUtils
# matplotlib.use('Agg')  # Use non-GUI backend
# import matplotlib.pyplot as plt

IMU_DT= 0.005 # 200Hz
CAM_DT = 0.05 # 10Hz
# sets random vibration to accel with RMS for x/y/z axis - 1/2/3 m/s^2, can be zero or changed to other values
FREQUENCY = 200
ENV_ACC = '[0.03 0.001 0.01]-random'
VIB_DEF_ACC = ImuUtils.vib_from_env(ENV_ACC, FREQUENCY)
ERR_ACC = ImuUtils.accel_high_accuracy
# sets sinusoidal vibration to gyro with frequency being 0.5 Hz and amp for x/y/z axis being 6/5/4 deg/s
ENV_GYRO = '[6 5 4]d-0.5Hz-sinusoidal'
VIB_DEF_GYRO = ImuUtils.vib_from_env(ENV_GYRO, FREQUENCY)
ERR_GYRO = ImuUtils.gyro_high_accuracy

def skew(vec):
    """
    Create a skew-symmetric matrix from a 3-element vector.
    """
    # vec = np.reshape(vec, (3,1))
    x, y, z = vec
    return np.array([
        [0, -z, y],
        [z, 0, -x],
        [-y, x, 0]])

def to_rotation(q):
    """
    Convert a quaternion to the corresponding rotation matrix.
    Pay attention to the convention used. The function follows the
    conversion in "Indirect Kalman Filter for 3D Attitude Estimation:
    A Tutorial for Quaternion Algebra", Equation (78).
    The input quaternion should be in the form [q1, q2, q3, q4(scalar)]
    """
    q = q / np.linalg.norm(q)
    vec = q[:3]
    w = q[3]

    R = (2*w*w-1)*np.identity(3) - 2*w*skew(vec) + 2*vec[:, None]*vec
    return R

def to_quaternion(R):
    """
    Convert a rotation matrix to a quaternion.
    Pay attention to the convention used. The function follows the
    conversion in "Indirect Kalman Filter for 3D Attitude Estimation:
    A Tutorial for Quaternion Algebra", Equation (78).
    The input quaternion should be in the form [q1, q2, q3, q4(scalar)]
    """
    if R[2, 2] < 0:
        if R[0, 0] > R[1, 1]:
            t = 1 + R[0,0] - R[1,1] - R[2,2]
            q = [t, R[0, 1]+R[1, 0], R[2, 0]+R[0, 2], R[1, 2]-R[2, 1]]
        else:
            t = 1 - R[0,0] + R[1,1] - R[2,2]
            q = [R[0, 1]+R[1, 0], t, R[2, 1]+R[1, 2], R[2, 0]-R[0, 2]]
    else:
        if R[0, 0] < -R[1, 1]:
            t = 1 - R[0,0] - R[1,1] + R[2,2]
            q = [R[0, 2]+R[2, 0], R[2, 1]+R[1, 2], t, R[0, 1]-R[1, 0]]
        else:
            t = 1 + R[0,0] + R[1,1] + R[2,2]
            q = [R[1, 2]-R[2, 1], R[2, 0]-R[0, 2], R[0, 1]-R[1, 0], t]

    q = np.array(q) # * 0.5 / np.sqrt(t)
    return q / np.linalg.norm(q)

def quaternion_normalize(q):
    """
    Normalize the given quaternion to unit quaternion.
    """
    return q / np.linalg.norm(q)

def quaternion_conjugate(q):
    """
    Conjugate of a quaternion.
    """
    return np.array([*-q[:3], q[3]])

def quaternion_multiplication(q1, q2):
    """
    Perform q1 * q2
    """
    q1 = q1 / np.linalg.norm(q1)
    q2 = q2 / np.linalg.norm(q2)

    L = np.array([
        [ q1[3],  q1[2], -q1[1], q1[0]],
        [-q1[2],  q1[3],  q1[0], q1[1]],
        [ q1[1], -q1[0],  q1[3], q1[2]],
        [-q1[0], -q1[1], -q1[2], q1[3]]
    ])

    q = L @ q2
    return q / np.linalg.norm(q)


def small_angle_quaternion(dtheta):
    """
    Convert the vector part of a quaternion to a full quaternion.
    This function is useful to convert delta quaternion which is  
    usually a 3x1 vector to a full quaternion.
    For more details, check Equation (238) and (239) in "Indirect Kalman 
    Filter for 3D Attitude Estimation: A Tutorial for quaternion Algebra".
    """
    dq = dtheta / 2.
    dq_square_norm = dq @ dq

    if dq_square_norm <= 1:
        q = np.array([*dq, np.sqrt(1-dq_square_norm)])
    else:
        q = np.array([*dq, 1.])
        q /= np.sqrt(1+dq_square_norm)
    return q


def from_two_vectors(v0, v1):
    """
    Rotation quaternion from v0 to v1.
    """
    v0 = v0 / np.linalg.norm(v0)
    v1 = v1 / np.linalg.norm(v1)
    d = v0 @ v1

    # if dot == -1, vectors are nearly opposite
    if d < -0.999999:
        axis = np.cross([1,0,0], v0)
        if np.linalg.norm(axis) < 0.000001:
            axis = np.cross([0,1,0], v0)
        q = np.array([*axis, 0.])
    elif d > 0.999999:
        q = np.array([0., 0., 0., 1.])
    else:
        s = np.sqrt((1+d)*2)
        axis = np.cross(v0, v1)
        vec = axis / s
        w = 0.5 * s
        q = np.array([*vec, w])
        
    q = q / np.linalg.norm(q)
    return quaternion_conjugate(q)   # hamilton -> JPL

def polar_decomp_quaternion(quat) -> tuple[float, np.ndarray]:
    """ Computes the polar representation of a quaternion [W X Y Z] -> [phi, n] using the following equation:
        \n \t q = |q| * e^(n * phi) = |q| * (cos(phi) + n * sin(phi))
        \n See [stack exchange](https://math.stackexchange.com/questions/1496308/how-can-i-express-a-quaternion-in-polar-form#:~:text=Sorted%20by:,sin%CE%B8=v%7Cv%7C) for impl details. 
        Args:
            quat (np.ndarray): quaternion to decompose
        Returns:
            phi (float): polar anglular component
            n (np.ndarray): unit vector of imaginary part of quaternion (3,)
    """
    phi = math.acos(quat[3] / np.linalg.norm(quat))
    # phi2 = math.asin(np.linalg.norm(quat[1:]) / np.linalg.norm(quat)) # two valid methods for computing phi
    n = quat[:3] / np.linalg.norm(quat[:3])
    return phi, n

def exponentiate_quaternion(quat: np.ndarray, exponent: float) -> np.ndarray:
    """ Computes the exponentiation of a quaternion (quat) by a non-negative power (exponent)
        \n the exponentation of a quaternion is done in polar form using the following equation:
        \n \t q^x = |q|^x * e^(n * phi * x) = |q|^x * (cos(phi * x) + n * sin(phi * x))
        \n see [Quaternion Wiki](https://en.wikipedia.org/wiki/Quaternion#Exponential,_logarithm,_and_power_functions) for impl details.
    
        Args:
            quat (np.ndarray): quaternion (base).
            exponent (float): exponent. Must be a real non-negative number
        Returns:
            result_quat (np.ndarray): result of quat^x
    """
    phi, n = polar_decomp_quaternion(quat)
    magnitude = np.linalg.norm(quat) ** exponent
    w = magnitude * math.cos(exponent * phi)
    v = magnitude * n * math.sin(exponent * phi)
    return np.concatenate((v, np.array([w])))

def slerp(q_1, q_2, t):
        a = quaternion_multiplication(q_1, quaternion_conjugate(q_2)/np.linalg.norm(q_2))   
        b = exponentiate_quaternion(a, t) # b = a^t
        target_quat = quaternion_multiplication(b, q_1) # b * q_1
        return target_quat

def euler_from_quat(q):
    q_w_x = q[3]*q[0]
    q_w_y = q[3]*q[1]
    q_w_z = q[3]*q[2]

    q_x_y = q[0]*q[1]
    q_x_z = q[0]*q[2]

    q_y_z = q[1]*q[2]
    phi = math.atan2(2*(q_w_x+q_y_z), 1 - 2*(q[0]**2 + q[1]**2))
    theta = -math.pi/2 + 2 * math.atan2(math.sqrt(1+2*(q_w_y - q_x_z)), math.sqrt(1 - 2*(q_w_y + q_x_z)))
    psi = math.atan2(2*(q_w_z+q_x_y), 1 - 2*(q[1]**2 + q[2]**2))
    return np.array([phi, theta, psi])


def csv_writer(measured_position_array: np.ndarray):
    np.savetxt("measured_data.txt", measured_position_array, delimiter=',', header='#timestamp,tx,ty,tz,qx,qy,qz,qw')

class Isometry3d(object):
    """
    3d rigid transform.
    """
    def __init__(self, R, t):
        self.R = R
        self.t = t

    def matrix(self):
        m = np.identity(4)
        m[:3, :3] = self.R
        m[:3, 3] = self.t
        return m

    def inverse(self):
        return Isometry3d(self.R.T, -self.R.T @ self.t)

    def __mul__(self, T1):
        R = self.R @ T1.R
        t = self.R @ T1.t + self.t
        return Isometry3d(R, t)
