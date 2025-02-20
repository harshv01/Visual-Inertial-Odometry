import numpy as np
import scipy
from scipy.stats import chi2

from utils import *
from feature import Feature

import time
from collections import namedtuple


class IMUState(object):
    # id for next IMU state
    next_id = 0

    # Gravity vector in the world frame
    gravity = np.array([0., 0., -9.81])

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
        self.position_null = np.zeros(3)
        self.velocity_null = np.zeros(3)

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
        self.cam_states = dict()  # <CAMStateID, CAMState>, ordered dict

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
        self.map_server = dict()  # <FeatureID, Feature>

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

        # print('---batch_imu_processing    ', time.time() - t)
        t = time.time()

        # Augment the state vector.
        self.state_augmentation(feature_msg.timestamp)

        # print('---state_augmentation      ', time.time() - t)
        t = time.time()

        # Add new observations for existing features or new features 
        # in the map server.
        self.add_feature_observations(feature_msg)

        # print('---add_feature_observations', time.time() - t)
        t = time.time()

        # Perform measurement update if necessary.
        # And prune features and camera states.
        self.remove_lost_features()

        # print('---remove_lost_features    ', time.time() - t)
        t = time.time()

        self.prune_cam_state_buffer()

        # print('---prune_cam_state_buffer  ', time.time() - t)
        # print('---msckf elapsed:          ', time.time() - start, f'({feature_msg.timestamp})')

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
        grav_bias = np.mean([iter.angular_velocity for iter in self.imu_msg_buffer], axis=0)
        self.state_server.imu_state.gyro_bias = grav_bias

        # Find the gravity in the IMU frame.
        grav_imu = np.mean([g.linear_acceleration for g in self.imu_msg_buffer], axis=0)

        # Normalize the gravity and save to IMUState
        grav_norm = np.linalg.norm(grav_imu)
        grav = np.array([0, 0, -grav_norm])
        IMUState.gravity = grav

        # Initialize the initial orientation, so that the estimation
        # is consistent with the inertial frame.
        self.state_server.imu_state.orientation = from_two_vectors(-grav, grav_imu)

    # Filter related functions
    # (batch_imu_processing, process_model, predict_new_state)
    def batch_imu_processing(self, time_bound):
        """
        IMPLEMENT THIS!!!!!
        """
        """
        Process the imu message given the time bound
        """
        # Process the imu messages in the imu_msg_buffer 
        # Execute process model.
        # Update the state info
        # Repeat until the time_bound is reached

        counter = 0
        for iter in self.imu_msg_buffer:
            if iter.timestamp < self.state_server.imu_state.timestamp:
                counter += 1
                continue
            if iter.timestamp > time_bound:
                break

            self.process_model(iter.timestamp, iter.angular_velocity, iter.linear_acceleration)
            counter += 1
            self.state_server.imu_state.timestamp = iter.timestamp

        # Set the current imu id to be the IMUState.next_id
        self.state_server.imu_state.id = self.state_server.imu_state.next_id

        # IMUState.next_id increments
        self.state_server.imu_state.next_id += 1

        # Remove all used IMU msgs.
        # for i in range(msg_count):
        #     self.imu_msg_buffer.pop(0)
        self.imu_msg_buffer = self.imu_msg_buffer[counter:]

    def process_model(self, time, m_gyro, m_acc):
        """
        IMPLEMENT THIS!!!!!
        """
        """
        Section III.A: The dynamics of the error IMU state following equation (2) in the "MSCKF" paper.
        """
        # Get the error IMU state
        w_cap = m_gyro - self.state_server.imu_state.gyro_bias
        a_cap = m_acc - self.state_server.imu_state.acc_bias
        dt = time - self.state_server.imu_state.timestamp
        q_hat_g_I = self.state_server.imu_state.orientation

        # Compute discrete transition F, Q matrices in Appendix A in "MSCKF" paper
        F = np.zeros((21, 21))
        G = np.zeros((21, 12))
        
        F[:3, :3] = -skew(w_cap)
        G[:3, :3] = -np.eye(3)

        F[:3, 3:6] = -np.eye(3)
        G[3:6, 3:6] = np.eye(3)

        F[6:9, :3] = -to_rotation(q_hat_g_I).T @ skew(a_cap)
        G[6:9, 6:9] = -to_rotation(q_hat_g_I).T

        G[12:15, -3:] = np.eye(3)
        F[6:9, 9:12] = -to_rotation(q_hat_g_I).T
        F[12:15, 6:9] = np.eye(3)


        # Approximate matrix exponential to the 3rd order, which can be
        # considered to be accurate enough assuming dt is within 0.01s.
        Fdt = F * dt
        F_dtt = Fdt @ Fdt
        F_dttt = F_dtt @ Fdt
        exp_3 = np.eye(21) + Fdt + F_dtt / 2.0 + F_dttt / 6.0  # Taylor Series

        # Propagate the state using 4th order Runge-Kutta
        self.predict_new_state(dt, w_cap, a_cap)

        # Modify the transition matrix
        Rnull = to_rotation(self.state_server.imu_state.orientation_null)
        exp_3[:3, :3] = to_rotation(self.state_server.imu_state.orientation) @ Rnull.T

        grav_r = Rnull @ self.state_server.imu_state.gravity
        grav_r2 = grav_r / (grav_r @ grav_r)

        A = exp_3[6:9, :3]
        w = skew(self.state_server.imu_state.velocity_null - self.state_server.imu_state.velocity)
        w = w @ self.state_server.imu_state.gravity
        exp_3[6:9, :3] = A - (A @ grav_r - w)[:, None] * grav_r2

        A2 = exp_3[12:15, :3]
        w2 = skew(dt * self.state_server.imu_state.velocity_null +
                  self.state_server.imu_state.position_null - self.state_server.imu_state.position)
        w2 = w2 @ self.state_server.imu_state.gravity
        exp_3[12:15, :3] = A2 - (A2 @ grav_r - w2)[:, None] * grav_r2

        # Propagate the state covariance matrix.
        Q = exp_3 @ G @ self.state_server.continuous_noise_cov @ G.T @ exp_3.T * dt
        self.state_server.state_cov[:21, :21] = exp_3 @ self.state_server.state_cov[:21, :21] @ exp_3.T + Q
        if len(self.state_server.cam_states) > 0:
            self.state_server.state_cov[:21, 21:] = exp_3 @ self.state_server.state_cov[:21, 21:]
            self.state_server.state_cov[21:, :21] = self.state_server.state_cov[21:, :21] @ exp_3.T


        # Fix the covariance to be symmetric
        self.state_server.state_cov += self.state_server.state_cov.T
        self.state_server.state_cov /= 2.0

        # Update the state correspondences to null space.
        self.state_server.imu_state.orientation_null = self.state_server.imu_state.orientation
        self.state_server.imu_state.position_null = self.state_server.imu_state.position
        self.state_server.imu_state.velocity_null = self.state_server.imu_state.velocity

    def predict_new_state(self, dt, gyro, acc):
        """
        IMPLEMENT THIS!!!!!
        """
        """Propagate the state using 4th order Runge-Kutta for equations (1) in "MSCKF" paper"""
        # compute norm of gyro
        gyro_norm = np.linalg.norm(gyro)

        # Get the Omega matrix, the equation above equation (2) in "MSCKF" paper
        w_matrix = np.vstack((np.hstack((-skew(gyro), gyro[:, None])),
                               np.hstack((-gyro, [0]))))

        # Get the orientation, velocity, position
        orientation = self.state_server.imu_state.orientation
        velocity = self.state_server.imu_state.velocity
        position = self.state_server.imu_state.position
        g_G = self.state_server.imu_state.gravity

        # Compute the dq_dt, dq_dt2 in equation (1) in "MSCKF" paper
        # dq_dt is derived the same as the OpenVINS explanation: https://docs.openvins.com/propagation.html
        dq_dt = None
        dq_dtt = None

        # check if gyro_norm too small and would cause divide by zeros error
        if gyro_norm < 1e-04:
            dq_dt = (np.eye(4) + w_matrix / 2.0 * dt) * np.cos(gyro_norm / 2.0) @ orientation
            dq_dtt = (np.eye(4) + w_matrix / 4.0 * dt) * np.cos(gyro_norm / 4.0) @ orientation
        else:
            dq_dt = (np.cos(gyro_norm / 2.0 * dt) * np.eye(4)
                     + 1. / gyro_norm * np.sin(gyro_norm / 2.0 * dt) * w_matrix) @ orientation
            dq_dtt = (np.cos(gyro_norm / 4.0 * dt) * np.eye(4)
                      + 1. / gyro_norm * np.sin(gyro_norm / 4.0 * dt) * w_matrix) @ orientation

        dq_dt_RT = to_rotation(dq_dt).T
        dq_dt2_RT = to_rotation(dq_dtt).T

        # Apply 4th order Runge-Kutta
        # k1 = f(tn, yn)
        k1vd = to_rotation(orientation).T @ acc + g_G
        k1pd = velocity
        k1_v = k1vd * dt/2. + velocity

        # k2 = f(tn+dt/2, yn+k1*dt/2)
        k2vd = dq_dt_RT @ acc + g_G
        k2pd = k1_v
        k2_v = velocity + k2vd * dt/2.

        # k3 = f(tn+dt/2, yn+k2*dt/2)
        k3vd = dq_dt2_RT @ acc + g_G
        k3pd = k2_v
        k3_v = velocity + k3vd * dt/2.

        # k4 = f(tn+dt, yn+k3*dt)
        k4vd = dq_dt_RT @ acc + g_G
        k4pd = k3_v

        # yn+1 = yn + dt/6*(k1+2*k2+2*k3+k4)
        q_norm = quaternion_normalize(dq_dt)
        # Use left, 2*middle1, 2*middle2, right
        new_v = velocity + dt/6. * (k1vd + 2*k2vd + 2*k3vd + k4vd)
        new_p = position + dt/6. * (k1pd + 2*k2pd + 2*k3pd + k4pd)

        # update the imu state
        self.state_server.imu_state.velocity = new_v
        self.state_server.imu_state.position = new_p
        self.state_server.imu_state.orientation = q_norm

    def state_augmentation(self, time):
        """
        IMPLEMENT THIS!!!!!
        """
        """
        Compute the state covariance matrix in equation (3) in the "MSCKF" paper.
        """
        # Get the imu_state, rotation from imu to cam0, and translation from cam0 to imu
        imu_R = self.state_server.imu_state.R_imu_cam0
        imu_t = self.state_server.imu_state.t_cam0_imu
        orientation = self.state_server.imu_state.orientation
        position = self.state_server.imu_state.position
        imu_id = self.state_server.imu_state.id

        # Add a new camera state to the state server.
        imu_r_world = to_rotation(orientation)
        Rw_cam0 = imu_R @ imu_r_world
        tcam0_w = position + imu_r_world.T @ imu_t
        new_cam_state = CAMState(imu_id)
        new_cam_state.timestamp = time
        new_cam_state.orientation = to_quaternion(Rw_cam0)
        new_cam_state.position = tcam0_w
        new_cam_state.orientation_null = new_cam_state.orientation
        new_cam_state.position_null = new_cam_state.position
        self.state_server.cam_states[imu_id] = new_cam_state

        # Update the covariance matrix of the state.
        # To simplify computation, the matrix J below is the nontrivial block
        # Appendix B of "MSCKF" paper.
        JI_mat = np.zeros((6, 21))
        JI_mat[:3, :3] = imu_R
        JI_mat[:3, 15:18] = np.eye(3)
        JI_mat[3:6, :3] = skew(imu_r_world.T @ imu_t)  
        JI_mat[3:6, 12:15] = np.eye(3)
        JI_mat[3:6, 18:] = np.eye(3)  

        # Resize the state covariance matrix.
        row, col = self.state_server.state_cov.shape[:2]
        new_state_cov = np.zeros((row+6, col+6))
        new_state_cov[:-6, :-6] = self.state_server.state_cov
        self.state_server.state_cov = new_state_cov

        # Fill in the augmented state covariance.
        II_cov = self.state_server.state_cov[:21, :col]
        # IC_cov = self.state_server.state_cov[:21, 21:cols-21]
        IC_cov = self.state_server.state_cov[:21, :21]

        # N = self.state_server.state_cov.shape[0] // 6
        # J = np.hstack([J_I, np.zeros((6, 6 * N))])

        self.state_server.state_cov[row:, :col] = JI_mat @ II_cov
        # self.state_server.state_cov[rows:, cols:] = J_I @ IC_cov
        self.state_server.state_cov[:row, col:] = self.state_server.state_cov[row:, :col].T
        CC_cov = JI_mat @ IC_cov @ JI_mat.T
        self.state_server.state_cov[-6:, -6:] = CC_cov

        # Fix the covariance to be symmetric
        state_cov = self.state_server.state_cov
        sym_mat = (state_cov + state_cov.T) / 2.0
        self.state_server.state_cov = sym_mat

    def add_feature_observations(self, feature_msg):
        """
        IMPLEMENT THIS!!!!!
        """
        # get the current imu state id and number of current features
        state_id_imu = self.state_server.imu_state.id
        state_features_N = len(self.map_server)
        tracked_features = 0

        # add all features in the feature_msg to self.map_server
        for msg in feature_msg.features:
            # Check if we have a new feature
            msg_id = msg.id
            if msg_id not in self.map_server.keys():
                self.map_server[msg_id] = Feature(msg_id, self.optimization_config)
            else:
                # increment counter
                tracked_features += 1
            # Update observations of feature
            self.map_server[msg_id].observations[state_id_imu] = np.array([msg.u0, msg.v0, msg.u1, msg.v1])

        # update the tracking rate
        eps = 1e-05
        self.tracking_rate = tracked_features / (state_features_N + eps)

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

        H_x = dz_dpc0 @ dpc0_dxc + dz_dpc1 @ dpc1_dxc  # shape: (4, 6)
        H_f = dz_dpc0 @ dpc0_dpg + dz_dpc1 @ dpc1_dpg  # shape: (4, 3)

        # Modifty the measurement Jacobian to ensure observability constrain.
        A = H_x  # shape: (4, 6)
        u = np.zeros(6)
        u[:3] = to_rotation(cam_state.orientation_null) @ IMUState.gravity
        u[3:] = skew(p_w - cam_state.position_null) @ IMUState.gravity

        H_x = A - (A @ u)[:, None] * u / (u @ u)
        H_f = -H_x[:4, 3:6]

        # Compute the residual.
        r = z - np.array([*p_c0[:2] / p_c0[2], *p_c1[:2] / p_c1[2]])

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
                         21 + len(self.state_server.cam_states) * 6))
        H_fj = np.zeros((jacobian_row_size, 3))
        r_j = np.zeros(jacobian_row_size)

        stack_count = 0
        for cam_id in valid_cam_state_ids:
            H_xi, H_fi, r_i = self.measurement_jacobian(cam_id, feature.id)

            # Stack the Jacobians.
            idx = list(self.state_server.cam_states.keys()).index(cam_id)
            H_xj[stack_count:stack_count + 4, 21 + 6 * idx:21 + 6 * (idx + 1)] = H_xi
            H_fj[stack_count:stack_count + 4, :3] = H_fi
            r_j[stack_count:stack_count + 4] = r_i
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
        if H.shape[0] == 0 or r.shape[0] == 0:
            return

        # Decompose the final Jacobian matrix to reduce computational complexity.
        reduce_H = H
        reduce_r = r
        if H.shape[0] > H.shape[1]:
            Q, R = np.linalg.qr(H, mode='reduced')
            reduce_H = R
            reduce_r = Q.T @ r

        # Compute the Kalman gain.
        cov_obs = self.config.observation_noise
        cov_p = self.state_server.state_cov
        mat_S = reduce_H @ cov_p @ reduce_H.T + cov_obs * np.eye(reduce_H.shape[0])
        K = np.linalg.solve(mat_S, reduce_H @ cov_p).T

        # Compute the error of the state.
        x_new = K @ reduce_r

        # Update the IMU state.
        x_imu_change = x_new[:21]
        # if np.linalg.norm(x_imu_residual[6:9]) > .5 or np.linalg.norm(x_imu_residual[12:15]) > 1.0:
        #     print("TOO LARGE")

        imu_dq = small_angle_quaternion(x_imu_change[:3])
        orientation = quaternion_multiplication(imu_dq, self.state_server.imu_state.orientation)
        self.state_server.imu_state.orientation = orientation
        self.state_server.imu_state.gyro_bias += x_imu_change[3:6]
        self.state_server.imu_state.velocity += x_imu_change[6:9]
        self.state_server.imu_state.acc_bias += x_imu_change[9:12]
        self.state_server.imu_state.position += x_imu_change[12:15]

        extrinsic_dq = small_angle_quaternion(x_imu_change[15:18])
        self.state_server.imu_state.R_imu_cam0 = to_rotation(extrinsic_dq) @ self.state_server.imu_state.R_imu_cam0
        self.state_server.imu_state.t_cam0_imu += x_imu_change[18:21]

        # Update the camera states.
        cam_states = self.state_server.cam_states
        for i, cam_state_k in enumerate(cam_states.keys()):
            idx = 21 + i * 6  # index starts after IMU, in increments of 6 for i-th cam
            x_cam_new = x_new[idx:idx + 6]
            cam_dq = small_angle_quaternion(x_cam_new[:3])
            cam_states[cam_state_k].orientation = quaternion_multiplication(cam_dq, cam_states[cam_state_k].orientation)
            cam_states[cam_state_k].position += x_cam_new[3:]

        # Update state covariance.
        KH_I = np.eye(K.shape[0]) - K @ reduce_H
        x_cov = KH_I @ self.state_server.state_cov

        # Fix the covariance to be symmetric
        x_cov_sym = (x_cov + x_cov.T) / 2.0
        self.state_server.state_cov = x_cov_sym

    def gating_test(self, H, r, dof):
        P1 = H @ self.state_server.state_cov @ H.T
        P2 = self.config.observation_noise * np.identity(len(H))
        gamma = r @ np.linalg.solve(P1 + P2, r)

        if (gamma < self.chi_squared_test_table[dof]):
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
                        21 + 6 * len(self.state_server.cam_states)))
        r = np.zeros(jacobian_row_size)
        stack_count = 0

        # Process the features which lose track.
        for feature_id in processed_feature_ids:
            feature = self.map_server[feature_id]

            cam_state_ids = []
            for cam_id, measurement in feature.observations.items():
                cam_state_ids.append(cam_id)

            H_xj, r_j = self.feature_jacobian(feature.id, cam_state_ids)

            if self.gating_test(H_xj, r_j, len(cam_state_ids) - 1):
                H_x[stack_count:stack_count + H_xj.shape[0], :H_xj.shape[1]] = H_xj
                r[stack_count:stack_count + len(r_j)] = r_j
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

            jacobian_row_size += 4 * len(involved_cam_state_ids) - 3

        # Compute the Jacobian and residual.
        H_x = np.zeros((jacobian_row_size, 21 + 6 * len(self.state_server.cam_states)))
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
                H_x[stack_count:stack_count + H_xj.shape[0], :H_xj.shape[1]] = H_xj
                r[stack_count:stack_count + len(r_j)] = r_j
                stack_count += H_xj.shape[0]

            for cam_id in involved_cam_state_ids:
                del feature.observations[cam_id]

        H_x = H_x[:stack_count]
        r = r[:stack_count]

        # Perform measurement update.
        self.measurement_update(H_x, r)

        for cam_id in rm_cam_state_ids:
            idx = list(self.state_server.cam_states.keys()).index(cam_id)
            cam_state_start = 21 + 6 * idx
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
        state_cov[3: 6, 3: 6] = self.config.gyro_bias_cov * np.identity(3)
        state_cov[6: 9, 6: 9] = self.config.velocity_cov * np.identity(3)
        state_cov[9:12, 9:12] = self.config.acc_bias_cov * np.identity(3)
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

        filename='result_estimates.txt'
        f=open(filename,'a')

        f.write(str(imu_state.timestamp))
        f.write(' ')
        f.write(str(imu_state.position[0]))
        f.write(' ')
        f.write(str(imu_state.position[1]))
        f.write(' ')
        f.write(str(imu_state.position[2]))
        f.write(' ')
        f.write(str(imu_state.orientation[0]))
        f.write(' ')
        f.write(str(imu_state.orientation[1]))
        f.write(' ')
        f.write(str(imu_state.orientation[2]))
        f.write(' ')
        f.write(str(imu_state.orientation[3]))
        f.write(' ')
        f.write('\n')

        f.close()

        T_i_w = Isometry3d(
            to_rotation(imu_state.orientation).T,
            imu_state.position)
        T_b_w = IMUState.T_imu_body * T_i_w * IMUState.T_imu_body.inverse()
        body_velocity = IMUState.T_imu_body.R @ imu_state.velocity

        R_w_c = imu_state.R_imu_cam0 @ T_i_w.R.T
        t_c_w = imu_state.position + T_i_w.R @ imu_state.t_cam0_imu
        T_c_w = Isometry3d(R_w_c.T, t_c_w)

        return namedtuple('vio_result', ['timestamp', 'pose', 'velocity', 'cam0_pose'])(
            time, T_b_w, body_velocity, T_c_w)
