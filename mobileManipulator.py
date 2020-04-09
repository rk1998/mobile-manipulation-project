from __future__ import print_function
import math
import gtsam
import gtsam.utils.plot as gtsam_plot
from gtsam import Pose2
import scipy
#from scipy import linalg, matrix
import numpy as np
from utils import *
#from sympy import Matrix

class FourLinkMM(object):

    def __init__(self):
        self.x_b = 0
        self.y_b = 0
        self.theta_b = 0

        self.L1 = 3.5
        self.L2 = 3.5
        self.L3 = 2.5
        self.L4 = 0.5


    def fwd_kinematics(self, q):
        """ Forward kinematics.
            Takes numpy array of joint angles, in radians.
        """
        # self.x_b = q[0]
        # self.y_b = q[1]
        # self.theta_b = q[2]
        tool_at_rest = Pose2(self.x_b +  (self.L1 + self.L2 + self.L3 + self.L4)*np.cos(self.theta_b),
                            self.y_b + (self.L1 + self.L2 + self.L3 + self.L4)*np.sin(self.theta_b), 0)
        unit_twist_1 = vector3(0, 0, q[0])
        unit_twist_2 = vector3(self.L1 * q[1], 0, q[1])
        unit_twist_3 = vector3((self.L1 + self.L2) * q[2], 0, q[2])
        unit_twist_4 = vector3((self.L1 + self.L2 + self.L3)*q[3], 0, q[3])

        map_1 = Pose2.Expmap(unit_twist_1)
        map_2 = Pose2.Expmap(unit_twist_2)
        map_3 = Pose2.Expmap(unit_twist_3)
        map_4 = Pose2.Expmap(unit_twist_4)

        end_effector_pose = compose(map_1, map_2, map_3, map_4, tool_at_rest)
        return end_effector_pose

    def jacobian(self, q):
        """
        Calculates manipulator Jacobian. Takes numpy array of joint angles,
        in radians
        """
        cos_b = np.cos(self.theta_b)
        sin_b = np.sin(self.theta_b)
        total = self.L1 + self.L2 + self.L3 + self.L4
        alpha = self.theta_b + q[0]
        beta = self.theta_b + q[0] + q[1]
        gamma = self.theta_b + q[0] + q[1] + q[2]
        delta = self.theta_b + q[0] + q[1] + q[2] + q[3]
        j_m_1 = -self.L1*np.cos(alpha) - self.L2*np.cos(beta) - self.L3*np.cos(gamma) - self.L4*np.cos(delta)
        j_m_2 = -self.L2*np.cos(beta) - self.L3*np.cos(gamma) - self.L4*np.cos(delta)
        j_m_3 = -self.L3*np.cos(gamma) - self.L4*np.cos(delta)
        j_m_4 = -self.L4*np.cos(delta)
        j_m_5 = -self.L1*np.sin(alpha) - self.L2*np.sin(beta) - self.L3*np.sin(gamma) - self.L4*np.sin(delta)
        j_m_6 = -self.L2*np.sin(beta) - self.L3*np.sin(gamma) - self.L4*np.sin(delta)
        j_m_7 = -self.L3*np.sin(gamma) - self.L4*np.sin(delta)
        j_m_8 = -self.L4*np.sin(delta)
        Jacobian = [[1, 0,j_m_1,j_m_1,j_m_2,j_m_3,j_m_4], \
                    [0, 1,j_m_5,j_m_5,j_m_6,j_m_7,j_m_8], \
                    [0, 0,1,1,1,1,1]]
        return np.array(Jacobian)

    def ik(self, sTt_desired, e=1e-9):
        """ Inverse kinematics.
            Takes desired Pose2 of tool T with respect to base S.
            Optional: e: error norm threshold
        """
        q = np.radians(vector4(30, 30, -30, 45))  # take initial estimate well within workspace
        base_config = generate_random_point_in_circle(sTt_desired, self.L1 + self.L2 + self.L3 + self.L4)
        # base = np.array([0.0, 0.0, 0.0])
        # q = np.hstack((base, q))
        self.x_b = base_config.x()
        self.y_b = base_config.y()
        self.theta_b = base_config.theta()
        error = 9999999
        max_iter = 2000
        i = 0
        while error >= e:
          jacobian_matrix = self.jacobian(q)
          manipulator_jacobian = jacobian_matrix[:, 3:]
          sTt_estimate = self.fwd_kinematics(q)
          error_vector = delta(sTt_estimate, sTt_desired)
          error = np.linalg.norm(error_vector)
          q = q + np.linalg.pinv(manipulator_jacobian).dot(error_vector)
          i = i+1

        # return result in interval [-pi,pi)
        return base_config, np.remainder(q+math.pi, 2*math.pi)-math.pi

    def velocity_in_null_space(self, J, u):
        """
        Given a velocity of the base (u) and the Jacobian (J). Compute velocity
        of manipulator thus the end-effector stays in place
        """
        J_b = J[:, 0:3]
        J_m =  J[:, 3:]
        J_mpinv = np.linalg.pinv(J_m)
        q_d = J_mpinv.dot(-J_b.dot(u))
        q_d = np.reshape(q_d, (4, 1))
        #q_d = Matrix(4,1, q_d)
        return q_d

    def null_space_projector(self, J):
        I = np.identity(7)
        J_pinv = np.linalg.pinv(J)
        N = (I - J_pinv.dot(J))
        return N
