from __future__ import print_function
import math
import gtsam
import gtsam.utils.plot as gtsam_plot
from gtsam import Pose2
import scipy
#from scipy import linalg, matrix
import numpy as np
from utils import *
import matplotlib.patches as mpatches
import matplotlib.transforms as mtransforms
import shapely
from shapely.geometry import Polygon
from shapely import affinity
#from sympy import Matrix

class FourLinkMM(object):

    def __init__(self):
        self.x_b = 0
        self.y_b = 0
        self.theta_b = 0
        self.len_b = 2
        self.d = np.sqrt(2*self.len_b*self.len_b/4)
        self.L1 = 3.5
        self.L2 = 3.5
        self.L3 = 2.5
        self.L4 = 0.5

        #boxes representing different parts of the mobile manipulator
        self.base_model = shapely.geometry.box(self.x_b-self.d*np.cos(self.theta_b+np.radians(45)),
                                               self.y_b-self.d*np.sin(self.theta_b+np.radians(45)),
                                               self.x_b-self.d*np.cos(self.theta_b+np.radians(45)) + self.len_b,
                                               self.y_b-self.d*np.sin(self.theta_b+np.radians(45)) + self.len_b)
        self.arm_model = {}
        self.arm_model['L1'] = shapely.geometry.box(0, 0, 3.5, 0.1)
        self.arm_model['L2'] = shapely.geometry.box(self.L1, 0, self.L1 + self.L2, 0.1)
        self.arm_model['L3'] = shapely.geometry.box(self.L1 + self.L2, 0, self.L1 + self.L2 + self.L3, 0.1)
        self.arm_model['L4'] = shapely.geometry.box(self.L1 + self.L2 + self.L3, 0, self.L1 + self.L2 + self.L3 + self.L4, 0.1)


    def update_manipulator_model(self, new_base_pose, q):
        """
        Updates position of model base and arm links.
        new_base_pose - Pose2 object for the base model
        q - vector of joint angles
        """
        diff_x = new_base_pose.x() - self.x_b
        diff_y = new_base_pose.y() - self.y_b
        self.x_b = new_base_pose.x()
        self.y_b = new_base_pose.y()
        self.theta_b = new_base_pose.theta()
        self.base_model = shapely.geometry.box(self.x_b-self.d*np.cos(self.theta_b+np.radians(45)),
                                               self.y_b-self.d*np.sin(self.theta_b+np.radians(45)),
                                               self.x_b-self.d*np.cos(self.theta_b+np.radians(45)) + self.len_b,
                                               self.y_b-self.d*np.sin(self.theta_b+np.radians(45)) + self.len_b)
        #link1
        sXl1 = Pose2(0, 0, self.theta_b)
        l1Zl1 = Pose2(0, 0, q[0])
        l1Xl2 = Pose2(self.L1, 0, 0)
        sTl2 = compose(sXl1, l1Zl1, l1Xl2)
        t1 = sTl2.translation()
        l1_angle = self.theta_b + q[0]
        l1_transform = Pose2(diff_x, diff_y, self.theta_b+q[0])

        #link 2
        l2Zl2 = Pose2(0, 0, q[1])
        l2Xl3 = Pose2(self.L2, 0, 0)
        sTl3 = compose(sTl2, l2Zl2, l2Xl3)
        t2 = sTl3.translation()
        l2_transform = Pose2(diff_x, diff_y, self.theta_b + q[0] + q[1])

        #link 3
        l3Zl3 = Pose2(0, 0, q[2])
        l3X4 = Pose2(self.L3, 0, 0)
        sTl4 = compose(sTl3, l3Zl3, l3X4)
        t3 = sTl4.translation()
        l3_transform = Pose2(diff_x, diff_y, self.theta_b + q[0] + q[1] + q[2])

        #link 4
        l4Zl4 = Pose2(0, 0, q[3])
        l4Xt = Pose2(self.L4, 0, 0)
        sTt = compose(sTl4, l4Zl4, l4Xt)
        t4 = sTt.translation()
        l4_transform = Pose2(diff_x, diff_y, self.theta_b + q[0] + q[1] + q[2] + q[3])
        l1_mat = l1_transform.matrix()
        l2_mat = l2_transform.matrix()
        l3_mat = l3_transform.matrix()
        l4_mat = l4_transform.matrix()

        l1_transform = [l1_mat[0][0], l1_mat[0][1], l1_mat[1][0], l1_mat[1][1],
                        l1_mat[0][2], l1_mat[1][2]]
        l2_transform = [l2_mat[0][0], l2_mat[0][1], l2_mat[1][0], l2_mat[1][1],
                        l2_mat[0][2], l2_mat[1][2]]
        l3_transform = [l3_mat[0][0], l3_mat[0][1], l3_mat[1][0], l3_mat[1][1],
                        l3_mat[0][2], l3_mat[1][2]]
        l4_transform = [l4_mat[0][0], l4_mat[0][1], l4_mat[1][0], l4_mat[1][1],
                        l4_mat[0][2], l4_mat[1][2]]
        self.arm_model['L1'] = affinity.affine_transform(self.arm_model['L1'], l1_transform)
        self.arm_model['L2'] = affinity.affine_transform(self.arm_model['L2'], l2_transform)
        self.arm_model['L3'] = affinity.affine_transform(self.arm_model['L3'], l3_transform)
        self.arm_model['L4'] = affinity.affine_transform(self.arm_model['L4'], l4_transform)

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


    def manipulator_jacobian(self, q):
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
        manipulator_jacobian = [[j_m_1,j_m_2,j_m_3,j_m_4],
                                [j_m_5,j_m_6,j_m_7,j_m_8],
                                [1, 1, 1, 1]]
        return np.array(manipulator_jacobian)

    def full_jacobian(self, q):
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

    def ik(self, sTt_desired, base_position=None, e=1e-9):
        """ Inverse kinematics.
            Takes desired Pose2 of tool T with respect to base S.
            Optional: e: error norm threshold
        """
        if base_position is not None:
            self.x_b = base_position.x()
            self.y_b = base_position.y()
            self.theta_b = base_position.z()
        # radius = self.L1 + self.L2 + self.L3 + self.L4
        # val = (sTt_desired.x() - self.x_b)**2 + (sTt_desired.y() - self.y_b)**2
        # print(val)
        # if  val > radius**2:
        #     return None
        q = np.radians(vector4(30, 30, -30, 45))  # take initial estimate well within workspace
        #base_config = generate_random_point_in_circle(sTt_desired, self.L1 + self.L2 + self.L3 + self.L4)
        # base = np.array([0.0, 0.0, 0.0])
        # q = np.hstack((base, q))
        # self.x_b = base_config.x()
        # self.y_b = base_config.y()
        # self.theta_b = base_config.theta()
        error = 9999999
        max_iter = 4000
        i = 0
        while error >= e and i < max_iter:
          manipulator_jacobian = self.manipulator_jacobian(q)
          # jacobian_matrix = self.jacobian(q)
          # manipulator_jacobian = jacobian_matrix[:, 3:]
          sTt_estimate = self.fwd_kinematics(q)
          error_vector = delta(sTt_estimate, sTt_desired)
          error = np.linalg.norm(error_vector)
          q = q + np.linalg.pinv(manipulator_jacobian).dot(error_vector)
          i = i+1

        # return result in interval [-pi,pi)
        return np.remainder(q+math.pi, 2*math.pi)-math.pi

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
