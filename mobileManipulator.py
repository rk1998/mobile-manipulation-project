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
        Updates manipulator model with new base pose and given joint angles
        Important to update model during planning to check for collisions
        """
        self.x_b = new_base_pose.x()
        self.y_b = new_base_pose.y()
        self.theta_b = new_base_pose.theta()
        base_model, link_1, link_2, link_3, link_4 = self.create_manipulator_model(new_base_pose, q)
        self.base_model = base_model
        self.arm_model['L1'] = Polygon(link_1.get_patch_transform().transform(link_1.get_path().vertices[:-1]))
        self.arm_model['L2'] = Polygon(link_2.get_patch_transform().transform(link_2.get_path().vertices[:-1]))
        self.arm_model['L3'] = Polygon(link_3.get_patch_transform().transform(link_3.get_path().vertices[:-1]))
        self.arm_model['L4'] = Polygon(link_4.get_patch_transform().transform(link_4.get_path().vertices[:-1]))

    def create_manipulator_model(self, new_base_pose, q):
        """
        Creates shapley polgons in the given manipulator config.
        new_base_pose - Pose2 object for the base model
        q - vector of joint angles
        """

        # self.x_b = new_base_pose.x()
        # self.y_b = new_base_pose.y()
        # self.theta_b = new_base_pose.theta()
        rect = mpatches.Rectangle([new_base_pose.x()-self.d*np.cos(new_base_pose.theta()+np.radians(45)),
        new_base_pose.y()-self.d*np.sin(new_base_pose.theta()+np.radians(45))],
         self.len_b, self.len_b, angle = new_base_pose.theta()*180/np.pi)
        base_model = Polygon(rect.get_patch_transform().transform(rect.get_path().vertices[:-1]))

        sXl1 = Pose2(0, 0, self.theta_b)
        l1Zl1 = Pose2(0, 0, q[0])
        l1Xl2 = Pose2(self.L1, 0, 0)
        sTl2 = compose(sXl1, l1Zl1, l1Xl2)
        t1 = sTl2.translation()
        # print(t1)

        l2Zl2 = Pose2(0, 0, q[1])
        l2Xl3 = Pose2(self.L2, 0, 0)
        sTl3 = compose(sTl2, l2Zl2, l2Xl3)
        t2 = sTl3.translation()
        # print(t2)

        l3Zl3 = Pose2(0, 0, q[2])
        l3X4 = Pose2(self.L3, 0, 0)
        sTl4 = compose(sTl3, l3Zl3, l3X4)
        t3 = sTl4.translation()
        # print(t3)

        l4Zl4 = Pose2(0, 0, q[3])
        l4Xt = Pose2(self.L4, 0, 0)
        sTt = compose(sTl4, l4Zl4, l4Xt)
        t4 = sTt.translation()
        # print(t4)
        link_1 = mpatches.Rectangle([self.x_b,self.y_b], 3.5, 0.1, angle=(self.theta_b+q[0])*180/np.pi, color='r')
        link_2 = mpatches.Rectangle([t1.x()+self.x_b,t1.y()+self.y_b], 3.5, 0.1, angle=(self.theta_b+q[0]+q[1])*180/np.pi, color='g')
        link_3 = mpatches.Rectangle([t2.x() + self.x_b, t2.y() + self.y_b], 2.5, 0.1, angle=(self.theta_b + q[0]+q[1]+q[2])*180/np.pi, color='b')
        link_4 = mpatches.Rectangle([t3.x()+ self.x_b , t3.y() + self.y_b], 0.5, 0.1, angle=(self.theta_b+q[0]+q[1]+q[2]+q[3])*180/np.pi, color='k')
        return base_model, link_1, link_2, link_3, link_4


    def check_collision_with_obstacles(self, obstacles, new_base_pose, q):
        """
        Checks if given base pose and joint config would collide
        with world obstacles
        obstacles - list of shapely Polygon objects
        new_base_pose - Pose2 of the base
        q - joint angles for the links of the manipulator arm
        """
        model = self.create_manipulator_model(new_base_pose, q):
        for i in range(0, len(model)):
            for obstacle in obstacles:
                if model[i].intersects(obstacle):
                    return True
        return False

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
