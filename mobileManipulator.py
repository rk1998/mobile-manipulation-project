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
        self.penalty = 20
        self.collision_penalty = 2

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
        self.arm_model['L1'] = link_1
        self.arm_model['L2'] = link_2
        self.arm_model['L3'] = link_3
        self.arm_model['L4'] = link_4

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
        x_b = new_base_pose.x()
        y_b = new_base_pose.y()
        theta_b = new_base_pose.theta()

        sXl1 = Pose2(0, 0, new_base_pose.theta())
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
        link_1 = mpatches.Rectangle([x_b, y_b], 3.5, 0.1, angle=(theta_b+q[0])*180/np.pi, color='r')
        link_2 = mpatches.Rectangle([t1.x()+x_b,t1.y()+y_b], 3.5, 0.1, angle=(theta_b+q[0]+q[1])*180/np.pi, color='g')
        link_3 = mpatches.Rectangle([t2.x() + x_b,t2.y() + y_b], 2.5, 0.1, angle=(theta_b + q[0]+q[1]+q[2])*180/np.pi, color='b')
        link_4 = mpatches.Rectangle([t3.x()+ x_b, t3.y() + y_b], 0.5, 0.1, angle=(theta_b+q[0]+q[1]+q[2]+q[3])*180/np.pi, color='k')
        link_1 = Polygon(link_1.get_patch_transform().transform(link_1.get_path().vertices[:-1]))
        link_2 = Polygon(link_2.get_patch_transform().transform(link_2.get_path().vertices[:-1]))
        link_3 = Polygon(link_3.get_patch_transform().transform(link_3.get_path().vertices[:-1]))
        link_4 = Polygon(link_4.get_patch_transform().transform(link_4.get_path().vertices[:-1]))

        return base_model, link_1, link_2, link_3, link_4


    def check_collision_with_obstacles(self, obstacles, new_base_pose, q):
        """
        Checks if given base pose and joint config would collide
        with world obstacles
        obstacles - list of shapely Polygon objects
        new_base_pose - Pose2 of the base
        q - joint angles for the links of the manipulator arm
        """
        model = self.create_manipulator_model(new_base_pose, q)

        for i in range(0, len(model)):
            for obstacle in obstacles:
                if model[i].intersects(obstacle):
                    return True
        return False

    def fwd_kinematics(self, q, base_pose=None):
        """ Forward kinematics.
            Takes numpy array of joint angles, in radians.
        """

        if base_pose is None:
            base_x = self.x_b
            base_y = self.y_b
            base_theta = self.theta_b
        else:
            base_x = base_pose.x()
            base_y = base_pose.y()
            base_theta = base_pose.theta()

        jointTransform1 = Pose2(0, 0, q[0] + base_theta)
        jointTransform2 = Pose2(0, 0, q[1])
        jointTransform3 = Pose2(0, 0, q[2])
        jointTransform4 = Pose2(0, 0, q[3])
        jointAngleOffset = Pose2(0, 0, 0)
        base_offset = Pose2(base_x, base_y, 0)
        link1pose = Pose2(self.L1, 0, 0)
        link2pose = Pose2(self.L2, 0, 0)
        link3pose = Pose2(self.L3, 0, 0)
        link4pose = Pose2(self.L4, 0, 0)
        end_effector_pose = compose(jointAngleOffset, base_offset, jointTransform1,
                                    link1pose, jointTransform2,
                                    link2pose, jointTransform3,
                                    link3pose, jointTransform4, link4pose)
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

    def full_jacobian(self, q, base_position=None):
        """
        Calculates manipulator Jacobian. Takes numpy array of joint angles,
        in radians
        """
        if base_position is not None:
            base_x = base_position.x()
            base_y = base_position.y()
            base_theta = base_position.theta()
        else:
            base_x = self.x_b
            base_y = self.y_b
            base_theta = self.theta_b
        cos_b = np.cos(base_theta)
        sin_b = np.sin(base_theta)
        total = self.L1 + self.L2 + self.L3 + self.L4
        alpha = base_theta + q[0]
        beta = base_theta + q[0] + q[1]
        gamma = base_theta + q[0] + q[1] + q[2]
        delta = base_theta + q[0] + q[1] + q[2] + q[3]
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

    def ik2(self, sTt_desired, obstacles, base_position=None, e=1e-4):
        """ Inverse kinematics. generates joint angles and base position to
            get end effector at sTt_desired
            Takes desired Pose2 of tool T with respect to base S.
            sTt_desired - desired end effector pose
            obstacles - world obstacles, needed for collision checking poses
            base_position - optional, pose of the base of the manipulator
            Optional: e: error norm threshold
        """
        base_x = None
        base_y = None
        base_theta = None
        if base_position is None:
            base_x = self.x_b
            base_y = self.y_b
            base_theta = self.theta_b
        else:
            base_x = base_position.x()
            base_y = base_position.y()
            base_theta = base_position.theta()
        radius = self.L1 + self.L2 + self.L3 + self.L4
        val = (sTt_desired.x() - base_x)**2 + (sTt_desired.y() - base_y)**2
        if val > (radius)**2:
            # random_base_pose = generate_random_point_in_circle(sTt_desired, radius)
            #initial guess
            base_x = np.random.random_sample()*10
            base_y = np.random.random_sample()*10
            base_theta = np.random.random_sample()*(2*np.pi) - (np.pi)
            q = np.radians(vector4(30, 30, -30, 45))  # take initial estimate well within workspac
            error = 9999999
            max_iter = 10000
            i = 0
            while error >= e and i < max_iter:
                J = self.full_jacobian(q, base_position=Pose2(base_x, base_y, base_theta))
                sTt_estimate = self.fwd_kinematics(q, base_pose=Pose2(base_x, base_y, base_theta))
                error_vector = delta(sTt_estimate, sTt_desired)
                error = np.linalg.norm(error_vector)
                q_del = np.linalg.inv(J.T.dot(J)  + (self.penalty**2)*np.identity(7)).dot(J.T.dot(error_vector))
                q = q + q_del[3:]
                base_x = base_x + q_del[0]
                base_y = base_y + q_del[1]
                base_theta = base_theta + q_del[2]
                i = i + 1
            return Pose2(base_x, base_y, base_theta), np.remainder(q+math.pi, 2*math.pi)-math.pi
        else:
            q = np.radians(vector4(30, 30, -30, 45))  # take initial estimate well within workspac
            error = 9999999
            max_iter = 10000
            i = 0
            while error >= e and i < max_iter:
              J = self.manipulator_jacobian(q)
              sTt_estimate = self.fwd_kinematics(q, base_pose=Pose2(base_x, base_y, base_theta))
              error_vector = delta(sTt_estimate, sTt_desired)
              error = np.linalg.norm(error_vector)
              q_del = np.linalg.inv(J.T.dot(J)  + (self.penalty**2)*np.identity(4)).dot(J.T.dot(error_vector))
              q = q + q_del
              # q = q + np.linalg.pinv(manipulator_jacobian).dot(error_vector)
              i = i+1
            # return result in interval [-pi,pi)
            return Pose2(base_x, base_y, base_theta), np.remainder(q+math.pi, 2*math.pi)-math.pi


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
