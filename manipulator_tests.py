from __future__ import print_function
import math
import unittest
from functools import reduce
from mobileManipulator import FourLinkMM
import gtsam
import gtsam.utils.plot as gtsam_plot
from gtsam import Pose2
import numpy as np
import shapely
from utils import *

base = np.array([0.0, 0.0, 0.0], dtype=np.float)
Q0 = np.radians(np.array([0,0,0,0], dtype=np.float))
Q1 = np.radians(np.array([-90,90,0,90], dtype=np.float))
Q2 = np.radians(np.array([-90, 90, 90,90], dtype=np.float))
Q3 = np.radians(np.array([-30, -45, -90, 0]), dtype=np.float)
# Q0 = np.hstack((base, Q0))
# Q1 = np.hstack((base, Q1))
# Q2 = np.hstack((base, Q2))
# Q3 = np.hstack((base, Q3))

class TestMobileManipulator(unittest.TestCase):
    """Unit tests for functions used below."""

    def setUp(self):
        self.arm = FourLinkMM()

    def assertPose2Equals(self, actual, expected, tol=1e-2):
        """Helper function that prints out actual and expected if not equal."""
        equal = actual.equals(expected, tol)
        if not equal:
            raise self.failureException(
                "Poses are not equal:\n{}!={}".format(actual, expected))

    #@unittest.skip("Skipping FK")
    def test_fk(self):
        """Make sure forward kinematics is correct for some known test configurations."""
        # at rest
        expected = Pose2(2*3.5 + 2.5 + 0.5, 0, math.radians(0))
        sTt = self.arm.fwd_kinematics(Q0)
        self.assertIsInstance(sTt, Pose2)
        self.assertPose2Equals(sTt, expected)

        # -30, -45, -90
        expected = Pose2(1.03, -5.9072, -2.879)
        sTt = self.arm.fwd_kinematics(Q3)
        self.assertPose2Equals(sTt, expected)

        #@unittest.skip("Skipping Jacobian")
    def test_jacobian(self):
        """Test Jacobian calculation."""
        # at rest
        expected = np.array([[1, 0, -10, -10, -6.5, -3,-0.5], [0, 1, 0, 0, 0, 0, 0], [0, 0, 1, 1, 1, 1, 1]], np.float)
        J = self.arm.full_jacobian(Q0)
        np.testing.assert_array_almost_equal(J, expected)

        # at -90, 90, 0
        expected = np.array([[1, 0, -6.0, -6.0, -6.0, -2.5, 0], [0, 1, 3.0, 3.0, -0.5, -0.5, -0.5], [0, 0, 1, 1, 1, 1, 1]], np.float)
        J = self.arm.full_jacobian(Q1)
        np.testing.assert_array_almost_equal(J, expected)


    def test_collision_check(self):
        obstacle_1 = shapely.geometry.box(5, 5, 8, 8)
        result = self.arm.check_collision_with_obstacles([obstacle_1], Pose2(0.0, 0.0, 0), Q0)
        self.assertFalse(result)

        obstacle_2 = shapely.geometry.box(3, -1.75, 4, 9)
        result = self.arm.check_collision_with_obstacles([obstacle_2], Pose2(0.0, 0.0, 0), Q3)
        self.assertTrue(result)
        result = self.arm.check_collision_with_obstacles([obstacle_1], Pose2(0.0, 0.0, 0), Q3)
        self.assertFalse(result)
    #@unittest.skip("Skipping Complete NullSpace")
    def test_velocity_in_nullspace(self):
        """Test Velocity_in_NullSpace."""
        v = 1
        w = 0.3
        dt = 0.1

        # at rest
        expected = np.array([[-0.0300000000000000], [0.0174311926605505], [-0.00183486238532109], [-0.0155963302752294]])
        J = self.arm.full_jacobian(Q1)
        # This is the current input for the base
        u = np.array([v*np.cos(self.arm.theta_b)*dt,v*np.sin(self.arm.theta_b)*dt,w*dt])
        u = u.reshape((3, 1))
        q_d = self.arm.velocity_in_null_space(J, u)
        np.testing.assert_array_almost_equal(q_d, expected)

        # at -90, 90, 0
        expected = np.array([[-0.0133494208494209], [0.0119208494208494], [-0.00526061776061778], [-0.0233108108108109]])
        self.arm.theta_b = np.radians(45)
        J = self.arm.full_jacobian(Q2)
        u = np.array([v*np.cos(self.arm.theta_b)*dt,v*np.sin(self.arm.theta_b)*dt,w*dt])
        u = u.reshape((3, 1))
        q_d = self.arm.velocity_in_null_space(J, u)
        np.testing.assert_array_almost_equal(q_d, expected)


    #@unittest.skip("Skipping IK")
    def test_ik2(self):
        print("testing ik2")
        base, actual = self.arm.ik2(Pose2(2*3.5 + 2.5 + 0.5, 0, math.radians(0)))
        print(base)
        np.testing.assert_array_almost_equal(actual, Q0, decimal=2)

        sTt_desired = Pose2(9,5,-1.5)
        base, actual = self.arm.ik2(sTt_desired)
        print(base)
        print(self.arm.fwd_kinematics(actual, base_pose=base))
        self.assertPose2Equals(self.arm.fwd_kinematics(actual, base_pose=base), sTt_desired, tol=1)

        sTt_desired = Pose2(20, 8,-3.14)
        base, actual = self.arm.ik2(sTt_desired)
        print(base)
        print(self.arm.fwd_kinematics(actual, base_pose=base))
        self.assertPose2Equals(self.arm.fwd_kinematics(actual, base_pose=base), sTt_desired, tol=1)


if __name__ == '__main__':
  unittest.main(argv=['first-arg-is-ignored'], exit=False)
