from __future__ import print_function
import math
import unittest
from functools import reduce
from mobileManipulator import FourLinkMM
import gtsam
import gtsam.utils.plot as gtsam_plot
from gtsam import Pose2
import numpy as np
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
        expected = Pose2(-6.34, 8.11025, -2.879)
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
    def test_ik(self):
        """Check iterative inverse kinematics function."""
        # at rest
        actual = self.arm.ik(Pose2(2*3.5 + 2.5 + 0.5, 0, math.radians(0)))
        np.testing.assert_array_almost_equal(actual, Q0, decimal=2)

        # -30, -45, -90
        sTt_desired = Pose2(-6.34, 8.11025, -2.879)
        actual = self.arm.ik(sTt_desired)
        self.assertPose2Equals(self.arm.fwd_kinematics(actual), sTt_desired)
        #np.testing.assert_array_almost_equal(actual, Q1, decimal=2)

if __name__ == '__main__':
  unittest.main(argv=['first-arg-is-ignored'], exit=False)
