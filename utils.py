import gtsam
import gtsam.utils.plot as gtsam_plot
from gtsam import Pose2
import numpy as np
import math
import random
from functools import reduce
import shapely
from shapely.geometry import Polygon, Point
import matplotlib.patches as mpatches
import matplotlib.transforms as mtransforms
# Some utility functions for Pose2
def vector3(x, y, z):
    """Create 3D double numpy array."""
    return np.array([x, y, z], dtype=np.float)

def vector4(w, x, y, z):
    """ Create 4D double numpy array."""
    return np.array([w, x, y, z], dtype=np.float)

def compose(*poses):
    """Compose all Pose2 transforms given as arguments from left to right."""
    return reduce((lambda x, y: x.compose(y)), poses)


def intersect_with_obstacles(pose, obstacles):
    len_b = 2
    d = np.sqrt(2*len_b*len_b/4)
    rect = mpatches.Rectangle([pose.x()-d*np.cos(pose.theta()+np.radians(45)),
            pose.y()-d*np.sin(pose.theta()+np.radians(45))],len_b, len_b, angle = pose.theta()*180/np.pi)
    base_model = Polygon(rect.get_patch_transform().transform(rect.get_path().vertices[:-1]))
    for obstacle in obstacles:
        if obstacle.intersects(base_model):
            return True
    return False

def generate_random_point_in_circle(circle_center, radius, obstacles):
    intersect = True
    pose = None
    while intersect:
        a = random.random() * 2 * np.pi
        r = radius * math.sqrt(random.random())
        x = r * np.cos(a) + circle_center.x()
        y = r * np.sin(a) + circle_center.y()
        pose = Pose2(x, y, a)
        intersect = intersect_with_obstacles(pose, obstacles)
    return pose

def vee(M):
    """Pose2 vee operator."""
    return vector3(M[0, 2], M[1, 2], M[1, 0])


def delta(g0, g1):
    """Difference between x,y,,theta components of SE(2) poses."""
    return vector3(g1.x() - g0.x(), g1.y() - g0.y(), g1.theta() - g0.theta())


def trajectory(g0, g1, N=20):
    """ Create an interpolated trajectory in SE(2), treating x,y, and theta separately.
        g0 and g1 are the initial and final pose, respectively.
        N is the number of *intervals*
        Returns N+1 poses
    """
    e = delta(g0, g1)
    return [Pose2(g0.x()+e[0]*t, g0.y()+e[1]*t, g0.theta()+e[2]*t) for t in np.linspace(0, 1, N)]
