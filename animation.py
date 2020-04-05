from __future__ import print_function

import math
import unittest
from functools import reduce

import matplotlib.pyplot as plt
import numpy as np
import scipy
from scipy import linalg, matrix

from sympy import Matrix
from mpl_toolkits.mplot3d import Axes3D  # pylint: disable=W0611

import matplotlib.patches as mpatches
import matplotlib.transforms as mtransforms
import mpl_toolkits.mplot3d.axes3d as p3

# Required to do animations in colab
from matplotlib import animation
#from IPython.display import HTML

import gtsam
import gtsam.utils.plot as gtsam_plot
from gtsam import Pose2

from utils import *

plt.style.use('ggplot')
plt.rcParams['figure.figsize'] = (8.0, 8.0)
plt.rc('animation', html='jshtml') # needed for animations!
arrowOptions = dict(head_width=.02,head_length=.02, width=0.01)