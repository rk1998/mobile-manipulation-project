from __future__ import print_function

import math
from functools import reduce

import matplotlib.pyplot as plt
import numpy as np
import scipy
from scipy import linalg, matrix

import matplotlib.patches as mpatches
import matplotlib.transforms as mtransforms
import mpl_toolkits.mplot3d.axes3d as p3

# Required to do animations in colab
from mobileManipulator import FourLinkMM
import gtsam
import gtsam.utils.plot as gtsam_plot
from gtsam import Pose2
import shapely
from shapely.geometry import Polygon
from shapely import affinity
from utils import *
from rrt_manipulator import *
from rrt_star_manipulator import *


def autolabel(rects, ax):
    """Attach a text label above each bar in *rects*, displaying its height."""
    for rect in rects:
        height = rect.get_height()
        ax.annotate('{}'.format(height),
                    xy=(rect.get_x() + rect.get_width() / 2, height),
                    xytext=(0, 3),  # 3 points vertical offset
                    textcoords="offset points",
                    ha='center', va='bottom')

def plot_metric(rrt_metric, rrt_star_metric, figure_title="", ylabel=""):
    labels=['Env1', 'Env2', 'Env3', 'Env4']
    x = np.arange(len(labels))
    width = 0.36
    fig, ax = plt.subplots()
    rects1 = ax.bar(x - width/2, rrt_metric, width, label='RRT')
    rects2 = ax.bar(x + width/2, rrt_star_metric, width, label='RRT*')
    ax.set_ylabel(ylabel)
    ax.set_title(figure_title)
    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    ax.legend()
    autolabel(rects1, ax)
    autolabel(rects2, ax)
    fig.tight_layout()
    plt.show()

max_x = 60
max_y = 60
QStart = Pose2(10, 0, 0)
start_config = (Pose2(0, 0, 0), vector4(0, 0, 0, 0), QStart)
arm = FourLinkMM()

#Setup environments
environments = []
goals = []
obstacles = [shapely.geometry.box(10, 10, 14, 14), shapely.geometry.box(25, 8, 35, 18), shapely.geometry.box(10, 35, 15, 40), shapely.geometry.box(33, 33, 38,38)]
env = Environment(max_x, max_y, obstacles=obstacles)
QGoal = Pose2(40, 25, np.pi/2)
environments.append(env)
goals.append(QGoal)

obstacles1 = [shapely.geometry.box(9, 9, 30, 30)]
QGoal1 = Pose2(35, 25, 0)
env1 = Environment(max_x, max_y, obstacles=obstacles1)
environments.append(env1)
goals.append(QGoal1)

QGoal2 = Pose2(34, 15, np.pi)
obstacles2 = [shapely.geometry.box(6, 6, 8, 20), shapely.geometry.box(8, 19, 24, 21), shapely.geometry.box(10, 10, 25, 13), shapely.geometry.box(25, 6, 27, 20)]
env2 = Environment(max_x, max_y, obstacles=obstacles2, num_obstacles=7, obstacle_size = 8)
environments.append(env2)
goals.append(QGoal2)

QGoal3 = Pose2(27.5, 25, 0)
obstacles3 = [Polygon([(8, 10), (18, 10), (18, 30)]), Polygon([(37, 30), (37, 10), (47, 10)]), shapely.geometry.box(19, 6, 35, 9)]
env3 = Environment(max_x, max_y, obstacles=obstacles3)
environments.append(env3)
goals.append(QGoal3)


#test each environment 10 times, get average iterations, path lengths and number of successes
#on each env
rrt_iterations_to_converge = []
rrt_path_lengths = []
rrt_successes = []
for i in range(len(environments)):
    print("Environment: " + str(i))
    avg_iteration = 0
    avg_path_length = 0
    num_convergances = 0
    env = environments[i]
    goal = goals[i]
    for j in range(10):
        path, graph_dictionary, graph, iterations, path_dist, converged = RRT(start_config, goal, env, arm, lim=0.5, step_size=1.5, num_iters=6000)
        if converged:
            avg_iteration += iterations
            avg_path_length += path_dist
            num_convergances +=1

    if num_convergances > 0:
        avg_iteration = avg_iteration / num_convergances
        avg_path_length = avg_path_length / num_convergances

    rrt_successes.append(num_convergances)
    rrt_iterations_to_converge.append(avg_iteration)
    rrt_path_lengths.append(avg_path_length)

print(rrt_iterations_to_converge)
print(rrt_successes)
print(rrt_path_lengths)

print("Testing RRT Star")
rrt_star_iterations_to_converge = []
rrt_star_path_lengths = []
rrt_star_successes = []
for i in range(len(environments)):
    print("Environment: " + str(i))
    avg_iteration = 0
    avg_path_length = 0
    num_convergances = 0
    env = environments[i]
    goal = goals[i]
    for j in range(10):
        path, graph_dictionary, graph, iterations, path_dist, converged = RRT_star(start_config, goal, env, arm, lim=0.5, step_size=1.5, num_iters=6000)
        if converged:
            avg_iteration += iterations
            avg_path_length += path_dist
            num_convergances +=1

    if num_convergances > 0:
        avg_iteration = avg_iteration / num_convergances
        avg_path_length = avg_path_length / num_convergances

    rrt_star_successes.append(num_convergances)
    rrt_star_iterations_to_converge.append(avg_iteration)
    rrt_star_path_lengths.append(avg_path_length)

print(rrt_star_iterations_to_converge)
print(rrt_star_successes)
print(rrt_star_path_lengths)

plot_metric(rrt_successes, rrt_star_successes,
            figure_title="Number of Successes per Environment", ylabel="Successes")
plot_metric(rrt_iterations_to_converge, rrt_star_iterations_to_converge,
            figure_title="Avg. Iterations to Converge", ylabel="Iterations")
plot_metric(rrt_path_lengths, rrt_star_path_lengths, figure_title="Avg Path Lengths", ylabel="Path Lengths (M)")
