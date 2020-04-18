import numpy as np
import sklearn
import networkx as nx
from matplotlib import pyplot as plt
import shapely
from shapely.geometry import Polygon, Point
import sys
from mobileManipulator import FourLinkMM
import gtsam
import gtsam.utils.plot as gtsam_plot
from gtsam import Pose2
from utils import *
import math
sys.setrecursionlimit(10**6)

# np.random.seed(42)

max_x = 50
max_y = 50

c_map = np.zeros((max_x, max_y))
no_obstacles = 5
o_size = 4


def RandomQ(Qgoal, arm, env):
    t1 = 0
    t2 = 0
    t3 = 0
    t4 = 0
    sTt_b = None
    end_effector = None
    q = vector4(0, 0, 0, 0)
    if np.random.random_sample() <= 0.15: #15% probability that it returns Qgoal
        sTt_b, q = arm.ik2(Qgoal, env.obstacles)
        return sTt_b, q, Qgoal
    collision = True
    while collision:
        x = np.random.random_sample()*env.max_x
        y = np.random.random_sample()*env.max_y
        theta = np.random.random_sample()*360 - 180
        end_effector = Pose2(x, y, math.radians(theta))
        if env.check_collision_with_obstacles(end_effector) == False:
            sTt_b, q = arm.ik2(end_effector, env.obstacles)
            collision = arm.check_collision_with_obstacles(env.obstacles, sTt_b, q)
            if not collision:
                break
        else:
            print("rand config in collision")
            collision = True


    return sTt_b, q, end_effector

class Environment:
    def __init__(self, max_x, max_y, obstacles=None, num_obstacles=5, obstacle_size=4):
        self.max_x = max_x
        self.max_y = max_y
        self.c_map = np.zeros((max_x, max_y))
        self.num_obstacles = num_obstacles
        self.o_size = obstacle_size
        if obstacles is None:
            self.obstacles = self.create_obstacles()
        else:
            self.obstacles = obstacles

    def create_obstacles(self):
        obstacles = []
        i = 0
        while i < no_obstacles:
          ob_pos_x, ob_pos_y = self.randomPosition()
          obstacle = shapely.geometry.box(ob_pos_x, ob_pos_y,
                    ob_pos_x + self.o_size, ob_pos_y + self.o_size)
          obstacles.append(obstacle)
          # if ob_pos[0] < self.o_size/2 and self.ob_pos[0] > max_x-o_size/2 and ob_pos[1] < o_size/2 and ob_pos[1] > max_y-o_size/2:
          #   continue
          # x1 = int(ob_pos[0]-o_size/2)
          # x2 = int(ob_pos[0]+o_size/2)
          # y1 = int(ob_pos[1]-o_size/2)
          # y2 = int(ob_pos[1]+o_size/2)
          # #print(x1, x2, y1, y2, ob_pos)
          # c_map[x1:x2, y1:y2] = np.ones((o_size, o_size))
          # obstacles.append(ob_pos)
          i += 1
        return obstacles
    def check_collision_with_obstacles(self, pose):

        point = Point(pose.x(), pose.y())
        for obstacle in self.obstacles:
            if obstacle.contains(point) or obstacle.exterior.distance(point) <= 5:
                return True
        return False

    def randomPosition(self):
        xc = np.random.randint(self.max_x)
        yc = np.random.randint(self.max_y)
        return xc, yc

class Node_manip():
    def __init__(self, idx, base_pose, Q, end_effector_pose, dictionary, G):
        self.idx = idx
        self.q = Q
        self.base_pose = base_pose
        self.end_effector_pose = end_effector_pose
        self.neighbors = []
        self.parent = None
        dictionary[idx] = (self.base_pose, self.q, self.end_effector_pose)
        G.add_node(idx)

class Tree_manip():
    def __init__(self, curr_iter, curr_node):
        self.root = curr_node

    def find_dist(self, q0, q1):
        pose_1 = q0.end_effector_pose
        dist_val = np.linalg.norm(delta(q1, pose_1))
        return dist_val

    def iterate(self, q, qc):

        if q.neighbors == []: #no neighbors
            return 10000, q

        m1 = 1000000
        e1 = None

        for i in range(len(q.neighbors)):
            new_dist = self.find_dist(q.neighbors[i], qc)

            if new_dist < m1:
                e1 = q.neighbors[i]
                m1 = new_dist

            #explore neighbor
            m2, e2 = self.iterate(e1, qc)
            if m2 < m1:
                m1 = m2
                e1 = e2

        return m1, e1

    def steer(self, Q_curr, end_eff_new, q_new, base_new, step_size, arm, env, distance_limit=2.5):
        end_eff_1 = Q_curr.end_effector_pose
        base_1 = Q_curr.base_pose
        if env.check_collision_with_obstacles(end_eff_1):
            return None, None, None
        # q1 = np.asarray(Q_curr)
        # q2 = np.asarray(Q_new) #go in direction of q2
        if np.linalg.norm(delta(end_eff_1, end_eff_new)) < distance_limit:
            return base_new, q_new, end_eff_new
        q_sub = delta(end_eff_1, end_eff_new)
        q_dir = q_sub/np.linalg.norm(q_sub)
        q_dir *= step_size

        q_new = vector3(end_eff_1.x(), end_eff_1.y(), end_eff_1.theta()) + q_dir
        end_eff_steer = Pose2(q_new[0], q_new[1], q_new[2])
        if env.check_collision_with_obstacles(end_eff_steer) == False:
            base_pose, q_steer = arm.ik2(end_eff_steer, env.obstacles)
            return base_pose, q_steer, end_eff_steer
        else:
            return None, None, None

    def link(self, q1, q2, G):
        if q2 not in q1.neighbors:
            q1.neighbors.append(q2)
            q2.parent = q1
            G.add_edge(q1.idx, q2.idx)
            return

    def find_closest(self, Q_curr):
        min_dist, explore = self.iterate(self.root, Q_curr)
        return explore


def RRT(start_config, Qgoal, env, arm, lim=0.5, step_size=3, num_iters=1000):
    base, q, Qstart = start_config
    dictionary = {}
    curr_iter = 0
    # num_iters = 1000
    G = nx.Graph()
    # lim = 0.2 #The robot can stop at a configuration (q1,q2,q3,q4) where q[i] is at most 15 degrees away from Qgoal[i]
    # step_size = 3
    path = []
    root = Node_manip(0, base, q, Qstart, dictionary, G)
    path.append((base, q, Qstart))
    graph = Tree_manip(0, root)

    till_now = [root]
    curr_node = root

    while curr_iter < num_iters:

        if curr_iter % 10 == 0:
            print("ITERATION: " + str(curr_iter))
        sTt_b_rand, q_rand, end_eff_rand = RandomQ(Qgoal, arm, env)
        #print('NEW ITERATION')

        nearest_node = graph.find_closest(end_eff_rand)
        #c1 = [nearest_node.t1, nearest_node.t2, nearest_node.t3, nearest_node.t4]

        sTt_b_steer, q_steer, end_eff_steer = graph.steer(nearest_node, end_eff_rand,
                                                        q_rand, sTt_b_rand, step_size, arm, env,
                                                        distance_limit=step_size)

        if sTt_b_rand is not None and q_steer is not None and end_eff_steer is not None:
            add_node = Node_manip(curr_iter, sTt_b_steer, q_steer, end_eff_steer, dictionary, G)
            graph.link(nearest_node, add_node, G)
            curr_node = add_node
            curr_iter += 1
            #print(till_now)
            # till_now.append(steer_node)
            add = add_node.end_effector_pose
            #print('add',add,Qgoal)
            if np.linalg.norm(delta(add, Qgoal)) <= lim:
                print('reached goal')
                break
        else:
            curr_iter += 1

    print("Iterations to converge: " + str(curr_iter))
    v_path = nx.algorithms.shortest_path(G, root.idx, add_node.idx)
    print('Vertices for shortest path:',v_path)

    for p in v_path:
      path.append(dictionary[p])
    print('Actual path:',path)
    return path, dictionary, G
