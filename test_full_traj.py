import argparse
import numpy as np
import sys
sys.path.append("/notebooks/project/iris-distro/build/install/lib/python2.7/dist-packages")
import irispy

from pydrake.common import FindResourceOrThrow
from pydrake.examples.quadrotor import QuadrotorPlant
from controller import QuadrotorController
from controller import FullPPTrajectory as PPTrajectory
from controller import rotation_matrix as rm
from pydrake.geometry import SceneGraph
from pydrake.math import RollPitchYaw
from pydrake.multibody.plant import MultibodyPlant
from pydrake.multibody.parsing import Parser
from pydrake.systems.analysis import Simulator
from pydrake.systems.framework import DiagramBuilder, VectorSystem
from pydrake.systems.meshcat_visualizer import MeshcatVisualizer
from pydrake.systems.rendering import MultibodyPositionToGeometryPose
from pydrake.systems.primitives import AffineSystem
from pydrake.attic.multibody.rigid_body_tree import RigidBodyTree, AddModelInstanceFromUrdfFile, FloatingBaseType
from pydrake.attic.multibody.rigid_body_plant import RigidBodyPlant

START_STATE = np.zeros(12) #np.random.randn(12,)
START_STATE[0] = 4.0 # x = 4.0
GOAL_STATE = np.zeros(12)
GOAL_STATE[0] = -4.0 # x = -4.0
degree = 3
n_segments = 1
DURATION = 4.0
A = np.array([      [1,0,0],
                    [-1,0,0],
                    [0,1,0],
                    [0,-1,0],
                    [0,0,0],
                    [0,0,-1]
                ])
b = np.array([5, 5, 5, 5, 5, 5])
REGIONS = [(A,b)]
H = np.array([[1.0]])
# start, goal, degree, n_segments, duration, regions, lb, ub, dt
y_traj = PPTrajectory(START_STATE[:3], GOAL_STATE[:3], degree, n_segments, DURATION, REGIONS, H)
