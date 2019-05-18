

# In order to see the visualization, please follow the instructions printed at
# the console to open a web browser.

# Alternatively, you may start the 3D visualizer yourself by running
# `meshcat-server` in your shell, and then opening the url printed to the
# screen in your web browser. Then use `python lqr.py --meshcat default` to run
# this script and see that visualization.

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

LOWER_BOUND = [-5, -5, -3]
UPPER_BOUND = [5, 5, 3]
RADIUS = .25 # maybe .175
OBJECTS = [
    ([1, 1, 1], np.array([5,5,5]), rm([1,0,0], 0))
]
bounds = irispy.Polyhedron.fromBounds(LOWER_BOUND, UPPER_BOUND)
obstacles = []
for object_ in OBJECTS:
    sz, trans, _ = object_
    sz_x, sz_y, sz_z = sz
    trans_x, trans_y, trans_z = trans
    obstacles.append(np.array([
        [trans_x+(sz_x+RADIUS), trans_y+(sz_y+RADIUS), trans_z+(sz_z+RADIUS)],
        [trans_x+(sz_x+RADIUS), trans_y-(sz_y+RADIUS), trans_z+(sz_z+RADIUS)],
        [trans_x+(sz_x+RADIUS), trans_y-(sz_y+RADIUS), trans_z-(sz_z+RADIUS)],
        [trans_x+(sz_x+RADIUS), trans_y+(sz_y+RADIUS), trans_z-(sz_z+RADIUS)],
        [trans_x-(sz_x+RADIUS), trans_y+(sz_y+RADIUS), trans_z-(sz_z+RADIUS)],
        [trans_x-(sz_x+RADIUS), trans_y-(sz_y+RADIUS), trans_z-(sz_z+RADIUS)],
        [trans_x-(sz_x+RADIUS), trans_y-(sz_y+RADIUS), trans_z+(sz_z+RADIUS)],
        [trans_x-(sz_x+RADIUS), trans_y+(sz_y+RADIUS), trans_z+(sz_z+RADIUS)],
    ]).T)
REGIONS = []
REGIONS.append(irispy.inflate_region(obstacles, np.array([0,0,0]), bounds)) # in front
# REGIONS.append(irispy.inflate_region(obstacles, np.array([-3,0,0]), bounds)) # in back
# REGIONS.append(irispy.inflate_region(obstacles, np.array([0,3,0]), bounds)) # right
# REGIONS.append(irispy.inflate_region(obstacles, np.array([0,-3,0]), bounds)) # left
REGIONS = [(r.polyhedron.getA(), r.polyhedron.getB()) for r in REGIONS]


START_STATE = np.zeros(12) #np.random.randn(12,)
START_STATE[0] = 4.0 # x = 4.0
GOAL_STATE = np.zeros(12)
GOAL_STATE[0] = -4.0 # x = -4.0
DURATION = 4.0

dt = .1

n_segments = 1
degree = 3
# start, goal, degree, n_segments, duration, regions, lb, ub, dt
y_traj = PPTrajectory(START_STATE[:3], GOAL_STATE[:3], degree, n_segments, DURATION, REGIONS, dt)
# y_traj = PPTrajectory(sample_times, num_vars, degree, continuity_degree)
# y_traj.add_constraint(t=0, derivative_order=0, lb=START_STATE[:3]) # initial position
# y_traj.add_constraint(t=0, derivative_order=1, lb=[0, 0, 0])       # initial velocity
# y_traj.add_constraint(t=0, derivative_order=2, lb=[0, 0, 0])       # initial acceleration
# y_traj.add_constraint(t=1, derivative_order=0, lb=[0, 1.0,  0])
# y_traj.add_constraint(t=2, derivative_order=0, lb=[0,  2.0,  0])
# y_traj.add_constraint(t=3, derivative_order=0, lb=[0, 3.0,  0])
# y_traj.add_constraint(t=tf, derivative_order=0, lb=[0, 4.0, 0])      # end at zero
# y_traj.add_constraint(t=tf, derivative_order=1, lb=[0, 0, 0])
# y_traj.add_constraint(t=tf, derivative_order=2, lb=[0, 0, 0])
# y_traj.generate()


# Build
builder = DiagramBuilder()

plant = builder.AddSystem(QuadrotorPlant())

controller = builder.AddSystem(QuadrotorController(plant, y_traj, DURATION))
builder.Connect(controller.get_output_port(0), plant.get_input_port(0))
builder.Connect(plant.get_output_port(0), controller.get_input_port(0))

# Set up visualization and env in MeshCat
from pydrake.geometry import Box
from pydrake.common.eigen_geometry import Isometry3
from pydrake.all import AddMultibodyPlantSceneGraph
from pydrake.multibody.tree import SpatialInertia, UnitInertia
env_plant, scene_graph = AddMultibodyPlantSceneGraph(builder)
for idx, object_ in enumerate(OBJECTS):
    sz, trans, rot = object_
    body = env_plant.AddRigidBody(str(idx+1), SpatialInertia(1.0, np.zeros(3), UnitInertia(1.0,1.0,1.0)))
    pos = Isometry3()
    pos.set_translation(trans) # np.array([0,1,1])
    pos.set_rotation(rot)         # rm([1,0,0], 90)
    env_plant.RegisterVisualGeometry(body, pos, Box(sz[0], sz[1], sz[2]), str(idx+1)+'v', np.array([1,1,1,1]), scene_graph)
env_plant.Finalize()
plant.RegisterGeometry(scene_graph)
builder.Connect(plant.get_geometry_pose_output_port(),
                scene_graph.get_source_pose_port(plant.source_id()))
meshcat = builder.AddSystem(MeshcatVisualizer(
    scene_graph, zmq_url=None,
    open_browser=None))
builder.Connect(scene_graph.get_pose_bundle_output_port(),
                meshcat.get_input_port(0))
# end setup for visualization

diagram = builder.Build()

simulator = Simulator(diagram)
simulator.set_target_realtime_rate(1.0)
context = simulator.get_mutable_context()
import pdb; pdb.set_trace()
# Run simulation
context.set_time(0.)
mutable_state_vector = context.get_continuous_state_vector().CopyToVector()
mutable_state_vector[:12] = START_STATE
context.SetContinuousState(mutable_state_vector)
simulator.Initialize()
simulator.StepTo(DURATION)

import pdb; pdb.set_trace()
