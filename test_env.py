

# In order to see the visualization, please follow the instructions printed at
# the console to open a web browser.

# Alternatively, you may start the 3D visualizer yourself by running
# `meshcat-server` in your shell, and then opening the url printed to the
# screen in your web browser. Then use `python lqr.py --meshcat default` to run
# this script and see that visualization.

import argparse
import numpy as np

from pydrake.common import FindResourceOrThrow
from pydrake.examples.quadrotor import QuadrotorPlant
from controller import QuadrotorController, PPTrajectory
from pydrake.geometry import SceneGraph
from pydrake.math import RollPitchYaw
from pydrake.multibody.plant import MultibodyPlant
from pydrake.multibody.parsing import Parser
from pydrake.systems.analysis import Simulator
from pydrake.systems.framework import DiagramBuilder, VectorSystem
from pydrake.systems.meshcat_visualizer import MeshcatVisualizer
from pydrake.systems.rendering import MultibodyPositionToGeometryPose
from pydrake.systems.primitives import AffineSystem
from pydrake.attic.multibody.rigid_body_tree import (
    RigidBodyTree, AddModelInstanceFromUrdfFile, 
    FloatingBaseType, AddModelInstancesFromSdfString,
    RigidBodyFrame)
from pydrake.attic.multibody.rigid_body_plant import RigidBodyPlant

START_STATE = np.zeros(12) #np.random.randn(12,)
START_STATE[2] = 2.0
DURATION = 4.0

sample_times = np.linspace(0, DURATION, 6)
num_vars = 3
degree = 5
continuity_degree = 4
tf = DURATION
y_traj = PPTrajectory(sample_times, num_vars, degree, continuity_degree)
y_traj.add_constraint(t=0, derivative_order=0, lb=START_STATE[:3]) # initial position
y_traj.add_constraint(t=0, derivative_order=1, lb=[0, 0, 0])       # initial velocity
y_traj.add_constraint(t=0, derivative_order=2, lb=[0, 0, 0])       # initial acceleration
y_traj.add_constraint(t=1, derivative_order=0, lb=[0, 0, 1.0])
y_traj.add_constraint(t=2, derivative_order=0, lb=[0, 0.5, 0.5])
y_traj.add_constraint(t=3, derivative_order=0, lb=[0.5, 0.5, 0.5])
y_traj.add_constraint(t=tf, derivative_order=0, lb=START_STATE[:3])      # end at zero
y_traj.add_constraint(t=tf, derivative_order=1, lb=[0, 0, 0])
y_traj.add_constraint(t=tf, derivative_order=2, lb=[0, 0, 0])
y_traj.generate()
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

# Tree...
# env = RigidBodyTree('./office.urdf', FloatingBaseType.kFixed)
# env_plant = builder.AddSystem(RigidBodyPlant(env))

# Set up visualization in MeshCat
from pydrake.geometry import Box
from pydrake.common.eigen_geometry import Isometry3
scene_graph = builder.AddSystem(SceneGraph())
from pydrake.all import AddMultibodyPlantSceneGraph
from pydrake.multibody.tree import SpatialInertia, UnitInertia
from controller import rotation_matrix as rm
env_plant, scene_graph = AddMultibodyPlantSceneGraph(builder)
# parser = Parser(env_plant, scene_graph)
body = env_plant.AddRigidBody('1', SpatialInertia(1.0, np.zeros(3), UnitInertia(1.0,1.0,1.0)))
pos = Isometry3()
pos.set_translation(np.array([0,1,1]))
pos.set_rotation(rm([1,0,0], 90))
env_plant.RegisterVisualGeometry(body, pos, Box(.1,.1,1), '1v', np.array([1,1,1,1]), scene_graph)
env_plant.Finalize()
# import pdb; pdb.set_trace()
# env_plant.Finalize()

# base_positions = [np.array([0,0,0])]
# tree = RigidBodyTree()
# for i, base_pos in enumerate(base_positions):
#     frame = RigidBodyFrame(
#         name="finger_%d_base_frame" % i,
#         body=tree.world(),
#         xyz=base_pos,
#         rpy=[0, 0, 0])
#     tree.addFrame(frame)
#     AddModelInstancesFromSdfString(
#         open("./warehouse.sdf", 'r').read(),
#         FloatingBaseType.kFixed,
#         frame, tree)
# env_plant = builder.AddSystem(RigidBodyPlant(tree))
# import pdb; pdb.set_trace()

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

# Run simulation
context.set_time(0.)
mutable_state_vector = context.get_continuous_state_vector().CopyToVector()
mutable_state_vector[:12] = START_STATE
context.SetContinuousState(mutable_state_vector)
simulator.Initialize()
simulator.StepTo(DURATION)

import pdb; pdb.set_trace()
