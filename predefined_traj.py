

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


START_STATE = np.zeros(12) #np.random.randn(12,)
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
y_traj.add_constraint(t=1, derivative_order=0, lb=[-0.5, -0.5, -0.5])
y_traj.add_constraint(t=2, derivative_order=0, lb=[1, 1, 0.5])
y_traj.add_constraint(t=3, derivative_order=0, lb=[0.5, 0.5, 0.5])
y_traj.add_constraint(t=tf, derivative_order=0, lb=[0, 0, 0])      # end at zero
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

# Set up visualization in MeshCat
scene_graph = builder.AddSystem(SceneGraph())
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
context.SetContinuousState(START_STATE)
simulator.Initialize()
simulator.StepTo(DURATION)

import pdb; pdb.set_trace()
