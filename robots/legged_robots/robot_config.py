# coding=utf-8
# Copyright 2020 The Google Research Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""The configuration parameters for our robots."""
import attr
from typing import Union, Text, List
from collections.abc import Sequence
import enum
import numpy as np


class MotorControlMode(enum.Enum):
    """The supported motor control modes."""
    POSITION = 1

    # Apply motor torques directly.
    TORQUE = 2

    # Apply a tuple (q, qdot, kp, kd, tau) for each motor. Here q, qdot are motor
    # position and velocities. kp and kd are PD gains. tau is the additional
    # motor torque. This is the most flexible control mode.
    HYBRID = 3

    HYBRID_COMPUTED_POS = 4
    HYBRID_COMPUTED_POS_VEL = 5

    HYBRID_COMPUTED_POS_SINGLE = 6
    HYBRID_COMPUTED_POS_TROT = 7


# Each hybrid action is a tuple (position, position_gain, velocity,
# velocity_gain, torque)
HYBRID_ACTION_DIMENSION = 5


class HybridActionIndex(enum.Enum):
    # The index of each component within the hybrid action tuple.
    POSITION = 0
    POSITION_GAIN = 1
    VELOCITY = 2
    VELOCITY_GAIN = 3
    TORQUE = 4


@attr.s
class HybridCommands(object):
    torque_estimate = attr.ib(type=np.ndarray, default=None)
    position_desired = attr.ib(type=np.ndarray, default=None)
    velocity_desired = attr.ib(type=np.ndarray, default=None)
    kp = attr.ib(type=float, default=0.0)
    kd = attr.ib(type=float, default=0.0)


@attr.s
class RobotSimNames(object):
    motor_names = attr.ib(type=List[Text], default=None)
    link_names = attr.ib(type=List[Text], default=None)
    link_leg_names = attr.ib(type=List[Text], default=None)
    link_foot_names = attr.ib(type=List[Text], default=None)
    link_single_leg_names = attr.ib(type=List[List[Text]], default=None)
    link_disable_collision_names = attr.ib(type=List[List[Text]], default=None)


@attr.s
class ScalarField(object):
    """A named scalar space with bounds."""
    name = attr.ib(type=str)
    upper_bound = attr.ib(type=float)
    lower_bound = attr.ib(type=float)


@attr.s
class RobotSimParams(object):
    JOINT_ANGLE_LIMIT = attr.ib(type=List[ScalarField],
                                default=[ScalarField(name="", upper_bound=np.inf, lower_bound=-np.inf)])
    JOINT_VELOCITY_LIMIT = attr.ib(type=List[ScalarField],
                                   default=[ScalarField(name="", upper_bound=np.inf, lower_bound=-np.inf)])
    JOINT_TORQUE_LIMIT = attr.ib(type=List[ScalarField],
                                 default=[ScalarField(name="", upper_bound=np.inf, lower_bound=-np.inf)])

    joint_angle_MinMax = attr.ib(type=List[List[float]], default=[[-np.inf], [np.inf]])
    joint_velocity_MinMax = attr.ib(type=List[List[float]], default=[[-np.inf], [np.inf]])
    joint_torque_MinMax = attr.ib(type=List[List[float]], default=[[-np.inf], [np.inf]])

    urdf_filepath = attr.ib(type=Text, default=None)
    num_motors = attr.ib(type=int, default=1)
    dofs_per_leg = attr.ib(type=int, default=1)
    on_rack = attr.ib(type=bool, default=True)
    enable_self_collision = attr.ib(type=bool, default=True)
    init_position = attr.ib(type=List[float], default=[0, 0, 0])
    init_orientation = attr.ib(type=List[float], default=[0, 0, 0, 1])
    init_motor_angles = attr.ib(type=Union[float, Sequence, np.ndarray], default=None)
    motor_offset = attr.ib(type=Union[float, Sequence, np.ndarray], default=None)
    motor_direction = attr.ib(type=Union[int, Sequence, np.ndarray], default=None)
    motor_control_mode = attr.ib(type=int, default=MotorControlMode.POSITION)
    motor_kp = attr.ib(type=Union[float, Sequence, np.ndarray], default=None)
    motor_kd = attr.ib(type=Union[float, Sequence, np.ndarray], default=None)
    sim_names = attr.ib(type=RobotSimNames, default=None)
    reset_at_current_pose = attr.ib(type=bool, default=False)
    control_latency = attr.ib(type=float, default=0.0)
    pd_latency = attr.ib(type=float, default=0.0)


@attr.s
class RobotRawObservation(object):
    jointPosition = attr.ib(type=np.ndarray, default=None)
    jointVelocity = attr.ib(type=np.ndarray, default=None)
    appliedJointMotorTorque = attr.ib(type=np.ndarray, default=None)
    basePosition_world = attr.ib(type=List[float], default=[-1, -1, -1])
    baseOrientation_world = attr.ib(type=List[float], default=[0, 0, 0, 1])
    baseRollPitchYaw_world = attr.ib(type=List[float], default=[0, 0, 0])
    baseLinearVelocity_world = attr.ib(type=List[float], default=[0, 0, 0])
    baseAngularVelocity_world = attr.ib(type=List[float], default=[0, 0, 0])
    baseLinearVelocity_body = attr.ib(type=List[float], default=[0, 0, 0])
    baseAngularVelocity_body = attr.ib(type=List[float], default=[0, 0, 0])
    baseOrientation_init = attr.ib(type=List[float], default=[0, 0, 0, 1])
    baseRollPitchYaw_init = attr.ib(type=List[float], default=[0, 0, 0])
    # baseRollPitchYaw_local = attr.ib(type=List[float], default=[0, 0, 0])
    # baseLinearVelocity_local = attr.ib(type=List[float], default=[0, 0, 0])
    # baseAngularVelocity_local = attr.ib(type=List[float], default=[0, 0, 0])
