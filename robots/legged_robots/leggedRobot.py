from __future__ import annotations
from typing import List, Union
import pybullet as p
from robots.legged_robots.robot_config import *
from robots.motors.bldc_motor import BLDC_motor
from collections.abc import Sequence
import numpy as np
import time
from utilities.pd_controller_stable_custom import PDControllerStable


class LeggedRobot:
    def __init__(self,
                 pybullet_client,
                 robot_params: RobotSimParams,
                 time_step: float = 1.0 / 240.0):

        self._pybullet_client = pybullet_client
        self._robot_params = robot_params
        self._urdf_filepath = robot_params.urdf_filepath
        self._num_motors = robot_params.num_motors
        self._dofs_per_leg = robot_params.dofs_per_leg
        self._enable_self_collision = robot_params.enable_self_collision
        self._on_rack = robot_params.on_rack
        self._init_position = robot_params.init_position
        self._init_orientation = robot_params.init_orientation
        self._time_step = time_step
        self._motor_control_mode = robot_params.motor_control_mode
        self._init_motor_angles = self._dtype_norm(robot_params.init_motor_angles, "init_motor_angles")
        self._motor_offset = self._dtype_norm(robot_params.motor_offset, "motor_offset")
        self._motor_direction = self._dtype_norm(robot_params.motor_direction, "motor_direction")
        self._motor_kp = self._dtype_norm(robot_params.motor_kp, "motor_kp")
        self._motor_kd = self._dtype_norm(robot_params.motor_kd, "motor_kd")

        self._motor_id_list = []
        self._link_id_list = []
        self._chassis_link_id_list = [-1]
        self._leg_link_id_list = []
        self._foot_link_id_list = []

        self._robot_raw_obs = RobotRawObservation()

        self._motor = BLDC_motor(self._num_motors)
        self._stablePD = PDControllerStable(self._pybullet_client)

        self._is_urdf_loaded = False

        self._num_robot_act = 0

        self.Reset()

    def _dtype_norm(self, data: Union[int, Sequence, np.ndarray], data_name: str) -> np.ndarray:
        if isinstance(data, (Sequence, np.ndarray)):
            if len(data) != self._num_motors:
                raise ValueError(f"The length of {data_name} is wrong!")
            else:
                return np.asarray(data)
        else:
            return np.full(self._num_motors, data)

    def Reset(self, reload_urdf=True, default_motor_angles=None, reset_time=3.0):
        if reload_urdf:
            if self._is_urdf_loaded:
                self._pybullet_client.removeBody(self.robotID)
            self._LoadRobotURDF()
            self._BuildUrdfIdLists()

            p.changeVisualShape(self.robotID, -1, rgbaColor=[174.0 / 255.0, 187.0 / 255.0, 143.0 / 255.0, 0.5])
            for i in range(len(self._link_id_list)):
                p.changeVisualShape(self.robotID, i, rgbaColor=[float(i / len(self._link_id_list)), 0.5,
                                                                1 - float(i / len(self._link_id_list)), 0.9])

            self._RemoveDefaultJointDamping()

            self.ResetPose()
        else:
            self._pybullet_client.resetBasePositionAndOrientation(
                self.robotID,
                self._init_position,
                self._init_orientation
            )
            self._pybullet_client.resetBaseVelocity(self.robotID, [0, 0, 0], [0, 0, 0])
            self.ResetPose()

        self._SettleDownForReset(default_motor_angles, reset_time)
        self._pybullet_client.setJointMotorControlArray(self.robotID,
                                                        self._motor_id_list,
                                                        p.POSITION_CONTROL,
                                                        forces=[0] * self._num_motors)
        self._num_robot_act = 0

    def Step(self, action=None, control_mode=None):

        if control_mode is None:
            control_mode = self._motor_control_mode

        self._StepInternal(action, control_mode)

        time_start = time.time()
        # time.sleep(self._time_step)
        time_spent = time.time() - time_start
        self._num_robot_act += 1
        # if self._num_robot_act % 100 == 0:
        #     print(f"time spent is: {time_spent * 1000} ms.")



    def _LoadRobotURDF(self):
        """Loads the URDF file for the robot."""
        flags = p.URDF_USE_INERTIA_FROM_FILE | (p.URDF_USE_SELF_COLLISION if self._enable_self_collision else 0)
        self.robotID = self._pybullet_client.loadURDF(
            self._urdf_filepath,
            self._init_position,
            self._init_orientation,
            useFixedBase=self._on_rack,
            flags=flags
        )
        if self.robotID < 0:
            raise Exception("The return of loadURDF function is negative!")
        else:
            self._is_urdf_loaded = True
            self._baseMass = self._pybullet_client.getDynamicsInfo(self.robotID, -1)[0]

    def _BuildUrdfIdLists(self):
        self._BuildUrdfName2IdDict()
        self._motor_id_list = [self._joint_name_to_id[joint_name] for joint_name in
                               self._robot_params.sim_names.motor_names]
        self._chassis_link_id_list = [-1]
        self._link_id_list = [self._link_name_to_id[link_name] for link_name in
                              self._robot_params.sim_names.link_names]
        self._leg_link_id_list = [self._link_name_to_id[link_name] for link_name in
                                  self._robot_params.sim_names.link_leg_names]
        # self._foot_link_id_list = [(self._link_name_to_id[link_name] for link_name in self._robot_params.sim_names.link_foot_names) if self._robot_params.sim_names.link_foot_names is not None else None]
        self._foot_link_id_list = [(self._link_name_to_id[link_name] for link_name in
                                    self._robot_params.sim_names.link_foot_names)
                                   if self._robot_params.sim_names.link_foot_names is not None else None]

    def _BuildUrdfName2IdDict(self):
        num_joints = self._pybullet_client.getNumJoints(self.robotID)
        self._joint_name_to_id = {}
        self._link_name_to_id = {}
        for i in range(num_joints):
            joint_info = self._pybullet_client.getJointInfo(self.robotID, i)
            self._joint_name_to_id[joint_info[1].decode("UTF-8")] = joint_info[0]
            self._link_name_to_id[joint_info[12].decode("UTF-8")] = joint_info[0]

    def _RemoveDefaultJointDamping(self):
        self._pybullet_client.changeDynamics(self.robotID,
                                             -1,
                                             lateralFriction=1.0,
                                             linearDamping=0,
                                             angularDamping=0)
        for i in range(len(self._link_id_list)):
            self._pybullet_client.changeDynamics(self.robotID,
                                                 self._link_id_list[i],
                                                 lateralFriction=1.0,
                                                 linearDamping=0,
                                                 angularDamping=0)

    def ResetPose(self):

        # for i in range(self._num_motors):
        #     p.resetJointState(self.robotID,
        #                       self._motor_id_list[i],
        #                       (self._init_motor_angles[i] + self._motor_offset[i]) * self._motor_direction[i],
        #                       targetVelocity=0,
        #                       physicsClientId=self._pybullet_client)

        for i in range(self._num_motors):
            self._pybullet_client.setJointMotorControl2(
                self.robotID,
                self._motor_id_list[i],
                p.POSITION_CONTROL,
                targetPosition=(self._init_motor_angles[i] + self._motor_offset[i]) * self._motor_direction[i],
                force=30,
                positionGain=0.05,
                maxVelocity=23
            )

        # self._pybullet_client.setJointMotorControlArray(self.robotID,
        #                             self._motor_id_list,
        #                             p.POSITION_CONTROL,
        #                             targetPositions=(self._init_motor_angles + self._motor_offset) * self._motor_direction,
        #                             #targetVelocities=[0]*self._num_motors,
        #                             forces=[20]*self._num_motors)

    def _SettleDownForReset(self, default_motor_angles, reset_time):
        if reset_time <= 0:
            return

        for _ in range(100):
            self._pybullet_client.stepSimulation()
            time.sleep(self._time_step)

        if default_motor_angles is None:
            return

        num_steps_to_reset = int(reset_time / self._time_step)
        current_motor_position = self._motor.motor_sensor.motorPosition
        delta_position = np.asarray(default_motor_angles) - current_motor_position

        self.ReceiveObservation()
        for i in range(num_steps_to_reset):
            inter_action = current_motor_position + float(i + 1) * delta_position / num_steps_to_reset
            self._StepInternal(inter_action, MotorControlMode.POSITION)
            time.sleep(self._time_step)

    def ReceiveObservation(self):
        joint_states = self._pybullet_client.getJointStates(self.robotID, self._motor_id_list)
        jointPosition = [state[0] for state in joint_states]
        jointVelocity = [state[1] for state in joint_states]
        appliedJointMotorTorque = [state[3] for state in joint_states]
        self._robot_raw_obs.jointPosition = np.multiply(jointPosition, self._motor_direction)
        self._robot_raw_obs.jointVelocity = np.multiply(jointVelocity, self._motor_direction)
        self._robot_raw_obs.appliedJointMotorTorque = np.multiply(appliedJointMotorTorque, self._motor_direction)

        self._motor.motor_sensor.motorPosition = self._robot_raw_obs.jointPosition
        self._motor.motor_sensor.motorVelocity = self._robot_raw_obs.jointVelocity
        self._motor.motor_sensor.motorTorque = self._robot_raw_obs.appliedJointMotorTorque

        """ ---relative to the world frame--- """
        self._robot_raw_obs.basePosition_world, self._robot_raw_obs.baseOrientation_world = self._pybullet_client.getBasePositionAndOrientation(
            self.robotID)

        self._robot_raw_obs.baseRollPitchYaw_world = self._pybullet_client.getEulerFromQuaternion(
            self._robot_raw_obs.baseOrientation_world)

        self._robot_raw_obs.baseLinearVelocity_world, self._robot_raw_obs.baseAngularVelocity_world = self._pybullet_client.getBaseVelocity(
            self.robotID)
        """ -------------------------------------- """

        _, baseOrientation_inv = p.invertTransform(position=[0, 0, 0],
                                                   orientation=self._robot_raw_obs.baseOrientation_world)

        """ ---relative to the local base frame--- """
        self._robot_raw_obs.baseLinearVelocity_local, _ = p.multiplyTransforms(
            [0, 0, 0],
            baseOrientation_inv,
            self._robot_raw_obs.baseLinearVelocity_world,
            p.getQuaternionFromEuler([0, 0, 0]))

        self._robot_raw_obs.baseAngularVelocity_local, _ = p.multiplyTransforms(
            [0, 0, 0],
            baseOrientation_inv,
            self._robot_raw_obs.baseAngularVelocity_world,
            p.getQuaternionFromEuler([0, 0, 0]))
        """ -------------------------------------- """

    def ApplyAction(self, motor_commands, motor_control_mode):
        if motor_commands is None:
            return

        control_mode = motor_control_mode
        if control_mode is None:
            control_mode = self._motor_control_mode

        if control_mode is MotorControlMode.POSITION:
            motor_commands = np.asarray(motor_commands)
            self._pybullet_client.setJointMotorControlArray(self.robotID,
                                                            self._motor_id_list,
                                                            p.POSITION_CONTROL,
                                                            targetPositions=motor_commands * self._motor_direction,
                                                            forces=[self._motor.torque_limit] * self._num_motors)

        elif control_mode is MotorControlMode.TORQUE:
            motor_commands = np.asarray(motor_commands)
            motor_torqes = np.clip(motor_commands, -1.0 * self._motor.torque_limit, self._motor.torque_limit)
            self._pybullet_client.setJointMotorControlArray(
                self.robotID,
                self._motor_id_list,
                p.TORQUE_CONTROL,
                forces=motor_torqes * self._motor_direction
            )
        elif control_mode is MotorControlMode.HYBRID:
            if not isinstance(motor_commands, HybridCommands):
                raise ValueError("The dataType of motor_commands is wrong!")
            elif motor_commands.torque_estimate is None:
                numBaseDofs = 6 if self._baseMass > 0 else 0
                tau = self._stablePD.computePD(self.robotID,
                                               self._motor_id_list,
                                               motor_commands.position_desired,
                                               motor_commands.velocity_desired,
                                               kps=[motor_commands.kp] * (numBaseDofs + self._num_motors),
                                               kds=[motor_commands.kd] * (numBaseDofs + self._num_motors),
                                               maxForces=[self._motor.torque_limit] * (numBaseDofs + self._num_motors),
                                               timeStep=self._time_step)
                motor_torqes = tau[numBaseDofs:] * -1
                # motor_commands.torque_estimate = motor_torqes
                # motor_torqes = self._motor.hybridCmd_to_torque(motor_commands)
            else:
                motor_torqes = self._motor.hybridCmd_to_torque(motor_commands)

            self._pybullet_client.setJointMotorControlArray(self.robotID,
                                                            self._motor_id_list,
                                                            p.TORQUE_CONTROL,
                                                            forces=motor_torqes * self._motor_direction
                                                            )

    def _StepInternal(self, action, motor_control_mode):
        self.ApplyAction(action, motor_control_mode)
        self._pybullet_client.stepSimulation()
        self.ReceiveObservation()

    def Terminate(self):
        pass

    def ReadBaseHeight(self):
        return self._robot_raw_obs.basePosition_world[2]

    def getRobotID(self):
        return self.robotID

    def GetBasePosition(self):
        return self._robot_raw_obs.basePosition_world
