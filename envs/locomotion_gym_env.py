import numpy as np
import gym
from gym import spaces
from gym.utils import seeding
import pybullet as p
import pybullet_data
import pkgutil
from pybullet_utils.bullet_client import BulletClient
from envs.locomotion_gym_config import SimulationParameters
from robots.legged_robots.robot_config import MotorControlMode, RobotSimParams
from envs.set_gui_sliders import set_gui_sliders
import time
from tasks.base_task import BaseTask
from sensors.sensor_wrappers import SensorWrapper
from tasks.contact_fall_for_legged_robot import ContactDetection


class LocomotionGymEnv(gym.Env):
    def __init__(self,
                 gym_config: SimulationParameters,
                 robot_class=None,
                 robot_params: RobotSimParams = None,
                 task: BaseTask = None):
        self.seed()
        self._gym_config = gym_config
        if robot_class is None:
            raise ValueError('robot_class cannot be None.')
        self._robot_class = robot_class

        # A dictionary containing the objects in the world other than the robot.
        self._world_dict = {}

        self._robot_params = robot_params

        self._task = task

        self._obs_sensor = None

        self._time_step = self._gym_config.time_step
        self._num_action_repeat = self._gym_config.num_action_repeat

        # self._num_bullet_solver_iterations = 12

        self._is_render = self._gym_config.enable_rendering and (not self._gym_config.egl_rendering)
        if self._is_render:
            self._pybullet_client = BulletClient(connection_mode=p.GUI)
            self._pybullet_client.configureDebugVisualizer(p.COV_ENABLE_GUI, gym_config.enable_rendering_gui)
        else:
            self._pybullet_client = BulletClient(connection_mode=p.DIRECT)

        self._pybullet_client.setAdditionalSearchPath(pybullet_data.getDataPath())

        self._last_frame_time = 0.0

        # set egl acceleration
        if self._gym_config.egl_rendering:
            egl = pkgutil.get_loader('eglRenderer')
            self._eglPlugin = self._pybullet_client.loadPlugin(egl.get_filename(), "_eglRendererPlugin")

        self._build_action_space()
        self._build_observation_space()

        # Set the default render options.
        self._camera_dist = self._gym_config.camera_distance
        self._camera_yaw = self._gym_config.camera_yaw
        self._camera_pitch = self._gym_config.camera_pitch
        self._render_width = self._gym_config.render_width
        self._render_height = self._gym_config.render_height

        self._hard_reset = True

        self._num_env_act = 0

        self.set_gui_sliders()

        self._contact_fall = ContactDetection(self)

        self._reset_time = self._gym_config.reset_time
        self.reset(start_motor_angles=None, reset_duration=self._reset_time)

        if self._task is not None:
            self._task.reset(self)

        self._hard_reset = self._gym_config.enable_hard_reset

        # global_values.global_userDebugParams = UserDebugParams(self._pybullet_client)
        # global_values.global_userDebugParams.setPbClient(self._pybullet_client)
        # set_gui_sliders(self._pybullet_client)

    def _build_action_space(self):
        motor_mode = self._robot_params.motor_control_mode
        action_upper_bound = []
        action_lower_bound = []

        if (motor_mode == MotorControlMode.POSITION) or (motor_mode == MotorControlMode.HYBRID_COMPUTED_POS):
            action_lower_bound.extend(self._robot_params.joint_angle_MinMax[0])
            action_upper_bound.extend(self._robot_params.joint_angle_MinMax[1])
            # action_config = self._robot_params.JOINT_ANGLE_LIMIT
            # for action in action_config:
            #     action_upper_bound.append(action.upper_bound)
            #     action_lower_bound.append(action.lower_bound)

        elif motor_mode == MotorControlMode.TORQUE:
            action_lower_bound.extend(self._robot_params.joint_torque_MinMax[0])
            action_upper_bound.extend(self._robot_params.joint_torque_MinMax[1])
            # action_config = self._robot_params.JOINT_TORQUE_LIMIT
            # for action in action_config:
            #     action_upper_bound.append(action.upper_bound)
            #     action_lower_bound.append(action.lower_bound)

        elif motor_mode == MotorControlMode.HYBRID_COMPUTED_POS_VEL:
            action_lower_bound.extend(self._robot_params.joint_angle_MinMax[0])
            action_lower_bound.extend(self._robot_params.joint_velocity_MinMax[0])
            action_upper_bound.extend(self._robot_params.joint_angle_MinMax[1])
            action_upper_bound.extend(self._robot_params.joint_velocity_MinMax[1])
            # for action_config in [self._robot_params.JOINT_ANGLE_LIMIT, self._robot_params.JOINT_VELOCITY_LIMIT]:
            #     for action in action_config:
            #         action_upper_bound.append(action.upper_bound)
            #         action_lower_bound.append(action.lower_bound)
        elif motor_mode == MotorControlMode.HYBRID_COMPUTED_POS_SINGLE:
            action_lower_bound.extend(self._robot_params.joint_angle_MinMax[0][:3])
            action_upper_bound.extend(self._robot_params.joint_angle_MinMax[1][:3])

        elif motor_mode == MotorControlMode.HYBRID_COMPUTED_POS_TROT:
            action_lower_bound.extend(self._robot_params.joint_angle_MinMax[0][:6])
            action_upper_bound.extend(self._robot_params.joint_angle_MinMax[1][:6])

        self.action_space = spaces.Box(np.array(action_lower_bound),
                                       np.array(action_upper_bound),
                                       dtype=np.float32)

    def _build_observation_space(self):
        if self._task is None:
            return
        else:
            self._obs_sensor = SensorWrapper(self._task.get_observation_sensors())
        self.observation_space = spaces.Box(self._obs_sensor.get_lower_bound(),
                                            self._obs_sensor.get_upper_bound(),
                                            dtype=np.float32)

    def seed(self, seed=None):
        self.np_random, self.np_random_seed = seeding.np_random(seed)
        return [self.np_random_seed]

    def reset(self,
              start_motor_angles=None,
              reset_duration=0.002,
              reset_visualization_camera=True,
              force_hard_reset=False
              ):
        if self._is_render:
            self._pybullet_client.configureDebugVisualizer(p.COV_ENABLE_RENDERING, 0)

        # Clear the simulation world and rebuild the robot interface.
        if force_hard_reset or self._hard_reset:
            self._pybullet_client.resetSimulation()
            # self._pybullet_client.setPhysicsEngineParameter(numSolverIterations=self._num_bullet_solver_iterations)
            self._pybullet_client.setTimeStep(self._time_step)
            self._pybullet_client.setGravity(0, 0, -9.8)

            # Rebuild the world.
            self._world_dict = {
                "ground": self._pybullet_client.loadURDF("plane_implicit.urdf")
            }

            # Rebuild the robot
            self._robot = self._robot_class(self._pybullet_client,
                                            self._robot_params,
                                            time_step=self._time_step,
                                            num_action_repeat=self._num_action_repeat)

            if self._obs_sensor is not None:
                self._obs_sensor.set_robot(self._robot)

        # Reset the pose of the robot.
        # self._robot.Reset(reload_urdf=False, default_motor_angles=start_motor_angles, reset_time=reset_duration)

        if self._is_render and reset_visualization_camera:
            self._pybullet_client.resetDebugVisualizerCamera(self._camera_dist,
                                                             self._camera_yaw,
                                                             self._camera_pitch,
                                                             [0, 0, 0])

        if self._is_render:
            self._pybullet_client.configureDebugVisualizer(p.COV_ENABLE_RENDERING, 1)
        self._robot.Reset(reload_urdf=False, default_motor_angles=start_motor_angles, reset_time=reset_duration)
        if self._obs_sensor is not None:
            self._obs_sensor.on_reset()

        self._contact_fall.reset()

        self._num_env_act = 0

        return self._get_observation()

    def step(self, action):
        time_spent = time.time() - self._last_frame_time
        self._last_frame_time = time.time()
        # if self._num_env_act % 1 == 0:
        #     print(f"time_spent: {time_spent * 1000} ms")

        if self._is_render:
            # Sleep, otherwise the computation takes less time than real time,
            # which will make the visualization like a fast-forward video.
            time_to_sleep = self._time_step * self._num_action_repeat - time_spent
            if time_to_sleep > 0:
                time.sleep(time_to_sleep)

        # action = action * np.asarray(self.action_space.high)
        self._robot.Step(action, None)

        if self._obs_sensor is not None:
            self._obs_sensor.on_step()

        self._num_env_act += 1

        return self._get_observation(), self._get_reward(), self.done, {}

    def render(self, mode='rgb_array'):
        if mode != 'rgb_array':
            raise ValueError('Unsupported render mode:{}'.format(mode))
        base_pos = self._robot.GetBasePosition()
        view_matrix = self._pybullet_client.computeViewMatrixFromYawPitchRoll(
            cameraTargetPosition=base_pos,
            distance=self._camera_dist,
            yaw=self._camera_yaw,
            pitch=self._camera_pitch,
            roll=0,
            upAxisIndex=2)
        proj_matrix = self._pybullet_client.computeProjectionMatrixFOV(
            fov=60,
            aspect=float(self._render_width) / self._render_height,
            nearVal=0.1,
            farVal=100.0)
        (_, _, px, _, _) = self._pybullet_client.getCameraImage(
            width=self._render_width,
            height=self._render_height,
            renderer=self._pybullet_client.ER_BULLET_HARDWARE_OPENGL,
            viewMatrix=view_matrix,
            projectionMatrix=proj_matrix)
        rgb_array = np.array(px)
        rgb_array = rgb_array[:, :, :3]
        return rgb_array

    def close(self):
        if hasattr(self, '_robot') and self._robot:
            self._robot.Terminate()

    def getRobotID(self):
        return self._robot.getRobotID()

    @property
    def robot(self):
        return self._robot

    @property
    def ground(self):
        return self._world_dict["ground"]

    def getClient(self):
        return self._pybullet_client

    @property
    def env_step_counter(self):
        return self._num_env_act

    @property
    def baseHeight(self):
        return self._robot.ReadBaseHeight()

    def set_gui_sliders(self):
        set_gui_sliders(self._pybullet_client)

    def _get_observation(self):
        if self._obs_sensor is not None:
            return self._obs_sensor.get_observation()
        else:
            return None

    def _get_reward(self):
        if self._task is not None:
            return self._task.reward()
        else:
            return None

    @property
    def done(self) -> bool:
        # if self._obs_sensor is not None:
        #     if self._obs_sensor.on_terminate():
        #         return True

        if self._contact_fall.is_contact_fall():
            return True

        if self._task is not None:
            return self._task.done()
        else:
            return False
