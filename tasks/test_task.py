from tasks.base_task import BaseTask
from sensors.robot_sensors import *
from sensors.sensor_wrappers import *
from sensors.basesensor import BaseSensor
from utilities import global_values
import numpy as np
from utilities.motion_util import *

BODY_HEIGHT_MIN = 0.080
BODY_HEIGHT_MAX = 0.415


class CommandBodyHeight(BaseSensor):
    def __init__(self,
                 train_or_test: str = "train",
                 name: Text = "Command_Body_Height",
                 lower_bound: float = -1.0,
                 upper_bound: float = 1.0,
                 dtype: Type[Any] = np.float64):
        assert train_or_test in ["train", "test"]
        self._train_or_test = train_or_test
        self._num_update = 0
        self._cmd = None

        super(CommandBodyHeight, self).__init__(
            name=name,
            shape=(1,),
            lower_bound=lower_bound,
            upper_bound=upper_bound,
            dtype=dtype)

    def _get_observation(self):
        if self._train_or_test == "test":
            self._cmd = global_values.global_userDebugParams.readValue("setBodyHeight", 0.6)
        elif self._num_update % 10000 == 0:
            self._cmd = np.random.uniform(-1, 1, 1)

        if isinstance(self._cmd, (float, int)):
            self._cmd = np.array([self._cmd])

        self._num_update += 1
        return self._cmd


class TestTask(BaseTask):
    def __init__(self, train_or_test: str = "train"):
        assert train_or_test in ["train", "test"]
        self._train_or_test = train_or_test
        self._obs_cmd: CommandBodyHeight
        super(TestTask, self).__init__()
        self._bodyH_lenth = BODY_HEIGHT_MAX - BODY_HEIGHT_MIN

    def _build_observation_sensors(self):
        sensor_motor_angle = MotorAngleSensor(num_motors=12)
        sensor_motor_vel = MotorVelocitySensor(num_motors=12)
        sensor_IMU = IMUSensor()
        self._observation_sensors.append(HistoricSensorWrapper([sensor_motor_angle, sensor_motor_vel], 3))
        self._observation_sensors.append(sensor_IMU)

        # command observation
        self._obs_cmd = CommandBodyHeight(self._train_or_test)
        self._observation_sensors.append(self._obs_cmd)

    def done(self):
        return False

    def reward(self):
        base_height_reward = self._calc_reward_base_height()
        base_rotation_reward = self._calc_reward_base_rotation()
        base_rotVel_reward = self._calc_reward_base_rotVelocity()

        breakPosVelBound_reward = self._calc_reward_breakPosVelBound()

        reward = 2 * base_height_reward + 0.5 * base_rotation_reward + 0.1 * base_rotVel_reward + breakPosVelBound_reward

        return reward

    def _calc_reward_base_height(self):
        base_height_curr = self._env.robot.ReadBaseHeight()
        base_height_desire = 0.025 + BODY_HEIGHT_MIN + self._bodyH_lenth * (1 - np.abs(self._obs_cmd.read_current_obs()[0]))

        base_height_err = np.abs(base_height_desire - base_height_curr)

        base_height_reward = np.exp(-20 * base_height_err)

        return base_height_reward

    def _calc_reward_base_rotation(self):
        base_rotation_curr = self._env.robot.GetRawBaseOrientation()
        base_rotation_desire = np.array([0, 0, 0, 1])

        base_rotation_diff = TransOfQuaternionsRef2World(base_rotation_curr, base_rotation_desire)
        _, base_rotation_diff_angle = Quaternion2AxisAngle(base_rotation_diff)
        base_rotation_diff_angle = NormalizeAngle2Pi(base_rotation_diff_angle)
        base_rotation_err = np.square(base_rotation_diff_angle)

        base_rotation_reward = np.exp(-10 * base_rotation_err)

        return base_rotation_reward

    def _calc_reward_base_rotVelocity(self):
        base_rotVel_curr = self._env.robot.GetRawBaseRPY_Rate()
        base_rotVel_desire = np.array([0, 0, 0])

        base_rotVel_diff = base_rotVel_desire - base_rotVel_curr
        base_rotVel_err = np.dot(base_rotVel_diff, base_rotVel_diff)

        base_rotVel_reward = np.exp(-0.2 * base_rotVel_err)

        return base_rotVel_reward

    def _calc_reward_breakPosVelBound(self):
        pos_over = 0.0
        vel_over = 0.0
        motor_pos_curr = self._env.robot.GetRawMotorAngles()
        motor_vel_curr = self._env.robot.GetRawMotorVelocities()

        motor_pos_lowerBound = self._env.robot.GetJointAngleLowerBound()
        motor_pos_upperBound = self._env.robot.GetJointAngleUpperBound()
        motor_vel_lowerBound = self._env.robot.GetJointVelocityLowerBound()
        motor_vel_upperBound = self._env.robot.GetJointVelocityUpperBound()

        motor_pos_lowOver = motor_pos_lowerBound - motor_pos_curr
        motor_pos_upOver = motor_pos_curr - motor_pos_upperBound
        motor_vel_lowOver = motor_vel_lowerBound - motor_vel_curr
        motor_vel_upOver = motor_vel_curr - motor_vel_upperBound

        for elem in (motor_pos_lowOver, motor_pos_upOver):
            pos_over += np.sum(elem[elem > 0])

        for elem in (motor_vel_lowOver, motor_vel_upOver):
            vel_over += np.sum(elem[elem > 0])

        reward_pos = np.exp(-5 * pos_over) - 1
        reward_vel = np.exp(-0.1 * vel_over) - 1

        reward = 1 * reward_pos + 1 * reward_vel

        return reward

        # if reward != 0:
        #     return reward
        # else:
        #     return 0
