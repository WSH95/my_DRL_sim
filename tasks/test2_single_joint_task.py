from tasks.base_task import BaseTask
from sensors.robot_sensors import *
from sensors.sensor_wrappers import *
from tasks.test_task import CommandBodyHeight


class TestSingleJointTask(BaseTask):
    def __init__(self, train_or_test: str = "train"):
        assert train_or_test in ["train", "test"]
        self._train_or_test = train_or_test
        self._obs_cmd: CommandBodyHeight
        super(TestSingleJointTask, self).__init__()

    def _build_observation_sensors(self):
        sensor_motor_angle = MotorAngleSensor(num_motors=1)
        sensor_motor_vel = MotorVelocitySensor(num_motors=1)
        self._observation_sensors.append(HistoricSensorWrapper([sensor_motor_angle, sensor_motor_vel], 3))

        # command observation
        self._obs_cmd = CommandBodyHeight(self._train_or_test)
        self._observation_sensors.append(self._obs_cmd)

    def done(self):
        return False

    def reward(self):
        joint_pos_reward = self._calc_reward_pos()

        reward = 2 * joint_pos_reward

        return reward

    def _calc_reward_pos(self):
        pos_curr = self._env.robot.GetRawMotorAngles()[0]
        pos_desire = self._obs_cmd.read_current_obs()[0] * 2 * np.pi
        pos_err = np.abs(pos_desire - pos_curr)

        motor_pos_lowerBound = self._env.robot.GetJointAngleLowerBound()[0]
        motor_pos_upperBound = self._env.robot.GetJointAngleUpperBound()[0]

        pos_over = 0

        if pos_curr < motor_pos_lowerBound:
            pos_over = motor_pos_lowerBound - pos_curr
        elif pos_curr > motor_pos_upperBound:
            pos_over = pos_curr - motor_pos_upperBound

        pos_reward = np.exp(-200 * pos_err) + (np.exp(-5 * pos_over) - 1)

        return pos_reward
