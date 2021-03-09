from tasks.base_task import BaseTask
from sensors.robot_sensors import *
from sensors.sensor_wrappers import *


class TestTask(BaseTask):
    def __init__(self):
        super(TestTask, self).__init__()

    def _build_observation_sensors(self):
        sensor_motor_angle = MotorAngleSensor(num_motors=12)
        sensor_motor_vel = MotorVelocitySensor(num_motors=12)
        sensor_IMU = IMUSensor()
        self._observation_sensors.append(HistoricSensorWrapper([sensor_motor_angle, sensor_motor_vel], 3))
        self._observation_sensors.append(sensor_IMU)

    def done(self):
        return False

    def reward(self):
        return 0
