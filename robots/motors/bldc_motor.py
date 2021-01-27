import numpy as np
from robots.legged_robots.robot_config import HybridCommands
from robots.motors.motor_sensor import MotorSensor

BLDC_MOTOR_TORQUE_LIMIT = 30


class BLDC_motor:
    def __init__(self, num_motors: int):
        self._num_motors = num_motors
        self.motor_sensor = MotorSensor(self._num_motors)
        self.torque_limit = BLDC_MOTOR_TORQUE_LIMIT

    def hybridCmd_to_torque(self, hybrid_cmd: HybridCommands):
        torque_obs = hybrid_cmd.torque_estimate + hybrid_cmd.kp * (
                    hybrid_cmd.position_desired - self.motor_sensor.motorPosition) + hybrid_cmd.kd * (
                                 hybrid_cmd.velocity_desired - self.motor_sensor.motorVelocity)

        return np.clip(torque_obs, -1.0 * self.torque_limit, self.torque_limit)

