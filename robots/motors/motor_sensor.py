import numpy as np


class MotorSensor:
    def __init__(self, num):
        self._motor_position = None
        self._motor_velocity = None
        self._motor_torque = None
        self._numSensor = num

    def _check_value(self, value, name: str):
        if not isinstance(value, np.ndarray):
            raise ValueError(f"The type of Data given to {name} is not np.ndarray!")
        elif len(value) != self._numSensor:
            raise ValueError(f"The length of Data given to {name} is wrong!")

    @property
    def motorPosition(self) -> np.ndarray:
        if self._motor_position is None:
            raise ValueError("There is no position data writen to the motor sensor!")
        return self._motor_position

    @motorPosition.setter
    def motorPosition(self, value: np.ndarray):
        self._check_value(value, "motorPosition")
        self._motor_position = value

    @property
    def motorVelocity(self) -> np.ndarray:
        if self._motor_velocity is None:
            raise ValueError("There is no velocity data writen to the motor sensor!")
        return self._motor_velocity

    @motorVelocity.setter
    def motorVelocity(self, value: np.ndarray):
        self._check_value(value, "motorVelocity")
        self._motor_velocity = value

    @property
    def motorTorque(self) -> np.ndarray:
        if self._motor_torque is None:
            raise ValueError("There is no torque data writen to the motor sensor!")
        return self._motor_torque

    @motorTorque.setter
    def motorTorque(self, value: np.ndarray):
        self._check_value(value, "motorTorque")
        self._motor_torque = value
