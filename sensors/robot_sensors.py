from sensors.basesensor import BaseSensor
import numpy as np
from typing import Text, Any, Union, Iterable, Type


class MotorAngleSensor(BaseSensor):
    def __init__(self,
                 num_motors: int,
                 delay_without_noise: bool = False,
                 noisy_reading: bool = True,
                 noise_stdev: float = 0.0,
                 name: Text = "MotorAngle",
                 lower_bound: Union[float, Iterable[float]] = -np.pi * 2,
                 upper_bound: Union[float, Iterable[float]] = np.pi * 2,
                 dtype: Type[Any] = np.float64):
        self._num_motors = num_motors
        self._delay_without_noise = delay_without_noise
        self._noisy_reading = noisy_reading
        self._noise_stdev = noise_stdev

        if self._delay_without_noise and self._noisy_reading:
            raise ValueError("delay_without_noise and noisy_reading cannot be enables together!")

        super(MotorAngleSensor, self).__init__(
            name=name,
            shape=(self._num_motors,),
            lower_bound=lower_bound,
            upper_bound=upper_bound,
            dtype=dtype)

    def _get_observation(self):
        if self._robot is None:
            raise Exception("The robot has not been setting yet!")

        if not self._noisy_reading:
            if self._delay_without_noise:
                motor_angles = self._robot.GetDelayedMotorAngles()
            else:
                motor_angles = self._robot.GetRawMotorAngles()
        else:
            motor_angles = self._robot.GetNoisyMotorAngles(self._noise_stdev)

        return motor_angles


class MotorVelocitySensor(BaseSensor):
    def __init__(self,
                 num_motors: int,
                 delay_without_noise: bool = False,
                 noisy_reading: bool = True,
                 noise_stdev: float = 0.0,
                 name: Text = "MotorVelocity",
                 lower_bound: Union[float, Iterable[float]] = -24.0,
                 upper_bound: Union[float, Iterable[float]] = 24.0,
                 dtype: Type[Any] = np.float64):
        self._num_motors = num_motors
        self._delay_without_noise = delay_without_noise
        self._noisy_reading = noisy_reading
        self._noise_stdev = noise_stdev

        if self._delay_without_noise and self._noisy_reading:
            raise ValueError("delay_without_noise and noisy_reading cannot be enables together!")

        super(MotorVelocitySensor, self).__init__(
            name=name,
            shape=(self._num_motors,),
            lower_bound=lower_bound,
            upper_bound=upper_bound,
            dtype=dtype)

    def _get_observation(self):
        if self._robot is None:
            raise Exception("The robot has not been setting yet!")

        if not self._noisy_reading:
            if self._delay_without_noise:
                motor_velocities = self._robot.GetDelayedMotorVelocities()
            else:
                motor_velocities = self._robot.GetRawMotorVelocities()
        else:
            motor_velocities = self._robot.GetNoisyMotorVelocities(self._noise_stdev)

        return motor_velocities


class IMUSensor(BaseSensor):
    def __init__(self,
                 channels: Iterable[Text] = None,
                 delay_without_noise: bool = False,
                 noisy_reading: bool = True,
                 noise_stdev: Iterable[float] = None,
                 name: Text = "IMU",
                 lower_bound: Union[float, Iterable[float]] = None,
                 upper_bound: Union[float, Iterable[float]] = None,
                 dtype: Type[Any] = np.float64):
        self._channels = channels if channels else ["R", "P", "dR", "dP"]
        self._num_channels = len(self._channels)
        self._delay_without_noise = delay_without_noise
        self._noisy_reading = noisy_reading
        self._noise_stdev = [0.0, 0.0] if noise_stdev is None else noise_stdev

        if self._delay_without_noise and self._noisy_reading:
            raise ValueError("delay_without_noise and noisy_reading cannot be enables together!")

        if lower_bound is None and upper_bound is None:
            lower_bound = []
            upper_bound = []
            for channel in self._channels:
                if channel in ["R", "P", "Y"]:
                    lower_bound.append(-2.0 * np.pi)
                    upper_bound.append(2.0 * np.pi)
                elif channel in ["Rcos", "Rsin", "Pcos", "Psin", "Ycos", "Ysin"]:
                    lower_bound.append(-1.)
                    upper_bound.append(1.)
                elif channel in ["dR", "dP", "dY"]:
                    lower_bound.append(-2000.0 * np.pi)
                    upper_bound.append(2000 * np.pi)
        elif lower_bound is None:
            lower_bound = []
            for channel in self._channels:
                if channel in ["R", "P", "Y"]:
                    lower_bound.append(-2.0 * np.pi)
                elif channel in ["Rcos", "Rsin", "Pcos", "Psin", "Ycos", "Ysin"]:
                    lower_bound.append(-1.)
                elif channel in ["dR", "dP", "dY"]:
                    lower_bound.append(-2000.0 * np.pi)
        elif upper_bound is None:
            upper_bound = []
            for channel in self._channels:
                if channel in ["R", "P", "Y"]:
                    upper_bound.append(2.0 * np.pi)
                elif channel in ["Rcos", "Rsin", "Pcos", "Psin", "Ycos", "Ysin"]:
                    upper_bound.append(1.)
                elif channel in ["dR", "dP", "dY"]:
                    upper_bound.append(2000 * np.pi)

        super(IMUSensor, self).__init__(
            name=name,
            shape=(self._num_channels,),
            lower_bound=lower_bound,
            upper_bound=upper_bound,
            dtype=dtype)

    def _get_observation(self):
        if self._robot is None:
            raise Exception("The robot has not been setting yet!")

        if not self._noisy_reading:
            if self._delay_without_noise:
                rpy = self._robot.GetDelayedBaseRPY()
                drpy = self._robot.GetDelayedBaseRPY_Rate()
            else:
                rpy = self._robot.GetRawBaseRPY()
                drpy = self._robot.GetRawBaseRPY_Rate()
        else:
            rpy = self._robot.GetNoisyBaseRPY(self._noise_stdev[0])
            drpy = self._robot.GetNoisyBaseRPY_Rate(self._noise_stdev[1])

        assert len(rpy) == 3, rpy
        assert len(drpy) == 3, drpy

        obs = np.zeros(self._num_channels)
        for i, channel in enumerate(self._channels):
            if channel == "R":
                obs[i] = rpy[0]
            if channel == "Rcos":
                obs[i] = np.cos(rpy[0])
            if channel == "Rsin":
                obs[i] = np.sin(rpy[0])

            if channel == "P":
                obs[i] = rpy[1]
            if channel == "Pcos":
                obs[i] = np.cos(rpy[1])
            if channel == "Psin":
                obs[i] = np.sin(rpy[1])

            if channel == "Y":
                obs[i] = rpy[2]
            if channel == "Ycos":
                obs[i] = np.cos(rpy[2])
            if channel == "Ysin":
                obs[i] = np.sin(rpy[2])

            if channel == "dR":
                obs[i] = drpy[0]
            if channel == "dP":
                obs[i] = drpy[1]
            if channel == "dY":
                obs[i] = drpy[2]

        return obs
