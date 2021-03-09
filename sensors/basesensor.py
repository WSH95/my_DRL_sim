from typing import Text, Tuple, Union, Iterable
import numpy as np


def AddSensorNoise(sensor_values, noise_stdev):
    if noise_stdev <= 0:
        return sensor_values
    noisyValue = sensor_values + np.random.normal(scale=noise_stdev, size=sensor_values.shape)
    return noisyValue


class BaseSensor(object):
    def __init__(self,
                 name: Text,
                 shape: Tuple[int, ...],
                 lower_bound: Union[float, Iterable[float]],
                 upper_bound: Union[float, Iterable[float]],
                 dtype=np.float64) -> None:
        self._name = name
        self._shape = shape
        self._dtype = dtype

        self._robot = None

        if isinstance(lower_bound, (float, int)):
            self._lower_bound = np.full(shape, lower_bound, dtype=self._dtype)
        else:
            self._lower_bound = np.array(lower_bound)

        if isinstance(upper_bound, (float, int)):
            self._upper_bound = np.full(shape, upper_bound, dtype=self._dtype)
        else:
            self._upper_bound = np.array(upper_bound)

    def get_shape(self) -> Tuple[int, ...]:
        return self._shape

    def get_dtype(self):
        return self._dtype

    def get_lower_bound(self) -> Iterable[float]:
        return self._lower_bound

    def get_upper_bound(self) -> Iterable[float]:
        return self._upper_bound

    def set_robot(self, robot):
        self._robot = robot

    def _get_observation(self):
        raise NotImplementedError()

    def get_observation(self) -> np.ndarray:
        return np.array(self._get_observation(), dtype=self._dtype)

    def on_reset(self):
        pass

    def on_step(self):
        pass

    def on_terminate(self):
        pass
