from sensors.basesensor import BaseSensor
from typing import Text, List
import enum
import numpy as np
from collections import deque


class SensorWrapper(BaseSensor):
    def __init__(self, wrapped_sensors: List[BaseSensor], name: Text = "WrappedSensors"):
        self._wrapped_sensors = wrapped_sensors
        lower_bound = np.concatenate([sensor.get_lower_bound() for sensor in self._wrapped_sensors])
        upper_bound = np.concatenate([sensor.get_upper_bound() for sensor in self._wrapped_sensors])
        shape = lower_bound.shape

        super(SensorWrapper, self).__init__(
            name=name,
            shape=shape,
            lower_bound=lower_bound,
            upper_bound=upper_bound,
            dtype=np.float64)

    def set_robot(self, robot):
        super(SensorWrapper, self).set_robot(robot)
        for sensor in self._wrapped_sensors:
            sensor.set_robot(robot)

    def on_reset(self):
        super(SensorWrapper, self).on_reset()

        for sensor in self._wrapped_sensors:
            sensor.on_reset()

    def on_step(self):
        super(SensorWrapper, self).on_step()

        for sensor in self._wrapped_sensors:
            sensor.on_step()

    def _get_observation(self):
        return np.concatenate([sensor.get_observation() for sensor in self._wrapped_sensors])


class FlattenMode(enum.Enum):
    SINGLE_ROW = 1
    MULTI_ROW = 2
    MULTI_COLUMN = 3


class HistoricSensorWrapper(BaseSensor):
    def __init__(self,
                 wrapped_sensors: List[BaseSensor],
                 num_history: int,
                 name: Text = "HistoricRobotSensors",
                 flatten_mode: FlattenMode = FlattenMode.SINGLE_ROW):
        self._wrapped_sensors = wrapped_sensors
        self._num_sensors = len(self._wrapped_sensors)
        self._num_history = num_history
        self._flatten_mode = flatten_mode

        lower_list = []
        upper_list = []

        for i in range(self._num_sensors):
            lower_list.append(np.tile(self._wrapped_sensors[i].get_lower_bound(), (self._num_history, 1)))
            upper_list.append(np.tile(self._wrapped_sensors[i].get_upper_bound(), (self._num_history, 1)))

        lower_bound = self.Flatten_to(lower_list, self._num_sensors, self._flatten_mode)
        upper_bound = self.Flatten_to(upper_list, self._num_sensors, self._flatten_mode)
        shape = lower_bound.shape

        super(HistoricSensorWrapper, self).__init__(
            name=name,
            shape=shape,
            lower_bound=lower_bound,
            upper_bound=upper_bound,
            dtype=np.float64)

        self._history_buffers = [deque(maxlen=self._num_history) for _ in range(self._num_sensors)]

    @staticmethod
    def Flatten_to(list_of_array: List, length_of_list: int, mode: FlattenMode):
        flatten_array = None
        if mode == FlattenMode.SINGLE_ROW:
            flatten_list = [list_of_array[i].flatten(order='C') for i in range(length_of_list)]
            flatten_array = np.concatenate(flatten_list)
        else:
            concatenated_array = np.concatenate(list_of_array, axis=1)
            if mode == FlattenMode.MULTI_ROW:
                flatten_array = concatenated_array.flatten(order='C')
            elif mode == FlattenMode.MULTI_COLUMN:
                flatten_array = concatenated_array.flatten(order='F')

        return flatten_array

    def set_robot(self, robot):
        super(HistoricSensorWrapper, self).set_robot(robot)
        for sensor in self._wrapped_sensors:
            sensor.set_robot(robot)

    def on_reset(self):
        super(HistoricSensorWrapper, self).on_reset()

        for sensor in self._wrapped_sensors:
            sensor.on_reset()

        for buf in self._history_buffers:
            buf.clear()

        for _ in range(self._num_history):
            self.on_step()

    def on_step(self):
        super(HistoricSensorWrapper, self).on_step()
        for i, sensor in enumerate(self._wrapped_sensors):
            sensor.on_step()
            self._history_buffers[i].appendleft(sensor.get_observation())

    def _get_observation(self):
        list_of_obs_array = [np.array(buf) for buf in self._history_buffers]
        return self.Flatten_to(list_of_obs_array, self._num_sensors, self._flatten_mode)


# a = np.array([[1.1, 2.1], [1.2, 2.2], [1.3, 2.3]])
# b = np.array([[3.1, 4.1, 5.1], [3.2, 4.2, 5.2], [3.3, 4.3, 5.3]])
# c = np.array([[6.1, 7.1], [6.2, 7.2], [6.3, 7.3]])
# d = [a, b, c]
# e = HistoricSensorWrapper.Flatten_to(d, 3, FlattenMode.SINGLE_ROW)
# f = HistoricSensorWrapper.Flatten_to(d, 3, FlattenMode.MULTI_ROW)
# g = HistoricSensorWrapper.Flatten_to(d, 3, FlattenMode.MULTI_COLUMN)
# print(a)
# print(b)
# print(c)
# print(f"Single Row: {e}")
# print(f"Multi Row: {f}")
# print(f"Multi Column: {g}")


