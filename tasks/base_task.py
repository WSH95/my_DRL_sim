class BaseTask:
    def __init__(self):
        self._observation_sensors = []
        self._build_observation_sensors()
        self._env = None

    def _build_observation_sensors(self):
        raise NotImplementedError()

    def get_observation_sensors(self):
        return self._observation_sensors

    def reset(self, env):
        self.set_env(env)

    def set_env(self, env):
        self._env = env

    def done(self):
        raise NotImplementedError()

    def reward(self):
        raise NotImplementedError()
