class UserDebugParams:
    def __init__(self, pybullet_client=None):
        self._pybullet_client = pybullet_client
        self._paramIds = {}
        if self._pybullet_client is not None:
            self._is_GUI = True if (self._pybullet_client.getConnectionInfo()['connectionMethod'] == 1) else False

        # self.AddSlider("start/terminate curve process", 1, 0, 0)

    def setPbClient(self, pybullet_client):
        self._pybullet_client = pybullet_client
        self._is_GUI = True if (self._pybullet_client.getConnectionInfo()['connectionMethod'] == 1) else False

    def AddSlider(self, paramName: str, rangeMin: float, rangeMax: float, startValue: float):
        if self._is_GUI:
            self._paramIds[paramName] = self._pybullet_client.addUserDebugParameter(paramName, rangeMin, rangeMax,
                                                                                    startValue)
        else:
            return

    def readValue(self, paramName: str, default_value: float = None):
        if default_value is not None:
            return default_value

        try:
            paramId = self._paramIds[paramName]
        except KeyError:
            raise KeyError(f'There is no button named {paramName}')
        else:
            return self._pybullet_client.readUserDebugParameter(paramId)

    def getID(self, paramName):
        try:
            return True, self._paramIds[paramName]
        except KeyError:
            return False, -1
