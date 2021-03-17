def is_contact_fall(env):
    pybullet_client = env.getClient()
    robot = env.robot
    foot_links = robot.GetFootLinkIDs()
    ground = env.ground

    contact_fall = False

    if env.env_step_counter > 0:
        robot_ground_contact_points = pybullet_client.getContactPoints(bodyA=robot.getRobotID(), bodyB=ground)
        robot_self_contact_points = pybullet_client.getContactPoints(bodyA=robot.getRobotID(), bodyB=robot.getRobotID())
        if robot_ground_contact_points:
            for contact_point in robot_ground_contact_points:
                if contact_point[3] not in foot_links:
                    print("Ground contact!")
                    contact_fall = True
                    break

        if robot_self_contact_points:
            print("Self contact!")
            contact_fall = True

    return contact_fall


class ContactDetection:
    def __init__(self, env):
        self._env = env
        self._pybullet_client = self._env.getClient()
        self._robotID = None
        self._groundID = None
        self._foot_links = None

    def reset(self):
        self._robotID = self._env.robot.getRobotID()
        self._groundID = self._env.ground
        self._foot_links = self._env.robot.GetFootLinkIDs()

    def is_contact_fall(self):
        contact_fall = False

        if self._env.env_step_counter > 0:
            robot_ground_contact_points = self._pybullet_client.getContactPoints(self._robotID, bodyB=self._groundID)
            robot_self_contact_points = self._pybullet_client.getContactPoints(bodyA=self._robotID, bodyB=self._robotID)
            if robot_ground_contact_points:
                for contact_point in robot_ground_contact_points:
                    if contact_point[3] not in self._foot_links:
                        print("Ground contact!")
                        contact_fall = True
                        break

            if robot_self_contact_points:
                print("Self contact!")
                contact_fall = True

        return contact_fall
