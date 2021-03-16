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
