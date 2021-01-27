import attr


@attr.s
class SimulationParameters(object):
    enable_rendering = attr.ib(type=bool, default=True)  # whether to show the interface. COV_ENABLE_RENDERING
    enable_rendering_gui = attr.ib(type=bool, default=True)  # COV_ENABLE_GUI
    camera_distance = attr.ib(type=float, default=1.0)
    camera_yaw = attr.ib(type=float, default=0)
    camera_pitch = attr.ib(type=float, default=-30)
    render_width = attr.ib(type=int, default=480)
    render_height = attr.ib(type=int, default=360)
    egl_rendering = attr.ib(type=bool, default=False)  # enable hardware accelerated OpenGL rendering without a X11 context
    enable_hard_reset = attr.ib(type=bool, default=False)
    reset_time = attr.ib(type=float, default=-1)
    time_step = attr.ib(type=float, default=1.0 / 240.0)
