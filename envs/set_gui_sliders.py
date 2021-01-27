from utilities import global_values


def set_gui_sliders(pybullet_client):
    slides = global_values.global_userDebugParams
    slides.setPbClient(pybullet_client)
    slides.AddSlider("terminate_curve_process", 1, 0, 0)
    slides.AddSlider("setBodyHeight", -1, 1, 0.6)
    slides.AddSlider("kp", 0, 1, 0.353)
    slides.AddSlider("kd", 0, 1, 0.758)
