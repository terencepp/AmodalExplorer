"""Camera-relative steps for the generation demo (PyTorch3D convention)."""
import copy
import math

import torch


def step_camera(camera, yaw_degrees, translation, scale):
    """Yaw right, translating in the old camera's OpenCV frame: right, down, forward."""
    camera = copy.deepcopy(camera)
    R = camera.R[0]
    center = -camera.T[0] @ R.T
    move = R.new_tensor(translation) * R.new_tensor([-1, -1, 1]) * scale
    center = center + move @ R.T
    a = math.radians(yaw_degrees)
    c, s = math.cos(a), math.sin(a)
    rotation = R.new_tensor([[c, 0, -s], [0, 1, 0], [s, 0, c]])
    R = R @ rotation
    camera.R = R[None]
    camera.T = (-center @ R)[None]
    return camera
