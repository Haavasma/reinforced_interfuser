from typing import Tuple
from gym_env.env import VisionModule


class InterFuserVisionModule(VisionModule):
    output_shape: Tuple[int] = (256,)

    def __call__(self):

        return
