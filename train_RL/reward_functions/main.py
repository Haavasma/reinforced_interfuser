from typing import Tuple
from gym_env.env import WorldState


def reward_function(state: WorldState) -> Tuple[float, bool]:
    return 0, False
