from reward_functions.main import (
    calculate_distance_reward,
    get_closest_waypoint,
)
from typing import Tuple

from gym_env.env import WorldState, ScenarioData


def evaluation_reward(state: WorldState, data: ScenarioData) -> Tuple[float, bool]:
    # Speed reward
    if state.done:
        return 1, True

    ego_vehicle_location = state.ego_vehicle_state.privileged.transform.location
    waypoints = data.global_plan_world_coord_privileged

    _, distance_diff = get_closest_waypoint(ego_vehicle_location, waypoints)

    # Distance reward
    distance_reward = calculate_distance_reward(distance_diff, max=3.0)
    if distance_reward < 0:
        return -10, True

    return 0, False
