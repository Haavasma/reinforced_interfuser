from dataclasses import dataclass
from typing import Any, List, Tuple
from episode_manager.episode_manager import Location
from gym_env.env import WorldState


@dataclass
class Reward:
    speed_reward: float
    angle_reward: float
    distance_reward: float

    def calculate_reward(self) -> float:
        assert self.speed_reward >= 0 and self.speed_reward <= 1
        assert self.angle_reward >= 0 and self.angle_reward <= 1
        assert self.distance_reward >= 0 and self.distance_reward <= 1

        return (self.speed_reward + self.angle_reward + self.distance_reward) / 3


def reward_function(state: WorldState) -> Tuple[float, bool]:
    # TODO: Implement this reward function

    # Calculate desired speed
    dist_to_hazard = closest_hazard(state)
    speed_limit = state.ego_vehicle_state.privileged.speed_limit
    speed = state.ego_vehicle_state.speed

    desired_speed = calculate_desired_speed(speed_limit, dist_to_hazard)
    speed_reward = calculate_speed_reward(speed, desired_speed)

    # TODO: Calculate angle reward

    ego_vehicle_location = state.ego_vehicle_state.privileged.ego_vehicle_location
    waypoints = state.scenario_state.global_plan

    closest_waypoint_index = get_closest_waypoint_index(ego_vehicle_location, waypoints)

    # Calculate angle between ego vehicle and closest waypoint
    angle_reward = 1

    # TODO: Calculate distance reward
    distance_reward = 1

    return Reward(speed_reward, angle_reward, distance_reward).calculate_reward(), False


def closest_hazard(state: WorldState) -> float:
    max_dist = 1000000000
    closest = max_dist

    for distance in [
        state.ego_vehicle_state.privileged.dist_to_vehicle,
        state.ego_vehicle_state.privileged.dist_to_pedestrian,
        state.ego_vehicle_state.privileged.dist_to_traffic_light,
    ]:
        if distance >= 0 and distance < closest:
            closest = distance

    return closest if closest != max_dist else -1


def get_closest_waypoint_index(
    ego_vehicle_location: Location, waypoints: List[Any]
) -> int:
    closest = 0
    for index, waypoint in enumerate(waypoints):
        if (
            vector_distance(
                waypoint.x,
                waypoint.y,
                ego_vehicle_location.x,
                ego_vehicle_location.y,
            )
            < closest
        ):
            closest = index

    return closest


def vector_distance(x1: float, y1: float, x2: float, y2: float) -> float:
    return ((x1 - x2) ** 2 + (y1 - y2) ** 2) ** 0.5


def calculate_desired_speed(speed_limit: float, distance: float) -> float:
    distance_cutoff = 20
    clipped_dist = min(distance, distance_cutoff)
    offset = 10
    desired_speed = 0

    # The distance is -1 if there is no hazard
    if distance < 0:
        desired_speed = speed_limit
    else:
        desired_speed = min(
            speed_limit, speed_limit * (max(clipped_dist - offset, 0) / offset)
        )

    return desired_speed


def calculate_speed_reward(speed: float, desired_speed: float) -> float:
    speed_diff = abs(speed - desired_speed)
    max_speed_diff = 3.0

    if speed > desired_speed:
        max_speed_diff = 0.5

    if desired_speed <= 0.001:
        max_speed_diff = 0.01

    if speed <= 0.5 and desired_speed > 2.0:
        return -1.0

    if speed_diff > max_speed_diff:
        return 0.0

    return (max_speed_diff - speed_diff) / max_speed_diff
