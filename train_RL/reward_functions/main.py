import math
from dataclasses import dataclass
from typing import Any, List, Tuple
from PIL.PngImagePlugin import o8

from episode_manager.agent_handler.models import Location, Transform
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
    # Speed reward
    dist_to_hazard = closest_hazard(state)
    speed_limit = 6.0
    speed = state.ego_vehicle_state.speed

    if len(state.ego_vehicle_state.privileged.collision_history.items()) > 0:
        print("COLLISION")
        return -100, True

    if state.scenario_state.done:
        return 1, True

    desired_speed = calculate_desired_speed(speed_limit, dist_to_hazard)
    speed_reward = calculate_speed_reward(speed, desired_speed)

    # print("\n------------------------------------")
    # print(f"DESIRED SPEED: {desired_speed}")
    # print(f"SPEED: {speed}")
    # print(f"REWARD: {speed_reward}")
    # print("------------------------------------\n")
    if speed_reward <= 0:
        return speed_reward, False

    # Angle reward
    ego_vehicle_location = state.ego_vehicle_state.privileged.transform.location
    waypoints = state.scenario_state.global_plan_world_coord

    closest_waypoint_index, distance_diff = _get_closest_waypoint(
        ego_vehicle_location, waypoints
    )

    wp_0, wp_1 = (
        waypoints[closest_waypoint_index][0].location,
        waypoints[(closest_waypoint_index + 1) % len(waypoints)][0].location,
    )

    angle = _calculate_angle(wp_0, wp_1)

    angle_diff = _calculate_radian_difference(
        angle, math.radians(state.ego_vehicle_state.compass - 90)
    )
    angle_reward = _calculate_angle_reward(angle_diff)
    if angle_reward < 0:
        return -1, False

    # Distance reward
    distance_reward = _calculate_distance_reward(distance_diff)
    if distance_reward < 0:
        return -50, True

    result = Reward(speed_reward, angle_reward, distance_reward).calculate_reward()

    return result, False


def _calculate_distance_reward(distance: float) -> float:
    max = 2.0

    if distance > max:
        return -1

    return 1 - distance / max


def _calculate_angle_reward(diff: float) -> float:
    # Angle reward
    max_diff = math.pi / 2

    if abs(diff) > max_diff:
        return -1.0

    reward = (max_diff - diff) / max_diff

    return reward


def _calculate_angle(point_1: Location, point_2: Location) -> float:
    dx = point_2.x - point_1.x
    dy = point_2.y - point_1.y

    return math.atan2(dy, dx)


def _calculate_radian_difference(angle_1: float, angle_2: float) -> float:
    diff = angle_1 - angle_2
    return abs(math.atan2(math.sin(diff), math.cos(diff)))


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


def _get_closest_waypoint(
    ego_vehicle_location: Location,
    waypoints: List[Tuple[Transform, Any]],
) -> Tuple[int, float]:
    closest_index = 0
    closest_distance = 1000000000.0
    for index, waypoint in enumerate(waypoints):
        waypoint_location = waypoint[0].location

        distance = vector_distance(
            waypoint_location.x,
            waypoint_location.y,
            ego_vehicle_location.x,
            ego_vehicle_location.y,
        )
        if distance < closest_distance:
            closest_index = index
            closest_distance = distance

    return closest_index, closest_distance


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
