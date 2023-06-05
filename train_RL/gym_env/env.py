from dataclasses import dataclass
import random
import time
from typing import Any, Callable, List, Optional, Protocol, Set, Tuple, TypedDict
import typing
from PIL import Image
from episode_manager.data import TrafficType, TrainingType
from episode_manager.renderer import WorldStateRenderer, generate_pygame_surface

from collections import deque
import gymnasium as gym
import numpy as np
import pygame
from episode_manager import EpisodeManager
from episode_manager.episode_manager import Action, WorldState, ScenarioData
from gymnasium.spaces import Box, Dict, Discrete
from gymnasium.utils import seeding
from srunner.tools.route_parser import RoadOption

from gym_env.route_planner import RoutePlanner, find_relative_target_waypoint
from gym_env.vision import VisionModule
from vision_modules.interfuser import create_carla_rgb_transform


class CarlaEnvironmentConfiguration(TypedDict):
    continuous_speed_range: Tuple[float, float]
    continuous_steering_range: Tuple[float, float]
    speed_goal_actions: List[float]
    steering_actions: List[float]
    discrete_actions: bool
    towns: Optional[List[str]]
    town_change_frequency: Optional[int]
    concat_images: bool
    traffic_type: TrafficType
    image_resize: Tuple[int, int]


def default_config() -> CarlaEnvironmentConfiguration:
    return {
        "continuous_speed_range": (-2.0, 10.0),
        "continuous_steering_range": (-0.3, 0.3),
        "speed_goal_actions": [],
        "steering_actions": [],
        "discrete_actions": True,
        "towns": ["Town01", "Town03", "Town04", "Town06"],
        "town_change_frequency": 10,
        "concat_images": False,
        "traffic_type": TrafficType.SCENARIO,
        "image_resize": (224, 224),
    }


class PIDController(Protocol):
    def __call__(
        self, target_vel: float, current_vel: float
    ) -> Tuple[float, float, bool]:
        return (0.0, 0.0, False)


@dataclass
class SpeedController(PIDController):
    kp: float = 0.5
    ki: float = 0.1
    kd: float = 0.2

    def __post_init__(self):
        self.last_error = 0.0
        self.integral = 0.0

    def __call__(
        self, target_vel: float, current_vel: float
    ) -> Tuple[float, float, bool]:
        error = target_vel - current_vel
        derivative = error - self.last_error
        self.integral += error
        self.last_error = error

        throttle = self.kp * error + self.ki * self.integral + self.kd * derivative
        brake = 0.0

        if throttle > 1.0:
            throttle = 1.0
        elif throttle < -1.0:
            throttle = -1.0

        if throttle < 0.0:
            brake = -throttle
            throttle = 0.0

        reverse = False
        if current_vel < 0.0 and target_vel < current_vel:
            reverse = True

        return throttle, brake, reverse


class Controller(object):
    def __init__(self, K_P=5.0, K_I=0.5, K_D=1.0, n=20):
        self._K_P = K_P
        self._K_I = K_I
        self._K_D = K_D

        self._window = deque([0 for _ in range(n)], maxlen=n)

    def step(self, error):
        self._window.append(error)

        if len(self._window) >= 2:
            integral = np.mean(self._window)
            derivative = self._window[-1] - self._window[-2]
        else:
            integral = 0.0
            derivative = 0.0

        return self._K_P * error + self._K_I * integral + self._K_D * derivative


class TestSpeedController(PIDController):
    def __init__(
        self,
        default_speed: float = 4.0,
        brake_speed: float = 0.4,
        brake_ratio: float = 1.1,
        clip_delta=0.25,
        clip_throttle=0.75,
    ):
        self._default_speed = default_speed
        self._brake_speed = brake_speed
        self._brake_ratio = brake_ratio
        self._clip_delta = clip_delta
        self._clip_throttle = clip_throttle
        self._controller = Controller()
        return

    def __call__(self, wanted_speed: float, current_speed: float):
        desired_speed = wanted_speed

        brake = (desired_speed < self._brake_speed) or (
            (current_speed / desired_speed) > self._brake_ratio
        )

        brake = (desired_speed < self._brake_speed) or (
            (current_speed / desired_speed) > self._brake_ratio
        )

        delta = np.clip(desired_speed - current_speed, 0.0, self._clip_delta)
        throttle = self._controller.step(delta)
        throttle = np.clip(throttle, 0.0, self._clip_throttle)
        throttle = throttle if not brake else 0.0

        brake_value = 0.0
        if brake:
            brake_value = 1.0

        return (throttle, brake_value, False)


@dataclass
class SteeringController:
    kp: float = 0.0
    ki: float = 0.0
    kd: float = 0.0

    def __call__(self, goal_angle: float, current_angle: float) -> float:
        return 0


@dataclass
class CarlaEnvironment(gym.Env):
    config: CarlaEnvironmentConfiguration
    carla_manager: EpisodeManager
    vision_module: Optional[VisionModule]
    reward_function: Callable[[WorldState, ScenarioData], Tuple[float, bool]]
    speed_controller: PIDController
    render_mode: str = "computer"

    def __post_init__(self):
        """
        Sets up action and observation space based on configurations
        """
        self.time = time.time()
        self._renderer = WorldStateRenderer()
        self._metrics: typing.Dict[str, Any] = {}
        self._n_episodes = 0
        self._steps = 0
        self._town: Optional[str] = None

        if self.config["towns"]:
            self._town = random.choice(self.config["towns"])

        self.amount_of_speed_actions = len(self.config["speed_goal_actions"])
        self.amount_of_steering_actions = len(self.config["steering_actions"])
        self._route_planner: Optional[RoutePlanner] = None
        self.metadata["render_fps"] = 10
        self._reward: Optional[float] = None

        self._rgb_transform = create_carla_rgb_transform(self.config["image_resize"])

        self._prev_obs: Optional[dict] = None

        print("INITIALIZING ENVIRONMENT")

        if self.config["discrete_actions"]:
            self.action_space = Discrete(
                self.amount_of_steering_actions * self.amount_of_speed_actions
            )

        else:
            self.action_space = Box(
                np.array(
                    [
                        self.config["continuous_speed_range"][0],
                        self.config["continuous_steering_range"][0],
                    ]
                ),
                np.array(
                    [
                        self.config["continuous_speed_range"][1],
                        self.config["continuous_steering_range"][1],
                    ]
                ),
            )

        observation_space_dict = (
            self._set_observation_space_without_vision()
            if self.vision_module is None
            else self._set_observation_space_with_vision()
        )

        self.observation_space = Dict(spaces=observation_space_dict)

        return

    @property
    def metrics(self) -> typing.Dict[str, Any]:
        return self._metrics

    def _set_observation_space_without_vision(self) -> dict:
        observation_space_dict = {}
        lidar = self.carla_manager.config.car_config.lidar

        if self.carla_manager.config.car_config.lidar["enabled"]:
            observation_space_dict["lidar"] = Box(
                low=0,
                high=255,
                shape=lidar["shape"],
                dtype=np.uint8,
            )

        camera_configs = self.carla_manager.config.car_config.cameras

        shape = (self.config["image_resize"][0], self.config["image_resize"][1], 3)

        if self.config["concat_images"]:
            height = camera_configs[0]["height"]
            assert all(camera["height"] == height for camera in camera_configs)

            observation_space_dict["image"] = Box(
                low=-np.inf,
                high=np.inf,
                shape=shape,
                dtype=np.float32,
            )
        else:
            for index, _ in enumerate(camera_configs):
                observation_space_dict[f"image_{index}"] = Box(
                    low=-np.inf,
                    high=np.inf,
                    shape=shape,
                    dtype=np.float32,
                )

        observation_space_dict = self._state_observation_space(observation_space_dict)

        return observation_space_dict

    def _set_observation_space_with_vision(self) -> dict:
        if self.vision_module is None:
            raise ValueError("Vision module is not set")

        observation_space_dict = {}

        if isinstance(self.vision_module.output_shape, list):
            for i, output_shape in enumerate(self.vision_module.output_shape):
                observation_space_dict[f"vision_encoding_{i}"] = Box(
                    low=self.vision_module.low,
                    high=self.vision_module.high,
                    shape=output_shape,
                    dtype=np.float32,
                )

        else:
            observation_space_dict["vision_encoding"] = Box(
                low=self.vision_module.low,
                high=self.vision_module.high,
                shape=self.vision_module.output_shape,
                dtype=np.float32,
            )

        observation_space_dict = self._state_observation_space(observation_space_dict)

        return observation_space_dict

    def _state_observation_space(self, observation_space_dict: dict) -> dict:
        observation_space_dict["state"] = Box(
            low=np.array([-2.0, -10.0, -10.0]),
            high=np.array([10.0, 10.0, 10.0]),
            dtype=np.float32,
        )

        observation_space_dict["command"] = Box(
            low=0, high=1, shape=(6,), dtype=np.int32
        )

        return observation_space_dict

    def reset(self, seed=None, options=None):
        # select random town from configurations
        self._n_episodes += 1

        # Change town if needed
        if (
            self.config["town_change_frequency"] is not None
            and self.config["towns"] is not None
        ):
            if self._n_episodes % self.config["town_change_frequency"] == 0:
                self._town = random.choice(self.config["towns"])

        self._metrics = self.carla_manager.stop_episode()
        self.state, self.data = self.carla_manager.start_episode(
            town=self._town, traffic_type=self.config["traffic_type"]
        )

        # Set up ruote planner with sparse waypoints
        self._route_planner = RoutePlanner()
        self._route_planner.set_route(self.data.global_plan, True)
        if self.vision_module is not None:
            self.vision_module.set_global_plan(self.data.global_plan)

        return self._get_obs(), self._metrics

    def render(self, mode="vision_module") -> Optional[np.ndarray]:
        surface = None
        if self.render_mode == "human":
            if self._renderer is None:
                self._renderer = WorldStateRenderer()

            self._renderer.render(self.state)
            return None
        if self.render_mode == "vision_module" and self.vision_module is not None:
            surface = np.array(self.vision_module.get_auxilliary_render())
            # return np.transpose(np.array(surface), axes=(1, 0, 2))

        if surface is None:
            additional_text = {}

            if self._reward is not None:
                additional_text["Reward: "] = self._reward

            if self._prev_obs is not None:
                additional_text["State: "] = self._prev_obs["state"]
                additional_text["COMMAND: "] = self._prev_obs["command"]

            pygame_surface = generate_pygame_surface(
                self.state,
                additional_text=additional_text,
            )
            surface = pygame.surfarray.array3d(pygame_surface).swapaxes(0, 1)

        return surface

    def _get_obs(self):
        self._prev_obs = (
            self._get_obs_without_vision()
            if self.vision_module is None
            else self._get_obs_with_vision()
        )

        return self._prev_obs

    def _get_obs_without_vision(self) -> dict:
        observation = self._setup_observation_state()

        if self.config["concat_images"]:
            concat_array = np.concatenate(
                [
                    image[:, :, :3]
                    for image in self.state.ego_vehicle_state.sensor_data.images
                ],
                axis=1,
            )

            image = (
                self._rgb_transform(Image.fromarray(concat_array))
                .numpy()
                .transpose(1, 2, 0)
            )

            observation["image"] = image

        else:
            for index, image in enumerate(
                self.state.ego_vehicle_state.sensor_data.images
            ):
                im = image[:, :, :3]
                image = (
                    self._rgb_transform(Image.fromarray(im)).numpy().transpose(1, 2, 0)
                )

                observation[f"image_{index}"] = image

        if self.carla_manager.config.car_config.lidar["enabled"]:
            observation[
                "lidar"
            ] = self.state.ego_vehicle_state.sensor_data.lidar_data.bev

        return observation

    def _get_obs_with_vision(self) -> dict:
        if self.vision_module is None:
            raise ValueError("Vision module is not set")

        observation = self._setup_observation_state()
        vision_encoding = self.vision_module(self.state)

        if isinstance(vision_encoding, list):
            for i, encoding in enumerate(vision_encoding):
                observation[f"vision_encoding_{i}"] = encoding

        else:
            observation["vision_encoding"] = vision_encoding
        # print("VISION ENCODING: ", vision_encoding)
        # print("SHAPE: ", vision_encoding.shape)
        # print("MEAN: ", vision_encoding.mean())
        # print("MAX: ", vision_encoding.max())
        # print("MIN: ", vision_encoding.min())
        # print("STD: ", vision_encoding.std())
        return observation

    def _setup_observation_state(self) -> dict:
        observation = {}
        if self._route_planner is None:
            raise ValueError("Route planner is not set")

        _, pos, target_point, next_cmd = self._route_planner.run_step(
            gps=self.state.ego_vehicle_state.gps
        )

        relative_target_waypoint = find_relative_target_waypoint(
            pos, target_point, self.state.ego_vehicle_state.compass
        )

        relative_target_waypoint = np.clip(relative_target_waypoint, -10.0, 10.0)

        command = np.zeros(6)

        if next_cmd == RoadOption.STRAIGHT:
            command[0] = 1.0
        elif next_cmd == RoadOption.LANEFOLLOW:
            command[1] = 1.0
        elif next_cmd == RoadOption.LEFT:
            command[2] = 1.0
        elif next_cmd == RoadOption.RIGHT:
            command[3] = 1.0
        elif next_cmd == RoadOption.CHANGELANELEFT:
            command[4] = 1.0
        elif next_cmd == RoadOption.CHANGELANERIGHT:
            command[5] = 1.0

        state = np.concatenate(
            (
                np.array([self.state.ego_vehicle_state.speed]),
                relative_target_waypoint,
            ),
            axis=0,
        )

        observation["state"] = state
        observation["command"] = command

        return observation

    def _get_target_point(self) -> np.ndarray:
        """"""
        return np.array([0.0])

    def stop_server(self):
        print("STOPPING SERVER")
        self.carla_manager.close()

    # def start_server(self):
    #     print("STARTING SERVER")
    #     self.carla_manager.reset()

    def set_mode(self, training_type: TrainingType):
        """
        Change the type of routes that the episode manager will use
        """
        self.carla_manager.config.training_type = training_type
        return

    def step(self, action):
        goal_speed = 0.0
        steering = 0.0

        self._steps += 1

        if self.config["discrete_actions"]:
            goal_speed = self.config["speed_goal_actions"][
                action // self.amount_of_steering_actions
            ]

            steering = self.config["steering_actions"][
                action % self.amount_of_steering_actions
            ]

        else:
            if not isinstance(action, np.ndarray):
                raise ValueError("Action must be a numpy array")

            goal_speed, steering = action[0], action[1]

        goal_speed = float(goal_speed)
        steering = float(steering)

        throttle, brake, reverse = self.speed_controller(
            goal_speed, self.state.ego_vehicle_state.speed
        )

        if brake >= 1.0:
            steering *= 0.5
            throttle = 0.0

        new_action = Action(throttle, brake, reverse, steering)

        # print("NEW ACTION: ", new_action, "\nGOAL SPEED: ", goal_speed)

        if self.vision_module is not None:
            new_action = self.vision_module.postprocess_action(new_action)

        # update state with result of using the new action

        self.state = self.carla_manager.step(new_action)

        self._reward, done = self.reward_function(self.state, self.data)

        result = (self._get_obs(), self._reward, done or self.state.done, False, {})
        self.time = time.time()

        return result

    def seed(self, seed=None):
        self.np_random, seed = seeding.np_random(seed)
        return [seed]


def transform_image(image: np.ndarray, shape: tuple):
    # Resize image and transpose it to have channels first (CHW format)
    return np.array(Image.fromarray(image).resize(shape)).transpose(2, 0, 1)


def shift_x_scale_crop(image, scale, crop, crop_shift=0):
    crop_h, crop_w = crop
    (width, height) = (int(image.width // scale), int(image.height // scale))
    im_resized = image.resize((width, height))
    image = np.array(im_resized)
    start_y = height // 2 - crop_h // 2
    start_x = width // 2 - crop_w // 2

    # only shift in x direction
    start_x += int(crop_shift // scale)
    cropped_image = image[start_y : start_y + crop_h, start_x : start_x + crop_w]
    cropped_image = np.transpose(cropped_image, (2, 0, 1))
    return cropped_image


def scale_crop(image, scale=1, start_x=0, crop_x=None, start_y=0, crop_y=None):
    (width, height) = (image.width // scale, image.height // scale)
    if scale != 1:
        image = image.resize((width, height))
    if crop_x is None:
        crop_x = width
    if crop_y is None:
        crop_y = height

    image = np.asarray(image)
    cropped_image = image[start_y : start_y + crop_y, start_x : start_x + crop_x]
    return cropped_image


def prepare_image(images: List[np.ndarray]) -> np.ndarray:
    rgb = []

    for image in images:
        rgb_pos = scale_crop(
            Image.fromarray(image),
            1,
            320,
            320,
            160,
            160,
        )
        rgb.append(rgb_pos)
    rgb = np.concatenate(rgb, axis=1)

    image = Image.fromarray(rgb)
    image_degrees = []
    rgb = np.expand_dims(
        shift_x_scale_crop(image, scale=1, crop=(160, 704), crop_shift=0), axis=0
    )
    image_degrees.append(rgb)
    image = np.concatenate(image_degrees, axis=0)

    return image
