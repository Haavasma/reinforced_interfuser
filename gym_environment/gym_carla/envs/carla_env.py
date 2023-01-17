#!/usr/bin/env python

# Copyright (c) 2019: Jianyu Chen (jianyuchen@berkeley.edu)
#
# This work is licensed under the terms of the MIT license.
# For a copy, see <https://opensource.org/licenses/MIT>.

from __future__ import division
import matplotlib.pyplot as plt
from abc import ABC, abstractmethod
from enum import Enum
from typing import Callable, Tuple
from typing_extensions import Protocol

import torch

import copy
import numpy as np
import random
import time
from skimage.transform import resize

import gym
from gym import spaces
from gym.utils import seeding
import carla

from gym_carla.envs.route_planner import RoutePlanner
from gym_carla.envs.misc import *

params = {
    "amount_of_frames": 1,
    "number_of_vehicles": 100,
    "number_of_walkers": 0,
    "display_size": 960,  # screen size of bird-eye render
    "max_past_step": 1,  # the number of past steps to draw
    "dt": 0.1,  # time interval between two frames
    "discrete": True,  # whether to use discrete control space
    "discrete_acc": [-3.0, 0.0, 1.5, 3.0],  # discrete value of accelerations
    "discrete_steer": [
        -0.3,
        -0.27,
        -0.24,
        -0.21,
        -0.18,
        -0.15,
        -0.12,
        -0.09,
        -0.06,
        -0.03,
        0.0,
        0.03,
        0.06,
        0.09,
        0.12,
        0.15,
        0.18,
        0.21,
        0.24,
        0.27,
        0.3,
    ],  # discrete value of steering angles
    "continuous_accel_range": [-3.0, 3.0],  # continuous acceleration range
    "continuous_steer_range": [-0.3, 0.3],  # continuous steering angle range
    "ego_vehicle_filter": "vehicle.lincoln*",  # filter for defining ego vehicle
    "port": 2000,  # connection port
    "town": "Town03",  # which town to simulate
    "task_mode": "random",  # mode of the task, [random, roundabout (only for Town03)]
    "max_time_episode": 1000,  # maximum timesteps per episode
    "max_waypt": 12,  # maximum number of waypoints
    "obs_range": 32,  # observation range (meter)
    "lidar_bin": 0.125,  # bin size of lidar sensor (meter)
    "d_behind": 12,  # distance behind the ego vehicle (meter)
    "out_lane_thres": 2.0,  # threshold for out of lane
    "desired_speed": 8,  # desired speed (m/s)
    "max_ego_spawn_times": 200,  # maximum times to spawn ego vehicle
    "display_route": True,  # whether to render the desired route
    "pixor_size": 64,  # size of the pixor labels
    "pixor": False,  # whether to output PIXOR observation
    "image_x": 960,
    "image_y": 480,
    "fov": 120,
    "camera_rot_0": [0.0, 0.0, -60.0],  # Roll Pitch Yaw of camera 0 in degree
    "camera_rot_1": [0.0, 0.0, 0.0],  # Roll Pitch Yaw of camera 1 in degree
    "camera_rot_2": [0.0, 0.0, 60.0],  # Roll Pitch Yaw of camera 2 in degree
    "lidar_rot": [0.0, 0.0, -90.0],  # Roll Pitch Yaw of lidar in degree
}


class TransfuserBackbone(Protocol):
    def __call__(
        self, rgb: torch.Tensor, lidar: torch.Tensor, ego_velocity: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        ...


class SegDecoder(Protocol):
    def __call__(self, x: torch.Tensor) -> torch.Tensor:
        ...


class DepthDecoder(Protocol):
    def __call__(self, x: torch.Tensor) -> torch.Tensor:
        ...


class CarlaInterFuser(ABC):
    @abstractmethod
    def __call__(
        self,
        rgb_l: torch.Tensor,
        rgb_r: torch.Tensor,
        rgb_front: torch.Tensor,
        rgb_focus: torch.Tensor,
        lidar_bev: torch.Tensor,
    ) -> torch.Tensor:
        pass


class VisionBackbone(Enum):
    transfuser = 1
    interfuser = 2


class CarlaEnvTransFuser(gym.Env):
    """An OpenAI gym wrapper for CARLA simulator."""

    def __init__(
        self,
        vision_backbone: TransfuserBackbone,
        vision_output_shape: Tuple,
        seg_decoder: SegDecoder,
        depth_decoder: DepthDecoder,
        pred_bev: SegDecoder,
        prepare_image: Callable[[np.ndarray, np.ndarray, np.ndarray], torch.Tensor],
        prepare_lidar: Callable[[np.ndarray], torch.Tensor],
        render_env=False,
        port=2000,
    ):

        self.render_env = render_env
        self.seg_decoder = seg_decoder
        self.depth_decoder = depth_decoder
        self.pred_bev = pred_bev
        if self.render_env:
            self._init_renderer()

        self.start_time = time.time()

        self.prepare_image = prepare_image
        self.prepare_lidar = prepare_lidar

        # parameters
        self.vision_model = vision_backbone
        self.display_size = params["display_size"]  # rendering screen size
        self.max_past_step = params["max_past_step"]
        self.number_of_vehicles = params["number_of_vehicles"]
        self.number_of_walkers = params["number_of_walkers"]
        self.dt = params["dt"]
        self.task_mode = params["task_mode"]
        self.max_time_episode = params["max_time_episode"]
        self.max_waypt = params["max_waypt"]
        self.obs_range = params["obs_range"]
        self.lidar_bin = params["lidar_bin"]
        self.d_behind = params["d_behind"]
        self.obs_size = int(self.obs_range / self.lidar_bin)
        self.out_lane_thres = params["out_lane_thres"]
        self.desired_speed = params["desired_speed"]
        self.max_ego_spawn_times = params["max_ego_spawn_times"]
        self.display_route = params["display_route"]
        if "pixor" in params.keys():
            self.pixor = params["pixor"]
            self.pixor_size = params["pixor_size"]
        else:
            self.pixor = False

        # Destination
        if params["task_mode"] == "roundabout":
            self.dests = [
                [4.46, -61.46, 0],
                [-49.53, -2.89, 0],
                [-6.48, 55.47, 0],
                [35.96, 3.33, 0],
            ]
        else:
            self.dests = None

        # action and observation spaces
        self.discrete = params["discrete"]
        self.discrete_act = [
            params["discrete_acc"],
            params["discrete_steer"],
        ]  # acc, steer
        self.n_acc = len(self.discrete_act[0])
        self.n_steer = len(self.discrete_act[1])
        if self.discrete:
            self.action_space = spaces.Discrete(self.n_acc * self.n_steer)
        else:
            self.action_space = spaces.Box(
                np.array(
                    [
                        params["continuous_accel_range"][0],
                        params["continuous_steer_range"][0],
                    ]
                ),
                np.array(
                    [
                        params["continuous_accel_range"][1],
                        params["continuous_steer_range"][1],
                    ]
                ),
                dtype=np.float32,
            )  # acc, steer

        # TODO: rework to parameterize the previous three time steps
        # as state, if there is room.

        state_range_min = np.array([-2, -1] + ([-5] * params["amount_of_frames"]))
        state_range_max = np.array([2, 1] + ([20] * params["amount_of_frames"]))
        vision_shape = ((vision_output_shape[0] * params["amount_of_frames"]),)

        observation_space_dict = {
            "vision_encoding": spaces.Box(
                low=-20,
                high=20,
                shape=vision_shape,
                dtype=np.float32,
            ),
            "state": spaces.Box(
                state_range_min,
                state_range_max,
                dtype=np.float32,
            ),
        }

        self.observation_space = spaces.Dict(spaces=observation_space_dict)

        # Connect to carla server and get world object
        print("connecting to Carla server...")
        client = carla.Client("localhost", port)
        client.set_timeout(10.0)
        self.world = client.load_world(params["town"])
        print("Carla server connected!")

        # Set weather
        self.world.set_weather(carla.WeatherParameters.ClearNoon)

        # Get spawn points
        self.vehicle_spawn_points = list(self.world.get_map().get_spawn_points())
        self.walker_spawn_points = []
        for i in range(self.number_of_walkers):
            spawn_point = carla.Transform()
            loc = self.world.get_random_location_from_navigation()
            if loc != None:
                spawn_point.location = loc
                self.walker_spawn_points.append(spawn_point)

        # Create the ego vehicle blueprint
        self.ego_bp = self._create_vehicle_bluepprint(
            params["ego_vehicle_filter"], color="49,8,8"
        )

        # Collision sensor
        self.collision_hist = []  # The collision history
        self.collision_hist_l = 1  # collision history length
        self.collision_bp = self.world.get_blueprint_library().find(
            "sensor.other.collision"
        )

        self.collision_sensor = None
        self.lidar_sensor = None
        self.camera_sensor_left = None
        self.camera_sensor_front = None
        self.camera_sensor_right = None

        # Lidar sensor
        self.lidar_data = np.array([])
        self.lidar_height = 2.1
        self.lidar_trans = carla.Transform(
            carla.Location(x=0.0, z=self.lidar_height),
            carla.Rotation(
                pitch=params["lidar_rot"][0],
                yaw=params["lidar_rot"][2],
                roll=params["lidar_rot"][1],
            ),
        )
        self.lidar_bp = self.world.get_blueprint_library().find("sensor.lidar.ray_cast")
        self.lidar_bp.set_attribute("channels", "32")
        self.lidar_bp.set_attribute("range", "5000")

        # Camera sensor
        self.camera_img_left = np.zeros(
            (params["image_y"], params["image_x"], 4), dtype=np.uint8
        )
        rot_left = carla.Rotation(
            pitch=params["camera_rot_0"][0],
            yaw=params["camera_rot_0"][2],
            roll=params["camera_rot_0"][1],
        )

        self.camera_img_front = np.zeros(
            (params["image_y"], params["image_x"], 4), dtype=np.uint8
        )
        rot_front = carla.Rotation(
            pitch=params["camera_rot_1"][0],
            yaw=params["camera_rot_1"][2],
            roll=params["camera_rot_1"][1],
        )

        self.camera_img_right = np.zeros(
            (params["image_y"], params["image_x"], 4), dtype=np.uint8
        )
        rot_right = carla.Rotation(
            pitch=params["camera_rot_2"][0],
            yaw=params["camera_rot_2"][2],
            roll=params["camera_rot_2"][1],
        )

        self.camera_trans_left = carla.Transform(carla.Location(x=1.3, z=2.3), rot_left)
        self.camera_trans_front = carla.Transform(
            carla.Location(x=1.3, z=2.3), rot_front
        )
        self.camera_trans_right = carla.Transform(
            carla.Location(x=1.3, z=2.3), rot_right
        )
        self.camera_bp = self.world.get_blueprint_library().find("sensor.camera.rgb")

        # self.gps_pos = np.array([])
        # self.gps_bp = self.world.get_blueprint_library().find("sensor.other.gnss")
        # self.gps_bp.set_attribute("sensor_tick", "0.01")

        # Modify the attributes of the blueprint to set image resolution and field of view.
        self.camera_bp.set_attribute("image_size_x", str(params["image_x"]))
        self.camera_bp.set_attribute("image_size_y", str(params["image_y"]))
        self.camera_bp.set_attribute("fov", str(params["fov"]))
        # Set the time in seconds between sensor captures
        self.camera_bp.set_attribute("sensor_tick", "0.5")

        # Set fixed simulation step for synchronous mode
        self.settings = self.world.get_settings()
        self.settings.fixed_delta_seconds = self.dt
        self.settings.substepping = True
        self.settings.max_substep_delta_time = 0.01
        self.settings.max_substeps = 10

        # Record the time of total steps and resetting steps
        self.reset_step = 0
        self.total_step = 0

        # Initialize the renderer
        self._init_renderer()

        # Get pixel grid points
        if self.pixor:
            x, y = np.meshgrid(
                np.arange(self.pixor_size), np.arange(self.pixor_size)
            )  # make a canvas with coordinates
            x, y = x.flatten(), y.flatten()
            self.pixel_grid = np.vstack((x, y)).T

    def reset(self):
        if self.collision_sensor is not None:
            self.collision_sensor.stop()
            self.collision_sensor.destroy()
        self.collision_sensor = None
        if self.lidar_sensor is not None:
            self.lidar_sensor.stop()
            self.lidar_sensor.destroy()
        self.lidar_sensor = None
        if self.camera_sensor_left is not None:
            self.camera_sensor_left.stop()
            self.camera_sensor_left.destroy()
        self.camera_sensor_left = None
        if self.camera_sensor_front is not None:
            self.camera_sensor_front.stop()
            self.camera_sensor_front.destroy()
        self.camera_sensor_front = None
        if self.camera_sensor_right is not None:
            self.camera_sensor_right.stop()
            self.camera_sensor_right.destroy()
        self.camera_sensor_right = None

        # Delete sensors, vehicles and walkers
        self._clear_all_actors(
            [
                "sensor.other.collision",
                "sensor.lidar.ray_cast",
                "sensor.camera.rgb",
                "vehicle.*",
                "sensor.other.imu",
                "controller.ai.walker",
                "walker.*",
            ]
        )

        # Disable sync mode
        self._set_synchronous_mode(True)

        # Spawn surrounding vehicles
        random.shuffle(self.vehicle_spawn_points)
        count = self.number_of_vehicles
        if count > 0:
            for spawn_point in self.vehicle_spawn_points:
                if self._try_spawn_random_vehicle_at(spawn_point, number_of_wheels=[4]):
                    count -= 1
                if count <= 0:
                    break
        while count > 0:
            if self._try_spawn_random_vehicle_at(
                random.choice(self.vehicle_spawn_points), number_of_wheels=[4]
            ):
                count -= 1

        # Spawn pedestrians
        random.shuffle(self.walker_spawn_points)
        count = self.number_of_walkers
        if count > 0:
            for spawn_point in self.walker_spawn_points:
                if self._try_spawn_random_walker_at(spawn_point):
                    count -= 1
                if count <= 0:
                    break
        while count > 0:
            if self._try_spawn_random_walker_at(
                random.choice(self.walker_spawn_points)
            ):
                count -= 1

        # Get actors polygon list
        self.vehicle_polygons = []
        vehicle_poly_dict = self._get_actor_polygons("vehicle.*")
        self.vehicle_polygons.append(vehicle_poly_dict)
        self.walker_polygons = []
        walker_poly_dict = self._get_actor_polygons("walker.*")
        self.walker_polygons.append(walker_poly_dict)

        # Spawn the ego vehicle
        ego_spawn_times = 0
        while True:
            if ego_spawn_times > self.max_ego_spawn_times:
                self.reset()

            if self.task_mode == "random":
                transform = random.choice(self.vehicle_spawn_points)
            if self.task_mode == "roundabout":
                self.start = [52.1 + np.random.uniform(-5, 5), -4.2, 178.66]  # random
                # self.start=[52.1,-4.2, 178.66] # static
                transform = set_carla_transform(self.start)
            if self._try_spawn_ego_vehicle_at(transform):
                break
            else:
                ego_spawn_times += 1
                time.sleep(0.1)

        # Add collision sensor
        self.collision_sensor = self.world.spawn_actor(
            self.collision_bp, carla.Transform(), attach_to=self.ego
        )
        self.collision_sensor.listen(lambda event: get_collision_hist(event))

        def get_collision_hist(event):
            impulse = event.normal_impulse
            intensity = np.sqrt(impulse.x**2 + impulse.y**2 + impulse.z**2)
            self.collision_hist.append(intensity)
            if len(self.collision_hist) > self.collision_hist_l:
                self.collision_hist.pop(0)

        self.collision_hist = []

        # Add lidar sensor
        self.lidar_sensor = self.world.spawn_actor(
            self.lidar_bp, self.lidar_trans, attach_to=self.ego
        )
        self.lidar_sensor.listen(lambda data: get_lidar_data(data))

        def get_lidar_data(data):
            self.lidar_data = data

        # Add camera sensor
        self.camera_sensor_left = self.world.spawn_actor(
            self.camera_bp, self.camera_trans_left, attach_to=self.ego
        )
        self.camera_sensor_front = self.world.spawn_actor(
            self.camera_bp, self.camera_trans_front, attach_to=self.ego
        )
        self.camera_sensor_right = self.world.spawn_actor(
            self.camera_bp, self.camera_trans_right, attach_to=self.ego
        )

        def get_camera_img_left(data):
            array = np.frombuffer(data.raw_data, dtype=np.dtype("uint8"))
            self.camera_img_left = array.reshape(
                params["image_y"], params["image_x"], 4
            )

        def get_camera_img_front(data):
            array = np.frombuffer(data.raw_data, dtype=np.dtype("uint8"))
            self.camera_img_front = array.reshape(
                params["image_y"], params["image_x"], 4
            )

        def get_camera_img_right(data):
            array = np.frombuffer(data.raw_data, dtype=np.dtype("uint8"))
            self.camera_img_right = array.reshape(
                params["image_y"], params["image_x"], 4
            )

        self.camera_sensor_left.listen(lambda data: get_camera_img_left(data))
        self.camera_sensor_front.listen(lambda data: get_camera_img_front(data))
        self.camera_sensor_right.listen(lambda data: get_camera_img_right(data))

        # Update timesteps
        self.time_step = 0
        self.reset_step += 1

        self.world.apply_settings(self.settings)

        self.routeplanner = RoutePlanner(self.ego, self.max_waypt)
        (
            self.waypoints,
            self.red_light,
            self.vehicle_front,
        ) = self.routeplanner.run_step()

        # Set ego information for render
        return self._get_obs()

    def step(self, action):
        # Calculate acceleration and steering
        if self.discrete:
            acc = self.discrete_act[0][action // self.n_steer]
            steer = self.discrete_act[1][action % self.n_steer]
        else:
            acc = action[0]
            steer = action[1]

        # Convert acceleration to throttle and brake
        if acc > 0:
            throttle = np.clip(acc / 3, 0, 1)
            brake = 0
        else:
            throttle = 0
            brake = np.clip(-acc / 8, 0, 1)

        # Apply control
        act = carla.VehicleControl(
            throttle=float(throttle), steer=float(-steer), brake=float(brake)
        )

        self.ego.apply_control(act)

        # Append actors polygon list
        vehicle_poly_dict = self._get_actor_polygons("vehicle.*")
        self.vehicle_polygons.append(vehicle_poly_dict)
        while len(self.vehicle_polygons) > self.max_past_step:
            self.vehicle_polygons.pop(0)
        walker_poly_dict = self._get_actor_polygons("walker.*")
        self.walker_polygons.append(walker_poly_dict)
        while len(self.walker_polygons) > self.max_past_step:
            self.walker_polygons.pop(0)

        # route planner
        self.waypoints, _, self.vehicle_front = self.routeplanner.run_step()

        # state information
        info = {"waypoints": self.waypoints, "vehicle_front": self.vehicle_front}

        # Update timesteps
        self.time_step += 1
        self.total_step += 1

        # Change camera view
        spectator = self.world.get_spectator()
        new_location = self.ego.get_location() + carla.Location(z=40)
        new_rotation = self.ego.get_transform().rotation
        new_rotation = carla.Rotation(pitch=-90, yaw=0, roll=0)

        spectator.set_transform(carla.Transform(new_location, new_rotation))

        return (
            self._get_obs(),
            self._get_reward(),
            self._terminal(),
            copy.deepcopy(info),
        )

    def seed(self, seed=None):
        self.np_random, seed = seeding.np_random(seed)
        return [seed]

    def render(self, _=None) -> np.ndarray:
        # update renderered images
        rgb_image = self.rgb.cpu().detach().numpy()[0]
        lidar_bev = self.lidar_bev.cpu().detach().numpy()[0]

        # segmentation_pred = self.seg_decoder(self.image_features_grid)
        depth_pred = self.depth_decoder(self.image_features_grid)
        pred_bev = self.pred_bev(self.lidar_features[0])

        bev = pred_bev[0].detach().cpu().numpy().argmax(axis=0) / 2.0
        bev = np.stack([bev, bev, bev], axis=2) * 255.0
        bev_image = bev.astype(np.uint8)

        depth = depth_pred.cpu().detach().numpy()

        new_lidar = np.zeros((256, 256, 3))
        new_lidar[:, :, 0] = lidar_bev[0, :, :]
        new_lidar[:, :, 1] = lidar_bev[1, :, :]
        new_lidar[:, :, 2] = lidar_bev[2, :, :]

        new_rgb = np.zeros((160, 704, 3))
        new_rgb[:, :, 0] = rgb_image[0, :, :]
        new_rgb[:, :, 1] = rgb_image[1, :, :]
        new_rgb[:, :, 2] = rgb_image[2, :, :]

        new_depth = np.zeros((160, 704, 3))
        new_depth[:, :, 0] = depth

        self.im1.set_data(new_rgb / 255)
        self.im2.set_data(new_lidar)
        self.im3.set_data(new_depth)
        self.im4.set_data(bev_image / 255)

        plt.pause(0.00001)
        return self.rgb.cpu().detach().numpy()

    def _create_vehicle_bluepprint(
        self, actor_filter, color=None, number_of_wheels=[4]
    ):
        """Create the blueprint for a specific actor type.

        Args:
          actor_filter: a string indicating the actor type, e.g, 'vehicle.lincoln*'.

        Returns:
          bp: the blueprint object of carla.
        """
        blueprints = self.world.get_blueprint_library().filter(actor_filter)
        blueprint_library = []
        for nw in number_of_wheels:
            blueprint_library = blueprint_library + [
                x for x in blueprints if int(x.get_attribute("number_of_wheels")) == nw
            ]
        bp = random.choice(blueprint_library)
        if bp.has_attribute("color"):
            if not color:
                color = random.choice(bp.get_attribute("color").recommended_values)
            bp.set_attribute("color", color)
        return bp

    def _init_renderer(self):
        """Initialize the birdeye view renderer."""
        plt.ion()
        ax1 = plt.subplot(2, 2, 1)
        ax2 = plt.subplot(2, 2, 2)
        ax3 = plt.subplot(2, 2, 3)
        ax4 = plt.subplot(2, 2, 4)

        self.im1 = ax1.imshow(np.zeros((160, 704, 3)))
        self.im2 = ax2.imshow(np.zeros(((256, 256, 3))))
        self.im3 = ax3.imshow(np.zeros(((160, 704, 3))))
        self.im4 = ax4.imshow(np.zeros(((64, 64, 3))))

        # plt.imshow(np.zeros((160, 704, 3)))

    def _set_synchronous_mode(self, synchronous=True):
        """Set whether to use the synchronous mode."""
        self.settings.synchronous_mode = synchronous
        self.world.apply_settings(self.settings)

    def _try_spawn_random_vehicle_at(self, transform, number_of_wheels=[4]):
        """Try to spawn a surrounding vehicle at specific transform with random bluprint.

        Args:
          transform: the carla transform object.

        Returns:
          Bool indicating whether the spawn is successful.
        """
        blueprint = self._create_vehicle_bluepprint(
            "vehicle.*", number_of_wheels=number_of_wheels
        )
        blueprint.set_attribute("role_name", "autopilot")
        vehicle = self.world.try_spawn_actor(blueprint, transform)
        if vehicle is not None:
            vehicle.set_autopilot()
            return True
        return False

    def _try_spawn_random_walker_at(self, transform):
        """Try to spawn a walker at specific transform with random bluprint.

        Args:
          transform: the carla transform object.

        Returns:
          Bool indicating whether the spawn is successful.
        """
        walker_bp = random.choice(self.world.get_blueprint_library().filter("walker.*"))
        # set as not invencible
        if walker_bp.has_attribute("is_invincible"):
            walker_bp.set_attribute("is_invincible", "false")
        walker_actor = self.world.try_spawn_actor(walker_bp, transform)

        if walker_actor is not None:
            walker_controller_bp = self.world.get_blueprint_library().find(
                "controller.ai.walker"
            )
            walker_controller_actor = self.world.spawn_actor(
                walker_controller_bp, carla.Transform(), walker_actor
            )
            # start walker
            walker_controller_actor.start()
            # set walk to random point
            walker_controller_actor.go_to_location(
                self.world.get_random_location_from_navigation()
            )
            # random max speed
            walker_controller_actor.set_max_speed(
                1 + random.random()
            )  # max speed between 1 and 2 (default is 1.4 m/s)
            return True
        return False

    def _try_spawn_ego_vehicle_at(self, transform):
        """Try to spawn the ego vehicle at specific transform.
        Args:
          transform: the carla transform object.
        Returns:
          Bool indicating whether the spawn is successful.
        """
        vehicle = None
        # Check if ego position overlaps with surrounding vehicles
        overlap = False
        for idx, poly in self.vehicle_polygons[-1].items():
            poly_center = np.mean(poly, axis=0)
            ego_center = np.array([transform.location.x, transform.location.y])
            dis = np.linalg.norm(poly_center - ego_center)
            if dis > 8:
                continue
            else:
                overlap = True
                break

        if not overlap:
            vehicle = self.world.try_spawn_actor(self.ego_bp, transform)

        if vehicle is not None:
            self.ego = vehicle
            return True

        return False

    def _get_actor_polygons(self, filt):
        """Get the bounding box polygon of actors.

        Args:
          filt: the filter indicating what type of actors we'll look at.

        Returns:
          actor_poly_dict: a dictionary containing the bounding boxes of specific actors.
        """
        actor_poly_dict = {}
        for actor in self.world.get_actors().filter(filt):
            # Get x, y and yaw of the actor
            trans = actor.get_transform()
            x = trans.location.x
            y = trans.location.y
            yaw = trans.rotation.yaw / 180 * np.pi
            # Get length and width
            bb = actor.bounding_box
            l = bb.extent.x
            w = bb.extent.y
            # Get bounding box polygon in the actor's local coordinate
            poly_local = np.array([[l, w], [l, -w], [-l, -w], [-l, w]]).transpose()
            # Get rotation matrix to transform to global coordinate
            R = np.array([[np.cos(yaw), -np.sin(yaw)], [np.sin(yaw), np.cos(yaw)]])
            # Get global bounding box polygon
            poly = np.matmul(R, poly_local).transpose() + np.repeat([[x, y]], 4, axis=0)
            actor_poly_dict[actor.id] = poly
        return actor_poly_dict

    def _get_obs(self):
        """Get the observations."""
        # State observation
        self.lidar_bev = torch.Tensor().to("cuda")
        self.rgb = torch.Tensor().to("cuda")
        self.speeds = []

        for _ in range(params["amount_of_frames"]):
            self.world.tick()
            point_cloud = []

            for location in self.lidar_data:
                point_cloud.append(
                    [location.point.x, location.point.y, location.point.z]
                )
            self.lidar_bev = torch.cat(
                (
                    self.lidar_bev,
                    torch.from_numpy(self.prepare_lidar(point_cloud)).to(
                        "cuda", dtype=torch.float32
                    ),
                ),
            )

            self.rgb = torch.cat(
                (
                    self.rgb,
                    self.prepare_image(
                        self.camera_img_left,
                        self.camera_img_front,
                        self.camera_img_right,
                    ),
                )
            )

            v = self.ego.get_velocity()
            self.speeds.append(np.sqrt(v.x**2 + v.y**2))

        ego_trans = self.ego.get_transform()
        ego_x = ego_trans.location.x
        ego_y = ego_trans.location.y
        ego_yaw = ego_trans.rotation.yaw / 180 * np.pi
        lateral_dis, w = get_preview_lane_dis(self.waypoints, ego_x, ego_y)
        delta_yaw = np.arcsin(
            np.cross(w, np.array(np.array([np.cos(ego_yaw), np.sin(ego_yaw)])))
        )
        state = np.array(
            [
                lateral_dis,
                -delta_yaw,
            ]
        )

        state = np.concatenate([state, self.speeds])

        lidar_features, image_features_grid, fused_features = self.vision_model(
            self.rgb, self.lidar_bev, torch.Tensor(self.speeds[-1].astype(np.uint8))
        )

        self.lidar_features = lidar_features
        self.fused_features = fused_features
        self.image_features_grid = image_features_grid
        vision_encoding = fused_features.flatten().cpu().detach().numpy()

        obs = {
            "vision_encoding": vision_encoding,
            "state": state,
        }

        print("state: ", state)

        if self.render_env:
            self.render(None)

        return obs

    def _get_desired_speed(self) -> int:
        # shortest distance to obstacle that is not -1
        max_speed = self.desired_speed

        def get_obstacle_distance(red_light: float, vehicle_front: float):
            obstacle_distances = [red_light, vehicle_front]
            obstacle_distances = [x for x in obstacle_distances if x != -1]
            return min(obstacle_distances) if obstacle_distances else -1

        obstacle_distance = get_obstacle_distance(self.red_light, self.vehicle_front)

        if obstacle_distance != -1:
            return min(max_speed, max_speed * (max(obstacle_distance - 10, 0) / 10))

        return max_speed

    def _get_reward(self):
        """Calculate the step reward."""

        # reward for collision
        if len(self.collision_hist) > 0:
            return -10

        print("DESIRED SPEED: ", self._get_desired_speed())
        max_speed_diff = 3.0
        v = self.ego.get_velocity()
        speed = np.sqrt(v.x**2 + v.y**2)

        if speed > self._get_desired_speed():
            # much more deminishing reward  for going above desired speed
            max_speed_diff = 0.5

        # TODO: no reward if too fast or too slow
        speed_diff = abs(speed - self._get_desired_speed())

        if speed <= 0.5 and self.desired_speed > 2.0:
            return -1.0

        if speed_diff > max_speed_diff:
            return 0

        r_speed = (max_speed_diff - speed_diff) / max_speed_diff

        # reward for out of lane
        ego_x, ego_y = get_pos(self.ego)
        dis, w = get_lane_dis(self.waypoints, ego_x, ego_y)
        if abs(dis) > self.out_lane_thres:
            return 0

        r_distance = (self.out_lane_thres - abs(dis)) / self.out_lane_thres

        # convert x, y vector to angle
        angle = np.degrees(np.arctan2(w[1], w[0]))

        # get difference between waypoint angle and ego vehicle Yaw
        rot_diff = 180 - abs(abs(angle - self.ego.get_transform().rotation.yaw) - 180)

        assert rot_diff >= 0

        max_diff = 30
        if rot_diff > max_diff:
            return 0

        r_angle = (max_diff - rot_diff) / max_diff

        r = (r_speed + r_distance + r_angle) / 3

        print("SPEED REWARD: ", r_speed)
        print("DISTANCE REWARD: ", r_distance)
        print("ANGLE REWARD: ", r_angle)
        print("REWARD: ", r)
        print("\n")

        return r

    def _terminal(self):
        """Calculate whether to terminate the current episode."""
        # Get ego state
        ego_x, ego_y = get_pos(self.ego)

        # If collides
        if len(self.collision_hist) > 0:
            return True

        # If reach maximum timestep
        if self.time_step > self.max_time_episode:
            return True

        # If at destination
        if self.dests is not None:  # If at destination
            for dest in self.dests:
                if np.sqrt((ego_x - dest[0]) ** 2 + (ego_y - dest[1]) ** 2) < 4:
                    return True

        # If out of lane
        dis, _ = get_lane_dis(self.waypoints, ego_x, ego_y)
        if abs(dis) > self.out_lane_thres:
            return True

        return False

    def _clear_all_actors(self, actor_filters):
        """Clear specific actors."""
        for actor_filter in actor_filters:
            for actor in self.world.get_actors().filter(actor_filter):
                if actor.is_alive:
                    if actor.type_id == "controller.ai.walker":
                        actor.stop()
                    actor.destroy()
