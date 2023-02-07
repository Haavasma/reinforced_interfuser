from dataclasses import dataclass
from enum import Enum
import pathlib
from typing import List
from typing_extensions import override
from manual_control import (
    World,
    HUD,
    GnssSensor,
    CollisionSensor,
    LaneInvasionSensor,
    IMUSensor,
    CameraManager,
    get_actor_display_name,
)
import time
import carla
import pygame


import numpy as np


from random import Random

from episode_manager.scenario_handler import ScenarioHandler


# Create enum with types Evaluation and Training
class TrainingType(Enum):
    TRAINING = "training_routes"
    VALIDATION = "validation_routes"
    EVALUATION = "evaluation_routes"


training_routes = [
    ["routes_town01_long.xml", "town01_all_scenarios.json"],
    ["routes_town01_short.xml", "town01_all_scenarios.json"],
    ["routes_town01_tiny.xml", "town01_all_scenarios.json"],
    ["routes_town02_long.xml", "town02_all_scenarios.json"],
    ["routes_town02_short.xml", "town02_all_scenarios.json"],
    ["routes_town02_tiny.xml", "town02_all_scenarios.json"],
    ["routes_town03_long.xml", "town03_all_scenarios.json"],
    ["routes_town03_short.xml", "town03_all_scenarios.json"],
    ["routes_town03_tiny.xml", "town03_all_scenarios.json"],
    ["routes_town04_long.xml", "town04_all_scenarios.json"],
    ["routes_town04_short.xml", "town04_all_scenarios.json"],
    ["routes_town04_tiny.xml", "town04_all_scenarios.json"],
    ["routes_town05_long.xml", "town05_all_scenarios.json"],
    ["routes_town05_short.xml", "town05_all_scenarios.json"],
    ["routes_town05_tiny.xml", "town05_all_scenarios.json"],
    ["routes_town06_long.xml", "town06_all_scenarios.json"],
    ["routes_town06_short.xml", "town06_all_scenarios.json"],
    ["routes_town06_tiny.xml", "town06_all_scenarios.json"],
    ["routes_town07_short.xml", "town07_all_scenarios.json"],
    ["routes_town07_tiny.xml", "town07_all_scenarios.json"],
    ["routes_town10_short.xml", "town10_all_scenarios.json"],
    ["routes_town10_tiny.xml", "town10_all_scenarios.json"],
]

validation_routes = [
    ["routes_town05_short.xml", "town05_all_scenarios.json"],
    ["routes_town05_tiny.xml", "town05_all_scenarios.json"],
]

evaluation_routes = [["routes_town05_long.xml", "town05_all_scenarios.json"]]


training_type_to_routes = {
    TrainingType.TRAINING.value: training_routes,
    TrainingType.VALIDATION.value: validation_routes,
    TrainingType.EVALUATION.value: evaluation_routes,
}


@dataclass
class Rotation:
    pitch: float
    yaw: float
    roll: float


@dataclass
class CarConfiguration:
    model: str
    transform: carla.Transform
    camera_rotations: List[Rotation]

    def __post_init__(self):
        self.carla_camera_rotations = [
            carla.Rotation(rotation.pitch, rotation.yaw, rotation.roll)
            for rotation in self.camera_rotations
        ]

    def get_carla_camera_rotations(self) -> List[carla.Rotation]:
        """
        parses and returns current set of rotations as
        """
        return self.carla_camera_rotations


@dataclass
class EpisodeManagerConfiguration:
    host: str
    port: int
    training_type: TrainingType
    route_directory: pathlib.Path
    car_config: CarConfiguration


@dataclass
class WorldState:
    images: List[np.ndarray]
    lidar: np.ndarray
    distance_to_red_light: float

    running: bool


@dataclass
class Action:
    throttle: float
    brake: float
    reverse: bool
    steer: float

    def carla_vehicle_control(self):
        return carla.VehicleControl(
            throttle=self.throttle,
            brake=self.brake,
            reverse=self.reverse,
            steer=self.steer,
        )


@dataclass
class EpisodeFiles:
    route: pathlib.Path
    scenario: pathlib.Path


class EpisodeManager:
    def __init__(
        self,
        config: EpisodeManagerConfiguration,
        scenario_handler: ScenarioHandler = ScenarioHandler(),
    ):
        self.config = config
        self.scenario_handler = scenario_handler

        def get_episodes(training_type: TrainingType) -> List[EpisodeFiles]:
            def get_path(dir: str, file: str):
                return config.route_directory / dir / file

            routes = []

            for path in training_type_to_routes[training_type.value]:
                routes.append(
                    EpisodeFiles(
                        route=get_path(training_type.value, path[0]),
                        scenario=get_path("scenarios", path[1]),
                    )
                )
            return routes

        self.routes = get_episodes(
            config.training_type,
        )

        self.client = carla.Client(self.config.host, self.config.port)
        self.client.set_timeout(20.0)
        self.sim_world = self.client.get_world()

        return

    def start_episode(self):
        """
        Starts a new route in the simulator based on the provided configurations
        """
        files = self.routes[Random().randint(0, len(self.routes))]

        print("Starting episode with route: " + str(files.route))

        # TODO: Pick a random scenario from the episodes, instead of hard-coding it to 0
        self.scenario_handler.start_episode(files.route, files.scenario, "0")

        pygame.init()
        pygame.font.init()

        width = 1280
        height = 720

        self.display = pygame.display.set_mode(
            (width, height), pygame.HWSURFACE | pygame.DOUBLEBUF
        )
        self.display.fill((0, 0, 0))
        pygame.display.flip()

        hud = HUD(width, height)
        print("Starting world coverage")
        self.world = AgentHandler(self.sim_world, hud, None)
        print("SET UP WORLD")

        return

    def step(self, ego_vehicle_action: Action) -> WorldState:
        """
        Runs one step/frame in the simulated scenario, performing the chosen action on the environment
        """
        self.scenario_handler.tick()

        self.world.player.apply_control(ego_vehicle_action.carla_vehicle_control())

        self.world.render(self.display)
        pygame.display.flip()

        return WorldState([], np.ndarray([]), 0, True)

    def stop_episode(self):
        return self.scenario_handler.stop_episode()


class AgentHandler(World):
    @override
    def restart(self):

        if self.restarted:
            return
        self.restarted = True

        self.player_max_speed = 1.589
        self.player_max_speed_fast = 3.713

        # Keep same camera config if the camera manager exists.
        cam_index = self.camera_manager.index if self.camera_manager is not None else 0
        cam_pos_index = (
            self.camera_manager.transform_index
            if self.camera_manager is not None
            else 0
        )

        # Get the ego vehicle
        while self.player is None:
            print("Waiting for the ego vehicle...")
            time.sleep(1)
            possible_vehicles = self.world.get_actors().filter("vehicle.*")
            for vehicle in possible_vehicles:
                if vehicle.attributes["role_name"] == "hero":
                    print("Ego vehicle found")
                    self.player = vehicle
                    break

        self.player_name = self.player.type_id

        # Set up the sensors.
        self.collision_sensor = CollisionSensor(self.player, self.hud)
        self.lane_invasion_sensor = LaneInvasionSensor(self.player, self.hud)
        self.gnss_sensor = GnssSensor(self.player)
        self.imu_sensor = IMUSensor(self.player)
        self.camera_manager = CameraManager(self.player, self.hud)
        self.camera_manager.transform_index = cam_pos_index
        self.camera_manager.set_sensor(cam_index, notify=False)
        actor_type = get_actor_display_name(self.player)
        self.hud.notification(actor_type)
