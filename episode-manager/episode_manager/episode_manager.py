from dataclasses import dataclass
from enum import Enum
import pathlib
import time
from typing import List
import carla


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
    sensor_data: np.ndarray
    running: bool


@dataclass
class Action:
    throttle: float
    brake: float
    reverse: bool


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

        self.scenario_handler = scenario_handler

        return

    def start_episode(self):
        """
        Starts a new route in the simulator based on the provided configurations
        """
        files = self.routes[Random().randint(0, len(self.routes))]
        # threading, start this episode in separate thread

        # TODO: Pick a random scenario from the episodes
        self.scenario_handler.start_episode(files.route, files.scenario, "0")

        print("EPISODE LOADED")

        return

    def step(self, ego_vehicle_action: Action) -> WorldState:
        """
        Runs one step/frame in the simulated scenario, performing the chosen action on the environment
        """

        print("action")
        self.scenario_handler.tick()
        return WorldState(np.array([]), self.scenario_handler.is_running())

    def stop_episode(self):
        return self.scenario_handler.stop_episode()

    def _act_on_environment(self, action: Action):

        return
