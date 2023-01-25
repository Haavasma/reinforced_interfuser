from dataclasses import dataclass

import numpy as np
from scenario_runner import ScenarioRunner


@dataclass
class EpisodeManagerConfiguration:
    port: int


@dataclass
class WorldState:
    sensor_data: np.ndarray


@dataclass
class Action:
    throttle: float
    brake: float
    reverse: bool


class EpisodeManager:
    def __init__(self, config: EpisodeManagerConfiguration):
        print(config)
        args = {}
        self.scenario_runner = ScenarioRunner(args)
        self.scenario_runner.run()

        return

    def start_episode(self, evaluation=False):
        """
        Starts a new route in the simulator,
        """

        print(evaluation)

        return

    def step(self, ego_vehicle_action: Action) -> WorldState:
        """
        Runs one step/frame in the simulated scenario, performing the chosen action on the environment
        """
        print(ego_vehicle_action)
        raise NotImplementedError

        # return WorldState(np.array([]))
