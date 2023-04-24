from ray import tune
import argparse
import numpy as np
from ray.rllib.algorithms.appo import APPO, APPOConfig
from ray.rllib.algorithms.apex_dqn.apex_dqn import ApexDQNConfig, ApexDQN
from ray.tune.registry import register_env
from typing import Any, Callable, List, Optional, TypedDict

import gymnasium as gym
from episode_manager import EpisodeManager
from gym_env.env import (
    CarlaEnvironment,
    CarlaEnvironmentConfiguration,
    TestSpeedController,
)
from matplotlib.pyplot import Subplot
from config import GlobalConfig
from episode_configs import baseline_config
from reward_functions.main import reward_function
from ray.air.integrations.wandb import WandbLoggerCallback

from vision_modules.transfuser import TransfuserVisionModule, setup_transfuser_backbone

rl_config = {"policy_type": "MultiInputPolicy", "total_timesteps": 1000000}


class TrainingConfig(TypedDict):
    ports: List[str]
    traffic_manager_ports: List[str]
    gpus: List[int]
    resume: bool
    eval: bool
    vision_module: str
    weights: str
    steps: int


def validate_training_config(config: TrainingConfig) -> None:
    """
    Throws an error if the config is not valid for the given system.
    """
    print("CONFIG: ", config)

    if len(config["ports"]) != len(config["traffic_manager_ports"]):
        raise ValueError("The number of ports and traffic manager ports must be equal")

    if len(config["ports"]) != len(config["gpus"]):
        raise ValueError("The number of ports and gpus must be equal")

    if config["vision_module"] != "" and config["weights"] == "":
        raise ValueError(
            "If a vision module is specified, a weights file must be specified as well"
        )

    for port, traffic_manager_port in zip(
        config["ports"], config["traffic_manager_ports"]
    ):
        if not port.isdigit():
            raise ValueError(f"Port {port} is not a valid port number")
        if not traffic_manager_port.isdigit():
            raise ValueError(
                f"Traffic manager port {traffic_manager_port} is not a valid port number"
            )


def train(config: TrainingConfig) -> None:
    experiment_name = "Scenario runner training baseline agent"

    validate_training_config(config)

    carla_config: CarlaEnvironmentConfiguration = {
        "speed_goal_actions": [0.0, 4.5, 6.0],
        "steering_actions": np.linspace(-1.0, 1.0, 31).tolist(),
        "discrete_actions": True,
        "continuous_speed_range": (0.0, 6.0),
        "continuous_steering_range": (-0.3, 0.3),
        "towns": ["Town01"],
        "town_change_frequency": 10,
    }

    create_env = make_carla_env(
        carla_config,
        config["ports"],
        config["traffic_manager_ports"],
        config["gpus"],
        config["vision_module"],
        config["weights"],
        seed=69,
    )

    name = "carla_env"
    register_env(name, create_env)

    algo_config = (
        APPOConfig()
        .rollouts(
            num_rollout_workers=len(config["ports"]),
            num_envs_per_worker=1,
            recreate_failed_workers=True,
            restart_failed_sub_environments=False,
            worker_health_probe_timeout_s=60,
            worker_restore_timeout_s=60,
        )
        .resources(num_gpus=1)
        .environment(name)
        .training()
        .framework("torch")
    )

    tune.run(
        APPO,
        config=algo_config.to_dict(),
        stop={"timesteps_total": 100_000},
        callbacks=[WandbLoggerCallback(project=experiment_name, log_config=True)],
    )


def make_carla_env(
    carla_config: CarlaEnvironmentConfiguration,
    ports: List[str],
    traffic_manager_ports: List[str],
    gpus: List[int],
    vision_module_name: str,
    weights_file: str,
    seed: int = 0,
) -> Callable[[Any], gym.Env]:
    def _init(env_config) -> gym.Env:
        i = env_config.worker_index - 1
        print("WORKER INDEX: ", i)
        episode_manager = EpisodeManager(
            baseline_config(
                port=int(ports[i]), traffic_manager_port=int(traffic_manager_ports[i])
            )
        )
        speed_controller = TestSpeedController()

        vision_module = None

        if vision_module_name == "transfuser":
            config = GlobalConfig(setting="eval")
            backbone = setup_transfuser_backbone(
                config, weights_file, device=f"cuda:{gpus[i]}"
            )
            vision_module = TransfuserVisionModule(backbone, config)

        elif vision_module == "interfuser":
            raise NotImplementedError("Interfuser not implemented yet")

        env = CarlaEnvironment(
            carla_config,
            episode_manager,
            vision_module,
            reward_function,
            speed_controller,
        )
        env.seed(seed + i)

        return env

    return _init


class ScriptArguments(TypedDict):
    port: List[int]
    traffic_manager_ports: List[int]
    gpus: List[int]


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Parse command line arguments")

    parser.add_argument(
        "--ports", type=str, default="2000", help="Port number (default: 2000)"
    )
    parser.add_argument(
        "--traffic-manager-ports",
        type=str,
        default="8000",
        help="Traffic manager port number (default: 8000, example: 8000,8001)",
    )

    parser.add_argument(
        "--resume", action="store_true", help="Resume training (default: False)"
    )

    parser.add_argument(
        "--vision-module",
        type=str,
        default="",
        help="Vision module (default: None)",
    )

    parser.add_argument("--steps", type=int, default=1_000_000, help="Number of steps")

    parser.add_argument("--weights", type=str, default="", help="Path to weights file")

    args = parser.parse_args()

    _ = [x.strip() for x in "".split(",")]

    ports = [x.strip() for x in (args.ports).split(",")]
    traffic_manager_ports = [x.strip() for x in (args.traffic_manager_ports).split(",")]

    train(
        {
            "ports": [port.strip() for port in (args.ports).split(",")],
            "traffic_manager_ports": [
                port.strip() for port in (args.traffic_manager_ports).split(",")
            ],
            "gpus": [0 for _ in ports],
            "resume": bool(args.resume),
            "vision_module": args.vision_module,
            "weights": args.weights,
            "eval": True,
            "steps": 1_000_000,
        }
    )
