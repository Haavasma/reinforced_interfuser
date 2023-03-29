import argparse
from copy import deepcopy
from typing import Callable, List, TypedDict

import gym
from episode_manager import EpisodeManager
from gym_env.env import (
    CarlaEnvironment,
    CarlaEnvironmentConfiguration,
    SpeedController,
)
from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import EvalCallback
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.utils import set_random_seed
from stable_baselines3.common.vec_env import DummyVecEnv, SubprocVecEnv

import wandb
from config import GlobalConfig
from episode_configs import baseline_config
from reward_functions.main import reward_function

from vision_modules.transfuser import TransfuserVisionModule, setup_transfuser_backbone
from wandb.integration.sb3 import WandbCallback

rl_config = {"policy_type": "MultiInputPolicy", "total_timesteps": 1000000}


class TrainingConfig(TypedDict):
    ports: List[str]
    traffic_manager_ports: List[str]
    resume: bool
    eval: bool
    vision_module: str
    weights: str


def validate_training_config(config: TrainingConfig) -> None:
    """
    Throws an error if the config is not valid for the given system.
    """
    print("CONFIG: ", config)

    if len(config["ports"]) != len(config["traffic_manager_ports"]):
        raise ValueError("The number of ports and traffic manager ports must be equal")

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
        "speed_goal_actions": [-1.0, 0.0, 2.0, 4.0, 5.0],
        "steering_actions": [
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
        ],
        "discrete_actions": True,
        "continuous_speed_range": (0, 0),
        "continuous_steering_range": (0, 0),
        "towns": ["Town01", "Town02"],
        "town_change_frequency": 10,
    }

    environments: List[Callable[[], gym.Env]] = []

    for i in range(len(config["ports"])):
        environments.append(
            make_carla_env(
                carla_config,
                int(config["ports"][i]),
                int(config["traffic_manager_ports"][i]),
                config["vision_module"],
                config["weights"],
                i,
                seed=69,
            )
        )

    env = SubprocVecEnv(environments)
    run = init_wandb(resume=config["resume"], name=experiment_name)

    wandb_callback = WandbCallback(
        # gradient_save_freq=10,
        model_save_path=f"./models/{run.id}/",
        model_save_freq=2048,
    )

    # eval_callback = EvalCallback(
    #     env,
    #     best_model_save_path=f"./models/{run.id}/best_model/",
    #     log_path=f"./models/{run.id}/logs/",
    #     eval_freq=10240,
    #     deterministic=True,
    #     render=False,
    #     n_eval_episodes=5,
    #     verbose=1,
    # )

    policy_kwargs = dict(net_arch=[1024, 512, dict(vf=[256], pi=[256])])

    rl_model = PPO(
        rl_config["policy_type"],
        env,
        verbose=2,
        gamma=0.95,
        n_steps=1024,
        # buffer_size=20_000,
        learning_rate=1e-4,
        # tau=0.005,
        # action_noise=NormalActionNoise(np.array([1.0, 0.0]), np.array([0.5, 0.3])),
        # tensorboard_log=f"runs/{run.id}",
        policy_kwargs=policy_kwargs,
        device="cuda",
    )

    # rl_model.load(f"./models/{run_id}/best_model/best_model.zip")
    # rl_model = PPO.load(f"./models/{run_id}/best_model/best_model", env=env)

    rl_model.save(f"./models/{run.id}/model")

    # obs = env.reset()
    # if eval:
    #     for _ in range(30000):
    #         action, _states = rl_model.predict(obs, deterministic=True)
    #         obs, rewards, dones, info = env.step(action)
    #         if dones:
    #             obs = env.reset()
    #
    #     return

    rl_model.learn(total_timesteps=50_000, callback=[wandb_callback])
    # rl_model.save(f"./models/{run.id}/model")

    return


def init_wandb(resume=False, name=None):
    return wandb.init(
        resume=resume,
        config=rl_config,
        name=name,
        monitor_gym=True,
        project="carla-rl",
        entity="haavasma",
        sync_tensorboard=True,
    )


def make_carla_env(
    carla_config: CarlaEnvironmentConfiguration,
    port: int,
    traffic_manager_port: int,
    vision_module_name: str,
    weights_file: str,
    rank: int,
    seed: int = 0,
) -> Callable[[], gym.Env]:
    def _init() -> gym.Env:
        episode_manager = EpisodeManager(
            baseline_config(port=port, traffic_manager_port=traffic_manager_port)
        )
        speed_controller = SpeedController()

        vision_module = None

        if vision_module_name == "transfuser":
            config = GlobalConfig(setting="eval")
            backbone = setup_transfuser_backbone(
                config, weights_file, device=f"cuda:{rank}"
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

        env.seed(seed + rank)
        return env

    set_random_seed(seed)
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
    parser.add_argument("--weights", type=str, default="", help="Path to weights file")

    args = parser.parse_args()

    _ = [x.strip() for x in "".split(",")]

    train(
        {
            "ports": [port.strip() for port in (args.ports).split(",")],
            "traffic_manager_ports": [
                port.strip() for port in (args.traffic_manager_ports).split(",")
            ],
            "resume": bool(args.resume),
            "vision_module": args.vision_module,
            "weights": args.weights,
            "eval": True,
        }
    )
