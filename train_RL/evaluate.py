import argparse
import copy
import os
import pathlib
import pickle
import uuid
from typing import Any, Callable, TypedDict

import gymnasium as gym
import numpy as np
from episode_manager import EpisodeManager
from episode_manager.data import TrafficType
from episode_manager.episode_manager import TrainingType
from ray import tune
from ray.rllib.algorithms.appo import APPOConfig
from ray.rllib.models import ModelCatalog
from ray.tune.registry import register_env

from config import GlobalConfig
from episode_configs import baseline_config, interfuser_config
from gym_env.env import (
    CarlaEnvironment,
    CarlaEnvironmentConfiguration,
    TestSpeedController,
)
from reward_functions.main import reward_function
from rl_lib.appo import CustomAPPO
from rl_lib.callback import CustomCallback
from rl_lib.complex_input_network import ConditionalComplexInputNetwork
from rl_lib.wandb_logging import CustomWandbLoggerCallback
from vision_modules.interfuser import InterFuserVisionModule
from vision_modules.interfuser_pretrained import InterFuserPretrainedVisionModule

# rl_config = {"policy_type": "MultiInputPolicy", "total_timesteps": 1_000_000}


class EvaluationConfig(TypedDict):
    vision_module: str
    weights: str
    traffic_type: TrafficType
    blind_ablation: bool
    checkpoint: str


def validate_training_config(config: EvaluationConfig) -> None:
    """
    Throws an error if the config is not valid for the given system.
    """

    if config["vision_module"] != "" and config["weights"] == "":
        raise ValueError(
            "If a vision module is specified, a weights file must be specified as well"
        )


def train(config: EvaluationConfig) -> None:
    validate_training_config(config)

    run_type = (
        (
            "baseline" if config["vision_module"] == "" else config["vision_module"]
        )  # Vision type
        + (f"_{config['traffic_type'].name}")  # Traffic type
        + (f"{'_blind' if config['blind_ablation'] else ''}")  # Blind ablation
    )

    run_id = get_run_name(
        run_type,
    )

    # run_id = "baseline_NO_TRAFFIC_5b6f404e2a0447839d0e52460fc7c12c"

    carla_config: CarlaEnvironmentConfiguration = {
        "speed_goal_actions": [0.0, 2.0, 4.0],
        "steering_actions": np.linspace(-1.0, 1.0, 31).tolist(),
        "discrete_actions": True,
        "continuous_speed_range": (0.0, 6.0),
        "continuous_steering_range": (-0.3, 0.3),
        "towns": None,
        "town_change_frequency": None,
        "concat_images": False,
        "traffic_type": config["traffic_type"],
        "image_resize": (100, 200),
    }

    env_name = "carla_env"
    create_env = make_carla_env(
        carla_config,
        config["vision_module"],
        config["weights"],
        seed=69,
        blind=config["blind_ablation"],
    )
    register_env(env_name, create_env)

    trainer_config = APPOConfig()

    # gpu_fraction = (config["gpus"] / (config["workers"] + (1 if config["workers"] > 1 else 0))) - 0.0001

    ModelCatalog.register_custom_model(
        "conditional_net", ConditionalComplexInputNetwork
    )

    conv_filters = [
        [16, [6, 8], [3, 4]],
        [32, [6, 6], 4],
        [256, [9, 9], 1],
    ]

    if config["vision_module"] == "interfuser_pretrained":
        conv_filters = [
            [16, [5, 5], 2],
            [32, [5, 5], 2],
            [256, [5, 5], 2],
        ]

    algo_config = (
        trainer_config.rollouts(
            num_rollout_workers=0,
            num_envs_per_worker=1,
            create_env_on_local_worker=True,
            recreate_failed_workers=False,
            ignore_worker_failures=False,
            restart_failed_sub_environments=False,
            validate_workers_after_construction=True,
            worker_health_probe_timeout_s=60,
            worker_restore_timeout_s=60,
            num_consecutive_worker_failures_tolerance=0,
        )
        .reporting(min_sample_timesteps_per_iteration=50)
        .resources(
            num_gpus=1,
        )
        .environment(env_name, disable_env_checking=True)
        .exploration()
        .training(
            gamma=0.95,
            lr=3e-4,
            model={
                "custom_model": "conditional_net",
                "post_fcnet_hiddens": [1024, 512, 256],
                "conv_filters": conv_filters,
                "use_attention": False,
                "framestack": True,
                "custom_model_config": {"num_conditional_inputs": 6},
            },
        )
        .evaluation(
            evaluation_interval=1,
            evaluation_duration_unit="episodes",
            evaluation_duration=25,
            evaluation_config={"env_config": {"is_eval": True}},
        )
        .callbacks(CustomCallback)
        .framework("torch")
    )

    checkpoint = config["checkpoint"]

    trainer = CustomAPPO

    tune.run(
        trainer,
        name=run_id,
        config=algo_config.to_dict(),
        stop={"training_iteration": 2},
        # resume=True if should_resume else False,
        # raise_on_failed_trial=True,
        checkpoint_freq=1,
        checkpoint_at_end=False,
        keep_checkpoints_num=5,
        restore=checkpoint,
        trial_name_creator=lambda _: run_id,
        # checkpoint_score_attr="episode_reward_mean",
        local_dir="./models/",
        fail_fast="RAISE",
        callbacks=[
            # CustomWandbLoggerCallback(
            #     project="Sensor fusion AD RL",
            #     group=run_id,
            #     log_config=True,
            #     upload_checkpoints=True,
            # ),
        ],
    )


def make_carla_env(
    carla_config: CarlaEnvironmentConfiguration,
    vision_module_name: str,
    weights_file: str,
    seed: int = 0,
    blind: bool = False,
) -> Callable[[Any], gym.Env]:
    def _init(env_config) -> gym.Env:
        i = env_config.worker_index
        print("WORKER INDEX: ", i)

        # is_eval is set for evaluation workers
        episode_config = baseline_config()
        if blind:
            print("BLIND MODE: ")
            episode_config.car_config.cameras = []
            episode_config.car_config.lidar["enabled"] = False

        vision_module = None
        if vision_module_name == "interfuser":
            vision_module = InterFuserVisionModule(
                weights_file,
                use_target_feature=False,
                render_imitation=False,
                postprocess=False,
            )
            episode_config = interfuser_config()

        elif vision_module_name == "interfuser_pretrained":
            vision_module = InterFuserPretrainedVisionModule(
                weights_file,
                use_target_feature=True,
                use_imitation_action=True,
                render_imitation=False,
                postprocess=False,
            )
            episode_config = interfuser_config()

        episode_config.training_type = TrainingType.EVALUATION

        episode_manager = EpisodeManager(
            episode_config, gpu_device=0, server_wait_time=15 + (i * 10)
        )
        speed_controller = TestSpeedController()

        env = CarlaEnvironment(
            carla_config,
            episode_manager,
            vision_module,
            reward_function,
            speed_controller,
            render_mode="vision_module",
        )
        env.seed(seed + i)

        return env

    return _init


def get_run_name(run_type: str) -> str:
    return f"{run_type}_{uuid.uuid4().hex}"


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Parse command line arguments")

    parser.add_argument(
        "--vision-module",
        type=str,
        default="",
        help="Vision module, (transfuser, interfuser) (default: None)",
    )

    parser.add_argument(
        "--no-traffic",
        action="store_true",
        help="Deactivates traffic for the training (default: False)",
    )

    parser.add_argument(
        "--no-scenarios",
        action="store_true",
        help="deactivates challenging scenarios during training (default: False)",
    )

    parser.add_argument(
        "--blind",
        action="store_true",
        help="Removes all sensors",
    )

    parser.add_argument("--weights", type=str, default="", help="Path to weights file")
    parser.add_argument(
        "--checkpoint", type=str, default="", help="Path to model checkpoint"
    )

    args = parser.parse_args()

    weights = str(pathlib.Path(args.weights).absolute().resolve())

    no_traffic = args.no_traffic
    no_scenarios = args.no_scenarios

    traffic_type = TrafficType.SCENARIO

    if no_scenarios:
        traffic_type = TrafficType.TRAFFIC
    if no_traffic:
        traffic_type = TrafficType.NO_TRAFFIC

    train(
        {
            "vision_module": args.vision_module,
            "weights": weights,
            "traffic_type": traffic_type,
            "blind_ablation": args.blind,
            "checkpoint": args.checkpoint,
        }
    )
