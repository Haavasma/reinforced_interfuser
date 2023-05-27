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
from vision_modules.transfuser import TransfuserVisionModule, setup_transfuser_backbone

# rl_config = {"policy_type": "MultiInputPolicy", "total_timesteps": 1_000_000}


class TrainingConfig(TypedDict):
    workers: int
    gpus: int
    resume: bool
    eval: bool
    vision_module: str
    weights: str
    steps: int
    traffic_type: TrafficType


def validate_training_config(config: TrainingConfig) -> None:
    """
    Throws an error if the config is not valid for the given system.
    """

    if config["vision_module"] != "" and config["weights"] == "":
        raise ValueError(
            "If a vision module is specified, a weights file must be specified as well"
        )


def train(config: TrainingConfig) -> None:
    validate_training_config(config)

    run_type = (
        "baseline" if config["vision_module"] == "" else config["vision_module"]
    ) + (f"_{config['traffic_type'].name}")

    run_id = get_run_name(
        run_type,
        resume=config["resume"],
    )

    carla_config: CarlaEnvironmentConfiguration = {
        "speed_goal_actions": [0.0, 2.0, 4.0],
        "steering_actions": np.linspace(-0.5, 0.5, 31).tolist(),
        "discrete_actions": True,
        "continuous_speed_range": (0.0, 6.0),
        "continuous_steering_range": (-0.3, 0.3),
        "towns": None,
        "town_change_frequency": None,
        "concat_images": False,
        "traffic_type": config["traffic_type"],
        "image_resize": (100, 200),
        "blind_ablation": False,
    }

    eval_config: CarlaEnvironmentConfiguration = copy.deepcopy(carla_config)
    eval_config["towns"] = None
    eval_config["town_change_frequency"] = None

    env_name = "carla_env"
    create_env = make_carla_env(
        carla_config,
        eval_config,
        config["gpus"],
        config["vision_module"],
        config["weights"],
        seed=69,
    )
    register_env(env_name, create_env)

    trainer_config = APPOConfig()

    #gpu_fraction = (config["gpus"] / (config["workers"] + (1 if config["workers"] > 1 else 0))) - 0.0001
    fraction = ((config["gpus"]) / (config["workers"] + 2))

    print("FRACTION: ", fraction)

    ModelCatalog.register_custom_model(
        "conditional_net", ConditionalComplexInputNetwork
    )

    algo_config = (
        trainer_config.rollouts(
            num_rollout_workers=config["workers"] if config["workers"] > 1 else 0,
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
        .reporting(min_sample_timesteps_per_iteration=2048)
        .resources(
            num_gpus=fraction,
            num_gpus_per_worker=fraction,
        )
        .environment(env_name, disable_env_checking=True)
        .exploration()
        .training(
            gamma=0.95,
            lr=3e-4,
            model={
                "custom_model": "conditional_net",
                "post_fcnet_hiddens": [1024, 512, 256],
                "conv_filters": [
                    [16, [6, 8], [3, 4]],
                    [32, [6, 6], 4],
                    [256, [9, 9], 1],
                ],
                "use_attention": False,
                "framestack": True,
                "custom_model_config": {"num_conditional_inputs": 6},
            },
        )
        .evaluation(
            evaluation_interval=10,
            evaluation_duration_unit="episodes",
            evaluation_duration=10,
            evaluation_config={"env_config": {"is_eval": True}},
        )
        .callbacks(CustomCallback)
        .framework("torch")
    )

    experiment_dir = os.path.join("./models", run_id)
    should_resume = config["resume"]

    checkpoint = ""

    if should_resume:
        has_checkpoints = False
        if not os.path.exists(experiment_dir):
            print(f"No experiment directory found at {experiment_dir}")
            should_resume = False
        else:
            for trial_dir in os.listdir(experiment_dir):
                trial_path = os.path.join(experiment_dir, trial_dir)
                print(f"TRIAL PATH: {trial_path}")
                if os.path.isdir(trial_path):
                    print(f"SUBDIRS: {os.listdir(trial_path)}")
                    tmp_checkpoints = [
                        entry
                        for entry in os.listdir(trial_path)
                        if entry.startswith("checkpoint_")
                    ]

                    if len(tmp_checkpoints) > 0:
                        tmp_checkpoints.sort()
                        if checkpoint == "" or tmp_checkpoints[-1] > checkpoint:
                            checkpoint = os.path.join(trial_path, tmp_checkpoints[-1])
                            print("NEW CHECKPOINT: ", checkpoint)

                        print(
                            f"Trial {trial_dir} has the following checkpoints: {tmp_checkpoints}"
                        )
                        has_checkpoints = True
                    else:
                        print(f"Trial {trial_dir} has no checkpoints")
        if not has_checkpoints:
            should_resume = False

    print("FINAL CHECKPOINT: ", checkpoint)

    # if not should_resume:
    #     os.makedirs(experiment_dir, exist_ok=True)
    #
    # checkpoint_dir = None
    # print("SETTING UP TRAINER")
    # trainer = CustomAPPO(algo_config)
    #
    # try:
    #     for i in range(50):
    #         print("TRAINING STEP: ", i)
    #         trainer.train()
    #         checkpoint_dir = trainer.save(experiment_dir)
    #
    #     if checkpoint_dir is None:
    #         raise Exception("No checkpoint directory found")
    #
    # except Exception as e:
    #     trainer.stop()
    #     raise e

    # trainer.evaluate(lambda num: 25 - num)

    trainer = CustomAPPO

    tune.run(
        trainer,
        name=run_id,
        config=algo_config.to_dict(),
        stop={"timesteps_total": config["steps"]},
        resume=True if should_resume else False,
        # raise_on_failed_trial=True,
        checkpoint_freq=1,
        checkpoint_at_end=False,
        keep_checkpoints_num=5,
        restore=checkpoint if should_resume else None,
        trial_name_creator=lambda _: run_id,
        # checkpoint_score_attr="episode_reward_mean",
        local_dir="./models/",
        fail_fast="RAISE",
        callbacks=[
            CustomWandbLoggerCallback(
                project="Sensor fusion AD RL",
                group=run_id,
                log_config=True,
                upload_checkpoints=True,
                resume=should_resume,
            ),
        ],
    )


def make_carla_env(
    carla_config: CarlaEnvironmentConfiguration,
    eval_config: CarlaEnvironmentConfiguration,
    gpus: int,
    vision_module_name: str,
    weights_file: str,
    seed: int = 0,
) -> Callable[[Any], gym.Env]:
    def _init(env_config) -> gym.Env:
        i = env_config.worker_index
        print("WORKER INDEX: ", i)

        # is_eval is set for evaluation workers
        evaluation = "is_eval" in env_config and env_config["is_eval"]

        print("EVALUATION: ", evaluation)

        episode_config = baseline_config()
        vision_module = None

        if vision_module_name == "transfuser":
            config = GlobalConfig(setting="eval")
            backbone = setup_transfuser_backbone(
                config, weights_file, device=f"cuda:{i%gpus}"
            )
            vision_module = TransfuserVisionModule(backbone, config)
        elif vision_module_name == "interfuser":
            vision_module = InterFuserVisionModule(
                weights_file,
                use_target_feature=True,
                render_imitation=False,
                postprocess=False,
            )
            episode_config = interfuser_config()

        episode_config.training_type = (
            TrainingType.EVALUATION if evaluation else TrainingType.TRAINING
        )

        episode_manager = EpisodeManager(
            episode_config, gpu_device=i % gpus, server_wait_time=15 + (i * 10)
        )
        speed_controller = TestSpeedController()

        env = CarlaEnvironment(
            eval_config if evaluation else carla_config,
            episode_manager,
            vision_module,
            reward_function,
            speed_controller,
        )
        env.seed(seed + i)

        return env

    return _init


def get_run_name(run_type: str, resume=False) -> str:
    run_id = ""
    if resume:
        with open("./models/run_name.pkl", "rb") as f:
            run_id = pickle.load(f)
    else:
        run_id = f"{run_type}_{uuid.uuid4().hex}"

    with open("./models/run_name.pkl", "wb") as f:
        pickle.dump(run_id, f)

    return run_id


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Parse command line arguments")

    parser.add_argument(
        "--workers", type=int, default=1, help="Amount of workers (default: 1)"
    )

    parser.add_argument(
        "--gpus", type=int, default=1, help="Amount of GPUS available (default: 1)"
    )

    parser.add_argument(
        "--resume", action="store_true", help="Resume training (default: False)"
    )

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

    parser.add_argument("--steps", type=int, default=1_000_000, help="Number of steps")

    parser.add_argument("--weights", type=str, default="", help="Path to weights file")

    args = parser.parse_args()

    workers = args.workers

    gpus = args.gpus

    weights = str(pathlib.Path(args.weights).absolute().resolve())

    steps = int(args.steps)

    no_traffic = args.no_traffic
    no_scenarios = args.no_scenarios

    traffic_type = TrafficType.SCENARIO

    if no_scenarios:
        traffic_type = TrafficType.TRAFFIC
    if no_traffic:
        traffic_type = TrafficType.NO_TRAFFIC

    train(
        {
            "workers": workers,
            "gpus": gpus,
            "resume": bool(args.resume),
            "vision_module": args.vision_module,
            "weights": weights,
            "eval": True,
            "steps": steps,
            "traffic_type": traffic_type,
        }
    )
