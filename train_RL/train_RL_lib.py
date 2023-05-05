import argparse
import os
import pickle
import uuid
from typing import Any, Callable, List, TypedDict, Union

import gymnasium as gym
from typing import Optional
import numpy as np
from episode_manager import EpisodeManager
from episode_manager.episode_manager import TrainingType
from gym_env.env import (
    CarlaEnvironment,
    CarlaEnvironmentConfiguration,
    TestSpeedController,
)
from gymnasium.wrappers.monitoring.video_recorder import VideoRecorder
from ray import tune
from ray.air.checkpoint import Checkpoint
from ray.air.integrations.wandb import WandbLoggerCallback
from ray.rllib.algorithms.algorithm import Algorithm
from ray.rllib.algorithms.appo import APPO, APPOConfig
from ray.rllib.algorithms.callbacks import DefaultCallbacks
from ray.rllib.algorithms.ppo import PPO
from ray.tune.registry import register_env

from config import GlobalConfig
from episode_configs import baseline_config
from reward_functions.main import reward_function
from vision_modules.transfuser import TransfuserVisionModule, setup_transfuser_backbone


class CustomLogger(tune.Callback):
    def on_train_result(self, *, algorithm: Algorithm, result: dict, **kwargs) -> None:
        print("Training result: ", result)

    def on_episode_end(self, worker, base_env, policies, episode, **kwargs):
        print("EPISODE: ", episode)
        print("WORKER: ", worker)


class CustomPPO(PPO):
    def save(
        self, checkpoint_dir: Optional[str] = None, prevent_upload: bool = False
    ) -> str:
        checkpoint_path = super().save(checkpoint_dir, prevent_upload)
        metrics_path = os.path.join(checkpoint_path, "metrics.pkl")

        with open(metrics_path, "wb") as f:
            pickle.dump(self._progress_metrics, f)

        evaluation_metrics_path = os.path.join(
            checkpoint_path, "evaluation_metrics.pkl"
        )

        with open(evaluation_metrics_path, "wb") as f:
            pickle.dump(self.evaluation_metrics, f)

        return checkpoint_path

    def restore(
        self,
        checkpoint_path: Union[str, Checkpoint],
        checkpoint_node_ip: Optional[str] = None,
        fallback_to_latest: bool = False,
    ):
        super().restore(checkpoint_path, checkpoint_node_ip, fallback_to_latest)
        metrics_path = os.path.join(os.path.dirname(checkpoint_path), "metrics.pkl")

        if os.path.exists(metrics_path):
            with open(metrics_path, "rb") as f:
                self._progress_metrics = pickle.load(f)

        evaluation_metrics_path = os.path.join(
            os.path.dirname(checkpoint_path), "evaluation_metrics.pkl"
        )

        if os.path.exists(evaluation_metrics_path):
            with open(evaluation_metrics_path, "rb") as f:
                self.evaluation_metrics = pickle.load(f)

        return


class CustomAPPO(APPO):
    def save(
        self, checkpoint_dir: Optional[str] = None, prevent_upload: bool = False
    ) -> str:
        checkpoint_path = super().save(checkpoint_dir, prevent_upload)
        metrics_path = os.path.join(checkpoint_path, "metrics.pkl")

        with open(metrics_path, "wb") as f:
            pickle.dump(self._progress_metrics, f)

        evaluation_metrics_path = os.path.join(
            checkpoint_path, "evaluation_metrics.pkl"
        )

        with open(evaluation_metrics_path, "wb") as f:
            pickle.dump(self.evaluation_metrics, f)

        return checkpoint_path

    def restore(
        self,
        checkpoint_path: Union[str, Checkpoint],
        checkpoint_node_ip: Optional[str] = None,
        fallback_to_latest: bool = False,
    ):
        super().restore(checkpoint_path, checkpoint_node_ip, fallback_to_latest)
        metrics_path = os.path.join(os.path.dirname(checkpoint_path), "metrics.pkl")

        if os.path.exists(metrics_path):
            with open(metrics_path, "rb") as f:
                self._progress_metrics = pickle.load(f)

        evaluation_metrics_path = os.path.join(
            os.path.dirname(checkpoint_path), "evaluation_metrics.pkl"
        )

        if os.path.exists(evaluation_metrics_path):
            with open(evaluation_metrics_path, "rb") as f:
                self.evaluation_metrics = pickle.load(f)

        return


# from ray.tune.integration.wandb import WandbLoggerCallback

rl_config = {"policy_type": "MultiInputPolicy", "total_timesteps": 999999}


class TrainingConfig(TypedDict):
    workers: int
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

    if config["vision_module"] != "" and config["weights"] == "":
        raise ValueError(
            "If a vision module is specified, a weights file must be specified as well"
        )


def train(config: TrainingConfig) -> None:
    validate_training_config(config)

    run_id = get_run_name(
        resume=config["resume"],
    )

    carla_config: CarlaEnvironmentConfiguration = {
        "speed_goal_actions": [0.0, 4.5, 6.0],
        "steering_actions": np.linspace(-1.0, 1.0, 31).tolist(),
        "discrete_actions": True,
        "continuous_speed_range": (0.0, 6.0),
        "continuous_steering_range": (-0.3, 0.3),
        "towns": ["Town01"],
        "town_change_frequency": 10,
        "concat_images": True,
    }

    create_env = make_carla_env(
        carla_config,
        config["gpus"],
        config["vision_module"],
        config["weights"],
        seed=69,
    )

    name = "carla_env"
    register_env(name, create_env)

    trainer_config = APPOConfig()  # if config["workers"] > 1 else PPOConfig()

    algo_config = (
        trainer_config.rollouts(
            num_rollout_workers=config["workers"],
            num_envs_per_worker=1,
            recreate_failed_workers=False,
            ignore_worker_failures=False,
            restart_failed_sub_environments=False,
            validate_workers_after_construction=True,
            worker_health_probe_timeout_s=60,
            worker_restore_timeout_s=60,
        )
        .resources(num_gpus=len(set(config["gpus"])))
        .environment(name)
        .training()
        .framework("torch")
    )

    trainer = CustomAPPO  # if config["workers"] > 1 else PPO
    tune.run(
        trainer,
        name=run_id,
        config=algo_config.to_dict(),
        stop={"timesteps_total": 1_000_000},
        resume="LOCAL+ERRORED" if config["resume"] else False,
        # raise_on_failed_trial=False,
        checkpoint_freq=1,
        checkpoint_at_end=True,
        local_dir="./models/",
        callbacks=[
            WandbLoggerCallback(
                project="Sensor fusion AD RL",
                log_config=True,
                upload_checkpoints=True,
                resume=config["resume"],
            ),
        ],
    )


def make_carla_env(
    carla_config: CarlaEnvironmentConfiguration,
    gpus: List[int],
    vision_module_name: str,
    weights_file: str,
    seed: int = 0,
    evaluation: bool = False,
) -> Callable[[Any], gym.Env]:
    def _init(env_config) -> gym.Env:
        i = env_config.worker_index - 1
        print("WORKER INDEX: ", i)
        episode_config = baseline_config()
        episode_config.training_type = (
            TrainingType.EVALUATION if evaluation else TrainingType.TRAINING
        )
        episode_manager = EpisodeManager(episode_config)
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


class CustomRLlibCallback(DefaultCallbacks):
    def on_episode_start(
        self, worker, base_env, policies, episode, env_index, **kwargs
    ):
        if not hasattr(self, "video_recorder"):
            self.video_recorder = None

        if self.video_recorder is None:
            env = base_env.get_sub_environments(env_index)
            self.video_recorder = VideoRecorder(
                env, base_path=os.path.join(worker.log_dir, "sample_videos")
            )

    def on_episode_step(self, worker, base_env, episode, env_index, **kwargs):
        print("ON EPISODE STEP: ", env_index)
        if self.video_recorder:
            base_env.get_sub_environments(env_index)
            self.video_recorder.capture_frame()

    def on_episode_end(self, worker, base_env, policies, episode, env_index, **kwargs):
        episode.custom_metrics["mean_episode_reward"] = np.mean(
            episode.batch_builder.policy_batches["default_policy"]["rewards"]
        )
        episode.custom_metrics["episode_length"] = len(
            episode.batch_builder.policy_batches["default_policy"]["rewards"]
        )

        if self.video_recorder:
            self.video_recorder.close()
            self.video_recorder = None

    def on_train_result(self, trainer, result, **kwargs):
        print("GOT RESULT: ", result["custom_metrics"])
        result["custom_metrics"]["mean_episode_reward"] = np.mean(
            [m["mean_episode_reward"] for m in result["hist_stats"]["custom_metrics"]]
        )
        result["custom_metrics"]["episode_length"] = np.mean(
            [m["episode_length"] for m in result["hist_stats"]["custom_metrics"]]
        )

        if not os.path.exists(os.path.join(trainer.logdir, "sample_videos")):
            os.makedirs(os.path.join(trainer.logdir, "sample_videos"))

        if self.video_recorder:
            self.video_recorder.enabled = False
            trainer.workers.foreach_worker(
                lambda worker: worker.foreach_env(lambda env: env.close())
            )
            self.video_recorder = None


def get_run_name(resume=False) -> str:
    run_id = ""
    if resume:
        with open("./models/run_name.pkl", "rb") as f:
            run_id = pickle.load(f)
    else:
        run_id = uuid.uuid4().hex

    with open("./models/run_name.pkl", "wb") as f:
        pickle.dump(run_id, f)

    return run_id


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Parse command line arguments")

    parser.add_argument(
        "--workers", type=int, default=1, help="Amount of workers (default: 1)"
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

    workers = args.workers

    _ = [x.strip() for x in "".split(",")]

    train(
        {
            "workers": workers,
            "gpus": [0 for _ in range(workers)],
            "resume": bool(args.resume),
            "vision_module": args.vision_module,
            "weights": args.weights,
            "eval": True,
            "steps": 1_000_000,
        }
    )
