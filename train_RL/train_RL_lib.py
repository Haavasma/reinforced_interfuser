import argparse
import glob
import os
import pathlib
import pickle
import uuid
import time
import copy
from typing import Any, Callable, Dict, List, TypedDict, Union

import gymnasium as gym
from typing import Optional
import numpy as np
from episode_manager import EpisodeManager
from episode_manager.episode_manager import TrainingType
from ray.rllib.env.base_env import BaseEnv
from ray.rllib.evaluation import RolloutWorker
from ray.rllib.evaluation.episode import Episode
from ray.rllib.evaluation.episode_v2 import EpisodeV2
from ray.rllib.policy import Policy
from ray.rllib.policy.sample_batch import SampleBatch
from ray.rllib.utils.typing import PolicyID, ResultDict
from ray.tune.experiment import Trial
from ray.tune.experiment.trial import Trial
import wandb
from gym_env.env import (
    CarlaEnvironment,
    CarlaEnvironmentConfiguration,
    TestSpeedController,
)
from gymnasium.wrappers.monitoring.video_recorder import VideoRecorder
from ray import tune
from ray.air.checkpoint import Checkpoint
from ray.air.integrations.wandb import WandbLoggerCallback, _QueueItem
from ray.rllib.algorithms.algorithm import Algorithm
from ray.rllib.algorithms.appo import APPO, APPOConfig
from ray.rllib.algorithms.callbacks import DefaultCallbacks
from ray.rllib.algorithms.ppo import PPO
from ray.tune.registry import register_env

from config import GlobalConfig
from episode_configs import baseline_config, interfuser_config
from reward_functions.main import reward_function
from vision_modules.interfuser import InterFuserVisionModule
from vision_modules.transfuser import TransfuserVisionModule, setup_transfuser_backbone


N_EPISODES_PER_VIDEO_ITERATION = 5


# class VideoUploader(tune.Callback):
#     def on_step_end(self, iteration: int, trials: List[Trial], **info):
#         videos = glob.glob("./videos/*.mp4")
#
#         print("VIDEOS: ", videos)
#         for video in videos:
#             print("LOGGING VIDEO: ", video)
#             wandb.log({"video": wandb.Video(video)})
#
#         for video in videos:
#             os.remove(video)
#
#         return super().on_step_end(iteration, trials, **info)


class CustomWandbLoggerCallback(WandbLoggerCallback):
    def log_trial_result(self, iteration: int, trial: Trial, result: Dict):
        videos = glob.glob("./videos/*.mp4")
        videos.sort()
        if len(videos) > 0:
            for video in videos:
                # move videos to trial specific folder
                video_name = os.path.basename(video)
                video_path = os.path.join(str(trial.logdir), "videos", video_name)
                pathlib.Path(os.path.dirname(video_path)).mkdir(
                    parents=True, exist_ok=True
                )

                print(f"MOVING {video} to {video_path}")
                os.rename(video, video_path)
                self._trial_queues[trial].put(
                    (_QueueItem.VIDEO, wandb.Video(video_path, fps=10, format="mp4"))
                )

        return super().log_trial_result(iteration, trial, result)

    # def on_trial_save(self, iteration: int, trials: List[Trial], trial: Trial, **info):
    #     print("UPLOADING VIDEOS")
    #     videos = glob.glob("./videos/*.mp4")
    #
    #     for video in videos:
    #         self._trial_queues[trial].put(
    #             (_QueueItem.RESULT, {"video": wandb.Video(video)})
    #         )
    #
    #     return super().on_trial_save(iteration, trials, trial, **info)

    # def on_step_end(self, iteration: int, trials: List[Trial], **info):
    #     return super().on_step_end(iteration, trials, **info)


class CustomCallback(DefaultCallbacks):
    episode_iteration: Dict[int, int] = {}

    def on_episode_start(
        self,
        *,
        worker: RolloutWorker,
        base_env: BaseEnv,
        policies: Dict[PolicyID, Policy],
        episode: Union[Episode, EpisodeV2],
        env_index: Optional[int] = None,
        **kwargs,
    ) -> None:
        # Collect all metrics and average them on the environments
        metrics = {}
        n_sub_envs = len(base_env.get_sub_environments())
        for env in base_env.get_sub_environments():
            for key, value in env.metrics.items():
                if key in metrics:
                    metrics[key] += value
                else:
                    metrics[key] = value

        for key, value in metrics.items():
            episode.custom_metrics[key] = value / n_sub_envs

        index = env_index if env_index is not None else 0

        iteration = self.episode_iteration.get(index, 0)

        if iteration % N_EPISODES_PER_VIDEO_ITERATION == 0:
            if not hasattr(self, "video_recorder"):
                self.video_recorder = None

            if self.video_recorder is None:
                env = base_env.get_sub_environments()[index]
                self.video_recorder = VideoRecorder(
                    env,
                    base_path=os.path.join(
                        "./videos/", f"{int(time.time())}_{uuid.uuid4()}"
                    ),
                )

        if index in self.episode_iteration:
            self.episode_iteration[index] += 1
        else:
            self.episode_iteration[index] = 1

        return super().on_episode_start(
            worker=worker,
            base_env=base_env,
            policies=policies,
            episode=episode,
            env_index=env_index,
            **kwargs,
        )

    def on_episode_step(self, worker, base_env, episode, env_index, **kwargs):
        if self.video_recorder:
            base_env.get_sub_environments()[env_index]
            self.video_recorder.capture_frame()

        return super().on_episode_step(
            worker=worker,
            base_env=base_env,
            episode=episode,
            env_index=env_index,
            **kwargs,
        )

    def on_episode_end(self, worker, base_env, policies, episode, env_index, **kwargs):
        if self.video_recorder:
            self.video_recorder.close()
            self.video_recorder = None

        return super().on_episode_end(
            worker=worker,
            base_env=base_env,
            policies=policies,
            episode=episode,
            env_index=env_index,
            **kwargs,
        )


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

    # def after_train_step(self, train_results: ResultDict) -> None:
    #     # iterate all videos set up in the callback:
    #     # for each video, log it to wandb
    #     videos = glob.glob("./videos/*.mp4")
    #     for video in videos:
    #         wandb.log({"Train/video": wandb.Video(video)})
    #
    #     for video in videos:
    #         os.remove(video)
    #
    #     return super().after_train_step(train_results)

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
    gpus: int
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
        "steering_actions": np.linspace(-0.5, 0.5, 31).tolist(),
        "discrete_actions": True,
        "continuous_speed_range": (0.0, 6.0),
        "continuous_steering_range": (-0.3, 0.3),
        "towns": ["Town01", "Town03", "Town04", "Town06"],
        "town_change_frequency": 10,
        "concat_images": True,
    }

    eval_config: CarlaEnvironmentConfiguration = copy.deepcopy(carla_config)
    eval_config["towns"] = ["Town02", "Town04", "Town05"]
    eval_config["town_change_frequency"] = 1

    create_env = make_carla_env(
        carla_config,
        eval_config,
        config["gpus"],
        config["vision_module"],
        config["weights"],
        seed=69,
    )

    name = "carla_env"
    register_env(name, create_env)

    trainer_config = APPOConfig()  # if config["workers"] > 1 else PPOConfig()

    gpu_fraction = (1 / (workers + 1)) - 0.01

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
            num_consecutive_worker_failures_tolerance=0,
        )
        .resources(
            num_gpus=gpu_fraction,
            num_gpus_per_worker=gpu_fraction,
        )
        .environment(name)
        .training(gamma=0.95, lr=1e-4)
        # .evaluation(
        #     evaluation_interval=50,
        #     evaluation_parallel_to_training=workers > 1,
        #     evaluation_num_episodes=5,
        #     evaluation_num_workers=1 if workers > 1 else 0,
        # )
        .callbacks(CustomCallback)
        .framework("torch")
    )

    experiment_dir = os.path.join("./models", run_id)
    should_resume = config["resume"]

    if should_resume:
        checkpoints = False
        if not os.path.exists(experiment_dir):
            print(f"No experiment directory found at {experiment_dir}")
            should_resume = False
        else:
            for trial_dir in os.listdir(experiment_dir):
                trial_path = os.path.join(experiment_dir, trial_dir)
                print(f"TRIAL PATH: {trial_path}")
                if os.path.isdir(trial_path):
                    print(f"SUBDIRS: {os.listdir(trial_path)}")
                    checkpoints = [
                        entry
                        for entry in os.listdir(trial_path)
                        if entry.startswith("checkpoint_")
                    ]

                    if checkpoints:
                        print(
                            f"Trial {trial_dir} has the following checkpoints: {checkpoints}"
                        )
                        checkpoints = True
                    else:
                        print(f"Trial {trial_dir} has no checkpoints")
        if not checkpoints:
            should_resume = False

    trainer = CustomAPPO  # if config["workers"] > 1 else PPO
    tune.run(
        trainer,
        name=run_id,
        config=algo_config.to_dict(),
        stop={"timesteps_total": 100_000},
        resume="LOCAL+ERRORED" if should_resume else False,
        # raise_on_failed_trial=True,
        checkpoint_freq=5,
        checkpoint_at_end=False,
        local_dir="./models/",
        fail_fast="RAISE",
        callbacks=[
            CustomWandbLoggerCallback(
                project="Sensor fusion AD RL",
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
        i = env_config.worker_index - 1

        print("WORKER INDEX: ", i)
        # Worker gets index 0 if it is the evaluator
        evaluation = i < 0

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
        time.sleep(5 * (i + 1))
        episode_manager = EpisodeManager(episode_config, gpu_device=i % gpus)
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


# class CustomRLlibCallback(DefaultCallbacks):
#     episode_iteration: int = 0
#
#     def __init__(self, n_episodes: int = 20):
#         self.episode_iteration = 0
#         self.n_episodes = n_episodes
#         return super().__init__()
#
#     def on_episode_start(
#         self, worker, base_env, policies, episode, env_index, **kwargs
#     ):
#         if self.episode_iteration % self.n_episodes == 0:
#             if not hasattr(self, "video_recorder"):
#                 self.video_recorder = None
#
#             if self.video_recorder is None:
#                 env = base_env.get_sub_environments(env_index)
#                 self.video_recorder = VideoRecorder(
#                     env, base_path=os.path.join(worker.log_dir, "video")
#                 )
#
#         self.episode_iteration += 1
#
#     def on_episode_step(self, worker, base_env, episode, env_index, **kwargs):
#         print("ON EPISODE STEP: ", env_index)
#
#         if self.video_recorder:
#             base_env.get_sub_environments()[env_index]
#             self.video_recorder.capture_frame()
#
#     def on_episode_end(self, worker, base_env, policies, episode, env_index, **kwargs):
#         if self.video_recorder:
#             self.video_recorder.close()
#             wandb.log({"train_video": wandb.Video(self.video_recorder.path)})
#             self.video_recorder = None


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

    parser.add_argument("--steps", type=int, default=1_000_000, help="Number of steps")

    parser.add_argument("--weights", type=str, default="", help="Path to weights file")

    args = parser.parse_args()

    workers = args.workers

    gpus = args.gpus

    weights = str(pathlib.Path(args.weights).absolute().resolve())

    _ = [x.strip() for x in "".split(",")]

    train(
        {
            "workers": workers,
            "gpus": gpus,
            "resume": bool(args.resume),
            "vision_module": args.vision_module,
            "weights": weights,
            "eval": True,
            "steps": 1_000_000,
        }
    )
