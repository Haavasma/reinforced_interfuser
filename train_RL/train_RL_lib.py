import argparse
import copy
import glob
import os
import pathlib
import pickle
import time
import urllib
import uuid
from typing import Any, Callable, Dict, List, Optional, TypedDict, Union
from episode_manager.data import TrafficType

import gymnasium as gym
import numpy as np
from episode_manager import EpisodeManager
from episode_manager.episode_manager import TrainingType
from gymnasium.wrappers.monitoring.video_recorder import VideoRecorder
from ray import logger, tune
from ray.air.checkpoint import Checkpoint
from ray.air.integrations.wandb import (
    WandbLoggerCallback,
    _QueueItem,
    _run_wandb_process_run_info_hook,
    _WandbLoggingActor,
)
from ray.rllib.algorithms.appo import APPO, APPOConfig
from ray.rllib.algorithms.callbacks import DefaultCallbacks
from ray.rllib.algorithms.ppo import PPO
from ray.rllib.env.base_env import BaseEnv
from ray.rllib.evaluation import RolloutWorker
from ray.rllib.evaluation.episode import Episode
from ray.rllib.evaluation.episode_v2 import EpisodeV2
from ray.rllib.policy import Policy
from ray.rllib.utils.typing import PolicyID
from ray.tune.experiment import Trial
from ray.tune.registry import register_env
from typing_extensions import override

import wandb
from config import GlobalConfig
from episode_configs import baseline_config, interfuser_config
from gym_env.env import (
    CarlaEnvironment,
    CarlaEnvironmentConfiguration,
    TestSpeedController,
)
from reward_functions.main import reward_function
from vision_modules.interfuser import InterFuserVisionModule
from vision_modules.transfuser import TransfuserVisionModule, setup_transfuser_backbone

N_EPISODES_PER_VIDEO_ITERATION = 10


class CustomWandbLoggingActor(_WandbLoggingActor):
    @override
    def run(self):
        # Since we're running in a separate process already, use threads.
        os.environ["WANDB_START_METHOD"] = "thread"
        run = self._wandb.init(*self.args, **self.kwargs)
        run.config.trial_log_path = self._logdir

        _run_wandb_process_run_info_hook(run)

        while True:
            item_type, item_content = self.queue.get()
            if item_type == _QueueItem.END:
                break

            if item_type == _QueueItem.CHECKPOINT:
                self._handle_checkpoint(item_content)
                continue

            if item_type == "VIDEO":
                print("GOT VIDEO TYPE: ", item_content)
                self._wandb.log({"video": item_content})
                continue

            assert item_type == _QueueItem.RESULT
            log, config_update = self._handle_result(item_content)

            try:
                self._wandb.config.update(config_update, allow_val_change=True)
                self._wandb.log(log)
            except urllib.error.HTTPError as e:
                # Ignore HTTPError. Missing a few data points is not a
                # big issue, as long as things eventually recover.
                logger.warn("Failed to log result to w&b: {}".format(str(e)))
        self._wandb.finish()


class CustomWandbLoggerCallback(WandbLoggerCallback):
    def __init__(
        self,
        project: Optional[str] = None,
        group: Optional[str] = None,
        api_key_file: Optional[str] = None,
        api_key: Optional[str] = None,
        excludes: Optional[List[str]] = None,
        log_config: bool = False,
        upload_checkpoints: bool = False,
        save_checkpoints: bool = False,
        **kwargs,
    ):
        self._logger_actor_cls = CustomWandbLoggingActor
        super().__init__(
            project,
            group,
            api_key_file,
            api_key,
            excludes,
            log_config,
            upload_checkpoints,
            save_checkpoints,
            **kwargs,
        )

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
                    ("VIDEO", wandb.Video(video_path, fps=10, format="mp4"))
                )

        return super().log_trial_result(iteration, trial, result)


class CustomCallback(DefaultCallbacks):
    episode_iteration: Dict[int, int] = {}

    def __init__(self, legacy_callbacks_dict: Dict[str, Any] = None):
        self.path = "./videos/"
        if not os.path.exists(self.path):
            os.makedirs(self.path)

        super().__init__(legacy_callbacks_dict)

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
                        self.path, f"{int(time.time())}_{uuid.uuid4()}"
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
        "concat_images": False,
        "traffic_type": TrafficType.NO_TRAFFIC,
        "concat_size": (240, 320),
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

    gpu_fraction = (config["gpus"] / (workers + (2 if workers > 1 else 1))) - 0.0001

    print("GPU FRACTION: ", gpu_fraction)

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
        .training(
            gamma=0.98,
            lr=1e-5,
            model={
                "fcnet_hiddens": [1024, 512],
                "conv_filters": [
                    [16, [6, 8], [3, 4]],
                    [32, [6, 6], 4],
                    [256, [9, 9], 1],
                ],
                "use_attention": False,
                "framestack": True,
            },
        )
        # .evaluation(
        #     evaluation_interval=10,
        #     evaluation_duration_unit="episodes",
        #     evaluation_parallel_to_training=workers > 1,
        #     evaluation_duration=10,
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
        stop={"timesteps_total": config["steps"]},
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
        episode_manager = EpisodeManager(
            episode_config, gpu_device=i % gpus, server_wait_time=10
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
