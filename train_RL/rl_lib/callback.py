import os
import pathlib
import time
import uuid
from typing import Any, Dict, Optional, Union

from gymnasium.wrappers.monitoring.video_recorder import VideoRecorder
from ray.rllib.algorithms.algorithm import Algorithm
from ray.rllib.algorithms.callbacks import DefaultCallbacks
from ray.rllib.env.base_env import BaseEnv
from ray.rllib.evaluation import RolloutWorker
from ray.rllib.evaluation.episode import Episode
from ray.rllib.evaluation.episode_v2 import EpisodeV2
from ray.rllib.policy import Policy
from ray.rllib.utils.typing import PolicyID
import random

N_EPISODES_PER_VIDEO_ITERATION = 50


class CustomCallback(DefaultCallbacks):
    # episode_iteration: Dict[int, int] = {}

    def __init__(self, legacy_callbacks_dict: Dict[str, Any] = None):
        path = str(pathlib.Path("./videos/").absolute().resolve())

        self.path = str(pathlib.Path(path).absolute().resolve())

        self.video_recorder: Optional[VideoRecorder] = None
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

        # iteration = self.episode_iteration.get(index, 0) + 1

        if random.randint(1, N_EPISODES_PER_VIDEO_ITERATION) == 1:
            if self.video_recorder is None:
                env = base_env.get_sub_environments()[index]

                self.video_recorder = VideoRecorder(
                    env,
                    base_path=os.path.join(
                        self.path, f"{int(time.time())}_{uuid.uuid4()}"
                    ),
                )

        return super().on_episode_start(
            worker=worker,
            base_env=base_env,
            policies=policies,
            episode=episode,
            env_index=env_index,
            **kwargs,
        )

    def on_episode_step(self, worker, base_env, episode, env_index, **kwargs):
        if self.video_recorder is not None:
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
        if self.video_recorder is not None:
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

    def on_evaluate_end(
        self, *, algorithm: Algorithm, evaluation_metrics: dict, **kwargs
    ) -> None:
        return super().on_evaluate_end(
            algorithm=algorithm, evaluation_metrics=evaluation_metrics, **kwargs
        )

    # def on_evaluate_start(self, *, algorithm: Algorithm, **kwargs) -> None:
    #     if algorithm.workers is not None:
    #         algorithm.workers.foreach_env(lambda env: env.close())
    #
    #     return super().on_evaluate_start(algorithm=algorithm, **kwargs)
