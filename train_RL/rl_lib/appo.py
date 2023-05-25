import os
import pickle
from typing import Callable, Optional, Union

from episode_manager.data import TrainingType
from ray.air.checkpoint import Checkpoint
from ray.rllib import RolloutWorker
from ray.rllib.algorithms.appo import APPO


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

    def train(self):
        return super().train()

    # def train(self):
    #     value = super().train()
    #     if self.workers is not None:
    #         self.workers.foreach_env(
    #             lambda env: env.stop_server() if hasattr(env, "stop_server") else []
    #         )
    #
    #     return value

    # def _before_evaluate(self):

    # if self.evaluation_workers is not None:
    #     self.evaluation_workers.foreach_env(
    #         lambda env: env.start_server() if hasattr(env, "start") else []
    #     )

    def stop_servers(self):
        if self.workers is not None:
            self.workers.foreach_env(
                lambda env: env.stop_server() if hasattr(env, "stop_server") else []
            )

        if self.evaluation_workers is not None:
            self.evaluation_workers.foreach_env(
                lambda env: env.stop_server() if hasattr(env, "stop_server") else []
            )

    def cleanup(self) -> None:
        print("CLEANING UP")

        if self.workers is not None:
            self.workers.foreach_env(
                lambda env: env.stop_server() if hasattr(env, "stop_server") else []
            )

        if self.evaluation_workers is not None:
            self.evaluation_workers.foreach_env(
                lambda env: env.stop_server() if hasattr(env, "stop_server") else []
            )

        return super().cleanup()

    def evaluate(self, duration_fn: Optional[Callable[[int], int]] = None) -> dict:
        if self.workers is not None:
            self.workers.foreach_env(
                lambda env: env.stop_server() if hasattr(env, "stop_server") else []
            )
            self.workers.foreach_env(lambda env: env.set_mode(TrainingType.EVALUATION))

        if self.evaluation_workers is not None:
            self.evaluation_workers.foreach_env(
                lambda env: env.set_mode(TrainingType.EVALUATION)
            )

        print("EVALUATING")
        value = super().evaluate(duration_fn)
        print("FINISHED EVALUATING")

        if self.workers is not None:
            self.workers.foreach_env(lambda env: env.set_mode(TrainingType.TRAINING))
            self.workers.foreach_env(
                lambda env: env.stop_server() if hasattr(env, "stop_server") else []
            )

        if self.evaluation_workers is not None:
            self.evaluation_workers.foreach_env(
                lambda env: env.set_mode(TrainingType.TRAINING)
            )
            self.evaluation_workers.foreach_env(
                lambda env: env.stop_server() if hasattr(env, "stop_server") else []
            )

        return value
        # if self.workers is not None:
        #     self.workers.foreach_env(
        #         lambda env: env.start_server() if hasattr(env, "start") else []
        #     )
