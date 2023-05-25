import os
import pickle
from typing import Optional, Union

from ray.air.checkpoint import Checkpoint
from ray.rllib.algorithms.ppo import PPO


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
