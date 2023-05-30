import os
import urllib

from ray import logger
from ray.air.integrations.wandb import (
    _QueueItem,
    _run_wandb_process_run_info_hook,
    _WandbLoggingActor,
)
from typing_extensions import override
import glob
import pathlib
from typing import Dict, List, Optional

from ray.tune.experiment import Trial
from ray.air.integrations.wandb import (
    WandbLoggerCallback,
)
from ray.tune.experiment.trial import Trial

import wandb


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
        path = str(pathlib.Path("./videos/").absolute().resolve())
        video_path = os.path.join(str(trial.logdir), "videos")
        videos = glob.glob(os.path.join(path, "*.mp4")) + glob.glob(
            os.path.join(video_path, "*.mp4")
        )

        videos.sort()
        processed_path = os.path.join(str(trial.logdir), "videos_processed")

        if len(videos) > 0:
            for video in videos:
                # move videos to trial specific processed folder
                video_name = os.path.basename(video)
                processed_file_name = os.path.join(processed_path, video_name)
                pathlib.Path(os.path.dirname(processed_file_name)).mkdir(
                    parents=True, exist_ok=True
                )

                print(f"MOVING {video} to {processed_file_name}")
                os.rename(video, processed_file_name)
                self._trial_queues[trial].put(
                    ("VIDEO", wandb.Video(processed_file_name, fps=10, format="mp4"))
                )

        return super().log_trial_result(iteration, trial, result)
