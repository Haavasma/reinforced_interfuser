import pathlib
import time

from srunner.scenariomanager.watchdog import thread
from episode_manager.episode_manager import (
    Action,
    CarConfiguration,
    EpisodeManager,
    EpisodeManagerConfiguration,
    TrainingType,
)


def main():

    manager = EpisodeManager(
        EpisodeManagerConfiguration(
            "localhost",
            2000,
            TrainingType.TRAINING,
            pathlib.Path("../routes"),
            CarConfiguration("temp", [], []),
        )
    )

    manager.start_episode()

    for _ in range(20):
        manager.step(Action(0.0, 0.0, False))
        time.sleep(0.1)

    manager.stop_episode()

    print("MANAGED TO DO SOME")

    return


if __name__ == "__main__":

    main()
