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

    for _ in range(400):
        state = manager.step(Action(1.0, 0.0, False, 0.0))
        if state.running is False:
            break
        time.sleep(0.1)

    manager.stop_episode()

    print("MANAGED TO DO SOME")

    return


if __name__ == "__main__":

    main()
