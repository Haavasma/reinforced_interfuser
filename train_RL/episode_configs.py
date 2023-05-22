from episode_manager import EpisodeManagerConfiguration
from episode_manager.episode_manager import (
    CarConfiguration,
    Location,
    Rotation,
    Transform,
)
from ray.rllib.evaluation.episode import Episode


_transfuser_img_size = (960, 480)
_fov = 103

TRANSFUSER_CONFIG = EpisodeManagerConfiguration(
    render_client=False,
    car_config=CarConfiguration(
        "TODO",
        [
            {
                "width": _transfuser_img_size[0],
                "height": _transfuser_img_size[1],
                "fov": _fov,
                "transform": Transform(Location(1.3, 0, 2.3), Rotation(0, -60, 0)),
            },
            {
                "width": _transfuser_img_size[0],
                "height": _transfuser_img_size[1],
                "fov": _fov,
                "transform": Transform(Location(1.3, 0, 2.3), Rotation(0, 0, 0)),
            },
            {
                "width": _transfuser_img_size[0],
                "height": _transfuser_img_size[1],
                "fov": _fov,
                "transform": Transform(Location(1.3, 0, 2.3), Rotation(0, 60, 0)),
            },
        ],
        {
            "enabled": True,
            "channels": 32,
            "range": 5000,
            "shape": (3, 256, 256),
            "points_per_second": 300000,
            "transform": Transform(Location(1.3, 0, 2.5), Rotation(0, -90, 0)),
        },
    ),
)


def interfuser_config() -> EpisodeManagerConfiguration:
    return EpisodeManagerConfiguration()


_baseline_image_size = (160, 120)


def baseline_config() -> EpisodeManagerConfiguration:
    return EpisodeManagerConfiguration(
        render_client=False,
        car_config=CarConfiguration(
            "TODO",
            [
                {
                    "width": _baseline_image_size[0],
                    "height": _baseline_image_size[1],
                    "fov": _fov,
                    "transform": Transform(Location(1.3, 0, 2.3), Rotation(0, -60, 0)),
                },
                {
                    "width": _baseline_image_size[0],
                    "height": _baseline_image_size[1],
                    "fov": _fov,
                    "transform": Transform(Location(1.3, 0, 2.3), Rotation(0, 0, 0)),
                },
                {
                    "width": _baseline_image_size[0],
                    "height": _baseline_image_size[1],
                    "fov": _fov,
                    "transform": Transform(Location(1.3, 0, 2.3), Rotation(0, 60, 0)),
                },
            ],
            {
                "enabled": False,
                "channels": 64,
                "range": 85,
                "shape": (3, 256, 256),
                "points_per_second": 300000,
                "transform": Transform(Location(1.3, 0, 2.5), Rotation(0, 90, 0)),
            },
        ),
    )
