from typing import List

import cv2
from episode_manager.agent_handler.models import CarConfiguration
from episode_manager.agent_handler.models.transform import Location, Rotation, Transform
import numpy as np
import pygame
import torch
from episode_manager.episode_manager import Action, WorldState
from gym_env.env import VisionModule
from PIL.Image import Image

from config import GlobalConfig
from model import LidarCenterNet
from transfuser import TransfuserBackbone


class TransfuserVisionModule(VisionModule):
    def __init__(self, model: TransfuserBackbone, config: GlobalConfig):
        self.model = model
        self.config = config

        self.output_shape = (256,)
        self.high = 1
        self.low = 0

    @staticmethod
    def get_car_config() -> CarConfiguration:
        return CarConfiguration(
            "tesla",
            [
                {
                    "height": 300,
                    "width": 400,
                    "fov": 100,
                    "transform": Transform(Location(1.3, 0, 2.3), Rotation(0, -60, 0)),
                },
                {
                    "height": 600,
                    "width": 800,
                    "fov": 100,
                    "transform": Transform(Location(1.3, 0, 2.3), Rotation(0, 0, 0)),
                },
                {
                    "height": 300,
                    "width": 400,
                    "fov": 100,
                    "transform": Transform(Location(1.3, 0, 2.3), Rotation(0, 60, 0)),
                },
            ],
            {
                "enabled": True,
                "channels": 64,
                "range": 85,
                "shape": (3, 256, 256),
                "points_per_second": 300000,
                "transform": Transform(Location(1.3, 0, 2.5), Rotation(0, 90, 0)),
            },
        )

    def __call__(self, world_state: WorldState) -> np.ndarray:
        rgb = self._parse_images(world_state.ego_vehicle_state.sensor_data.images)
        lidar_bev = world_state.ego_vehicle_state.sensor_data.lidar_data.bev

        _, _, fused_features = self.model(
            rgb,
            lidar_bev,
            torch.Tensor(
                np.array([world_state.ego_vehicle_state.speed]).astype(np.uint8)
            ),
        )

        return fused_features.flatten().cpu().detach().numpy()

    def _parse_images(self, images: List[np.ndarray]) -> torch.Tensor:
        if not len(images) == 3:
            raise ValueError(
                f"Amount of sensor data images must be 3! found {len(images)}"
            )

        return self._prepare_image(images[0], images[1], images[2])

    def postprocess_action(self, action: Action) -> Action:
        return action

    def get_auxilliary_render(self) -> pygame.Surface:
        """
        Returns a pygame surface that visualizes auxilliary predections from the vision module
        """
        raise NotImplementedError

    def _prepare_image(
        self, left: np.ndarray, front: np.ndarray, right: np.ndarray
    ) -> torch.Tensor:
        for image in [left, front, right]:
            rgb_pos = cv2.cvtColor(image[:, :, :3], cv2.COLOR_BGR2RGB)
            rgb_pos = _scale_crop(
                Image.fromarray(rgb_pos),
                self.config.scale,
                self.config.img_width,
                self.config.img_width,
                self.config.img_resolution[0],
                self.config.img_resolution[0],
            )
            rgb.append(rgb_pos)
        rgb = np.concatenate(rgb, axis=1)

        image = Image.fromarray(rgb)
        image_degrees = []
        rgb = torch.from_numpy(
            _shift_x_scale_crop(
                image,
                scale=self.config.scale,
                crop=self.config.img_resolution,
                crop_shift=0,
            )
        ).unsqueeze(0)
        image_degrees.append(rgb.to("cuda", dtype=torch.float32))
        image = torch.cat(image_degrees, dim=0)

        return image


def _shift_x_scale_crop(image, scale, crop, crop_shift=0):
    crop_h, crop_w = crop
    (width, height) = (int(image.width // scale), int(image.height // scale))
    im_resized = image.resize((width, height))
    image = np.array(im_resized)
    start_y = height // 2 - crop_h // 2
    start_x = width // 2 - crop_w // 2

    # only shift in x direction
    start_x += int(crop_shift // scale)
    cropped_image = image[start_y : start_y + crop_h, start_x : start_x + crop_w]
    cropped_image = np.transpose(cropped_image, (2, 0, 1))
    return cropped_image


def _scale_crop(image, scale=1, start_x=0, crop_x=None, start_y=0, crop_y=None):
    (width, height) = (image.width // scale, image.height // scale)
    if scale != 1:
        image = image.resize((width, height))
    if crop_x is None:
        crop_x = width
    if crop_y is None:
        crop_y = height

    image = np.asarray(image)
    cropped_image = image[start_y : start_y + crop_y, start_x : start_x + crop_x]
    return cropped_image


def setup_transfuser_backbone(
    config: GlobalConfig, file_path: str, device: str = "cuda:0"
) -> TransfuserBackbone:
    model = LidarCenterNet(
        config, device, "transFuser", "regnety_032", "regnety_032", use_velocity=False
    )

    state_dict = torch.load(file_path, map_location=device)

    state_dict = {k[7:]: v for k, v in state_dict.items()}
    model.load_state_dict(state_dict, strict=False)
    model.cuda()
    model.eval()

    backbone: TransfuserBackbone = model._model

    return backbone
