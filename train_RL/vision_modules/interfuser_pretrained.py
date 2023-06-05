import datetime
import math
import os
import pathlib
import time
from collections import deque
from typing import List, Optional, Tuple

import carla
import cv2
import numpy as np
import torch
from episode_manager.episode_manager import Action, WorldState
from gym_env.route_planner import RoutePlanner, find_relative_target_waypoint
from gym_env.vision import VisionModule
from leaderboard.autoagents import autonomous_agent
from PIL import Image
from team_code.interfuser_config import GlobalConfig
from team_code.interfuser_controller import InterfuserController
from team_code.render import render, render_self_car, render_waypoints
from team_code.tracker import Tracker
from timm.models import create_model
from torchvision import transforms


SAVE_PATH = None
IMAGENET_DEFAULT_MEAN = (0.485, 0.456, 0.406)
IMAGENET_DEFAULT_STD = (0.229, 0.224, 0.225)

try:
    import pygame
except ImportError:
    raise RuntimeError("cannot import pygame, make sure pygame package is installed")


class InterFuserPretrainedVisionModule(VisionModule):
    output_shape: List[Tuple] = [
        (20, 20, 7),
        (256,),
        (256,),
    ]

    # output_shape: Tuple = (256,)
    low: float = -np.inf
    high: float = np.inf

    def __init__(
        self,
        model_path: str,
        use_target_feature: bool = True,
        use_imitation_action: bool = False,
        render_imitation=False,
        postprocess=False,
        gpu_device: int = 0,
    ):
        self.use_target_feature = use_target_feature
        self.use_imitation_action = use_imitation_action
        self.render_imitation = render_imitation
        self.postprocess = postprocess

        self.throttle = 0.0
        self.brake = 1.0
        self.steer = 0.0

        self._hic = DisplayInterface(should_render_live=self.render_imitation)
        self.lidar_processed = list()
        self.track = autonomous_agent.Track.SENSORS
        self.step = -1
        self.wall_start = time.time()
        self.initialized = False
        self.rgb_front_transform = create_carla_rgb_transform(224)
        self.rgb_left_transform = create_carla_rgb_transform(128)
        self.rgb_right_transform = create_carla_rgb_transform(128)
        self.rgb_center_transform = create_carla_rgb_transform(128, need_scale=False)
        self.tracker = Tracker()
        self.input_buffer = {
            "rgb": deque(),
            "rgb_left": deque(),
            "rgb_right": deque(),
            "rgb_rear": deque(),
            "lidar": deque(),
            "gps": deque(),
            "thetas": deque(),
        }

        self.config = GlobalConfig()
        self.config.model_path = model_path
        self.config.model = "interfuser_pretrained"
        self.skip_frames = self.config.skip_frames
        self.controller = InterfuserController(self.config)
        self.net = create_model(self.config.model)
        path_to_model_file = self.config.model_path
        print("load model: %s" % path_to_model_file)
        self.net.load_state_dict(
            torch.load(path_to_model_file, map_location=f"cuda:{gpu_device}")[
                "state_dict"
            ]
        )
        self.net.cuda()
        self.net.eval()
        self.softmax = torch.nn.Softmax(dim=1)
        self.traffic_meta_moving_avg = np.zeros((400, 7))
        self.momentum = self.config.momentum
        self.prev_lidar = None
        self.prev_control = None
        self.prev_surround_map = None

        self.save_path = None
        if SAVE_PATH is not None:
            now = datetime.datetime.now()
            string = pathlib.Path(os.environ["ROUTES"]).stem + "_"
            string += "_".join(
                map(
                    lambda x: "%02d" % x,
                    (now.month, now.day, now.hour, now.minute, now.second),
                )
            )

            self.save_path = pathlib.Path(SAVE_PATH) / string
            self.save_path.mkdir(parents=True, exist_ok=False)
            (self.save_path / "meta").mkdir(parents=True, exist_ok=False)

        return

    def set_global_plan(self, global_plan):
        self._route_planner = RoutePlanner()
        self._route_planner.set_route(global_plan, True)
        self.initialized = True

        return

    def tick(self, world_state: WorldState):
        vehicle_state = world_state.ego_vehicle_state
        images = vehicle_state.sensor_data.images

        if len(images) != 3:
            raise ValueError("Expected 3 images, got %d" % len(images))

        if images[0].shape != (300, 400, 4):
            raise ValueError(
                "Expected image shape (300, 400, 3), got %s" % str(images[0].shape)
            )

        if images[1].shape != (600, 800, 4):
            raise ValueError(
                "Expected image shape (600, 800, 3), got %s" % str(images[1].shape)
            )

        if images[2].shape != (300, 400, 4):
            raise ValueError(
                "Expected image shape (300, 400, 3), got %s" % str(images[2].shape)
            )

        rgb = images[1][:, :, :3]
        rgb_left = cv2.cvtColor(images[0][:, :, :3], cv2.COLOR_BGR2RGB)
        rgb_right = cv2.cvtColor(images[2][:, :, :3], cv2.COLOR_BGR2RGB)
        gps = vehicle_state.gps
        speed = vehicle_state.speed
        compass = vehicle_state.compass
        if (
            math.isnan(compass) is True
        ):  # It can happen that the compass sends nan for a few frames
            compass = 0.0

        result = {
            "rgb": rgb,
            "rgb_left": rgb_left,
            "rgb_right": rgb_right,
            "gps": gps,
            "speed": speed,
            "compass": compass,
        }

        pos = self._get_position(result)

        lidar_processed = vehicle_state.sensor_data.lidar_data.bev

        if lidar_processed.shape != (3, 224, 224):
            raise ValueError(
                "Expected lidar shape (3, 224, 224), got %s"
                % str(lidar_processed.shape)
            )

        if self.step % 2 == 0 or self.step < 4:
            self.prev_lidar = lidar_processed
        result["lidar"] = self.prev_lidar

        result["gps"] = pos
        _, position, tar_point, next_cmd = self._route_planner.run_step(gps)
        result["next_command"] = next_cmd.value
        result["measurements"] = [pos[0], pos[1], compass, speed]

        relative_target_waypoint = find_relative_target_waypoint(
            position, tar_point, compass
        )

        result["target_point"] = relative_target_waypoint

        return result

    @torch.no_grad()
    def __call__(self, world_state: WorldState, postprocess=True):
        self.step += 1
        # if self.step % self.skip_frames != 0 and self.step > 4:
        #     return self.prev_control

        self.tick_data = self.tick(world_state)
        self.velocity = self.tick_data["speed"]
        command = self.tick_data["next_command"]
        rgb = (
            self.rgb_front_transform(Image.fromarray(self.tick_data["rgb"]))
            .unsqueeze(0)
            .cuda()
            .float()
        )
        rgb_left = (
            self.rgb_left_transform(Image.fromarray(self.tick_data["rgb_left"]))
            .unsqueeze(0)
            .cuda()
            .float()
        )
        rgb_right = (
            self.rgb_right_transform(Image.fromarray(self.tick_data["rgb_right"]))
            .unsqueeze(0)
            .cuda()
            .float()
        )
        rgb_center = (
            self.rgb_center_transform(Image.fromarray(self.tick_data["rgb"]))
            .unsqueeze(0)
            .cuda()
            .float()
        )

        cmd_one_hot = [0, 0, 0, 0, 0, 0]
        cmd = command - 1
        cmd_one_hot[cmd] = 1
        cmd_one_hot.append(self.velocity)
        mes = np.array(cmd_one_hot)
        mes = torch.from_numpy(mes).float().unsqueeze(0).cuda()

        input_data = {}
        input_data["rgb"] = rgb
        input_data["rgb_left"] = rgb_left
        input_data["rgb_right"] = rgb_right
        input_data["rgb_center"] = rgb_center
        input_data["measurements"] = mes
        input_data["target_point"] = (
            torch.from_numpy(self.tick_data["target_point"]).float().cuda().view(1, -1)
        )
        input_data["lidar"] = (
            torch.from_numpy(self.tick_data["lidar"]).float().cuda().unsqueeze(0)
        )

        with torch.no_grad():
            (
                traffic_meta,
                waypoints,
                is_junction,
                traffic_light_state,
                stop_sign,
                bev_feature,
                target_feature,
                traffic_state_feature,
            ) = self.net(input_data)

        self.traffic_meta = traffic_meta.detach().cpu().numpy()[0]
        self.bev_feature = bev_feature.detach().cpu().numpy()[0]
        self.pred_waypoints = waypoints.detach().cpu().numpy()[0]
        self.is_junction = (
            self.softmax(is_junction).detach().cpu().numpy().reshape(-1)[0]
        )
        self.traffic_light_state = (
            self.softmax(traffic_light_state).detach().cpu().numpy().reshape(-1)[0]
        )
        self.stop_sign = self.softmax(stop_sign).detach().cpu().numpy().reshape(-1)[0]

        traffic_meta_np = traffic_meta.detach().cpu().numpy()[0].reshape(20, 20, -1)

        return [
            traffic_meta_np.reshape(20, 20, -1),
            target_feature.squeeze(0).cpu().numpy(),
            traffic_state_feature.detach().cpu().numpy()[0],
        ]

        # return target_feature.squeeze(0).cpu().numpy()

    def _get_position(self, tick_data):
        gps = tick_data["gps"]
        gps = (gps - self._route_planner.mean) * self._route_planner.scale
        return gps

    def postprocess_action(self, action: Action) -> Action:
        if self.postprocess or self.use_imitation_action:
            if self.step % 2 == 0 or self.step < 4:
                traffic_meta = self.tracker.update_and_predict(
                    self.traffic_meta.reshape(20, 20, -1),
                    self.tick_data["gps"],
                    self.tick_data["compass"],
                    self.step // 2,
                )
                traffic_meta = traffic_meta.reshape(400, -1)
                self.traffic_meta_moving_avg = (
                    self.momentum * self.traffic_meta_moving_avg
                    + (1 - self.momentum) * traffic_meta
                )
            self.traffic_meta = self.traffic_meta_moving_avg
            self.tick_data["raw"] = self.traffic_meta
            self.tick_data["bev_feature"] = self.bev_feature

            steer, throttle, brake, self.meta_infos = self.controller.run_step(
                self.velocity,
                self.pred_waypoints,
                self.is_junction,
                self.traffic_light_state,
                self.stop_sign,
                self.traffic_meta_moving_avg,
            )

            if brake < 0.05:
                brake = 0.0
            if brake > 0.1:
                throttle = 0.0

            self.throttle = throttle
            self.brake = brake
            self.steer = steer

        if self.use_imitation_action:
            new_action = Action(
                throttle=self.throttle,
                steer=self.steer,
                brake=1.0 if self.brake else 0.0,
                reverse=False,
            )

            return new_action

        if self.postprocess:
            return Action(
                throttle=self.throttle,
                steer=action.steer,
                brake=action.brake,
                reverse=action.reverse,
            )
        else:
            return action

    def get_auxilliary_render(self) -> Optional[pygame.Surface]:
        if not self.postprocess and not self.use_imitation_action:
            return None

        if self.step % 2 == 0 or self.step < 4:
            traffic_meta = self.tracker.update_and_predict(
                self.traffic_meta.reshape(20, 20, -1),
                self.tick_data["gps"],
                self.tick_data["compass"],
                self.step // 2,
            )
            traffic_meta = traffic_meta.reshape(400, -1)
            self.traffic_meta_moving_avg = (
                self.momentum * self.traffic_meta_moving_avg
                + (1 - self.momentum) * traffic_meta
            )
        self.traffic_meta = self.traffic_meta_moving_avg
        self.tick_data["raw"] = self.traffic_meta
        self.tick_data["bev_feature"] = self.bev_feature

        self.control = carla.VehicleControl(
            throttle=self.throttle, steer=self.steer, brake=self.brake, reverse=False
        )
        surround_map, box_info = render(
            self.traffic_meta.reshape(20, 20, 7), pixels_per_meter=20
        )
        surround_map = surround_map[:400, 160:560]
        surround_map = np.stack([surround_map, surround_map, surround_map], 2)

        self_car_map = render_self_car(
            loc=np.array([0, 0]),
            ori=np.array([0, -1]),
            box=np.array([2.45, 1.0]),
            color=[1, 1, 0],
            pixels_per_meter=20,
        )[:400, 160:560]

        pred_waypoints = self.pred_waypoints.reshape(-1, 2)
        safe_index = 10
        for i in range(10):
            if (
                pred_waypoints[i, 0] ** 2 + pred_waypoints[i, 1] ** 2
                > (self.meta_infos[3] + 0.5) ** 2
            ):
                safe_index = i
                break
        wp1 = render_waypoints(
            pred_waypoints[:safe_index], pixels_per_meter=20, color=(0, 255, 0)
        )[:400, 160:560]
        wp2 = render_waypoints(
            pred_waypoints[safe_index:], pixels_per_meter=20, color=(255, 0, 0)
        )[:400, 160:560]
        wp = wp1 + wp2

        surround_map = np.clip(
            (
                surround_map.astype(np.float32)
                + self_car_map.astype(np.float32)
                + wp.astype(np.float32)
            ),
            0,
            255,
        ).astype(np.uint8)

        map_t1, box_info = render(
            self.traffic_meta.reshape(20, 20, 7), pixels_per_meter=20, t=1
        )
        map_t1 = map_t1[:400, 160:560]
        map_t1 = np.stack([map_t1, map_t1, map_t1], 2)
        map_t1 = np.clip(
            map_t1.astype(np.float32) + self_car_map.astype(np.float32), 0, 255
        ).astype(np.uint8)
        map_t1 = cv2.resize(map_t1, (200, 200))
        map_t2, box_info = render(
            self.traffic_meta.reshape(20, 20, 7), pixels_per_meter=20, t=2
        )
        map_t2 = map_t2[:400, 160:560]
        map_t2 = np.stack([map_t2, map_t2, map_t2], 2)
        map_t2 = np.clip(
            map_t2.astype(np.float32) + self_car_map.astype(np.float32), 0, 255
        ).astype(np.uint8)
        map_t2 = cv2.resize(map_t2, (200, 200))

        if self.step % 2 != 0 and self.step > 4:
            self.control = self.prev_control
        else:
            self.prev_control = self.control
            self.prev_surround_map = surround_map

        self.tick_data["map"] = self.prev_surround_map
        self.tick_data["map_t1"] = map_t1
        self.tick_data["map_t2"] = map_t2
        self.tick_data["rgb_raw"] = self.tick_data["rgb"]
        self.tick_data["rgb_left_raw"] = self.tick_data["rgb_left"]
        self.tick_data["rgb_right_raw"] = self.tick_data["rgb_right"]

        self.tick_data["rgb"] = cv2.resize(self.tick_data["rgb"], (800, 600))
        self.tick_data["rgb_left"] = cv2.resize(self.tick_data["rgb_left"], (200, 150))
        self.tick_data["rgb_right"] = cv2.resize(
            self.tick_data["rgb_right"], (200, 150)
        )
        self.tick_data["rgb_focus"] = cv2.resize(
            self.tick_data["rgb_raw"][244:356, 344:456], (150, 150)
        )
        self.tick_data["control"] = "throttle: %.2f, steer: %.2f, brake: %.2f" % (
            self.throttle,
            self.steer,
            self.brake,
        )

        self.tick_data["meta_infos"] = self.meta_infos
        self.tick_data["box_info"] = "car: %d, bike: %d, pedestrian: %d" % (
            box_info["car"],
            box_info["bike"],
            box_info["pedestrian"],
        )
        self.tick_data["mes"] = "speed: %.2f" % self.velocity
        self.tick_data["time"] = "time: %.3f" % time.time()
        surface = self._hic.run_interface(self.tick_data)
        self.tick_data["surface"] = surface

        return surface


def create_carla_rgb_transform(
    input_size, need_scale=True, mean=IMAGENET_DEFAULT_MEAN, std=IMAGENET_DEFAULT_STD
) -> transforms.Compose:
    if isinstance(input_size, (tuple, list)):
        img_size = input_size[-2:]
    else:
        img_size = input_size
    tfl = []

    if isinstance(input_size, (tuple, list)):
        input_size_num = input_size[-1]
    else:
        input_size_num = input_size

    if need_scale:
        if input_size_num == 112:
            tfl.append(Resize2FixedSize((170, 128)))
        elif input_size_num == 128:
            tfl.append(Resize2FixedSize((195, 146)))
        elif input_size_num == 224:
            tfl.append(Resize2FixedSize((341, 256)))
        elif input_size_num == 256:
            tfl.append(Resize2FixedSize((288, 288)))
        else:
            tfl.append(Resize2FixedSize(input_size))
            # raise ValueError("Can't find proper crop size")
    tfl.append(transforms.CenterCrop(img_size))
    tfl.append(transforms.ToTensor())
    tfl.append(transforms.Normalize(mean=torch.tensor(mean), std=torch.tensor(std)))

    return transforms.Compose(tfl)


class DisplayInterface(object):
    def __init__(self, should_render_live: bool = False):
        self._width = 1200
        self._height = 600
        self._surface = None

        self._should_render_live = should_render_live
        if self._should_render_live == "human":
            pygame.init()
            pygame.font.init()

        self._clock = pygame.time.Clock()
        if self._should_render_live:
            self._display = pygame.display.set_mode(
                (self._width, self._height), pygame.HWSURFACE | pygame.DOUBLEBUF
            )

            pygame.display.set_caption("Human Agent")

    def run_interface(self, input_data):
        rgb = input_data["rgb"]
        input_data["rgb_left"]
        input_data["rgb_right"]
        input_data["rgb_focus"]
        map = input_data["map"]
        surface = np.zeros((600, 1200, 3), np.uint8)
        surface[:, :800] = rgb
        surface[:400, 800:1200] = map
        surface[400:600, 800:1000] = input_data["map_t1"]
        surface[400:600, 1000:1200] = input_data["map_t2"]
        surface[:150, :200] = input_data["rgb_left"]
        surface[:150, 600:800] = input_data["rgb_right"]
        surface[:150, 325:475] = input_data["rgb_focus"]
        surface = cv2.putText(
            surface,
            input_data["control"],
            (20, 580),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            (0, 0, 255),
            1,
        )
        surface = cv2.putText(
            surface,
            input_data["meta_infos"][0],
            (20, 560),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            (0, 0, 255),
            1,
        )
        surface = cv2.putText(
            surface,
            input_data["meta_infos"][1],
            (20, 540),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            (0, 0, 255),
            1,
        )
        surface = cv2.putText(
            surface,
            input_data["time"],
            (20, 520),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            (0, 0, 255),
            1,
        )
        surface = cv2.putText(
            surface,
            str(input_data["target_point"]),
            (20, 500),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            (0, 0, 255),
            1,
        )
        surface = cv2.putText(
            surface,
            "Left  View",
            (40, 135),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.75,
            (0, 0, 0),
            2,
        )
        surface = cv2.putText(
            surface,
            "Focus View",
            (335, 135),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.75,
            (0, 0, 0),
            2,
        )
        surface = cv2.putText(
            surface,
            "Right View",
            (640, 135),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.75,
            (0, 0, 0),
            2,
        )

        surface = cv2.putText(
            surface,
            "Future Prediction",
            (940, 420),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            (255, 0, 0),
            2,
        )
        surface = cv2.putText(
            surface, "t", (1160, 385), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 0, 0), 2
        )
        surface = cv2.putText(
            surface, "0", (1170, 385), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 2
        )
        surface = cv2.putText(
            surface, "t", (960, 585), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 0, 0), 2
        )
        surface = cv2.putText(
            surface, "1", (970, 585), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 2
        )
        surface = cv2.putText(
            surface, "t", (1160, 585), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 0, 0), 2
        )
        surface = cv2.putText(
            surface, "2", (1170, 585), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 2
        )

        surface[:150, 198:202] = 0
        surface[:150, 323:327] = 0
        surface[:150, 473:477] = 0
        surface[:150, 598:602] = 0
        surface[148:152, :200] = 0
        surface[148:152, 325:475] = 0
        surface[148:152, 600:800] = 0
        surface[430:600, 998:1000] = 255
        surface[0:600, 798:800] = 255
        surface[0:600, 1198:1200] = 255
        surface[0:2, 800:1200] = 255
        surface[598:600, 800:1200] = 255
        surface[398:400, 800:1200] = 255

        # display image
        self._surface = pygame.surfarray.make_surface(surface.swapaxes(0, 1))

        if self._should_render_live:
            if self._surface is not None:
                self._display.blit(self._surface, (0, 0))
            pygame.display.flip()
            pygame.event.get()

        return surface

    def _quit(self):
        pygame.quit()


class Resize2FixedSize:
    def __init__(self, size):
        self.size = size

    def __call__(self, pil_img):
        pil_img = pil_img.resize(self.size)
        return pil_img
