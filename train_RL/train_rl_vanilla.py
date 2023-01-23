from typing import List
import cv2

from stable_baselines3 import PPO

# from stable_baselines3.common.vec_env import DummyVecEnv
from stable_baselines3.common.callbacks import EvalCallback
from stable_baselines3.common.monitor import Monitor
import numpy as np
from config import GlobalConfig
import torch
from PIL import Image

from gym_carla.envs.vanilla_rl_env import CarlaEnv


from wandb.integration.sb3 import WandbCallback
import wandb

rl_config = {"policy_type": "MultiInputPolicy", "total_timesteps": 1000000}
# experiment_name = f"CARLA_1670767940"

pixels_per_meter = 2


def main():
    """ """
    # TODO implement pure CNN learner
    # Test out training with IDUN
    resume = False
    run_id = "m6jibona"
    eval = True
    experiment_name = f"PPO training vanilla RL"

    config = GlobalConfig(setting="eval")
    run = init_wanda(resume=resume, run_id=run_id, name=experiment_name)

    config.n_layer = 4
    config.use_target_point_image = True

    # Model was trained with Sync. Batch Norm. Need to convert it otherwise parameters will load incorrectly.
    # model = torch.nn.SyncBatchNorm.convert_sync_batchnorm(model)

    def prepare_image(
        left: np.ndarray, front: np.ndarray, right: np.ndarray
    ) -> torch.Tensor:
        rgb = []

        for image in [left, front, right]:
            rgb_pos = cv2.cvtColor(image[:, :, :3], cv2.COLOR_BGR2RGB)
            rgb_pos = scale_crop(
                Image.fromarray(rgb_pos),
                config.scale,
                config.img_width,
                config.img_width,
                config.img_resolution[0],
                config.img_resolution[0],
            )
            rgb.append(rgb_pos)
        rgb = np.concatenate(rgb, axis=1)

        image = Image.fromarray(rgb)
        image_degrees = []
        rgb = torch.from_numpy(
            shift_x_scale_crop(
                image, scale=config.scale, crop=config.img_resolution, crop_shift=0
            )
        ).unsqueeze(0)
        image_degrees.append(rgb.to("cuda", dtype=torch.float32))
        image = torch.cat(image_degrees, dim=0)

        return image

    def shift_x_scale_crop(image, scale, crop, crop_shift=0):
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

    def scale_crop(image, scale=1, start_x=0, crop_x=None, start_y=0, crop_y=None):
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

    # @jit(nopython=True)
    def prepare_lidar(point_cloud: List[List[float]]) -> np.ndarray:
        point_cloud = np.array(point_cloud)
        lidar_transformed = point_cloud

        def lidar_to_histogram_features(lidar):
            """
            Convert LiDAR point cloud into 2-bin histogram over 256x256 grid
            """

            def splat_points(point_cloud):
                # 256 x 256 grid
                hist_max_per_pixel = 5
                x_meters_max = 16
                y_meters_max = 32
                xbins = np.linspace(
                    -x_meters_max, x_meters_max, 32 * pixels_per_meter + 1
                )
                ybins = np.linspace(-y_meters_max, 0, 32 * pixels_per_meter + 1)
                hist = np.histogramdd(
                    point_cloud[..., :2],
                    bins=(xbins, ybins),
                )[0]
                hist[hist > hist_max_per_pixel] = hist_max_per_pixel
                overhead_splat = hist / hist_max_per_pixel
                return overhead_splat

            below = lidar[lidar[..., 2] <= -2.3]
            above = lidar[lidar[..., 2] > -2.3]
            below_features = splat_points(below)
            above_features = splat_points(above)
            features = np.stack([above_features, below_features], axis=-1)
            features = np.transpose(features, (2, 0, 1)).astype(np.float32)
            features = np.rot90(features, -1, axes=(1, 2)).copy()
            return features

        lidar_transformed[:, 1] *= -1  # invert
        lidar_transformed = np.expand_dims(
            lidar_to_histogram_features(lidar_transformed), 0
        )
        lidar_transformed_degrees = lidar_transformed
        lidar_bev = lidar_transformed_degrees[::-1]

        lidar_bev = np.append(
            lidar_bev,
            np.zeros((1, 1, 32 * pixels_per_meter, 32 * pixels_per_meter)),
            axis=1,
        )

        return lidar_bev

    env = CarlaEnv(
        prepare_image,
        prepare_lidar,
        render_env=False,
    )

    env = Monitor(env)

    wandb_callback = WandbCallback(
        # gradient_save_freq=2048,
        model_save_path=f"./models/{run.id}/",
        model_save_freq=2048,
    )

    eval_callback = EvalCallback(
        env,
        best_model_save_path=f"./models/{run.id}/best_model/",
        log_path=f"./models/{run.id}/logs/",
        eval_freq=10240,
        deterministic=True,
        render=False,
        n_eval_episodes=5,
        callback_after_eval=wandb_callback,
        verbose=1,
    )

    policy_kwargs = dict(net_arch=[1024, 512, dict(vf=[256], pi=[256])])

    rl_model = PPO(
        rl_config["policy_type"],
        env,
        verbose=2,
        gamma=0.95,
        # n_steps=2048,
        # batch_size=64,
        # buffer_size=20_000,
        learning_rate=1e-4,
        # tau=0.005,
        # action_noise=NormalActionNoise(np.array([1.0, 0.0]), np.array([0.5, 0.3])),
        tensorboard_log=f"runs/{run.id}",
        policy_kwargs=policy_kwargs,
        device="cuda",
    )

    print("LOADING MODEL")
    # rl_model.load(f"./models/{run_id}/best_model/best_model.zip")
    rl_model = PPO.load(f"./models/{run_id}/model", env=env)

    # rl_model.save(f"./models/{run.id}/model")

    if eval:
        obs = env.reset()
        for _ in range(30000):
            action, _states = rl_model.predict(obs, deterministic=True)
            obs, rewards, dones, info = env.step(action)
            if dones:
                obs = env.reset()

        return

    rl_model.learn(total_timesteps=50_000, callback=[wandb_callback, eval_callback])
    rl_model.save(f"./models/{run.id}/model")
    # rl_model.save(f"./models/ppo_carla_{time.time()}")


def init_wanda(resume=False, run_id=None, name=None):
    return wandb.init(
        resume=resume,
        id=run_id,
        config=rl_config,
        name=name,
        monitor_gym=True,
        project="carla-rl",
        entity="haavasma",
        sync_tensorboard=True,
    )


if __name__ == "__main__":
    main()
