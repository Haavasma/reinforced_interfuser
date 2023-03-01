from typing import List, Tuple
import cv2

from episode_manager import EpisodeManager, EpisodeManagerConfiguration
from episode_manager.agent_handler import AgentHandler
from episode_manager.episode_manager import (
    CarConfiguration,
    LidarConfiguration,
    Location,
    RGBCameraConfiguration,
    Rotation,
    TrainingType,
    Transform,
)


from stable_baselines3 import PPO

# from stable_baselines3.common.vec_env import DummyVecEnv

from stable_baselines3.common.callbacks import EvalCallback
from stable_baselines3.common.monitor import Monitor
from model import LidarCenterNet
import numpy as np
from config import GlobalConfig
import torch
from PIL import Image

from gym_env.env import (
    CarlaEnvironment,
    CarlaEnvironmentConfiguration,
    PIDController,
)
from reward_functions.main import reward_function


from transfuser import TransfuserBackbone

from wandb.integration.sb3 import WandbCallback
import wandb

from vision_modules.transfuser import TransfuserVisionModule


rl_config = {"policy_type": "MultiInputPolicy", "total_timesteps": 1000000}
# experiment_name = f"CARLA_1670767940"


def main():
    """ """

    # TODO implement pure CNN learner
    # Test out training with IDUN
    resume = False
    run_id = None
    eval = True
    experiment_name = f"PPO custom policy"

    config = GlobalConfig(setting="eval")
    # run = init_wanda(resume=resume, name=experiment_name)

    config.n_layer = 4
    config.use_target_point_image = True

    backbone = setup_transfuser_backbone(
        config,
        "/lhome/haavasma/Documents/fordypningsoppgave/repositories/models/transfuser/model_ckpt/models_2022/transfuser/model_seed2_39.pth",
    )

    transfuser_img_size = (960, 480)
    fov = 103

    transfuser_vision_module = TransfuserVisionModule(backbone, config)

    episode_config = EpisodeManagerConfiguration(
        render_client=True,
        car_config=CarConfiguration(
            "test",
            [
                RGBCameraConfiguration(
                    transfuser_img_size[0],
                    transfuser_img_size[1],
                    fov,
                    Transform(Location(1.3, 0, 2.3), Rotation(0, -60, 0)),
                ),
                RGBCameraConfiguration(
                    transfuser_img_size[0],
                    transfuser_img_size[1],
                    fov,
                    Transform(Location(1.3, 0, 2.3), Rotation(0, 0, 0)),
                ),
                RGBCameraConfiguration(
                    transfuser_img_size[0],
                    transfuser_img_size[1],
                    fov,
                    Transform(Location(1.3, 0, 2.3), Rotation(0, 60, 0)),
                ),
            ],
            LidarConfiguration(enabled=True),
        ),
    )
    episode_manager = EpisodeManager(episode_config)
    speed_controller = PIDController()

    env = CarlaEnvironment(
        CarlaEnvironmentConfiguration(
            speed_goal_actions=[-2.0, -1.0, 0.0, 2.0, 4.0, 5.0],
            steering_actions=[
                -0.3,
                -0.27,
                -0.24,
                -0.21,
                -0.18,
                -0.15,
                -0.12,
                -0.09,
                -0.06,
                -0.03,
                0.0,
                0.03,
                0.06,
                0.09,
                0.12,
                0.15,
                0.18,
                0.21,
                0.24,
                0.27,
                0.3,
            ],
            discrete_actions=True,
        ),
        episode_manager,
        transfuser_vision_module,
        reward_function,
        speed_controller,
    )

    env = Monitor(env)

    # wandb_callback = WandbCallback(
    #     # gradient_save_freq=10,
    #     model_save_path=f"./models/{run.id}/",
    #     model_save_freq=2048,
    # )

    # eval_callback = EvalCallback(
    #     env,
    #     best_model_save_path=f"./models/{run.id}/best_model/",
    #     log_path=f"./models/{run.id}/logs/",
    #     eval_freq=10240,
    #     deterministic=True,
    #     render=False,
    #     n_eval_episodes=5,
    #     # callback_after_eval=wandb_callback,
    #     verbose=1,
    # )

    policy_kwargs = dict(net_arch=[1024, 512, dict(vf=[256], pi=[256])])

    rl_model = PPO(
        rl_config["policy_type"],
        env,
        verbose=2,
        gamma=0.95,
        n_steps=2048,
        # buffer_size=20_000,
        learning_rate=1e-4,
        # tau=0.005,
        # action_noise=NormalActionNoise(np.array([1.0, 0.0]), np.array([0.5, 0.3])),
        # tensorboard_log=f"runs/{run.id}",
        policy_kwargs=policy_kwargs,
        device="cuda",
    )

    # rl_model.load(f"./models/{run_id}/best_model/best_model.zip")
    # rl_model = PPO.load(f"./models/{run_id}/best_model/best_model", env=env)

    # rl_model.save(f"./models/{run.id}/model")

    # obs = env.reset()
    # if eval:
    #     for _ in range(30000):
    #         action, _states = rl_model.predict(obs, deterministic=True)
    #         obs, rewards, dones, info = env.step(action)
    #         if dones:
    #             obs = env.reset()
    #
    #     return

    rl_model.learn(
        total_timesteps=50_000
    )  # , callback=[wandb_callback, eval_callback])
    rl_model.save(f"./models/{run.id}/model")
    # rl_model.save(f"./models/ppo_carla_{time.time()}")


def init_wanda(resume=False, name=None):
    return wandb.init(
        resume=resume,
        config=rl_config,
        name=name,
        monitor_gym=True,
        project="carla-rl",
        entity="haavasma",
        sync_tensorboard=True,
    )


def setup_transfuser_backbone(
    config: GlobalConfig, file_path: str
) -> TransfuserBackbone:
    model = LidarCenterNet(
        config, "cuda", "transFuser", "regnety_032", "regnety_032", use_velocity=False
    )

    # Model was trained with Sync. Batch Norm. Need to convert it otherwise parameters will load incorrectly.
    # model = torch.nn.SyncBatchNorm.convert_sync_batchnorm(model)

    state_dict = torch.load(
        file_path,
        map_location="cuda:0",
    )

    state_dict = {k[7:]: v for k, v in state_dict.items()}
    model.load_state_dict(state_dict, strict=False)
    model.cuda()
    model.eval()

    backbone: TransfuserBackbone = model._model

    return backbone


if __name__ == "__main__":
    main()
