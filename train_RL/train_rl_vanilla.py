from episode_manager import EpisodeManager, EpisodeManagerConfiguration
from gym_env.env import (
    CarlaEnvironment,
    PIDController,
)
from stable_baselines3 import PPO

# from stable_baselines3.common.vec_env import DummyVecEnv
from stable_baselines3.common.callbacks import EvalCallback
from stable_baselines3.common.monitor import Monitor

import wandb
from episode_configs import BASELINE_CONFIG
from reward_functions.main import reward_function
from wandb.integration.sb3 import WandbCallback

rl_config = {"policy_type": "MultiInputPolicy", "total_timesteps": 1000000}
# experiment_name = f"CARLA_1670767940"


def main():
    """ """

    resume = False
    eval = True
    experiment_name = f"PPO custom policy"

    episode_manager = EpisodeManager(BASELINE_CONFIG)
    speed_controller = PIDController()
    run = init_wanda(resume=resume, name=experiment_name)

    env = CarlaEnvironment(
        {
            "speed_goal_actions": [-2.0, -1.0, 0.0, 2.0, 4.0, 5.0],
            "steering_actions": [
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
            "discrete_actions": True,
            "continuous_speed_range": (0, 0),
            "continuous_steering_range": (0, 0),
        },
        episode_manager,
        None,
        reward_function,
        speed_controller,
    )

    env = Monitor(env)

    wandb_callback = WandbCallback(
        # gradient_save_freq=10,
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
        verbose=1,
    )

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

    rl_model.learn(total_timesteps=50_000, callback=[wandb_callback, eval_callback])
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


if __name__ == "__main__":
    main()
