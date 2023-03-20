from setuptools import setup

setup(
    name="train_rl",
    version="0.0.1",
    py_modules=["./"],
    install_requires=[
        "carla_gym_env @ git+https://github.com/Haavasma/Carla_scenario_runner_gym_environment.git"
    ],
)
