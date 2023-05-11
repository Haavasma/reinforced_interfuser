from setuptools import setup

setup(
    name="train_rl",
    version="0.0.1",
    py_modules=["./"],
    install_requires=[
        "episode_manager @ git+https://github.com/Haavasma/episode_manager.git"
    ],
)
