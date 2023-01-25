from setuptools import setup, find_packages


setup(
    packages=[
        package for package in find_packages() if package.startswith("episode_manager")
    ],
    name="episode_manager",
    version="0.9.13",
    install_requires=[
        "scenario_runner @ git+https://github.com/Haavasma/scenario_runner.git@v0.9.13-setup-script"
    ],
)


# Set up the scenario runner from carla and install the package including scenario_runner.py and the srunner pacakage
