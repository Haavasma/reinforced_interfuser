from setuptools import setup, find_packages

setup(
    name="gym_carla",
    version="0.0.1",
    packages=[
        package for package in find_packages() if package.startswith("gym_carla")
    ],
)
