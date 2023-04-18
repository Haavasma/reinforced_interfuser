import os
from pathlib import Path
import random
import shutil

from torchvision.utils import pathlib

routes = {}

ROUTE_FOLDER = Path("../routes").absolute().resolve()
DATA_FOLDER = Path("../expert_data").absolute().resolve()
DATA_GEN_PATH = Path("./").absolute().resolve()

CHECKPOINT_FOLDER = DATA_GEN_PATH / "checkpoints"

BASHS_DIR = Path("scripts/bashs").absolute().resolve()

SCENARIOS_FILE = ROUTE_FOLDER / "all_towns_traffic_scenarios.json"

routes[f"{ROUTE_FOLDER}/routes_training_town01.xml"] = SCENARIOS_FILE
routes[f"{ROUTE_FOLDER}/routes_training_town03.xml"] = SCENARIOS_FILE
routes[f"{ROUTE_FOLDER}/routes_training_town04.xml"] = SCENARIOS_FILE
routes[f"{ROUTE_FOLDER}/routes_training_town06.xml"] = SCENARIOS_FILE


def main():
    ip_ports = []

    for port in range(20000, 20028, 2):
        ip_ports.append(("localhost", port, port + 500))

    carla_seed = 2000
    traffic_seed = 2000

    configs = []

    if not os.path.exists(CHECKPOINT_FOLDER):
        os.mkdir(CHECKPOINT_FOLDER)

    for i in range(14):
        new_config = f"weather-{i}.yaml"
        configs.append(new_config)
        os.mkdir(f"{CHECKPOINT_FOLDER}/weather-%d" % i)

    if os.path.exists(BASHS_DIR):
        shutil.rmtree(BASHS_DIR)
    os.mkdir(BASHS_DIR)

    for i in range(14):
        os.mkdir(f"{BASHS_DIR}/weather-%d" % i)
        for route in routes:
            ip, port, tm_port = ip_ports[i]
            script = generate_script(
                ip,
                port,
                tm_port,
                route,
                routes[route],
                carla_seed,
                traffic_seed,
                configs[i],
            )
            fw = open(
                f"{BASHS_DIR}/weather-%d/%s.sh"
                % (i, route.split("/")[-1].split(".")[0]),
                "w",
            )
            for line in script:
                fw.write(line)
    print(f"Finished writing bashes to {BASHS_DIR}.")


def generate_script(
    ip, port, tm_port, route, scenario, carla_seed, traffic_seed, config_path
):
    lines = []
    lines.append("export HOST=%s\n" % ip)
    lines.append("export PORT=%d\n" % port)
    lines.append("export TM_PORT=%d\n" % tm_port)
    lines.append("export ROUTES=%s\n" % route)
    lines.append("export SCENARIOS=%s\n" % scenario)
    lines.append("export CARLA_SEED=%d\n" % port)
    lines.append("export TRAFFIC_SEED=%d\n" % port)
    lines.append("export TEAM_CONFIG=yamls/%s\n" % config_path)
    lines.append("export SAVE_PATH=%s\n" % DATA_FOLDER)
    lines.append(
        "export CHECKPOINT_ENDPOINT=%s/checkpoints/%s/%s.json\n"
        % (
            DATA_GEN_PATH,
            config_path.split(".")[0],
            route.split("/")[1].split(".")[0],
        )
    )
    lines.append("\n")
    base = open(os.path.join(os.path.dirname(__file__), "base_script.sh")).readlines()

    for line in lines:
        base.insert(11, line)

    return base


if __name__ == "__main__":
    main()
