import os
import random

routes = {}

ROUTE_FOLDER = "routes"
DATA_FOLDER = "expert_data"
DATA_GEN_PATH = "data_gen"

BASHS_DIR = f"{DATA_GEN_PATH}/scripts/bashs"

SCENARIOS_FILE = f"{ROUTE_FOLDER}/all_towns_traffic_scenarios.json"

routes[f"{ROUTE_FOLDER}/routes_training_town01.xml"] = SCENARIOS_FILE
routes[f"{ROUTE_FOLDER}/routes_training_town03.xml"] = SCENARIOS_FILE
routes[f"{ROUTE_FOLDER}/routes_training_town04.xml"] = SCENARIOS_FILE
routes[f"{ROUTE_FOLDER}/routes_training_town06.xml"] = SCENARIOS_FILE


ip_ports = []

for port in range(20000, 20028, 2):
    ip_ports.append(("localhost", port, port + 500))


carla_seed = 2000
traffic_seed = 2000

configs = []
for i in range(14):
    configs.append("weather-%d.yaml" % i)


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
        "export CHECKPOINT_ENDPOINT=%s/results/%s/results/%s.json\n"
        % (DATA_GEN_PATH, config_path.split(".")[0], route.split("/")[1].split(".")[0])
    )
    lines.append("\n")
    base = open(os.path.join(os.path.dirname(__file__), "base_script.sh")).readlines()

    for line in lines:
        base.insert(9, line)

    return base


for i in range(14):
    if not os.path.exists(BASHS_DIR):
        os.mkdir(BASHS_DIR)
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
            f"{BASHS_DIR}/weather-%d/%s.sh" % (i, route.split("/")[1].split(".")[0]),
            "w",
        )
        for line in script:
            fw.write(line)
