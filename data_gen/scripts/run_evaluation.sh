#!/bin/bash

# export CARLA_ROOT=carla
# export CARLA_SERVER=${CARLA_ROOT}/CarlaUE4.sh
# export PYTHONPATH=$PYTHONPATH:${CARLA_ROOT}/PythonAPI
# export PYTHONPATH=$PYTHONPATH:${CARLA_ROOT}/PythonAPI/carla
# export PYTHONPATH=$PYTHONPATH:$CARLA_ROOT/PythonAPI/carla/dist/carla-0.9.10-py3.7-linux-x86_64.egg
# export PYTHONPATH=$PYTHONPATH:leaderboard
# export PYTHONPATH=$PYTHONPATH:leaderboard/team_code
# export PYTHONPATH=$PYTHONPATH:scenario_runner

export DATAGEN_ROOT=data_gen

export LEADERBOARD_ROOT=$DATAGEN_ROOT/leaderboard

export CHALLENGE_TRACK_CODENAME=SENSORS
export PORT=2000 # same as the carla server port
export TM_PORT=2500 # port for traffic manager, required when spawning multiple servers/clients
export DEBUG_CHALLENGE=0
export REPETITIONS=1 # multiple evaluation runs
export ROUTES=routes/routes_test_town02.xml
export TEAM_AGENT=$DATAGEN_ROOT/team_code/auto_pilot.py # agent
export TEAM_CONFIG=yamls/weather-0.yaml
export CHECKPOINT_ENDPOINT=$DATAGEN_ROOT/results/sample_result.json # results file
export SCENARIOS=routes/all_towns_traffic_scenarios.json
export SAVE_PATH=expert_data/ # path for saving episodes while evaluating
# export RESUME=True

python3 ${LEADERBOARD_ROOT}/leaderboard_evaluator.py \
--scenarios=${SCENARIOS}  \
--routes=${ROUTES} \
--repetitions=${REPETITIONS} \
--track=${CHALLENGE_TRACK_CODENAME} \
--checkpoint=${CHECKPOINT_ENDPOINT} \
--agent=${TEAM_AGENT} \
--agent-config=${TEAM_CONFIG} \
--debug=${DEBUG_CHALLENGE} \
--record=${RECORD_PATH} \
--resume=${RESUME} \
--port=${PORT} \
--traffic-manager-port=${TM_PORT}
