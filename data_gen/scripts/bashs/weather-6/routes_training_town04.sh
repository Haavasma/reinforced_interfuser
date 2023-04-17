#!/bin/bash
export DATAGEN_ROOT=data_gen

export LEADERBOARD_ROOT=$DATAGEN_ROOT/leaderboard
export TEAM_AGENT=$DATAGEN_ROOT/team_code/auto_pilot.py # agent

export CHALLENGE_TRACK_CODENAME=SENSORS
export RESUME=True


export CHECKPOINT_ENDPOINT=data_gen/results/weather-6/results/routes_training_town04.json
export SAVE_PATH=expert_data
export TEAM_CONFIG=yamls/weather-6.yaml
export TRAFFIC_SEED=20012
export CARLA_SEED=20012
export SCENARIOS=routes/all_towns_traffic_scenarios.json
export ROUTES=routes/routes_training_town04.xml
export TM_PORT=20512
export PORT=20012
export HOST=localhost

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
