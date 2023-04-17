#!/bin/bash
export DATAGEN_ROOT=data_gen

export LEADERBOARD_ROOT=$DATAGEN_ROOT/leaderboard
export TEAM_AGENT=$DATAGEN_ROOT/team_code/auto_pilot.py # agent

export CHALLENGE_TRACK_CODENAME=SENSORS
export RESUME=True


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
