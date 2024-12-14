#!/bin/bash
nohup python -u actor.py --config examples/ppo/running_actor.yaml > ./log/actor.log 2>&1 &
