#!/bin/bash
nohup python -u actor/actor.py --config actor/examples/ppo/curling_actor.yaml > ./actor.log 2>&1 &
sleep 3
CUDA_VISIBLE_DEVICES=0 nohup python -u learner/learner.py --config learner/examples/ppo/curling_learner.yaml > ./learner.log 2>&1 &
