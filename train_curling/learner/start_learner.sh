#!/bin/bash
CUDA_VISIBLE_DEVICES=0 nohup python -u learner.py --config examples/ppo/running_learner.yaml > ./log/learner.log 2>&1 &
