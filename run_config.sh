#!/usr/bin/env bash
# Claim C1 (single-task, one hyperparameter set): offline evaluation of the
# officially published 5M single-task DMControl checkpoints. Verifies the
# released agents achieve the reported asymptotic single-task performance.
MODE=eval
TASK=walker-walk
TASKS="walker-walk cheetah-run hopper-hop dog-run"
MODEL_SIZE=5
EVAL_EPISODES=10
SEED=1
SAVE_VIDEO=false
STEPS=1000000
