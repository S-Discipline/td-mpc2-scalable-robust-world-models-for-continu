#!/usr/bin/env bash
# Scaling claim (C2): offline evaluation of the officially published mt80
# multi-task checkpoints at four model sizes. Measures the normalized score
# (DMControl reward/10, Meta-World success x100, averaged over 80 tasks).
MODE=eval
TASK=mt80
MODEL_SIZE=5
MODEL_SIZES="1 5 19 48"
EVAL_EPISODES=10
SEED=1
SAVE_VIDEO=false
STEPS=1000000
