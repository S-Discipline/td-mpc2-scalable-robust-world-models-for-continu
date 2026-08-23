#!/usr/bin/env bash
# Scaling claim (C2): offline evaluation of the officially published mt30
# multi-task checkpoints (30 DMControl tasks) at four model sizes. Measures
# the normalized score (DMControl reward/10, averaged over 30 tasks).
#
# SUBSTITUTION: the paper's headline scaling used the mt80 task set (80 tasks,
# DMControl + Meta-World). Meta-World is not installable on this compute box
# without breaking the working MuJoCo/gym setup, so we reproduce the scaling
# claim on the mt30 DMControl-only set instead. The paper reports mt30 scaling:
# 1M=18.9, 5M=28.3, 19M=54.2, 48M=59.4, 317M=71.4.
MODE=eval
TASK=mt30
MODEL_SIZE=5
MODEL_SIZES="1 5 19 48"
EVAL_EPISODES=10
SEED=1
SAVE_VIDEO=false
STEPS=1000000
