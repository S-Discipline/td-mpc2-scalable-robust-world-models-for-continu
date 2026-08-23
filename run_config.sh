#!/usr/bin/env bash
# Committed per-branch run configuration.
#
# Each experiment node edits this file (a code change on its branch) to select
# the exact reproduction job. The run command (`bash run.sh`) stays fixed.
#
# MODE: eval (offline evaluation of a checkpoint) | train (online RL)
# For eval: set TASK, MODEL_SIZE, optionally CHECKPOINT (else auto-downloaded
#   from the published HuggingFace repo). EVAL_EPISODES (default 10).
# For train: set TASK, MODEL_SIZE, STEPS, and any single-task overrides.

MODE=eval
TASK=walker-walk
MODEL_SIZE=5
EVAL_EPISODES=10
SEED=1
SAVE_VIDEO=false

# Training
STEPS=1000000
# Optional single-task overrides; leave empty to use config defaults.
BATCH_SIZE=
SIMNORM_DIM=
NUM_Q=
