#!/usr/bin/env bash
# Claim C1 (single-task data-efficiency) + C3 (SimNorm ablation, ON arm).
# Retrain a 5M TD-MPC2 agent on walker-walk from scratch for a short budget
# with the default architecture (SimNorm enabled). torch.compile disabled
# (it is very slow on the 24-SM L4 and not needed for a small task).
MODE=train
TASK=walker-walk
MODEL_SIZE=5
STEPS=250000
BATCH_SIZE=
SEED=1
SAVE_VIDEO=false
COMPILE=false
