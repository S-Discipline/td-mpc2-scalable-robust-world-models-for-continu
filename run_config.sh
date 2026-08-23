#!/usr/bin/env bash
# Claim C3 (SimNorm ablation, OFF arm). Retrain a 5M TD-MPC2 agent on
# walker-walk from scratch for a short budget with SimNorm replaced by an
# identity (no latent normalization). Compare against the sibling full-agent
# run (SimNorm on). torch.compile disabled (L4 speed).
MODE=train
TASK=walker-walk
MODEL_SIZE=5
STEPS=250000
BATCH_SIZE=
SEED=1
SAVE_VIDEO=false
COMPILE=false
