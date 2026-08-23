#!/usr/bin/env bash
# Reproduction driver for TD-MPC2 (arXiv 2310.16828).
#
# Dispatches to train or evaluate based on the committed run_config.sh.
# This is the single fixed run command for every experiment node; nodes vary
# only the code/config on their branch, never this entrypoint.
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
CONFIG="${ROOT}/run_config.sh"

if [[ ! -f "${CONFIG}" ]]; then
	echo "ERROR: ${CONFIG} not found" >&2
	exit 1
fi

# shellcheck disable=SC1090
source "${CONFIG}"

: "${MODE:?run_config.sh must set MODE=eval or MODE=train}"
: "${TASK:?run_config.sh must set TASK}"
: "${MODEL_SIZE:?run_config.sh must set MODEL_SIZE}"

echo "======================================"
echo "TD-MPC2 reproduction run"
echo "  mode       : ${MODE}"
echo "  task       : ${TASK}"
echo "  model_size : ${MODEL_SIZE}"
echo "  steps      : ${STEPS:-}"
echo "  checkpoint : ${CHECKPOINT:-}"
echo "======================================"

export PYTHONPATH="${ROOT}:${ROOT}/tdmpc2:${PYTHONPATH:-}"

# Use any available python; prefer a project venv if present.
PY="${PYTHON_BIN:-python3}"
echo "using python: ${PY} ($(${PY} --version 2>&1))"
${PY} -c "import torch; print('torch', torch.__version__, 'cuda', torch.cuda.is_available())"

checkpoint_file=""
if [[ "${MODE}" == "eval" ]]; then
	if [[ -z "${CHECKPOINT:-}" ]]; then
		# Derive from task/model_size and download the published checkpoint.
		base_url="https://huggingface.co/nicklashansen/tdmpc2/resolve/main"
		if [[ "${TASK}" == mt30 || "${TASK}" == mt80 ]]; then
			CHECKPOINT="${ROOT}/ckpts/${TASK}-${MODEL_SIZE}M.pt"
			url="${base_url}/multitask/${TASK}-${MODEL_SIZE}M.pt"
		else
			domain="dmcontrol"
			CHECKPOINT="${ROOT}/ckpts/${TASK}-1.pt"
			url="${base_url}/${domain}/${TASK}-1.pt"
		fi
		mkdir -p "${ROOT}/ckpts"
		if [[ ! -f "${CHECKPOINT}" ]]; then
			echo "download checkpoint: ${url}"
			wget -q -O "${CHECKPOINT}" "${url}"
		fi
		checkpoint_file="${CHECKPOINT}"
		echo "checkpoint ready: ${checkpoint_file}"
	fi
	cd "${ROOT}/tdmpc2"
	# data_dir is unused by evaluation but must resolve the config's '???'.
	${PY} evaluate.py task="${TASK}" model_size="${MODEL_SIZE}" \
		checkpoint="${checkpoint_file}" \
		eval_episodes="${EVAL_EPISODES:-10}" \
		seed="${SEED:-1}" \
		save_video="${SAVE_VIDEO:-false}" \
		data_dir="${ROOT}/data" \
		enable_wandb=false
elif [[ "${MODE}" == "train" ]]; then
	cd "${ROOT}/tdmpc2"
	args=(task="${TASK}" model_size="${MODEL_SIZE}" steps="${STEPS:-1000000}" enable_wandb=false data_dir="${ROOT}/data")
	# Optional overrides passed via run_config.sh
	for key in batch_size seed obs mpc enable_wandb wandb_project simnorm_dim num_q latent_dim mlp_dim enc_dim num_enc_layers; do
		val="$(eval "echo \${${key}:-}")"
		if [[ -n "${val}" ]]; then
			args+=("${key}=${val}")
		fi
	done
	echo "train args: ${args[*]}"
	${PY} train.py "${args[@]}"
else
	echo "ERROR: unknown MODE '${MODE}'" >&2
	exit 1
fi
echo "======================================"
echo "TD-MPC2 reproduction run COMPLETE"
echo "======================================"
