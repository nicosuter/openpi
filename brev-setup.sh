set -euo pipefail

SECRETS_FILE="${BREV_SETUP_SECRETS_FILE:-secrets}"

if [ ! -f "$SECRETS_FILE" ]; then
  echo "Expected secrets file at $SECRETS_FILE" >&2
  exit 1
fi

source "$SECRETS_FILE"

if [ -z "${HF_TOKEN:-}" ]; then
  echo "HF_TOKEN must be set in $SECRETS_FILE" >&2
  exit 1
fi

if [ -z "${WANDB_API_KEY:-}" ]; then
  echo "WANDB_API_KEY must be set in $SECRETS_FILE" >&2
  exit 1
fi

git submodule update --init --recursive

curl -LsSf https://astral.sh/uv/install.sh | sh

if [ ! -d ../google-cloud-sdk ]; then
  export CLOUDSDK_CORE_DISABLE_PROMPTS=1
  curl -sSL https://sdk.cloud.google.com | bash

  "$HOME/google-cloud-sdk/install.sh" --quiet \
    --path-update=true \
    --command-completion=true \
    --rc-path="$HOME/.bashrc"
fi

source "$HOME/.bashrc"

uv add "huggingface_hub[cli]"

GIT_LFS_SKIP_SMUDGE=1 uv sync
GIT_LFS_SKIP_SMUDGE=1 uv pip install -e .

uv run huggingface-cli login --token "$HF_TOKEN"
uv run wandb login --relogin "$WANDB_API_KEY"

XLA_PYTHON_CLIENT_MEM_FRACTION=0.9 uv run scripts/train.py pi05_libero --exp-name=my_experiment --overwrite # fails after downloading data
uv run scripts/compute_norm_stats.py --config-name pi05_libero # normalizes data
XLA_PYTHON_CLIENT_MEM_FRACTION=0.9 uv run scripts/train.py pi05_libero --exp-name=my_experiment --overwrite # finally trains
