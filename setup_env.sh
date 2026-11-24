#!/usr/bin/env bash
# Create the conda environment `cabsfda` (Python 3.11, PyTorch 2.7.1, pinned packages) and check it.
#
#   bash setup_env.sh                 # picks the PyTorch CUDA build from nvidia-smi
#   TORCH_CUDA=cu118 bash setup_env.sh  # force a build: cu118 | cu126 | cu128
#   ENV_NAME=cabsfda2 bash setup_env.sh
#
# CUDA builds of torch 2.7.1: cu126 (PyPI default, driver >= 525), cu118 (driver >= 450),
# cu128 (needed for Blackwell GPUs such as B200 / RTX 50xx; driver >= 570).
set -eo pipefail          # no -u: conda activate scripts use unset variables
cd "$(dirname "$0")"
ENV_NAME=${ENV_NAME:-cabsfda}

if ! command -v conda >/dev/null; then
  echo "conda not found (run 'source ~/miniconda3/etc/profile.d/conda.sh' or install Miniconda)"; exit 1
fi
eval "$(conda shell.bash hook)"

if [[ -z "${TORCH_CUDA:-}" ]]; then
  if command -v nvidia-smi >/dev/null; then
    DRIVER=$(nvidia-smi --query-gpu=driver_version --format=csv,noheader | head -1)
    GPU=$(nvidia-smi --query-gpu=name --format=csv,noheader | head -1)
    MAJOR=${DRIVER%%.*}
    if [[ "$GPU" =~ (B100|B200|GB200|RTX\ 50|RTX\ PRO\ 6000\ Blackwell) ]]; then TORCH_CUDA=cu128
    elif (( MAJOR >= 525 )); then TORCH_CUDA=cu126
    else TORCH_CUDA=cu118; fi
    echo "GPU: $GPU, driver $DRIVER -> PyTorch build $TORCH_CUDA"
  else
    echo "nvidia-smi not found: installing the default (cu126) build"; TORCH_CUDA=cu126
  fi
fi

if conda env list | awk '{print $1}' | grep -qx "$ENV_NAME"; then
  echo "conda env '$ENV_NAME' exists; remove it first (conda env remove -n $ENV_NAME) or set ENV_NAME"; exit 1
fi
conda create -y -n "$ENV_NAME" python=3.11 pip
conda activate "$ENV_NAME"
python -m pip install --upgrade pip
if [[ "$TORCH_CUDA" == "cu126" ]]; then
  pip install torch==2.7.1 torchvision==0.22.1
else
  pip install torch==2.7.1 torchvision==0.22.1 --index-url "https://download.pytorch.org/whl/$TORCH_CUDA"
fi
pip install -r requirements.txt

python - <<'EOF'
import torch, torchvision
print("torch", torch.__version__, "CUDA", torch.version.cuda, "| torchvision", torchvision.__version__)
print("GPUs visible:", torch.cuda.device_count())
assert torch.cuda.is_available(), "CUDA not available: wrong PyTorch build for this driver (set TORCH_CUDA)"
x = torch.randn(256, 256, device="cuda"); print("GPU matmul ok on", torch.cuda.get_device_name(0), tuple((x @ x).shape))
EOF

# ImageNet weights (ResNet-18/50/101) into ~/.cache/torch, so jobs never download in parallel
python -c "import torchvision.models as m; m.resnet18(weights='IMAGENET1K_V1'); m.resnet50(weights='IMAGENET1K_V2'); m.resnet101(weights='IMAGENET1K_V2'); print('ImageNet weights cached')"
echo "environment '$ENV_NAME' ready: conda activate $ENV_NAME"
