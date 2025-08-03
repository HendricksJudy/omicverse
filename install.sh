#!/usr/bin/env bash
set -euo pipefail

#————————————————————————
# Check Python version (only 3.10 and 3.11 supported)
#————————————————————————
PYTHON_VERSION=$(python -c 'import sys; print(f"{sys.version_info.major}.{sys.version_info.minor}")')
if [[ "$PYTHON_VERSION" != "3.10" && "$PYTHON_VERSION" != "3.11" ]]; then
  echo "❌ Error: Python version $PYTHON_VERSION is not supported"
  echo "   OmicVerse requires Python 3.10 or 3.11"
  exit 1
fi
echo "✅ Python $PYTHON_VERSION detected"

#————————————————————————
# 0. Speed‐test PyPI mirrors & pick the fastest 📡
#————————————————————————
MIRRORS=(
  "https://pypi.tuna.tsinghua.edu.cn/simple"
  "https://pypi.org/simple"
)
declare -A LATENCIES

echo "⏳ Testing mirror latencies, please wait..."
for m in "${MIRRORS[@]}"; do
  t=$(curl -o /dev/null -s -w "%{time_total}" --connect-timeout 3 -I "$m" || echo 999)
  LATENCIES["$m"]=$t
  echo "  $m → ${t}s"
done

BEST_MIRROR="${MIRRORS[0]}"
best_time=${LATENCIES[$BEST_MIRROR]}
for m in "${MIRRORS[@]}"; do
  if (( $(echo "${LATENCIES[$m]} < $best_time" | bc -l) )); then
    BEST_MIRROR=$m
    best_time=${LATENCIES[$m]}
  fi
done
echo "✔️ Selected fastest mirror: $BEST_MIRROR"
PIP_INDEX="-i $BEST_MIRROR"

#————————————————————————
# helper: install a conda pkg if missing 🐍
#————————————————————————
conda_install_pkg(){
  pkg=$1
  if conda list --no-pip | awk '{print $1}' | grep -xq "$pkg"; then
    echo "✅ Skipping conda:$pkg (already installed)"
  else
    echo "🔄 Installing conda:$pkg"
    conda install -c conda-forge -y "$pkg"
  fi
}

#————————————————————————
# helper: install pip pkgs if missing 🛠️
#————————————————————————
pip_install_pkg(){
  missing=()
  for pkg in "$@"; do
    if pip show "$pkg" >/dev/null 2>&1; then
      echo "✅ Already have pip:$pkg"
    else
      echo "❌ Missing pip:$pkg"
      missing+=("$pkg")
    fi
  done

  if [ "${#missing[@]}" -gt 0 ]; then
    echo "🔄 Installing missing pip packages: ${missing[*]}"
    pip install "${missing[@]}" $PIP_INDEX
  else
    echo "✅ All pip packages already installed"
  fi
}

#————————————————————————
# 1. Conda: core packages 🐾
#————————————————————————
conda_install_pkg s_gd2
conda_install_pkg opencv

#————————————————————————
# 2. Torch: use existing or install latest 🔥
#————————————————————————
if pip show torch >/dev/null 2>&1; then
  TORCH_VERSION="$(python - << 'PYCODE'
import torch
print(torch.__version__.split("+")[0])
PYCODE
)"
  echo "⚡ Detected local torch==$TORCH_VERSION, skipping installation"
else
  # auto-detect latest torch.*.* version
  TORCH_VERSION="$(pip index versions torch 2>/dev/null \
    | grep -oP 'Available versions: \K[0-9]+\.[0-9]+\.[0-9]+' \
    | head -1)"
  echo "🌟 Installing torch==$TORCH_VERSION and letting pip pick matching torchvision/torchaudio"
  pip install \
    torch \
    torchvision \
    torchaudio \
    $PIP_INDEX
fi

#————————————————————————
# 3. Detect CUDA & prepare PyG wheel URL 🚀
#————————————————————————
CUDA_TAG="$(python - << 'PYCODE'
import torch
if torch.cuda.is_available() and torch.version.cuda:
    print("cu" + torch.version.cuda.replace(".", ""))
else:
    print("cpu")
PYCODE
)"
echo "🔍 CUDA tag: $CUDA_TAG"
PYG_WHL_URL="https://data.pyg.org/whl/torch-${TORCH_VERSION}+${CUDA_TAG}.html"
echo "🔗 PyG wheel index: $PYG_WHL_URL"

#————————————————————————
# 4. Install PyG & extensions 🧩
#————————————————————————
pip_install_pkg torch_geometric


#————————————————————————
# 5. Install OmicVerse 🧬
#————————————————————————
pip_install_pkg omicverse

#————————————————————————
# 6. Other deep‐bio packages 🌱
#————————————————————————
pip_install_pkg \
  tangram-sc \
  fa2-modified \
  pot \
  libpysal \
  openai \
  patsy \
  combat

pip_install_pkg \
  pymde \
  opencv-python \
  scikit-image \
  memento-de

#————————————————————————
# 7. Dynamics & analysis tools 🔬
#————————————————————————
pip_install_pkg \
  harmonypy \
  intervaltree \
  fbpca \
  scvi-tools \
  mofax \
  metatime \
  s-gd2

pip_install_pkg \
  mellon \
  scvelo \
  cellrank \
  einops \
  dynamo-release \
  squidpy \
  pertpy

pip_install_pkg \
  toytree \
  arviz \
  ete3 \
  pymde \
  torchdr



#————————————————————————
# 8. Version‐locked packages 🔒
#————————————————————————


#————————————————————————
# 9. Miscellaneous tools 🛠️
#————————————————————————
#pip_install_pkg backports.tarfile openpyxl 

python -c "import omicverse as ov; ov.plot_set()"

echo "🎉 All set! (torch==$TORCH_VERSION, CUDA tag==$CUDA_TAG) 🚀"
