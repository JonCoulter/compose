#!/bin/bash
# Download CUB-200-2011 from CaltechDATA and build a tiny MPO-compatible layout under <project>/datasets.
# https://data.caltech.edu/records/65de6-vp158
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
CUB_DIR="${PROJECT_ROOT}/datasets/cub"
ARCHIVE="${CUB_DIR}/CUB_200_2011.tgz"
ARCHIVE_URL="https://data.caltech.edu/records/65de6-vp158/files/CUB_200_2011.tgz?download=1"
MPO_DATA_DIR="${PROJECT_ROOT}/datasets"

mkdir -p "${CUB_DIR}"

if [[ ! -f "${ARCHIVE}" ]]; then
  echo "Downloading CUB_200_2011.tgz (~1.2 GB)..."
  wget -c -O "${ARCHIVE}.partial" "${ARCHIVE_URL}"
  mv "${ARCHIVE}.partial" "${ARCHIVE}"
else
  echo "Archive already present: ${ARCHIVE}"
fi

if [[ ! -d "${CUB_DIR}/CUB_200_2011/images" ]]; then
  echo "Extracting ${ARCHIVE}..."
  tar -xzf "${ARCHIVE}" -C "${CUB_DIR}"
else
  echo "Already extracted: ${CUB_DIR}/CUB_200_2011"
fi

echo "Building MPO cuckoo JSON + images symlink..."
# 3 cuckoo species have 60+53+59=172 images total; disjoint train+test cannot exceed 172 (86+86 uses all).
python3 "${SCRIPT_DIR}/build_cub_mpo_sample.py" \
  --cub-root "${CUB_DIR}" \
  --mpo-data-dir "${MPO_DATA_DIR}" \
  --task cuckoo \
  --classes 3 \
  --train-total 86 \
  --test-total 86

echo "Done. Use MPO with: --data_dir ${MPO_DATA_DIR}"
echo "Match TRAIN_SIZE/TEST_SIZE in MPO/main.sh (currently 86/86)."
