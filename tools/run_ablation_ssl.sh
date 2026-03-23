#!/bin/bash
# ---------------------------------------------------------------
# Ablation Study Runner for DAPCN-SSL Satellite Segmentation
#
# Usage:
#   bash tools/run_ablation_ssl.sh [GROUP] [GPU_ID] [SEED]
#
# Arguments:
#   GROUP   - Which ablation group to run:
#               all       : Run all experiments (default)
#               baseline  : Only the full model (reference)
#               component : Group A — component ablation
#               hyperparam: Group B — hyperparameter sensitivity
#               boundary  : Group C — boundary mode comparison
#   GPU_ID  - CUDA device index (default: 0)
#   SEED    - Random seed (default: 0)
#
# Examples:
#   bash tools/run_ablation_ssl.sh all 0 0
#   bash tools/run_ablation_ssl.sh component 1 42
#   bash tools/run_ablation_ssl.sh boundary 0 0
#
# Output:
#   work_dirs/ablation_ssl/<config_name>/  per experiment
# ---------------------------------------------------------------

set -e

GROUP=${1:-all}
GPU_ID=${2:-0}
SEED=${3:-0}

CONFIG_DIR="configs/daformer/ablation_ssl"
BASELINE_CONFIG="configs/daformer/satellite_ssl_dapcn_daformer_mitb5.py"

# Colour output helpers
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
CYAN='\033[0;36m'
NC='\033[0m' # No Color

log_info()  { echo -e "${CYAN}[INFO]${NC}  $1"; }
log_start() { echo -e "${YELLOW}[START]${NC} $1"; }
log_done()  { echo -e "${GREEN}[DONE]${NC}  $1"; }

# ---------------------------------------------------------------
# Define experiment groups
# ---------------------------------------------------------------

BASELINE=(
    "${BASELINE_CONFIG}"
)

COMPONENT=(
    "${CONFIG_DIR}/A1_no_boundary.py"
    "${CONFIG_DIR}/A2_no_prototype.py"
    "${CONFIG_DIR}/A3_no_warmup.py"
    "${CONFIG_DIR}/A4_no_classmix.py"
    "${CONFIG_DIR}/A5_no_rcs.py"
    "${CONFIG_DIR}/A6_baseline_selftraining.py"
)

HYPERPARAM=(
    "${CONFIG_DIR}/B1_boundary_lambda_01.py"
    "${CONFIG_DIR}/B1_boundary_lambda_03.py"
    "${CONFIG_DIR}/B1_boundary_lambda_07.py"
    "${CONFIG_DIR}/B1_boundary_lambda_10.py"
    "${CONFIG_DIR}/B2_proto_lambda_005.py"
    "${CONFIG_DIR}/B2_proto_lambda_02.py"
    "${CONFIG_DIR}/B2_proto_lambda_05.py"
    "${CONFIG_DIR}/B3_max_groups_32.py"
    "${CONFIG_DIR}/B3_max_groups_64.py"
    "${CONFIG_DIR}/B3_max_groups_128.py"
    "${CONFIG_DIR}/B4_warmup_500.py"
    "${CONFIG_DIR}/B4_warmup_2000.py"
    "${CONFIG_DIR}/B4_warmup_5000.py"
)

BOUNDARY=(
    "${CONFIG_DIR}/C1_boundary_sobel.py"
    "${CONFIG_DIR}/C2_boundary_laplacian.py"
    "${CONFIG_DIR}/C3_boundary_hybrid.py"
)

# ---------------------------------------------------------------
# Assemble experiment list
# ---------------------------------------------------------------

EXPERIMENTS=()

case "${GROUP}" in
    all)
        EXPERIMENTS+=("${BASELINE[@]}" "${COMPONENT[@]}" "${HYPERPARAM[@]}" "${BOUNDARY[@]}")
        ;;
    baseline)
        EXPERIMENTS+=("${BASELINE[@]}")
        ;;
    component)
        EXPERIMENTS+=("${BASELINE[@]}" "${COMPONENT[@]}")
        ;;
    hyperparam)
        EXPERIMENTS+=("${BASELINE[@]}" "${HYPERPARAM[@]}")
        ;;
    boundary)
        EXPERIMENTS+=("${BASELINE[@]}" "${BOUNDARY[@]}")
        ;;
    *)
        echo "Unknown group: ${GROUP}"
        echo "Valid groups: all, baseline, component, hyperparam, boundary"
        exit 1
        ;;
esac

TOTAL=${#EXPERIMENTS[@]}
log_info "Ablation study: ${GROUP} group — ${TOTAL} experiments on GPU ${GPU_ID}, seed ${SEED}"
echo ""

# ---------------------------------------------------------------
# Run experiments sequentially
# ---------------------------------------------------------------

IDX=0
FAILED=()

for CONFIG in "${EXPERIMENTS[@]}"; do
    IDX=$((IDX + 1))
    EXP_NAME=$(basename "${CONFIG}" .py)
    WORK_DIR="work_dirs/ablation_ssl/${EXP_NAME}_s${SEED}"

    log_start "[${IDX}/${TOTAL}] ${EXP_NAME}"

    if [ -f "${WORK_DIR}/latest.pth" ]; then
        log_info "  Skipping — checkpoint already exists at ${WORK_DIR}/latest.pth"
        continue
    fi

    CUDA_VISIBLE_DEVICES=${GPU_ID} python tools/train.py \
        "${CONFIG}" \
        --work-dir "${WORK_DIR}" \
        --seed "${SEED}" \
        --deterministic \
        2>&1 | tee "${WORK_DIR}/train.log" || {
            echo -e "\033[0;31m[FAIL]\033[0m ${EXP_NAME}"
            FAILED+=("${EXP_NAME}")
            continue
        }

    log_done "${EXP_NAME} — results at ${WORK_DIR}"
    echo ""
done

# ---------------------------------------------------------------
# Summary
# ---------------------------------------------------------------

echo ""
echo "========================================"
echo " Ablation Study Complete"
echo "========================================"
echo " Group:       ${GROUP}"
echo " Total:       ${TOTAL}"
echo " Succeeded:   $((TOTAL - ${#FAILED[@]}))"
echo " Failed:      ${#FAILED[@]}"

if [ ${#FAILED[@]} -gt 0 ]; then
    echo ""
    echo " Failed experiments:"
    for F in "${FAILED[@]}"; do
        echo "   - ${F}"
    done
fi

echo "========================================"
