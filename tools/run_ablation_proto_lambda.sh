#!/bin/bash
# ---------------------------------------------------------------
# Ablation: Proto Lambda (λ_proto) Sensitivity
#
# Runs 5 experiments sweeping proto_lambda: {0.0, 0.05, 0.1, 0.2, 0.5}
# Includes baseline (0.1) + A2 removal (0.0) + B2 variants.
#
# Usage:
#   bash tools/run_ablation_proto_lambda.sh [GPU_ID] [SEED]
#
# Examples:
#   bash tools/run_ablation_proto_lambda.sh 0 0
#   bash tools/run_ablation_proto_lambda.sh 1 42
# ---------------------------------------------------------------

set -e

GPU_ID=${1:-0}
SEED=${2:-0}

CONFIG_DIR="configs/daformer/ablation_ssl"
BASELINE="configs/daformer/satellite_ssl_dapcn_daformer_mitb5.py"

GREEN='\033[0;32m'
YELLOW='\033[1;33m'
CYAN='\033[0;36m'
RED='\033[0;31m'
NC='\033[0m'

log_info()  { echo -e "${CYAN}[INFO]${NC}  $1"; }
log_start() { echo -e "${YELLOW}[START]${NC} $1"; }
log_done()  { echo -e "${GREEN}[DONE]${NC}  $1"; }
log_fail()  { echo -e "${RED}[FAIL]${NC}  $1"; }

# ---------------------------------------------------------------
# Experiment list: proto_lambda = {0.0, 0.05, 0.1 (baseline), 0.2, 0.5}
# ---------------------------------------------------------------

declare -A EXPERIMENTS=(
    ["proto_lambda_000"]="${CONFIG_DIR}/A2_no_prototype.py"
    ["proto_lambda_005"]="${CONFIG_DIR}/B2_proto_lambda_005.py"
    ["proto_lambda_010_baseline"]="${BASELINE}"
    ["proto_lambda_020"]="${CONFIG_DIR}/B2_proto_lambda_02.py"
    ["proto_lambda_050"]="${CONFIG_DIR}/B2_proto_lambda_05.py"
)

# Ordered keys for sequential execution
ORDERED_KEYS=(
    "proto_lambda_000"
    "proto_lambda_005"
    "proto_lambda_010_baseline"
    "proto_lambda_020"
    "proto_lambda_050"
)

TOTAL=${#ORDERED_KEYS[@]}
log_info "Proto Lambda Ablation: ${TOTAL} experiments on GPU ${GPU_ID}, seed ${SEED}"
echo ""
echo "  proto_lambda = 0.0   (A2: no prototype loss)"
echo "  proto_lambda = 0.05  (B2a: low)"
echo "  proto_lambda = 0.1   (baseline)"
echo "  proto_lambda = 0.2   (B2b: moderate)"
echo "  proto_lambda = 0.5   (B2c: high)"
echo ""

IDX=0
FAILED=()

for KEY in "${ORDERED_KEYS[@]}"; do
    IDX=$((IDX + 1))
    CONFIG="${EXPERIMENTS[$KEY]}"
    WORK_DIR="work_dirs/ablation_ssl/${KEY}_s${SEED}"

    log_start "[${IDX}/${TOTAL}] ${KEY}"

    # Skip if already completed
    if [ -f "${WORK_DIR}/latest.pth" ]; then
        log_info "  Skipping — checkpoint exists at ${WORK_DIR}/latest.pth"
        continue
    fi

    mkdir -p "${WORK_DIR}"

    CUDA_VISIBLE_DEVICES=${GPU_ID} python tools/train.py \
        "${CONFIG}" \
        --work-dir "${WORK_DIR}" \
        --seed "${SEED}" \
        --deterministic \
        2>&1 | tee "${WORK_DIR}/train.log" || {
            log_fail "${KEY}"
            FAILED+=("${KEY}")
            continue
        }

    log_done "${KEY} — results at ${WORK_DIR}"
    echo ""
done

# ---------------------------------------------------------------
# Summary
# ---------------------------------------------------------------

echo ""
echo "========================================"
echo " Proto Lambda Ablation Complete"
echo "========================================"
echo " Sweep:     {0.0, 0.05, 0.1, 0.2, 0.5}"
echo " GPU:       ${GPU_ID}"
echo " Seed:      ${SEED}"
echo " Total:     ${TOTAL}"
echo " Succeeded: $((TOTAL - ${#FAILED[@]}))"
echo " Failed:    ${#FAILED[@]}"

if [ ${#FAILED[@]} -gt 0 ]; then
    echo ""
    echo " Failed experiments:"
    for F in "${FAILED[@]}"; do
        echo "   - ${F}"
    done
fi

echo ""
echo " Collect results:"
echo "   python tools/collect_ablation_results.py"
echo "========================================"
