#!/bin/bash
set -e
exec > >(tee /workspace/run_calibration.log) 2>&1
echo "=== calibration-only runs: RF k-sweep + ESM15B acq sweep ==="

WORKDIR=$(mktemp -d /tmp/protein_XXXXXXXXXX)
git clone -q https://github.com/Erhad/protein.git $WORKDIR
cd $WORKDIR/v1

PY_VER=$(python3 -c "import sys; print('{}.{}'.format(sys.version_info.major, sys.version_info.minor))")
curl -sS -f "https://bootstrap.pypa.io/pip/${PY_VER}/get-pip.py" | python3 || curl -sS "https://bootstrap.pypa.io/get-pip.py" | python3
python3 -m pip install --prefer-binary numpy pandas scikit-learn joblib scipy -q --break-system-packages 2>/dev/null || \
python3 -m pip install --prefer-binary numpy pandas scikit-learn joblib scipy -q
python3 -m pip install --prefer-binary torch --index-url https://download.pytorch.org/whl/cpu -q --break-system-packages 2>/dev/null || \
python3 -m pip install --prefer-binary torch --index-url https://download.pytorch.org/whl/cpu -q

VOL=/workspace/v1/data
find "$VOL" \( -name "*.npy" -o -name "*.npz" -o -name "*.csv" \) | while read src; do
    rel="${src#$VOL/}"; dst="data/$rel"
    mkdir -p "$(dirname $dst)" && ln -sf "$src" "$dst"
done
echo "Data symlinked"

mkdir -p results/raw results/calibration
mkdir -p /workspace/v1/results/raw /workspace/v1/results/calibration

# Symlink existing JSONL so skip-logic works
find /workspace/v1/results/raw -name "*.jsonl" | while read src; do
    ln -sf "$src" "results/raw/$(basename $src)"
done

# Helper: run cal_only for a landscape+method, copy npz on completion
run_cal() {
    local landscape=$1 method=$2
    echo "CAL  $landscape  $method"
    python3 experiments/run_batch.py \
        --method "$method" --landscapes "$landscape" \
        --batch_sizes 96 --seeds 100 --workers 1 --cal_only
    find results/calibration -name "*.npz" -newer /tmp/.cal_marker 2>/dev/null | while read f; do
        cp -n "$f" /workspace/v1/results/calibration/ 2>/dev/null || true
    done
}
touch /tmp/.cal_marker

# ── RF k-sweep: onehot + esm15b_mean, k=1/5/10/100, all 4 proteins ──────────
for PROT in gb1 trpb t7 tev; do
    for LAND_SUFFIX in onehot esm2_15b; do
        LAND="${PROT}_${LAND_SUFFIX}"
        for METHOD in rf_ts_k1 rf_ts_k5 rf_ts_k10; do
            run_cal "$LAND" "$METHOD"
        done
        run_cal "$LAND" evolvepro   # rfk100 greedy
    done
done

# ── ESM15B DNN acq sweep: dnn256 + dnn500, acq=ts/greedy/ucb/ei, all proteins
for PROT in gb1 trpb t7 tev; do
    LAND="${PROT}_esm2_15b"
    for METHOD in dnn_ts dnn_greedy dnn_ucb dnn_ei dnn_ts_s dnn_greedy_s dnn_ucb_s dnn_ei_s; do
        run_cal "$LAND" "$METHOD"
    done
done

echo "=== calibration done ==="
