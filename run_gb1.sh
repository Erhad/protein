#!/bin/bash
set -e
exec > >(tee /workspace/run_gb1.log) 2>&1
echo "=== gb1 pod starting — 30 jobs ==="

WORKDIR=$(mktemp -d /tmp/protein_XXXXXXXXXX)
git clone -q https://github.com/Erhad/protein.git $WORKDIR
cd $WORKDIR/v1

PY_VER=$(python3 -c "import sys; print('{}.{}'.format(sys.version_info.major, sys.version_info.minor))")
curl -sS -f "https://bootstrap.pypa.io/pip/${PY_VER}/get-pip.py" | python3 || curl -sS "https://bootstrap.pypa.io/get-pip.py" | python3
python3 -m pip install --prefer-binary numpy pandas scikit-learn joblib -q --break-system-packages 2>/dev/null || python3 -m pip install --prefer-binary numpy pandas scikit-learn joblib -q
python3 -m pip install --prefer-binary torch --index-url https://download.pytorch.org/whl/cpu -q --break-system-packages 2>/dev/null || python3 -m pip install --prefer-binary torch --index-url https://download.pytorch.org/whl/cpu -q

VOL=/workspace/v1/data
find "$VOL" \( -name "*.npy" -o -name "*.npz" -o -name "*.csv" \) | while read src; do
    rel="${src#$VOL/}"; dst="data/$rel"
    mkdir -p "$(dirname $dst)" && ln -sf "$src" "$dst"
done
echo "Data symlinked"

mkdir -p results/raw results/calibration
mkdir -p /workspace/results/raw /workspace/results/calibration
find /workspace/results/raw -name "*.jsonl" | while read src; do
    ln -sf "$src" "results/raw/$(basename $src)"
done
find /workspace/results/raw -name "*.jsonl" | while read src; do
    ln -sf "$src" "results/raw/$(basename $src)"
done

# gb1_onehot / rf_ts_k5
RUN_NAME=$(python3 -c "import sys; sys.path.insert(0,'.'); from experiments.run_single import _make_run_name; print(_make_run_name('gb1_onehot','rf_ts_k5',96,None,False,True))")
RESULT_FILE="/workspace/results/raw/${RUN_NAME}.jsonl"
if [ -f "$RESULT_FILE" ] && [ "$(wc -l < "$RESULT_FILE")" -ge 100 ]; then
    echo "SKIP gb1_onehot rf_ts_k5"
else
    echo "RUN  gb1_onehot rf_ts_k5"
    python3 experiments/run_batch.py --method rf_ts_k5 --landscapes gb1_onehot --batch_sizes 96 --seeds 100 --workers 8 --track_calibration --double_mut_init
    cp results/raw/*.jsonl /workspace/results/raw/ 2>/dev/null || true
    mv results/calibration/*.npz /workspace/results/calibration/ 2>/dev/null || true
fi

# gb1_onehot / dnn_ts
RUN_NAME=$(python3 -c "import sys; sys.path.insert(0,'.'); from experiments.run_single import _make_run_name; print(_make_run_name('gb1_onehot','dnn_ts',96,None,False,True))")
RESULT_FILE="/workspace/results/raw/${RUN_NAME}.jsonl"
if [ -f "$RESULT_FILE" ] && [ "$(wc -l < "$RESULT_FILE")" -ge 100 ]; then
    echo "SKIP gb1_onehot dnn_ts"
else
    echo "RUN  gb1_onehot dnn_ts"
    python3 experiments/run_batch.py --method dnn_ts --landscapes gb1_onehot --batch_sizes 96 --seeds 100 --workers 8 --track_calibration --double_mut_init
    cp results/raw/*.jsonl /workspace/results/raw/ 2>/dev/null || true
    mv results/calibration/*.npz /workspace/results/calibration/ 2>/dev/null || true
fi

# gb1_onehot / dnn_ts_s
RUN_NAME=$(python3 -c "import sys; sys.path.insert(0,'.'); from experiments.run_single import _make_run_name; print(_make_run_name('gb1_onehot','dnn_ts_s',96,None,False,True))")
RESULT_FILE="/workspace/results/raw/${RUN_NAME}.jsonl"
if [ -f "$RESULT_FILE" ] && [ "$(wc -l < "$RESULT_FILE")" -ge 100 ]; then
    echo "SKIP gb1_onehot dnn_ts_s"
else
    echo "RUN  gb1_onehot dnn_ts_s"
    python3 experiments/run_batch.py --method dnn_ts_s --landscapes gb1_onehot --batch_sizes 96 --seeds 100 --workers 8 --track_calibration --double_mut_init
    cp results/raw/*.jsonl /workspace/results/raw/ 2>/dev/null || true
    mv results/calibration/*.npz /workspace/results/calibration/ 2>/dev/null || true
fi

# gb1_esmc_mean / rf_ts_k5
RUN_NAME=$(python3 -c "import sys; sys.path.insert(0,'.'); from experiments.run_single import _make_run_name; print(_make_run_name('gb1_esmc_mean','rf_ts_k5',96,None,False,True))")
RESULT_FILE="/workspace/results/raw/${RUN_NAME}.jsonl"
if [ -f "$RESULT_FILE" ] && [ "$(wc -l < "$RESULT_FILE")" -ge 100 ]; then
    echo "SKIP gb1_esmc_mean rf_ts_k5"
else
    echo "RUN  gb1_esmc_mean rf_ts_k5"
    python3 experiments/run_batch.py --method rf_ts_k5 --landscapes gb1_esmc_mean --batch_sizes 96 --seeds 100 --workers 8 --track_calibration --double_mut_init
    cp results/raw/*.jsonl /workspace/results/raw/ 2>/dev/null || true
    mv results/calibration/*.npz /workspace/results/calibration/ 2>/dev/null || true
fi

# gb1_esmc_mean / dnn_ts
RUN_NAME=$(python3 -c "import sys; sys.path.insert(0,'.'); from experiments.run_single import _make_run_name; print(_make_run_name('gb1_esmc_mean','dnn_ts',96,None,False,True))")
RESULT_FILE="/workspace/results/raw/${RUN_NAME}.jsonl"
if [ -f "$RESULT_FILE" ] && [ "$(wc -l < "$RESULT_FILE")" -ge 100 ]; then
    echo "SKIP gb1_esmc_mean dnn_ts"
else
    echo "RUN  gb1_esmc_mean dnn_ts"
    python3 experiments/run_batch.py --method dnn_ts --landscapes gb1_esmc_mean --batch_sizes 96 --seeds 100 --workers 8 --track_calibration --double_mut_init
    cp results/raw/*.jsonl /workspace/results/raw/ 2>/dev/null || true
    mv results/calibration/*.npz /workspace/results/calibration/ 2>/dev/null || true
fi

# gb1_esmc_mean / dnn_ts_s
RUN_NAME=$(python3 -c "import sys; sys.path.insert(0,'.'); from experiments.run_single import _make_run_name; print(_make_run_name('gb1_esmc_mean','dnn_ts_s',96,None,False,True))")
RESULT_FILE="/workspace/results/raw/${RUN_NAME}.jsonl"
if [ -f "$RESULT_FILE" ] && [ "$(wc -l < "$RESULT_FILE")" -ge 100 ]; then
    echo "SKIP gb1_esmc_mean dnn_ts_s"
else
    echo "RUN  gb1_esmc_mean dnn_ts_s"
    python3 experiments/run_batch.py --method dnn_ts_s --landscapes gb1_esmc_mean --batch_sizes 96 --seeds 100 --workers 8 --track_calibration --double_mut_init
    cp results/raw/*.jsonl /workspace/results/raw/ 2>/dev/null || true
    mv results/calibration/*.npz /workspace/results/calibration/ 2>/dev/null || true
fi

# gb1_esmc_sitemean / rf_ts_k5
RUN_NAME=$(python3 -c "import sys; sys.path.insert(0,'.'); from experiments.run_single import _make_run_name; print(_make_run_name('gb1_esmc_sitemean','rf_ts_k5',96,None,False,True))")
RESULT_FILE="/workspace/results/raw/${RUN_NAME}.jsonl"
if [ -f "$RESULT_FILE" ] && [ "$(wc -l < "$RESULT_FILE")" -ge 100 ]; then
    echo "SKIP gb1_esmc_sitemean rf_ts_k5"
else
    echo "RUN  gb1_esmc_sitemean rf_ts_k5"
    python3 experiments/run_batch.py --method rf_ts_k5 --landscapes gb1_esmc_sitemean --batch_sizes 96 --seeds 100 --workers 8 --track_calibration --double_mut_init
    cp results/raw/*.jsonl /workspace/results/raw/ 2>/dev/null || true
    mv results/calibration/*.npz /workspace/results/calibration/ 2>/dev/null || true
fi

# gb1_esmc_sitemean / dnn_ts
RUN_NAME=$(python3 -c "import sys; sys.path.insert(0,'.'); from experiments.run_single import _make_run_name; print(_make_run_name('gb1_esmc_sitemean','dnn_ts',96,None,False,True))")
RESULT_FILE="/workspace/results/raw/${RUN_NAME}.jsonl"
if [ -f "$RESULT_FILE" ] && [ "$(wc -l < "$RESULT_FILE")" -ge 100 ]; then
    echo "SKIP gb1_esmc_sitemean dnn_ts"
else
    echo "RUN  gb1_esmc_sitemean dnn_ts"
    python3 experiments/run_batch.py --method dnn_ts --landscapes gb1_esmc_sitemean --batch_sizes 96 --seeds 100 --workers 8 --track_calibration --double_mut_init
    cp results/raw/*.jsonl /workspace/results/raw/ 2>/dev/null || true
    mv results/calibration/*.npz /workspace/results/calibration/ 2>/dev/null || true
fi

# gb1_esmc_sitemean / dnn_ts_s
RUN_NAME=$(python3 -c "import sys; sys.path.insert(0,'.'); from experiments.run_single import _make_run_name; print(_make_run_name('gb1_esmc_sitemean','dnn_ts_s',96,None,False,True))")
RESULT_FILE="/workspace/results/raw/${RUN_NAME}.jsonl"
if [ -f "$RESULT_FILE" ] && [ "$(wc -l < "$RESULT_FILE")" -ge 100 ]; then
    echo "SKIP gb1_esmc_sitemean dnn_ts_s"
else
    echo "RUN  gb1_esmc_sitemean dnn_ts_s"
    python3 experiments/run_batch.py --method dnn_ts_s --landscapes gb1_esmc_sitemean --batch_sizes 96 --seeds 100 --workers 8 --track_calibration --double_mut_init
    cp results/raw/*.jsonl /workspace/results/raw/ 2>/dev/null || true
    mv results/calibration/*.npz /workspace/results/calibration/ 2>/dev/null || true
fi

# gb1_esm2_15b / rf_ts_k5
RUN_NAME=$(python3 -c "import sys; sys.path.insert(0,'.'); from experiments.run_single import _make_run_name; print(_make_run_name('gb1_esm2_15b','rf_ts_k5',96,None,False,True))")
RESULT_FILE="/workspace/results/raw/${RUN_NAME}.jsonl"
if [ -f "$RESULT_FILE" ] && [ "$(wc -l < "$RESULT_FILE")" -ge 100 ]; then
    echo "SKIP gb1_esm2_15b rf_ts_k5"
else
    echo "RUN  gb1_esm2_15b rf_ts_k5"
    python3 experiments/run_batch.py --method rf_ts_k5 --landscapes gb1_esm2_15b --batch_sizes 96 --seeds 100 --workers 8 --track_calibration --double_mut_init
    cp results/raw/*.jsonl /workspace/results/raw/ 2>/dev/null || true
    mv results/calibration/*.npz /workspace/results/calibration/ 2>/dev/null || true
fi

# gb1_esm2_15b / dnn_ts
RUN_NAME=$(python3 -c "import sys; sys.path.insert(0,'.'); from experiments.run_single import _make_run_name; print(_make_run_name('gb1_esm2_15b','dnn_ts',96,None,False,True))")
RESULT_FILE="/workspace/results/raw/${RUN_NAME}.jsonl"
if [ -f "$RESULT_FILE" ] && [ "$(wc -l < "$RESULT_FILE")" -ge 100 ]; then
    echo "SKIP gb1_esm2_15b dnn_ts"
else
    echo "RUN  gb1_esm2_15b dnn_ts"
    python3 experiments/run_batch.py --method dnn_ts --landscapes gb1_esm2_15b --batch_sizes 96 --seeds 100 --workers 8 --track_calibration --double_mut_init
    cp results/raw/*.jsonl /workspace/results/raw/ 2>/dev/null || true
    mv results/calibration/*.npz /workspace/results/calibration/ 2>/dev/null || true
fi

# gb1_esm2_15b / dnn_ts_s
RUN_NAME=$(python3 -c "import sys; sys.path.insert(0,'.'); from experiments.run_single import _make_run_name; print(_make_run_name('gb1_esm2_15b','dnn_ts_s',96,None,False,True))")
RESULT_FILE="/workspace/results/raw/${RUN_NAME}.jsonl"
if [ -f "$RESULT_FILE" ] && [ "$(wc -l < "$RESULT_FILE")" -ge 100 ]; then
    echo "SKIP gb1_esm2_15b dnn_ts_s"
else
    echo "RUN  gb1_esm2_15b dnn_ts_s"
    python3 experiments/run_batch.py --method dnn_ts_s --landscapes gb1_esm2_15b --batch_sizes 96 --seeds 100 --workers 8 --track_calibration --double_mut_init
    cp results/raw/*.jsonl /workspace/results/raw/ 2>/dev/null || true
    mv results/calibration/*.npz /workspace/results/calibration/ 2>/dev/null || true
fi

# gb1_esm2_15b_sitemean / rf_ts_k5
RUN_NAME=$(python3 -c "import sys; sys.path.insert(0,'.'); from experiments.run_single import _make_run_name; print(_make_run_name('gb1_esm2_15b_sitemean','rf_ts_k5',96,None,False,True))")
RESULT_FILE="/workspace/results/raw/${RUN_NAME}.jsonl"
if [ -f "$RESULT_FILE" ] && [ "$(wc -l < "$RESULT_FILE")" -ge 100 ]; then
    echo "SKIP gb1_esm2_15b_sitemean rf_ts_k5"
else
    echo "RUN  gb1_esm2_15b_sitemean rf_ts_k5"
    python3 experiments/run_batch.py --method rf_ts_k5 --landscapes gb1_esm2_15b_sitemean --batch_sizes 96 --seeds 100 --workers 8 --track_calibration --double_mut_init
    cp results/raw/*.jsonl /workspace/results/raw/ 2>/dev/null || true
    mv results/calibration/*.npz /workspace/results/calibration/ 2>/dev/null || true
fi

# gb1_esm2_15b_sitemean / dnn_ts
RUN_NAME=$(python3 -c "import sys; sys.path.insert(0,'.'); from experiments.run_single import _make_run_name; print(_make_run_name('gb1_esm2_15b_sitemean','dnn_ts',96,None,False,True))")
RESULT_FILE="/workspace/results/raw/${RUN_NAME}.jsonl"
if [ -f "$RESULT_FILE" ] && [ "$(wc -l < "$RESULT_FILE")" -ge 100 ]; then
    echo "SKIP gb1_esm2_15b_sitemean dnn_ts"
else
    echo "RUN  gb1_esm2_15b_sitemean dnn_ts"
    python3 experiments/run_batch.py --method dnn_ts --landscapes gb1_esm2_15b_sitemean --batch_sizes 96 --seeds 100 --workers 8 --track_calibration --double_mut_init
    cp results/raw/*.jsonl /workspace/results/raw/ 2>/dev/null || true
    mv results/calibration/*.npz /workspace/results/calibration/ 2>/dev/null || true
fi

# gb1_esm2_15b_sitemean / dnn_ts_s
RUN_NAME=$(python3 -c "import sys; sys.path.insert(0,'.'); from experiments.run_single import _make_run_name; print(_make_run_name('gb1_esm2_15b_sitemean','dnn_ts_s',96,None,False,True))")
RESULT_FILE="/workspace/results/raw/${RUN_NAME}.jsonl"
if [ -f "$RESULT_FILE" ] && [ "$(wc -l < "$RESULT_FILE")" -ge 100 ]; then
    echo "SKIP gb1_esm2_15b_sitemean dnn_ts_s"
else
    echo "RUN  gb1_esm2_15b_sitemean dnn_ts_s"
    python3 experiments/run_batch.py --method dnn_ts_s --landscapes gb1_esm2_15b_sitemean --batch_sizes 96 --seeds 100 --workers 8 --track_calibration --double_mut_init
    cp results/raw/*.jsonl /workspace/results/raw/ 2>/dev/null || true
    mv results/calibration/*.npz /workspace/results/calibration/ 2>/dev/null || true
fi

# gb1_onehot / random
RUN_NAME=$(python3 -c "import sys; sys.path.insert(0,'.'); from experiments.run_single import _make_run_name; print(_make_run_name('gb1_onehot','random',96,None,False,False))")
RESULT_FILE="/workspace/results/raw/${RUN_NAME}.jsonl"
if [ -f "$RESULT_FILE" ] && [ "$(wc -l < "$RESULT_FILE")" -ge 100 ]; then
    echo "SKIP gb1_onehot random"
else
    echo "RUN  gb1_onehot random"
    python3 experiments/run_batch.py --method random --landscapes gb1_onehot --batch_sizes 96 --seeds 100 --workers 8  
    cp results/raw/*.jsonl /workspace/results/raw/ 2>/dev/null || true
    mv results/calibration/*.npz /workspace/results/calibration/ 2>/dev/null || true
fi

# gb1_esm2_15b / dnn_greedy
RUN_NAME=$(python3 -c "import sys; sys.path.insert(0,'.'); from experiments.run_single import _make_run_name; print(_make_run_name('gb1_esm2_15b','dnn_greedy',96,None,False,True))")
RESULT_FILE="/workspace/results/raw/${RUN_NAME}.jsonl"
if [ -f "$RESULT_FILE" ] && [ "$(wc -l < "$RESULT_FILE")" -ge 100 ]; then
    echo "SKIP gb1_esm2_15b dnn_greedy"
else
    echo "RUN  gb1_esm2_15b dnn_greedy"
    python3 experiments/run_batch.py --method dnn_greedy --landscapes gb1_esm2_15b --batch_sizes 96 --seeds 100 --workers 8  --double_mut_init
    cp results/raw/*.jsonl /workspace/results/raw/ 2>/dev/null || true
    mv results/calibration/*.npz /workspace/results/calibration/ 2>/dev/null || true
fi

# gb1_esm2_15b / dnn_greedy_s
RUN_NAME=$(python3 -c "import sys; sys.path.insert(0,'.'); from experiments.run_single import _make_run_name; print(_make_run_name('gb1_esm2_15b','dnn_greedy_s',96,None,False,True))")
RESULT_FILE="/workspace/results/raw/${RUN_NAME}.jsonl"
if [ -f "$RESULT_FILE" ] && [ "$(wc -l < "$RESULT_FILE")" -ge 100 ]; then
    echo "SKIP gb1_esm2_15b dnn_greedy_s"
else
    echo "RUN  gb1_esm2_15b dnn_greedy_s"
    python3 experiments/run_batch.py --method dnn_greedy_s --landscapes gb1_esm2_15b --batch_sizes 96 --seeds 100 --workers 8  --double_mut_init
    cp results/raw/*.jsonl /workspace/results/raw/ 2>/dev/null || true
    mv results/calibration/*.npz /workspace/results/calibration/ 2>/dev/null || true
fi

# gb1_esm2_15b / dnn_ucb
RUN_NAME=$(python3 -c "import sys; sys.path.insert(0,'.'); from experiments.run_single import _make_run_name; print(_make_run_name('gb1_esm2_15b','dnn_ucb',96,None,False,True))")
RESULT_FILE="/workspace/results/raw/${RUN_NAME}.jsonl"
if [ -f "$RESULT_FILE" ] && [ "$(wc -l < "$RESULT_FILE")" -ge 100 ]; then
    echo "SKIP gb1_esm2_15b dnn_ucb"
else
    echo "RUN  gb1_esm2_15b dnn_ucb"
    python3 experiments/run_batch.py --method dnn_ucb --landscapes gb1_esm2_15b --batch_sizes 96 --seeds 100 --workers 8 --track_calibration --double_mut_init
    cp results/raw/*.jsonl /workspace/results/raw/ 2>/dev/null || true
    mv results/calibration/*.npz /workspace/results/calibration/ 2>/dev/null || true
fi

# gb1_esm2_15b / dnn_ucb_s
RUN_NAME=$(python3 -c "import sys; sys.path.insert(0,'.'); from experiments.run_single import _make_run_name; print(_make_run_name('gb1_esm2_15b','dnn_ucb_s',96,None,False,True))")
RESULT_FILE="/workspace/results/raw/${RUN_NAME}.jsonl"
if [ -f "$RESULT_FILE" ] && [ "$(wc -l < "$RESULT_FILE")" -ge 100 ]; then
    echo "SKIP gb1_esm2_15b dnn_ucb_s"
else
    echo "RUN  gb1_esm2_15b dnn_ucb_s"
    python3 experiments/run_batch.py --method dnn_ucb_s --landscapes gb1_esm2_15b --batch_sizes 96 --seeds 100 --workers 8 --track_calibration --double_mut_init
    cp results/raw/*.jsonl /workspace/results/raw/ 2>/dev/null || true
    mv results/calibration/*.npz /workspace/results/calibration/ 2>/dev/null || true
fi

# gb1_esm2_15b / dnn_ei
RUN_NAME=$(python3 -c "import sys; sys.path.insert(0,'.'); from experiments.run_single import _make_run_name; print(_make_run_name('gb1_esm2_15b','dnn_ei',96,None,False,True))")
RESULT_FILE="/workspace/results/raw/${RUN_NAME}.jsonl"
if [ -f "$RESULT_FILE" ] && [ "$(wc -l < "$RESULT_FILE")" -ge 100 ]; then
    echo "SKIP gb1_esm2_15b dnn_ei"
else
    echo "RUN  gb1_esm2_15b dnn_ei"
    python3 experiments/run_batch.py --method dnn_ei --landscapes gb1_esm2_15b --batch_sizes 96 --seeds 100 --workers 8 --track_calibration --double_mut_init
    cp results/raw/*.jsonl /workspace/results/raw/ 2>/dev/null || true
    mv results/calibration/*.npz /workspace/results/calibration/ 2>/dev/null || true
fi

# gb1_esm2_15b / dnn_ei_s
RUN_NAME=$(python3 -c "import sys; sys.path.insert(0,'.'); from experiments.run_single import _make_run_name; print(_make_run_name('gb1_esm2_15b','dnn_ei_s',96,None,False,True))")
RESULT_FILE="/workspace/results/raw/${RUN_NAME}.jsonl"
if [ -f "$RESULT_FILE" ] && [ "$(wc -l < "$RESULT_FILE")" -ge 100 ]; then
    echo "SKIP gb1_esm2_15b dnn_ei_s"
else
    echo "RUN  gb1_esm2_15b dnn_ei_s"
    python3 experiments/run_batch.py --method dnn_ei_s --landscapes gb1_esm2_15b --batch_sizes 96 --seeds 100 --workers 8 --track_calibration --double_mut_init
    cp results/raw/*.jsonl /workspace/results/raw/ 2>/dev/null || true
    mv results/calibration/*.npz /workspace/results/calibration/ 2>/dev/null || true
fi

# gb1_esmc_sitemean / rf_ts_k5
RUN_NAME=$(python3 -c "import sys; sys.path.insert(0,'.'); from experiments.run_single import _make_run_name; print(_make_run_name('gb1_esmc_sitemean','rf_ts_k5',96,None,False,False))")
RESULT_FILE="/workspace/results/raw/${RUN_NAME}.jsonl"
if [ -f "$RESULT_FILE" ] && [ "$(wc -l < "$RESULT_FILE")" -ge 100 ]; then
    echo "SKIP gb1_esmc_sitemean rf_ts_k5"
else
    echo "RUN  gb1_esmc_sitemean rf_ts_k5"
    python3 experiments/run_batch.py --method rf_ts_k5 --landscapes gb1_esmc_sitemean --batch_sizes 96 --seeds 100 --workers 8 --track_calibration 
    cp results/raw/*.jsonl /workspace/results/raw/ 2>/dev/null || true
    mv results/calibration/*.npz /workspace/results/calibration/ 2>/dev/null || true
fi

# gb1_esmc_sitemean / dnn_ts
RUN_NAME=$(python3 -c "import sys; sys.path.insert(0,'.'); from experiments.run_single import _make_run_name; print(_make_run_name('gb1_esmc_sitemean','dnn_ts',96,None,False,False))")
RESULT_FILE="/workspace/results/raw/${RUN_NAME}.jsonl"
if [ -f "$RESULT_FILE" ] && [ "$(wc -l < "$RESULT_FILE")" -ge 100 ]; then
    echo "SKIP gb1_esmc_sitemean dnn_ts"
else
    echo "RUN  gb1_esmc_sitemean dnn_ts"
    python3 experiments/run_batch.py --method dnn_ts --landscapes gb1_esmc_sitemean --batch_sizes 96 --seeds 100 --workers 8 --track_calibration 
    cp results/raw/*.jsonl /workspace/results/raw/ 2>/dev/null || true
    mv results/calibration/*.npz /workspace/results/calibration/ 2>/dev/null || true
fi

# gb1_onehot / rf_ts_k1
RUN_NAME=$(python3 -c "import sys; sys.path.insert(0,'.'); from experiments.run_single import _make_run_name; print(_make_run_name('gb1_onehot','rf_ts_k1',96,None,False,True))")
RESULT_FILE="/workspace/results/raw/${RUN_NAME}.jsonl"
if [ -f "$RESULT_FILE" ] && [ "$(wc -l < "$RESULT_FILE")" -ge 100 ]; then
    echo "SKIP gb1_onehot rf_ts_k1"
else
    echo "RUN  gb1_onehot rf_ts_k1"
    python3 experiments/run_batch.py --method rf_ts_k1 --landscapes gb1_onehot --batch_sizes 96 --seeds 100 --workers 8 --track_calibration --double_mut_init
    cp results/raw/*.jsonl /workspace/results/raw/ 2>/dev/null || true
    mv results/calibration/*.npz /workspace/results/calibration/ 2>/dev/null || true
fi

# gb1_onehot / rf_ts_k10
RUN_NAME=$(python3 -c "import sys; sys.path.insert(0,'.'); from experiments.run_single import _make_run_name; print(_make_run_name('gb1_onehot','rf_ts_k10',96,None,False,True))")
RESULT_FILE="/workspace/results/raw/${RUN_NAME}.jsonl"
if [ -f "$RESULT_FILE" ] && [ "$(wc -l < "$RESULT_FILE")" -ge 100 ]; then
    echo "SKIP gb1_onehot rf_ts_k10"
else
    echo "RUN  gb1_onehot rf_ts_k10"
    python3 experiments/run_batch.py --method rf_ts_k10 --landscapes gb1_onehot --batch_sizes 96 --seeds 100 --workers 8 --track_calibration --double_mut_init
    cp results/raw/*.jsonl /workspace/results/raw/ 2>/dev/null || true
    mv results/calibration/*.npz /workspace/results/calibration/ 2>/dev/null || true
fi

# gb1_onehot / evolvepro
RUN_NAME=$(python3 -c "import sys; sys.path.insert(0,'.'); from experiments.run_single import _make_run_name; print(_make_run_name('gb1_onehot','evolvepro',96,None,False,True))")
RESULT_FILE="/workspace/results/raw/${RUN_NAME}.jsonl"
if [ -f "$RESULT_FILE" ] && [ "$(wc -l < "$RESULT_FILE")" -ge 100 ]; then
    echo "SKIP gb1_onehot evolvepro"
else
    echo "RUN  gb1_onehot evolvepro"
    python3 experiments/run_batch.py --method evolvepro --landscapes gb1_onehot --batch_sizes 96 --seeds 100 --workers 8  --double_mut_init
    cp results/raw/*.jsonl /workspace/results/raw/ 2>/dev/null || true
    mv results/calibration/*.npz /workspace/results/calibration/ 2>/dev/null || true
fi

# gb1_esm2_15b / rf_ts_k1
RUN_NAME=$(python3 -c "import sys; sys.path.insert(0,'.'); from experiments.run_single import _make_run_name; print(_make_run_name('gb1_esm2_15b','rf_ts_k1',96,None,False,True))")
RESULT_FILE="/workspace/results/raw/${RUN_NAME}.jsonl"
if [ -f "$RESULT_FILE" ] && [ "$(wc -l < "$RESULT_FILE")" -ge 100 ]; then
    echo "SKIP gb1_esm2_15b rf_ts_k1"
else
    echo "RUN  gb1_esm2_15b rf_ts_k1"
    python3 experiments/run_batch.py --method rf_ts_k1 --landscapes gb1_esm2_15b --batch_sizes 96 --seeds 100 --workers 8 --track_calibration --double_mut_init
    cp results/raw/*.jsonl /workspace/results/raw/ 2>/dev/null || true
    mv results/calibration/*.npz /workspace/results/calibration/ 2>/dev/null || true
fi

# gb1_esm2_15b / rf_ts_k10
RUN_NAME=$(python3 -c "import sys; sys.path.insert(0,'.'); from experiments.run_single import _make_run_name; print(_make_run_name('gb1_esm2_15b','rf_ts_k10',96,None,False,True))")
RESULT_FILE="/workspace/results/raw/${RUN_NAME}.jsonl"
if [ -f "$RESULT_FILE" ] && [ "$(wc -l < "$RESULT_FILE")" -ge 100 ]; then
    echo "SKIP gb1_esm2_15b rf_ts_k10"
else
    echo "RUN  gb1_esm2_15b rf_ts_k10"
    python3 experiments/run_batch.py --method rf_ts_k10 --landscapes gb1_esm2_15b --batch_sizes 96 --seeds 100 --workers 8 --track_calibration --double_mut_init
    cp results/raw/*.jsonl /workspace/results/raw/ 2>/dev/null || true
    mv results/calibration/*.npz /workspace/results/calibration/ 2>/dev/null || true
fi

# gb1_esm2_15b / evolvepro
RUN_NAME=$(python3 -c "import sys; sys.path.insert(0,'.'); from experiments.run_single import _make_run_name; print(_make_run_name('gb1_esm2_15b','evolvepro',96,None,False,True))")
RESULT_FILE="/workspace/results/raw/${RUN_NAME}.jsonl"
if [ -f "$RESULT_FILE" ] && [ "$(wc -l < "$RESULT_FILE")" -ge 100 ]; then
    echo "SKIP gb1_esm2_15b evolvepro"
else
    echo "RUN  gb1_esm2_15b evolvepro"
    python3 experiments/run_batch.py --method evolvepro --landscapes gb1_esm2_15b --batch_sizes 96 --seeds 100 --workers 8  --double_mut_init
    cp results/raw/*.jsonl /workspace/results/raw/ 2>/dev/null || true
    mv results/calibration/*.npz /workspace/results/calibration/ 2>/dev/null || true
fi

echo "=== gb1 done ==="
