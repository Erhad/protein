"""
Spawn one RunPod CPU pod per protein. Each pod runs all jobs for that protein
sequentially, writing results to the network volume, then self-terminates.

Usage:
    export RUNPOD_API_KEY=your_key
    python launch_runpod.py --volume_id <id>

    # Tail pods (G4→G3→G2, to run in parallel with existing G1 pods):
    python launch_runpod.py --volume_id <id> --tail

    # Dry run:
    python launch_runpod.py --volume_id <id> --dry_run
"""

import argparse, os, time, base64
import requests

PROTEINS = ["gb1", "trpb", "tev", "t7"]

CALIBRATION_METHODS = {"rf_ts_k1", "rf_ts_k5", "rf_ts_k10",
                       "dnn_ts", "dnn_ts_s",
                       "dnn_ucb", "dnn_ucb_s",
                       "dnn_ei", "dnn_ei_s"}

def protein_subjobs(p, tail=False):
    """Return ordered list of (landscape, method, dmzs) for one protein.

    tail=True: G4→G3→G2 order (for a second pod running in parallel with
    an existing G1 pod; skip logic prevents duplicate work).
    """
    emb = {
        "onehot":            f"{p}_onehot",
        "esmc_mean":         f"{p}_esmc_mean",
        "esmc_sitemean":     f"{p}_esmc_sitemean",
        "esm2_15b":          f"{p}_esm2_15b",
        "esm2_15b_sitemean": f"{p}_esm2_15b_sitemean",
    }
    g1 = []
    for landscape in emb.values():
        for method in ["rf_ts_k5", "dnn_ts", "dnn_ts_s"]:
            g1.append((landscape, method, True))
    g1.append((f"{p}_onehot", "random", False))

    g2 = []
    for method in ["dnn_greedy", "dnn_greedy_s", "dnn_ucb", "dnn_ucb_s", "dnn_ei", "dnn_ei_s"]:
        g2.append((f"{p}_esm2_15b", method, True))

    g3 = []
    for method in ["rf_ts_k5", "dnn_ts"]:
        g3.append((f"{p}_esmc_sitemean", method, False))

    g4 = []
    for landscape in [f"{p}_onehot", f"{p}_esm2_15b"]:
        for method in ["rf_ts_k1", "rf_ts_k10", "evolvepro"]:
            g4.append((landscape, method, True))

    if tail:
        return g4 + g3 + g2
    return g1 + g2 + g3 + g4


def make_startup_cmd(protein, seeds=100, workers=4, tail=False):
    subjobs = protein_subjobs(protein, tail=tail)

    # Build the per-subjob run blocks
    run_blocks = []
    for landscape, method, dmzs in subjobs:
        dmzs_arg  = "True" if dmzs else "False"
        dmzs_flag = "--double_mut_init" if dmzs else ""
        cal_flag  = "--track_calibration" if method in CALIBRATION_METHODS else ""
        run_blocks.append(f"""
# ── {landscape} / {method} / dmzs={dmzs} ──
RUN_NAME=$(python3 -c "
import sys; sys.path.insert(0, '.')
from experiments.run_single import _make_run_name
print(_make_run_name('{landscape}', '{method}', 96, None, False, {dmzs_arg}))
")
RESULT_FILE="/workspace/results/raw/${{RUN_NAME}}.jsonl"
if [ -f "$RESULT_FILE" ] && [ "$(wc -l < "$RESULT_FILE")" -ge {seeds} ]; then
    echo "SKIP {landscape} {method} — already complete"
else
    echo "RUN  {landscape} {method}"
    python3 experiments/run_batch.py \\
        --method {method} \\
        --landscapes {landscape} \\
        --batch_sizes 96 \\
        --seeds {seeds} \\
        --workers {workers} \\
        {cal_flag} \\
        {dmzs_flag}
    cp results/raw/*.jsonl   /workspace/results/raw/          2>/dev/null || true
    cp results/calibration/*.npz /workspace/results/calibration/ 2>/dev/null || true
fi""")

    all_runs = "\n".join(run_blocks)

    script = f"""#!/bin/bash
set -e
echo "=== Pod {protein} starting — {len(subjobs)} jobs ==="

WORKDIR=$(mktemp -d /tmp/protein_XXXXXXXXXX)
git clone -q https://github.com/Erhad/protein.git $WORKDIR
cd $WORKDIR/v1

# Install deps
python3 -m pip install numpy -q
python3 -m pip install torch --index-url https://download.pytorch.org/whl/cpu -q
python3 -m pip install pandas scikit-learn joblib -q

# Symlink volume data (embeddings + li2024 ZS CSVs) into local data/ tree
if [ -d /workspace/v1/data ]; then VOL=/workspace/v1/data
else echo "ERROR: /workspace/v1/data not found on volume" && exit 1; fi
find "$VOL" \\( -name "*.npy" -o -name "*.npz" -o -name "*.csv" \\) | while read src; do
    rel="${{src#$VOL/}}"
    dst="data/$rel"
    mkdir -p "$(dirname $dst)"
    ln -sf "$src" "$dst"
done
echo "Data symlinked from $VOL"

mkdir -p results/raw results/calibration
mkdir -p /workspace/results/raw /workspace/results/calibration

# Symlink existing volume results into local results/ so run_single.py's seed-skip logic fires
find /workspace/results/raw -name "*.jsonl" | while read src; do
    dst="results/raw/$(basename $src)"
    ln -sf "$src" "$dst"
done

{all_runs}

echo "=== Pod {protein} done ==="
runpodctl remove pod $RUNPOD_POD_ID
"""
    encoded = base64.b64encode(script.encode()).decode()
    return f"bash -c 'echo {encoded} | base64 -d | bash'"


GRAPHQL_URL = "https://api.runpod.io/graphql"


def get_running_pod_names(api_key):
    query = "{ myself { pods { name } } }"
    resp = requests.post(GRAPHQL_URL,
        headers={"Authorization": f"Bearer {api_key}"},
        json={"query": query}, timeout=15)
    pods = resp.json().get("data", {}).get("myself", {}).get("pods", [])
    return {p["name"] for p in pods}


def deploy_cpu_pod(api_key, name, cmd, volume_id, instance_id="cpu3g-8-32",
                   disk_gb=20, image="runpod/base:0.4.0-cuda11.8.0"):
    mutation = """
    mutation DeployCpuPod($input: deployCpuPodInput!) {
      deployCpuPod(input: $input) {
        id
        name
        desiredStatus
      }
    }
    """
    variables = {"input": {
        "name": name,
        "imageName": image,
        "instanceId": instance_id,
        "containerDiskInGb": disk_gb,
        "dockerArgs": cmd,
        "networkVolumeId": volume_id,
        "volumeMountPath": "/workspace",
    }}
    resp = requests.post(
        GRAPHQL_URL,
        headers={"Authorization": f"Bearer {api_key}"},
        json={"query": mutation, "variables": variables},
        timeout=30,
    )
    data = resp.json()
    errors = data.get("errors")
    if errors:
        raise RuntimeError(errors[0]["message"])
    return data["data"]["deployCpuPod"]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--volume_id",   required=True)
    parser.add_argument("--seeds",       type=int, default=100)
    parser.add_argument("--workers",     type=int, default=4)
    parser.add_argument("--instance_id", default="cpu3g-8-32",
                        help="RunPod CPU instance type (e.g. cpu3g-8-32, cpu3g-4-16)")
    parser.add_argument("--proteins",    nargs="+", default=PROTEINS,
                        help="Subset of proteins to launch (default: all 4)")
    parser.add_argument("--tail",         action="store_true",
                        help="Run G4→G3→G2 (tail pod, parallel with existing G1 pod)")
    parser.add_argument("--dry_run",     action="store_true")
    args = parser.parse_args()

    pod_suffix = "-tail" if args.tail else ""
    print(f"Proteins: {args.proteins}  Instance: {args.instance_id}  "
          f"Workers: {args.workers}  Seeds: {args.seeds}  Tail: {args.tail}")

    if args.dry_run:
        for p in args.proteins:
            subjobs = protein_subjobs(p, tail=args.tail)
            print(f"\n{p}: {len(subjobs)} jobs")
            for landscape, method, dmzs in subjobs:
                print(f"  {landscape:35s} {method:20s} dmzs={dmzs}")
        return

    api_key = os.environ["RUNPOD_API_KEY"]
    print("Fetching currently running pods...")
    running_names = get_running_pod_names(api_key)
    if running_names:
        print(f"  Already running: {sorted(running_names)}")

    pending = [p for p in args.proteins if f"protein-{p}{pod_suffix}" not in running_names]
    cmds    = {p: make_startup_cmd(p, seeds=args.seeds, workers=args.workers, tail=args.tail)
               for p in pending}

    attempt = 0
    while pending:
        attempt += 1
        still_pending = []
        for protein in pending:
            pod_name = f"protein-{protein}{pod_suffix}"
            try:
                pod = deploy_cpu_pod(api_key, pod_name, cmds[protein], args.volume_id,
                                     instance_id=args.instance_id)
                pod_id = pod.get("id", "?")
                print(f"  Spawned {pod_name} → pod {pod_id}")
            except Exception as e:
                msg = str(e)
                if "no longer any instances available" in msg.lower() or "no instances" in msg.lower():
                    still_pending.append(protein)
                else:
                    print(f"  FAILED {pod_name} (permanent): {e}")
            time.sleep(0.5)

        pending = still_pending
        if pending:
            wait = min(30 * attempt, 300)
            print(f"  [{attempt}] {len(pending)} pods still pending ({pending}). "
                  f"Retrying in {wait}s...")
            time.sleep(wait)

    print("All pods launched.")


if __name__ == "__main__":
    main()
