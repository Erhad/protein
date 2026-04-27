"""
Spawn one RunPod CPU pod per job, all reading from the same network volume.
Each pod runs its assigned conditions, writes results to the volume, self-terminates.

Usage:
    export RUNPOD_API_KEY=your_key
    python launch_runpod.py --volume_id <id> --region <region>

    # Dry run (print jobs without spawning):
    python launch_runpod.py --volume_id <id> --region <region> --dry_run
"""

import argparse, os, time, base64
import requests
import runpod

# ── Job definitions ────────────────────────────────────────────────────────────
# Each job = one pod. pod runs all (landscape, method) combos for its group.

PROTEINS = ["gb1", "trpb", "tev", "t7"]

def make_jobs():
    jobs = []

    for p in PROTEINS:
        emb = {
            "onehot":           f"{p}_onehot",
            "esmc_mean":        f"{p}_esmc_mean",
            "esmc_sitemean":    f"{p}_esmc_sitemean",
            "esm2_15b":         f"{p}_esm2_15b",
            "esm2_15b_sitemean":f"{p}_esm2_15b_sitemean",
        }

        # G1: embedding comparison — DMZS, TS, k=5
        for emb_key, landscape in emb.items():
            for method in ["rf_ts_k5", "dnn_ts", "dnn_ts_s"]:
                jobs.append({"name": f"G1-{p}-{emb_key}-{method}",
                             "landscape": landscape, "method": method,
                             "dmzs": True})
        # G1: random baseline (embedding-agnostic, run once per protein)
        jobs.append({"name": f"G1-{p}-random",
                     "landscape": f"{p}_onehot", "method": "random",
                     "dmzs": False})

        # G2: acquisition sweep — ESM2-15B mean, DMZS
        for method in ["dnn_greedy", "dnn_greedy_s", "dnn_ucb", "dnn_ucb_s",
                       "dnn_ei", "dnn_ei_s"]:
            jobs.append({"name": f"G2-{p}-esm2_15b-{method}",
                         "landscape": f"{p}_esm2_15b", "method": method,
                         "dmzs": True})

        # G3: init ablation — ESMc sitemean, random init only (DMZS covered in G1)
        for method in ["rf_ts_k5", "dnn_ts"]:
            jobs.append({"name": f"G3-{p}-esmc_sitemean-{method}-random",
                         "landscape": f"{p}_esmc_sitemean", "method": method,
                         "dmzs": False})

        # G4: k-sweep — DMZS, RF only, onehot + ESM2-15B (k=5 covered in G1)
        for landscape in [f"{p}_onehot", f"{p}_esm2_15b"]:
            for method in ["rf_ts_k1", "rf_ts_k10", "evolvepro"]:
                jobs.append({"name": f"G4-{p}-{landscape.split('_',1)[1]}-{method}",
                             "landscape": landscape, "method": method,
                             "dmzs": True})

    return jobs


CALIBRATION_METHODS = {"rf_ts_k1", "rf_ts_k5", "rf_ts_k10",
                       "dnn_ts", "dnn_ts_s",
                       "dnn_ucb", "dnn_ucb_s",
                       "dnn_ei", "dnn_ei_s"}

def make_startup_cmd(job, seeds=100, workers=16):
    landscape = job["landscape"]
    method    = job["method"]
    dmzs_arg  = "True" if job["dmzs"] else "False"
    dmzs_flag = "--double_mut_init" if job["dmzs"] else ""
    cal_flag  = "--track_calibration" if method in CALIBRATION_METHODS else ""
    name      = job["name"]

    script = f"""#!/bin/bash
set -e
echo "=== Pod {name} starting ==="

# Setup — mktemp guarantees a fresh unique dir even if /tmp is shared
WORKDIR=$(mktemp -d /tmp/protein_XXXXXXXXXX)
git clone -q https://github.com/Erhad/protein.git $WORKDIR
cd $WORKDIR/v1

# Install numpy first (needed by run_single.py import for completeness check)
python3 -m pip install numpy -q

# Completeness check — exit early if result file already has enough seeds
RUN_NAME=$(python3 -c "
import sys; sys.path.insert(0, '.')
from experiments.run_single import _make_run_name
print(_make_run_name('{landscape}', '{method}', 96, None, False, {dmzs_arg}))
")
RESULT_FILE="/workspace/results/raw/${{RUN_NAME}}.jsonl"
if [ -f "$RESULT_FILE" ]; then
    LINES=$(wc -l < "$RESULT_FILE")
    if [ "$LINES" -ge {seeds} ]; then
        echo "=== {name} already complete ($LINES lines in $RESULT_FILE) — exiting ==="
        runpodctl remove pod $RUNPOD_POD_ID
        exit 0
    fi
fi

# Install the rest (torch is 180 MB, defer until we know we need to run)
python3 -m pip install torch --index-url https://download.pytorch.org/whl/cpu -q
python3 -m pip install pandas scikit-learn joblib -q

# Point data/ at the network volume.
# git clone creates data/ with old unused CSVs; remove it so ln -s works.
if   [ -d /workspace/protein/v1/data ]; then VOL=/workspace/protein/v1/data
elif [ -d /workspace/v1/data ];         then VOL=/workspace/v1/data
else echo "ERROR: data not found on volume" && exit 1; fi
rm -rf data
ln -s $VOL data
echo "data/ → $VOL"

mkdir -p results/raw

# Run
python3 experiments/run_batch.py \\
    --method {method} \\
    --landscapes {landscape} \\
    --batch_sizes 96 \\
    --seeds {seeds} \\
    --workers {workers} \\
    {cal_flag} \\
    {dmzs_flag}

# Copy results to volume
mkdir -p /workspace/results/raw /workspace/results/calibration
cp results/raw/*.jsonl /workspace/results/raw/ 2>/dev/null || true
cp results/calibration/*.npz /workspace/results/calibration/ 2>/dev/null || true

echo "=== Pod {name} done ==="
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


def deploy_cpu_pod(api_key, name, cmd, volume_id, instance_id="cpu3g-2-16",
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
    parser.add_argument("--workers",     type=int, default=16)
    parser.add_argument("--instance_id", default="cpu3g-8-32",
                        help="RunPod CPU instance type (e.g. cpu3g-8-32 = 8 vCPU / 32 GB)")
    parser.add_argument("--dry_run",     action="store_true")
    parser.add_argument("--jobs",        nargs="*", help="subset of job names to run")
    args = parser.parse_args()

    api_key = os.environ["RUNPOD_API_KEY"]

    jobs = make_jobs()
    if args.jobs:
        jobs = [j for j in jobs if j["name"] in args.jobs]
    print(f"Total jobs: {len(jobs)}")

    if args.dry_run:
        for j in jobs:
            print(f"  {j['name']:55s}  {j['method']:20s}  {'DMZS' if j['dmzs'] else 'random'}")
        return

    print("Fetching currently running pods...")
    running_names = get_running_pod_names(api_key)
    if running_names:
        print(f"  Already running: {sorted(running_names)}")

    spawned, failed, skipped = [], [], []
    for job in jobs:
        if job["name"] in running_names:
            print(f"  SKIP {job['name']} — already running")
            skipped.append(job["name"])
            continue
        cmd = make_startup_cmd(job, seeds=args.seeds, workers=args.workers)
        try:
            pod = deploy_cpu_pod(api_key, job["name"], cmd, args.volume_id,
                                 instance_id=args.instance_id)
            pod_id = pod.get("id", "?")
            print(f"  Spawned {job['name']} → pod {pod_id}")
            spawned.append((job["name"], pod_id))
            running_names.add(job["name"])
        except Exception as e:
            print(f"  FAILED {job['name']}: {e}")
            failed.append(job["name"])
        time.sleep(0.3)

    print(f"\nSpawned {len(spawned)}/{len(jobs)} pods."
          f"  Skipped {len(skipped)} already running.")
    if failed:
        print(f"Failed ({len(failed)}): {failed[:5]}")
    print("Results will appear in /workspace/results/raw/ on the volume.")


if __name__ == "__main__":
    main()
