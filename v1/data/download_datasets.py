"""
Download reference protein fitness landscape datasets via HuggingFace datasets.

Datasets:
  - GFP   : Sarkisyan et al. 2016 (MaveDB API)
  - GB1   : Wu et al. 2016 (SaProtHub/Dataset-GB1-fitness)
  - TrpB  : Johnston et al. 2024 (SaProtHub/Dataset-TrpB_fitness_landsacpe)
"""

import os
import sys
import urllib.request
import urllib.error

DATA_DIR = os.path.dirname(os.path.abspath(__file__))


def download_gfp():
    dest = os.path.join(DATA_DIR, "gfp", "sequences_scores.csv")
    url = "https://api.mavedb.org/api/v1/score-sets/urn:mavedb:00000080-a-1/scores"
    print("[gfp] GFP (avGFP) Sarkisyan et al. 2016 — MaveDB")
    req = urllib.request.Request(url, headers={"User-Agent": "protein-thesis/1.0"})
    with urllib.request.urlopen(req, timeout=120) as r:
        data = r.read()
    with open(dest, "wb") as f:
        f.write(data)
    print(f"  OK  : {len(data)/1024:.1f} KB → {dest}\n")


def download_hf(hf_id: str, dest: str, label: str):
    from datasets import load_dataset
    import csv

    print(f"[{label}] Loading {hf_id} from HuggingFace ...")
    ds = load_dataset(hf_id, split="train")
    print(f"  Rows: {len(ds):,}  Cols: {ds.column_names}")

    with open(dest, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=ds.column_names)
        writer.writeheader()
        for row in ds:
            writer.writerow(row)

    size_mb = os.path.getsize(dest) / 1024 / 1024
    print(f"  OK  : {size_mb:.1f} MB → {dest}\n")


def main():
    targets = sys.argv[1:] if sys.argv[1:] else ["gfp", "gb1", "trpb"]

    valid = {"gfp", "gb1", "trpb"}
    unknown = [t for t in targets if t not in valid]
    if unknown:
        print(f"Unknown: {unknown}. Choose from: {sorted(valid)}")
        sys.exit(1)

    if "gfp" in targets:
        download_gfp()

    if "gb1" in targets:
        download_hf(
            hf_id="SaProtHub/Dataset-GB1-fitness",
            dest=os.path.join(DATA_DIR, "gb1", "gb1_fitness.csv"),
            label="gb1",
        )

    if "trpb" in targets:
        download_hf(
            hf_id="SaProtHub/Dataset-TrpB_fitness_landsacpe",
            dest=os.path.join(DATA_DIR, "trpb", "trpb_fitness.csv"),
            label="trpb",
        )

    print("All done.")


if __name__ == "__main__":
    main()
