# Set these:
V1_PATH       = "/content/drive/MyDrive/protein/v1"
GB1_EMB_PATH  = "/content/drive/MyDrive/protein/v1/data/gb1/embeddings_esm2_650m_4site.npz"
TRPB_EMB_PATH = "/content/drive/MyDrive/protein/v1/data/trpb/embeddings_esm2_650m_4site.npz"

from google.colab import drive; drive.mount("/content/drive")
import sys, subprocess, json, os, numpy as np
sys.path.insert(0, V1_PATH)
subprocess.run(["pip", "install", "gpytorch", "-q"], check=True)

from datasets import load_dataset
from methods.evolvepro import EVOLVEpro

gb1_fit  = np.array(load_dataset("SaProtHub/Dataset-GB1-fitness",            split="train")["label"], dtype=np.float32)
trpb_fit = np.array(load_dataset("SaProtHub/Dataset-TrpB_fitness_landsacpe", split="train")["label"], dtype=np.float32)
gb1_emb  = np.load(GB1_EMB_PATH)["embeddings"]
trpb_emb = np.load(TRPB_EMB_PATH)["embeddings"]

INITIAL_N, TOTAL_BUDGET = 96, 480
OUT = os.path.join(V1_PATH, "results", "raw"); os.makedirs(OUT, exist_ok=True)

for landscape, emb, fit in [("gb1", gb1_emb, gb1_fit), ("trpb", trpb_emb, trpb_fit)]:
    for bs in [16, 96]:
        out_path = f"{OUT}/{landscape}_evolvepro_{bs}.jsonl"
        done = {json.loads(l)["seed"] for l in open(out_path)} if os.path.exists(out_path) else set()
        for seed in range(100):
            if seed in done: continue
            rng = np.random.default_rng(seed); opt = EVOLVEpro(seed=seed)
            labeled = list(rng.choice(len(fit), INITIAL_N, replace=False).tolist())
            pool    = list(set(range(len(fit))) - set(labeled))
            while len(labeled) < TOTAL_BUDGET:
                b = min(bs, TOTAL_BUDGET - len(labeled))
                opt.train(emb[labeled], fit[labeled])
                sel = opt.select(emb[pool], b)
                chosen = [pool[i] for i in sel]; labeled += chosen
                pool = [i for i in pool if i not in set(chosen)]
            open(out_path, "a").write(json.dumps({"landscape": landscape, "method": "evolvepro", "batch_size": bs, "seed": seed, "selection_order": labeled}) + "\n")
            print(f"Done {landscape} bs={bs} seed={seed}  max={fit[labeled].max():.4f}")
