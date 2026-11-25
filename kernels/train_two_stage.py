# E8 training kernel (no submission). Two-stage LoRA24 on OUR frozen prompt-grouped folds, see EXPERIMENTS.md.
# Recipe reused with attribution from the public kernels lucasboesen/name-that-llm-train-lora24-s7 (itself from
# lightningv08/ntl-lora24-fivefold-training) and the two-stage idea of nishantdahal/two-stage-lora-on-the-public-pool.
# GPU 0: stage1 (1 epoch on the in-style public pool, class-weighted) -> ts_0..ts_4 (2 epochs on official fit rows,
#        OOF features for held-out rows) -> ts_final (all official rows; adapter saved).
# GPU 1: ctl_0..ctl_4 (public setting: 3 epochs on official fit rows from the base, no stage 1) = official-only control.
# Each job writes lora_out/job_<name>/ when it finishes, so a timeout keeps completed jobs.
import glob
import json
import os
import pathlib
import subprocess
import sys
import time


SMOKE = os.environ.get("NTL_SMOKE")  # local CPU smoke test with a tiny random model
SEED = 8
LAYERS = [2, 4, 6, 8, 12, 16, 20, 24]
KEEP, MAX_LEN, ACCUM = 24, 447, 6
R, ALPHA, LDROP = 16, 32, 0.05
STAGE1 = dict(epochs=1, lr_adapter=1e-4, lr_head=1e-3)
STAGE2 = dict(epochs=2, lr_adapter=5e-5, lr_head=5e-4)
CONTROL = dict(epochs=3, lr_adapter=1e-4, lr_head=1e-3)
WD, WARMUP = 0.01, 0.10
JOBS = {
    0: ["stage1", "ts_0", "ts_1", "ts_2", "ts_3", "ts_final", "ts_4"],
    1: ["ctl_0", "ctl_1", "ctl_2", "ctl_3", "ctl_4"],
}

WORKER = None  # Filled by build_kernel.py for standalone Kaggle execution.

T0 = time.time()
if SMOKE:
    DATA_DIR = pathlib.Path("data/raw")
    POOL_DIR = pathlib.Path(SMOKE)
    model_dir = str(POOL_DIR / "tiny_model")
    JOBS = {0: ["stage1", "ts_0", "ts_final"], 1: ["ctl_0"]}
    KEEP, LAYERS, MAX_LEN = 2, [1, 2], 32
else:
    DATA_DIR = pathlib.Path("/kaggle/input/competitions/name-that-llm")
    POOL_DIR = pathlib.Path(
        next(iter(glob.glob("/kaggle/input/**/pool_instyle.parquet", recursive=True)))
    ).parent
    model_dir = None
    for c in sorted(glob.glob("/kaggle/input/**/config.json", recursive=True)):
        try:
            j = json.load(open(c))
        except Exception:
            continue
        if (
            j.get("model_type") == "qwen3"
            and j.get("hidden_size") == 4096
            and j.get("num_hidden_layers") == 36
        ):
            model_dir = os.path.dirname(c)
            break
print("base", model_dir, "pool", POOL_DIR, flush=True)
assert model_dir
cfg = dict(
    train_csv=str(DATA_DIR / "train.csv"),
    splits_csv=str(POOL_DIR / "official_splits.csv"),
    pool_parquet=str(POOL_DIR / "pool_instyle.parquet"),
    model_dir=model_dir,
    seed=SEED,
    layers=LAYERS,
    keep=KEEP,
    max_len=MAX_LEN,
    accum=ACCUM,
    wd=WD,
    warmup=WARMUP,
    r=R,
    alpha=ALPHA,
    ldrop=LDROP,
    stage1=STAGE1,
    stage2=STAGE2,
    control=CONTROL,
    limit=(40 if SMOKE else 0),
)
worker_source = WORKER or pathlib.Path(__file__).with_name("lora_worker.py").read_text()
pathlib.Path("worker.py").write_text(worker_source)
procs = []
for g, jl in JOBS.items():
    env = dict(
        os.environ, CUDA_VISIBLE_DEVICES=str(g), PYTORCH_CUDA_ALLOC_CONF="expandable_segments:True"
    )
    procs.append(
        (
            g,
            subprocess.Popen(
                [sys.executable, "worker.py", json.dumps(cfg), ",".join(jl)],
                env=env,
                stdout=open(f"worker{g}.log", "w"),
                stderr=subprocess.STDOUT,
            ),
        )
    )
for g, p in procs:
    p.wait()
    print(f"---- worker {g} (rc {p.returncode})")
    print(open(f"worker{g}.log").read()[-6000:])
done = sorted(glob.glob("lora_out/job_*/COMPLETE.json"))
print("completed jobs", [json.load(open(d)) for d in done], f"total {time.time() - T0:.0f}s")
if any(process.returncode for _, process in procs):
    raise SystemExit("Training failed. Check the worker logs above.")
