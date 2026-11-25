import json
import math
import os
import random
import sys
import time
import importlib.util as _iu

_fs = _iu.find_spec  # Kaggle ships a torchao that peft rejects; hide it (unused)
_iu.find_spec = (
    lambda name, *a, **k: None if name.split(".")[0] == "torchao" else _fs(name, *a, **k)
)
import numpy as np
import pandas as pd
import torch
from torch import nn
from transformers import AutoConfig, AutoModel, AutoTokenizer
from peft import LoraConfig, PeftModel, get_peft_model
import peft.import_utils as _pi

_pi.is_torchao_available = lambda *a, **k: False
from safetensors.torch import load_file, save_file

cfg = json.loads(sys.argv[1])
jobs = sys.argv[2].split(",")
dev = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
use_amp = dev.type == "cuda"
train = pd.read_csv(cfg["train_csv"])
splits = pd.read_csv(cfg["splits_csv"])
train = train.merge(splits[["id", "split", "fold"]], on="id", how="left", validate="one_to_one")
assert train.split.notna().all()
pool = pd.read_parquet(cfg["pool_parquet"])
if cfg.get("limit"):
    train = train.groupby("prompt_id").head(3).head(cfg["limit"]).reset_index(drop=True)
    pool = pool.groupby("pid").head(3).head(cfg["limit"]).reset_index(drop=True)
LABELS = sorted(pd.read_csv(cfg["train_csv"]).model.unique())
assert len(LABELS) == 18
lab = {l: i for i, l in enumerate(LABELS)}
assert set(pool.model) <= set(LABELS)
tok = AutoTokenizer.from_pretrained(cfg["model_dir"])
tok.padding_side = "right"
if tok.pad_token is None:
    tok.pad_token = tok.eos_token


def encode(texts):
    texts = [t if isinstance(t, str) and t.strip() else tok.eos_token for t in texts]
    return [
        ids + [tok.eos_token_id]
        for ids in tok(texts, add_special_tokens=False, truncation=True, max_length=cfg["max_len"])[
            "input_ids"
        ]
    ]


OFF = dict(
    tokens=encode(train.text.tolist()),
    y=train.model.map(lab).values,
    pid=train.prompt_id.values,
    ids=train.id.values,
)
POOL = dict(
    tokens=encode(pool.text.tolist()),
    y=pool.model.map(lab).values,
    pid=pool.pid.values,
    ids=np.arange(len(pool)),
)
print(f"official {len(train)} rows, pool {len(pool)} rows, device {dev}", flush=True)
L = cfg["layers"]


def triplets(pid, rows, rng):
    out = []
    for p in np.unique(pid[rows]):
        m = rng.permutation(rows[pid[rows] == p])
        for s in range(0, len(m), 3):
            b = list(m[s : s + 3])
            while (
                len(b) < 3
            ):  # fill a short chunk with other answers to the same prompt (or itself if alone)
                rest = [x for x in m if x not in b]
                b.append(int(rng.choice(rest)) if rest else int(b[0]))
            out.append(b)
    out = [out[i] for i in rng.permutation(len(out))]
    while len(out) % cfg["accum"]:
        out.append(out[rng.randint(len(out))])
    return out


def run_job(job, D, fit, valid, hp, init_dir, class_weight):
    t0 = time.time()
    outdir = os.path.join("lora_out", f"job_{job}")
    os.makedirs(outdir, exist_ok=True)
    random.seed(cfg["seed"])
    np.random.seed(cfg["seed"])
    torch.manual_seed(cfg["seed"])
    torch.cuda.manual_seed_all(cfg["seed"])
    rng = np.random.RandomState(cfg["seed"] + abs(hash(job)) % 1000)
    mc = AutoConfig.from_pretrained(cfg["model_dir"])
    mc.num_hidden_layers, mc.use_cache = cfg["keep"], False
    if getattr(mc, "layer_types", None) is not None:
        mc.layer_types = mc.layer_types[: cfg["keep"]]
    H = mc.hidden_size
    base = AutoModel.from_pretrained(
        cfg["model_dir"], config=mc, dtype=torch.float16 if use_amp else torch.float32
    ).to(dev)
    base.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
    base.enable_input_require_grads()
    norms = nn.ModuleList([nn.LayerNorm(H) for _ in L]).to(dev).float()
    plain = nn.Linear(H * len(L), len(LABELS)).to(dev).float()
    centered = nn.Linear(H * len(L), len(LABELS)).to(dev).float()
    if init_dir:
        model = PeftModel.from_pretrained(
            base, os.path.join(init_dir, "adapter"), is_trainable=True
        )
        hs = load_file(os.path.join(init_dir, "heads.safetensors"))
        norms.load_state_dict(
            {k[len("norms.") :]: v for k, v in hs.items() if k.startswith("norms.")}
        )
        plain.load_state_dict(
            {k[len("plain.") :]: v for k, v in hs.items() if k.startswith("plain.")}
        )
        centered.load_state_dict(
            {k[len("centered.") :]: v for k, v in hs.items() if k.startswith("centered.")}
        )
        print(f"job {job}: initialised from {init_dir}", flush=True)
    else:
        model = get_peft_model(
            base,
            LoraConfig(
                r=cfg["r"],
                lora_alpha=cfg["alpha"],
                lora_dropout=cfg["ldrop"],
                bias="none",
                target_modules=[
                    "q_proj",
                    "k_proj",
                    "v_proj",
                    "o_proj",
                    "gate_proj",
                    "up_proj",
                    "down_proj",
                ],
            ),
        )
    for p in model.parameters():
        if p.requires_grad:
            p.data = p.data.float()
    drop = nn.Dropout(0.1)
    ad = [p for p in model.parameters() if p.requires_grad]
    hd = list(norms.parameters()) + list(plain.parameters()) + list(centered.parameters())
    opt = torch.optim.AdamW(
        [{"params": ad, "lr": hp["lr_adapter"]}, {"params": hd, "lr": hp["lr_head"]}],
        weight_decay=cfg["wd"],
    )
    plan = [triplets(D["pid"], fit, rng) for _ in range(hp["epochs"])]
    total = sum(len(e) for e in plan) // cfg["accum"]
    warm = max(1, int(cfg["warmup"] * total))
    sch = torch.optim.lr_scheduler.LambdaLR(
        opt,
        lambda s: (s + 1) / warm
        if s < warm
        else 0.5 * (1 + math.cos(math.pi * (s - warm) / max(1, total - warm))),
    )
    scaler = torch.amp.GradScaler(init_scale=256.0, enabled=use_amp)
    cw = None
    if class_weight:
        cnt = np.bincount(D["y"][fit], minlength=len(LABELS)).astype(np.float64)
        w = 1.0 / np.maximum(cnt, 1)
        w = w / w.mean()
        cw = torch.tensor(w, dtype=torch.float32, device=dev)
        print(f"job {job}: class weights min {w.min():.2f} max {w.max():.2f}", flush=True)

    def rep(idx):
        batch = tok.pad({"input_ids": [D["tokens"][i] for i in idx]}, return_tensors="pt").to(dev)
        with torch.autocast("cuda", dtype=torch.float16, enabled=use_amp):
            hs = model(**batch, output_hidden_states=True, use_cache=False).hidden_states
        m = batch["attention_mask"].unsqueeze(-1).float()
        m[:, 0] = 0
        den = m.sum(1).clamp(min=1)
        with torch.autocast("cuda", enabled=False):
            return torch.cat([nm((hs[l].float() * m).sum(1) / den) for l, nm in zip(L, norms)], -1)

    model.train()
    norms.train()
    step = 0
    skipped = 0
    losses = []
    for ep, batches in enumerate(plan):
        for i, b in enumerate(batches):
            z = rep(b)
            yy = torch.as_tensor(D["y"][b], device=dev)
            loss = nn.functional.cross_entropy(
                plain(drop(z)), yy, weight=cw
            ) + nn.functional.cross_entropy(
                centered(drop(z - z.mean(0, keepdim=True))), yy, weight=cw
            )
            scaler.scale(loss / cfg["accum"]).backward()
            losses.append(float(loss.detach()))
            if (i + 1) % cfg["accum"] == 0:
                scaler.unscale_(opt)
                torch.nn.utils.clip_grad_norm_(ad + hd, 1.0)
                before = scaler.get_scale()
                scaler.step(opt)
                scaler.update()
                if scaler.get_scale() < before:
                    skipped += 1
                opt.zero_grad(set_to_none=True)
                sch.step()
                step += 1
                if step % 30 == 0 or step == total:
                    print(
                        f"job {job} ep {ep} step {step}/{total} loss {np.mean(losses[-180:]):.4f} skipped {skipped} "
                        f"({time.time() - t0:.0f}s)",
                        flush=True,
                    )
    model.eval()
    norms.eval()
    Z = np.zeros((len(valid), H * len(L)), np.float32)
    order = np.argsort([len(D["tokens"][i]) for i in valid], kind="stable")
    with torch.inference_mode():
        for s in range(0, len(order), 6):
            pos = order[s : s + 6]
            Z[pos] = rep(valid[pos]).cpu().numpy()
    assert np.isfinite(Z).all()
    W = {n: t.weight.detach().cpu().numpy() for n, t in (("plain", plain), ("center", centered))}
    B = {n: t.bias.detach().cpu().numpy() for n, t in (("plain", plain), ("center", centered))}
    np.savez_compressed(
        os.path.join(outdir, "readouts.npz"),
        classes=np.array(LABELS),
        pool_layers=np.array(L),
        plain_coef=W["plain"],
        plain_intercept=B["plain"],
        center_coef=W["center"],
        center_intercept=B["center"],
    )
    acc = float("nan")
    if len(valid):
        np.savez_compressed(
            os.path.join(outdir, "valid_features.npz"),
            ids=D["ids"][valid],
            y=D["y"][valid],
            features=Z.astype(np.float16),
        )
        lg = Z @ W["plain"].T + B["plain"]
        acc = float((lg.argmax(1) == D["y"][valid]).mean())
    if job in ("stage1", "ts_final"):
        model.save_pretrained(os.path.join(outdir, "adapter"), safe_serialization=True)
        sd = {}
        for pre, mod in (("norms", norms), ("plain", plain), ("centered", centered)):
            sd.update(
                {f"{pre}.{k}": v.detach().cpu().contiguous() for k, v in mod.state_dict().items()}
            )
        save_file(sd, os.path.join(outdir, "heads.safetensors"))
        tok.save_pretrained(os.path.join(outdir, "tokenizer"))
    json.dump(
        {
            "job": job,
            "valid_acc": acc,
            "n_fit": int(len(fit)),
            "n_valid": int(len(valid)),
            "steps": step,
            "skipped": skipped,
            "secs": time.time() - t0,
            "hp": hp,
            "init": init_dir,
        },
        open(os.path.join(outdir, "COMPLETE.json"), "w"),
    )
    print(
        f"job {job} done: valid acc {acc:.4f} steps {step} skipped {skipped} ({time.time() - t0:.0f}s)",
        flush=True,
    )
    del model, base, opt, norms, plain, centered
    if use_amp:
        torch.cuda.empty_cache()


dev_rows = np.where(train.split.values == "dev")[0]
fold = train.fold.values
for job in jobs:
    if os.path.exists(os.path.join("lora_out", f"job_{job}", "COMPLETE.json")):
        print(f"job {job} already complete, skipping", flush=True)
        continue
    if job == "stage1":
        run_job(job, POOL, np.arange(len(pool)), np.arange(0), cfg["stage1"], None, True)
    else:
        kind, part = job.split("_")
        init = os.path.join("lora_out", "job_stage1") if kind == "ts" else None
        if init:
            assert os.path.exists(os.path.join(init, "COMPLETE.json")), "stage1 missing"
        hp = cfg["stage2"] if kind == "ts" else cfg["control"]
        if part == "final":
            fit, valid = np.arange(len(train)), np.arange(0)
        else:
            k = int(part)
            fit = dev_rows[fold[dev_rows] != k]
            valid = dev_rows[fold[dev_rows] == k]
        run_job(job, OFF, fit, valid, hp, init, False)
