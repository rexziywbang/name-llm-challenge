# E17 submission kernel: E16 plus CHEN Han's native Qwen3-Embedding-8B plain head.
# Deliberately excluded: n14b, centred/twin heads, balancing, typography, sibling/group logic, and post-argmax inference.
import gc
import glob
import json
import os
import shutil
import subprocess
import tempfile
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import transformers
from safetensors.torch import load_file
from scipy.special import log_softmax
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from transformers import AutoConfig, AutoModel, AutoTokenizer

T0 = time.time()
IN = (
    os.environ.get("NTL_INPUT_DIR")
    or next(
        (
            d
            for d in ("/kaggle/input/competitions/name-that-llm", "/kaggle/input/name-that-llm")
            if os.path.exists(f"{d}/test.csv")
        ),
        None,
    )
    or os.path.dirname(next(iter(glob.glob("/kaggle/input/**/test.csv", recursive=True))))
)
OUT = os.environ.get("NTL_OUTPUT_PATH", "/kaggle/working/submission.parquet")
ADAPTER_ROOT = os.environ.get("NTL_ADAPTER_ROOT", "/kaggle/input")
E6_ADAPTERS = os.environ.get(
    "NTL_E6_ADAPTERS", "q8v5,q8v5_f0,q8v5_f1,q8v5_f2,q8v5_f3,q8v5_f4"
).split(",")
Q14_ADAPTERS = ["q14v5", "q14v5_f0", "q14v5_f1", "q14v5_f2", "q14v5_f3", "q14v5_f4"]
KEEP8, KEEP14, KEEP25 = 24, 27, 32
LAYERS, BS, C_HIDDEN, MAX_LEN = [2, 4, 6, 8, 12, 16, 20, 24], 8, 0.002, 384
NTL_SMOKE = os.environ.get("NTL_SMOKE") == "1"
assert len(E6_ADAPTERS) == len(Q14_ADAPTERS) == 6

LABELS = [
    "gpt-5.5",
    "gpt-5.6-luna",
    "gpt-5.6-sol",
    "gpt-5.6-terra",
    "gpt-6-astra",
    "gemini-3.1-pro-preview",
    "gemini-3.6-flash",
    "gemini-3.7-flash",
    "gemini-3.8-flash",
    "claude-fable-5",
    "claude-opus-4-6",
    "claude-opus-4-8",
    "claude-opus-5",
    "claude-sonnet-5",
    "qwen3.8-max",
    "glm-5.3",
    "glm-5.3-flash",
    "kimi-k3",
]
classes = np.array(sorted(LABELS))
assert len(classes) == 18


NATIVE_WORKER_SOURCE = None  # Filled by build_kernel.py for standalone Kaggle execution.

from ensemble import blend_e17, native_log_probabilities


def resolve_model_dir(env_name, required_path_part):
    override = os.environ.get(env_name)
    if override:
        return override
    matches = [
        Path(path).parent
        for path in glob.glob("/kaggle/input/models/**/config.json", recursive=True)
        if "qwen-3" in path.lower() and required_path_part in path.lower()
    ]
    assert len(matches) == 1, (env_name, matches)
    return str(matches[0])


def resolve_q25_model_dir():
    override = os.environ.get("NTL_MODEL25_DIR")
    if override:
        model_dir = override
    else:
        matches = [
            Path(path).parent
            for path in glob.glob("/kaggle/input/models/**/config.json", recursive=True)
            if "qwen2.5" in path.lower() and "/14b/" in path.lower() and "/7b/" not in path.lower()
        ]
        assert len(matches) == 1, matches
        model_dir = str(matches[0])
    lowered = model_dir.lower()
    assert "qwen2.5" in lowered and "14b" in lowered and "7b" not in lowered, model_dir
    return model_dir


def resolve_native_assets_dir():
    override = os.environ.get("NTL_NATIVE_ASSETS_DIR")
    if NTL_SMOKE and override:
        assets_dir = Path(override)
        assert (assets_dir / "manifest.json").is_file(), assets_dir
        return assets_dir
    matches = glob.glob(
        "/kaggle/input/**/ntl-native-q8-multidepth-fivefold-0679/manifest.json",
        recursive=True,
    )
    assert len(matches) == 1, matches
    return Path(matches[0]).parent


def resolve_native_base_dir():
    base_dir = Path(resolve_model_dir("NTL_NATIVE_BASE_DIR", "qwen-3-embedding/transformers/8b/1"))
    native_config = AutoConfig.from_pretrained(base_dir)
    assert native_config.num_hidden_layers == 36
    assert native_config.hidden_size == 4096
    return base_dir


MODEL8_DIR = resolve_model_dir("NTL_MODEL_DIR", "8b-base")
MODEL14_DIR = resolve_model_dir("NTL_MODEL14_DIR", "14b-base")
MODEL25_DIR = resolve_q25_model_dir()
assert "14b-base" not in MODEL8_DIR.lower(), MODEL8_DIR
ADAPTER_CONFIGS = list(Path(ADAPTER_ROOT).rglob("adapter_config.json"))
print(
    "models",
    MODEL8_DIR,
    MODEL14_DIR,
    MODEL25_DIR,
    "E6 adapters",
    E6_ADAPTERS,
    "Q14 adapters",
    Q14_ADAPTERS,
    "cuda",
    torch.cuda.is_available(),
    flush=True,
)

train = pd.read_csv(f"{IN}/train.csv", dtype={"id": str})
test = pd.read_csv(f"{IN}/test.csv", dtype={"id": str})
for frame in (train, test):
    frame["text"] = frame["text"].fillna("").astype(str)
assert train.id.is_unique and test.id.is_unique
if os.environ.get("NTL_TRAIN_LIMIT"):
    train = train.sample(int(os.environ["NTL_TRAIN_LIMIT"]), random_state=0).reset_index(drop=True)
assert set(train.model) == set(LABELS)
y = train.model.to_numpy()

NATIVE_ASSETS_DIR = resolve_native_assets_dir()
NATIVE_BASE_DIR = resolve_native_base_dir()
native_tmp = tempfile.TemporaryDirectory(prefix="ntl-native-")
native_root = Path(native_tmp.name)
native_worker_path = native_root / "worker.py"
native_worker_path.write_text(
    NATIVE_WORKER_SOURCE or Path(__file__).with_name("native_embedding.py").read_text(),
    encoding="utf-8",
)
native_query_path = native_root / "queries.csv"
test[["id", "text"]].to_csv(native_query_path, index=False)
native_output_path = native_root / "scores.npz"
subprocess.run(
    [
        sys.executable,
        "-u",
        str(native_worker_path),
        "--base",
        str(NATIVE_BASE_DIR),
        "--assets",
        str(NATIVE_ASSETS_DIR),
        "--query",
        str(native_query_path),
        "--out",
        str(native_output_path),
        "--work",
        str(native_root / "work"),
    ],
    check=True,
)
with np.load(native_output_path, allow_pickle=False) as native_data:
    native_ids = native_data["ids"].astype(str)
    native_labels = native_data["labels"].astype(str).tolist()
    native_plain_raw = np.asarray(native_data["plain"])
native_tmp.cleanup()
native_manifest = json.loads((NATIVE_ASSETS_DIR / "manifest.json").read_text())
manifest_classes = [str(label) for label in native_manifest["classes"]]
assert native_labels == manifest_classes
assert native_ids.tolist() == test.id.astype(str).tolist()
assert native_plain_raw.shape == (5, len(test), len(classes)), native_plain_raw.shape
assert np.isfinite(native_plain_raw).all()
native_scales = []
for fold in range(5):
    calibration = json.loads((NATIVE_ASSETS_DIR / f"calibration/fold{fold}.json").read_text())
    native_scales.append(float(calibration["models"]["qwen8"]["plain_scale"]))
logPnat, nat_plain_folds, native_order = native_log_probabilities(
    native_plain_raw, manifest_classes, classes, native_scales
)
assert logPnat.shape == (len(test), len(classes))
assert nat_plain_folds.shape == (5, len(test), len(classes))
assert np.isfinite(logPnat).all() and np.isfinite(nat_plain_folds).all()
native_timing = json.loads(native_output_path.with_suffix(".json").read_text())
native_seconds = float(native_timing["seconds_including_load_and_staging"])
assert len(native_timing["fold_timings"]) == 5
assert len(native_timing["peak_reserved"]) == 2
print(f"elapsed NAT worker {native_seconds:.1f}s", flush=True)
for fold_timing in native_timing["fold_timings"]:
    print(
        f"NAT fold{fold_timing['fold']} seconds {fold_timing['seconds']:.1f}s",
        flush=True,
    )
print(
    "NAT peak reserved GB per GPU",
    [round(value / 1e9, 3) for value in native_timing["peak_reserved"]],
    flush=True,
)

if torch.cuda.is_available():
    MODEL_DTYPE = torch.float16
    DEVICE_MAP = "auto"
else:
    MODEL_DTYPE = torch.float32
    DEVICE_MAP = None


def load_truncated_model(model_dir, layer_count):
    config = AutoConfig.from_pretrained(model_dir)
    config.num_hidden_layers = layer_count
    config.use_cache = False
    if getattr(config, "layer_types", None) is not None:
        config.layer_types = config.layer_types[:layer_count]
    model = AutoModel.from_pretrained(
        model_dir, config=config, dtype=MODEL_DTYPE, device_map=DEVICE_MAP
    ).eval()
    return model, config


def locate_adapter(adapter_name):
    if isinstance(adapter_name, Path):
        bundle_dir = adapter_name
        adapter_dir = (
            bundle_dir / "adapter"
            if (bundle_dir / "adapter" / "adapter_config.json").is_file()
            else bundle_dir
        )
        config_path = adapter_dir / "adapter_config.json"
        assert config_path.is_file(), config_path
        return bundle_dir, adapter_dir, config_path
    matches = [path for path in ADAPTER_CONFIGS if adapter_name in path.parts]
    assert len(matches) == 1, (adapter_name, matches)
    config_path = matches[0]
    bundle_dir = next(
        (
            parent
            for parent in (config_path.parent, *config_path.parent.parents)
            if parent.name == adapter_name
        ),
        None,
    )
    assert bundle_dir is not None, config_path
    return bundle_dir, config_path.parent, config_path


@torch.no_grad()
def merge_adapter(adapter_name, model, layer_count):
    bundle_dir, adapter_dir, config_path = locate_adapter(adapter_name)
    with config_path.open() as f:
        adapter_config = json.load(f)
    scale = adapter_config["lora_alpha"] / adapter_config["r"]
    state_dict = load_file(str(adapter_dir / "adapter_model.safetensors"))
    modules = dict(model.named_modules())
    deltas = []
    original_weights = [] if NTL_SMOKE else None
    lora_a_keys = [key for key in state_dict if ".lora_A." in key]
    suffixes = {key.split(".lora_A.", 1)[0].rsplit(".", 1)[-1] for key in lora_a_keys}
    target_modules = set(adapter_config["target_modules"])
    assert suffixes <= target_modules, (adapter_name, suffixes, target_modules)
    for key in lora_a_keys:
        b_key = key.replace(".lora_A.", ".lora_B.", 1)
        assert b_key in state_dict, (adapter_name, key, b_key)
    for key in lora_a_keys:
        matrix_a = state_dict[key]
        name = key.replace("base_model.model.", "", 1).split(".lora_A.")[0]
        candidates = [name]
        if name.startswith("model."):
            candidates.append(name[len("model.") :])
        else:
            candidates.append(f"model.{name}")
        module_name = next((candidate for candidate in candidates if candidate in modules), None)
        assert module_name is not None, (adapter_name, key)
        weight = modules[module_name].weight
        b_key = key.replace(".lora_A.", ".lora_B.", 1)
        matrix_b = state_dict[b_key]
        if original_weights is not None:
            original_weights.append(weight.detach().clone())
        weight += (
            (matrix_b.to(weight.device, torch.float32) @ matrix_a.to(weight.device, torch.float32))
            * scale
        ).to(weight.dtype)
        deltas.append((weight, matrix_b, matrix_a))
    merged = len(deltas)
    assert merged == layer_count * len(suffixes), (
        adapter_name,
        merged,
        layer_count,
        sorted(suffixes),
    )
    print(f"[{adapter_name}] adapter module suffixes {sorted(suffixes)}", flush=True)
    del modules, state_dict
    return bundle_dir, deltas, scale, merged, original_weights


@torch.no_grad()
def unmerge_adapter(deltas, scale, original_weights=None):
    for index, (weight, matrix_b, matrix_a) in enumerate(deltas):
        weight -= (
            (matrix_b.to(weight.device, torch.float32) @ matrix_a.to(weight.device, torch.float32))
            * scale
        ).to(weight.dtype)
        if original_weights is not None:
            assert torch.allclose(weight, original_weights[index], rtol=1e-6, atol=1e-6)
    if original_weights is not None:
        print(
            f"NTL_SMOKE merge/unmerge restored {len(original_weights)} module weights", flush=True
        )


def featurize_e6(texts, tag, model, tokenizer, hidden_size, device):
    texts = [text if text.strip() else tokenizer.eos_token for text in texts]
    encoded = tokenizer(texts, add_special_tokens=False, truncation=True, max_length=MAX_LEN)[
        "input_ids"
    ]
    order = np.argsort([len(tokens) for tokens in encoded])
    features = np.zeros((len(encoded), hidden_size * len(LAYERS)), dtype=np.float32)
    start = time.time()
    with torch.no_grad():
        for offset in range(0, len(order), BS):
            batch = order[offset : offset + BS]
            max_batch_len = max(len(encoded[index]) for index in batch)
            input_ids = torch.full((len(batch), max_batch_len), tokenizer.pad_token_id)
            attention = torch.zeros((len(batch), max_batch_len), dtype=torch.long)
            for row, index in enumerate(batch):
                input_ids[row, : len(encoded[index])] = torch.tensor(encoded[index])
                attention[row, : len(encoded[index])] = 1
            hidden_states = model(
                input_ids=input_ids.to(device),
                attention_mask=attention.to(device),
                output_hidden_states=True,
                use_cache=False,
            ).hidden_states
            mask = attention.to(device).unsqueeze(-1).float()
            mask[:, 0] = 0
            denominator = mask.sum(1).clamp(min=1)
            features[batch] = (
                torch.nan_to_num(
                    torch.cat(
                        [
                            (hidden_states[layer].float() * mask).sum(1) / denominator
                            for layer in LAYERS
                        ],
                        -1,
                    )
                )
                .cpu()
                .numpy()
            )
            if offset % (BS * 100) == 0:
                print(f"  {tag} {offset}/{len(order)} {time.time() - T0:.0f}s", flush=True)
    print(f"{tag} features {features.shape} in {time.time() - start:.1f}s", flush=True)
    return features


# (i) E6: six public adapters, balanced LR probe per adapter, mean of log-softmax rows.
e6_start = time.time()
tokenizer8 = AutoTokenizer.from_pretrained(MODEL8_DIR)
if tokenizer8.pad_token is None:
    tokenizer8.pad_token = tokenizer8.eos_token
model8, config8 = load_truncated_model(MODEL8_DIR, KEEP8)
device8 = next(model8.parameters()).device
hidden_size8 = config8.hidden_size
logP6_sum = np.zeros((len(test), len(classes)), dtype=np.float64)
for adapter in E6_ADAPTERS:
    _, deltas, scale, merged, original_weights = merge_adapter(adapter, model8, KEEP8)
    print(f"[{adapter}] merged {merged} modules {time.time() - T0:.0f}s", flush=True)
    Htr = featurize_e6(
        train.text.tolist(), f"{adapter} train", model8, tokenizer8, hidden_size8, device8
    )
    scaler = StandardScaler().fit(Htr)
    classifier = LogisticRegression(C=C_HIDDEN, max_iter=2000, class_weight="balanced").fit(
        scaler.transform(Htr), y
    )
    assert (classifier.classes_ == classes).all(), classifier.classes_
    print(
        f"[{adapter}] probe fitted {time.time() - T0:.0f}s; train acc "
        f"{classifier.score(scaler.transform(Htr), y):.3f}",
        flush=True,
    )
    del Htr
    Hte = featurize_e6(
        test.text.tolist(), f"{adapter} test", model8, tokenizer8, hidden_size8, device8
    )
    logP6_sum += classifier.predict_log_proba(scaler.transform(Hte))
    del Hte, classifier, scaler
    unmerge_adapter(deltas, scale, original_weights)
    print(f"[{adapter}] unmerged {merged} modules", flush=True)
    del deltas
logP6 = log_softmax(logP6_sum / len(E6_ADAPTERS), axis=1)
assert logP6.shape == (len(test), len(classes)) and np.isfinite(logP6).all()
e6_elapsed = time.time() - e6_start
print(f"elapsed E6 {e6_elapsed:.1f}s", flush=True)

del model8, tokenizer8
gc.collect()
if torch.cuda.is_available():
    for device_index in range(torch.cuda.device_count()):
        torch.cuda.synchronize(device_index)
    torch.cuda.empty_cache()
gpu_allocated = (
    [torch.cuda.memory_allocated(device_index) for device_index in range(torch.cuda.device_count())]
    if torch.cuda.is_available()
    else []
)
assert max(gpu_allocated, default=0) <= 32 * 1024**2, gpu_allocated
print(f"8B model freed; allocated GPU bytes {gpu_allocated}", flush=True)


def load_q14_head(bundle_dir, hidden_size):
    heads_path = bundle_dir / "heads.pt"
    assert heads_path.is_file(), heads_path
    heads = torch.load(heads_path, map_location="cpu")
    head_hidden_size = int(heads["norms.0.weight"].numel())
    assert head_hidden_size == hidden_size, (head_hidden_size, hidden_size)
    if not NTL_SMOKE:
        assert hidden_size == 5120, hidden_size
    assert heads["plain.weight"].shape == (len(classes), hidden_size * len(LAYERS))
    assert heads["plain.bias"].shape == (len(classes),)
    norms = []
    for layer_index in range(len(LAYERS)):
        norm = torch.nn.LayerNorm(hidden_size)
        norm.load_state_dict(
            {
                "weight": heads[f"norms.{layer_index}.weight"],
                "bias": heads[f"norms.{layer_index}.bias"],
            }
        )
        norm.eval()
        norms.append(norm)
    return heads, norms


def selected_hidden_states_are_finite(hidden_states, layers):
    return all(bool(torch.isfinite(hidden_states[layer]).all()) for layer in layers)


@torch.no_grad()
def featurize_q14(encoded, tag, model, tokenizer, norms, hidden_size, model_dtype, device, layers):
    encoded = [tokens if tokens else [tokenizer.pad_token_id] for tokens in encoded]
    order = np.argsort([len(tokens) for tokens in encoded])
    features = np.zeros((len(encoded), hidden_size * len(layers)), dtype=np.float32)
    bf16_redo_batches = 0
    start = time.time()
    for offset in range(0, len(order), BS):
        batch = order[offset : offset + BS]
        max_batch_len = max(len(encoded[index]) for index in batch)
        input_ids = torch.full((len(batch), max_batch_len), tokenizer.pad_token_id)
        attention = torch.zeros((len(batch), max_batch_len), dtype=torch.long)
        for row, index in enumerate(batch):
            input_ids[row, : len(encoded[index])] = torch.tensor(encoded[index])
            attention[row, : len(encoded[index])] = 1
        input_batch = input_ids.to(device)
        attention_batch = attention.to(device)
        hidden_states = model(
            input_ids=input_batch,
            attention_mask=attention_batch,
            output_hidden_states=True,
            use_cache=False,
        ).hidden_states
        if not selected_hidden_states_are_finite(hidden_states, layers):
            model.to(torch.bfloat16)
            try:
                hidden_states = model(
                    input_ids=input_batch,
                    attention_mask=attention_batch,
                    output_hidden_states=True,
                    use_cache=False,
                ).hidden_states
            finally:
                model.to(model_dtype)
            bf16_redo_batches += 1
            assert selected_hidden_states_are_finite(hidden_states, layers), (
                f"{tag} batch {offset} remains non-finite in bf16"
            )
        mask = attention.unsqueeze(-1).float()
        mask[:, 0] = 0
        pooled_layers = []
        for layer_index, norm in zip(layers, norms):
            layer_hidden = hidden_states[layer_index]
            layer_mask = mask.to(layer_hidden.device)
            denominator = layer_mask.sum(1).clamp(min=1)
            pooled = ((layer_hidden.float() * layer_mask).sum(1) / denominator).cpu()
            pooled_layers.append(norm(pooled))
        features[batch] = torch.cat(pooled_layers, dim=1).numpy()
        if offset % (BS * 100) == 0:
            print(f"  {tag} {offset}/{len(order)} {time.time() - T0:.0f}s", flush=True)
    assert np.isfinite(features).all(), tag
    print(
        f"{tag} features {features.shape} in {time.time() - start:.1f}s; "
        f"bf16 redo batches {bf16_redo_batches}",
        flush=True,
    )
    return features, bf16_redo_batches


# (ii) q14v5: load the 14B base only after the 8B E6 model has been freed.
q14_start = time.time()
tokenizer14 = AutoTokenizer.from_pretrained(MODEL14_DIR)
if tokenizer14.pad_token is None:
    tokenizer14.pad_token = tokenizer14.eos_token
model14, config14 = load_truncated_model(MODEL14_DIR, KEEP14)
device14 = next(model14.parameters()).device
hidden_size14 = config14.hidden_size
logP14_sum = np.zeros((len(test), len(classes)), dtype=np.float64)
bf16_redo_count = 0
for adapter in Q14_ADAPTERS:
    bundle_dir, deltas, scale, merged, original_weights = merge_adapter(adapter, model14, KEEP14)
    print(f"[{adapter}] merged {merged} modules {time.time() - T0:.0f}s", flush=True)
    heads, norms = load_q14_head(bundle_dir, hidden_size14)
    encoded14 = tokenizer14(test.text.tolist(), truncation=True, max_length=MAX_LEN)["input_ids"]
    Hte, adapter_bf16_redos = featurize_q14(
        encoded14,
        f"{adapter} test",
        model14,
        tokenizer14,
        norms,
        hidden_size14,
        MODEL_DTYPE,
        device14,
        LAYERS,
    )
    logits = Hte @ heads["plain.weight"].float().numpy().T + heads["plain.bias"].float().numpy()
    assert logits.shape == (len(test), len(classes)) and np.isfinite(logits).all()
    logP14_sum += log_softmax(logits, axis=1)
    bf16_redo_count += adapter_bf16_redos
    unmerge_adapter(deltas, scale, original_weights)
    print(
        f"[{adapter}] unmerged {merged} modules; bf16 redo batches {adapter_bf16_redos}",
        flush=True,
    )
    del heads, norms, Hte, logits, deltas, original_weights
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
logP14 = logP14_sum / len(Q14_ADAPTERS)
assert logP14.shape == (len(test), len(classes)) and np.isfinite(logP14).all()
q14_elapsed = time.time() - q14_start
print(f"elapsed q14v5 {q14_elapsed:.1f}s; bf16 redo batches {bf16_redo_count}", flush=True)
if NTL_SMOKE:
    q14_logp_snapshot = logP14.copy()

gc.collect()

del model14, tokenizer14
gc.collect()
if torch.cuda.is_available():
    torch.cuda.empty_cache()
print("Qwen3-14B model freed before Qwen2.5-14B load", flush=True)


def locate_q25_bundle():
    matches = [
        path.parent
        for path in Path(ADAPTER_ROOT).rglob("meta.json")
        if path.parent.name == "lora_qwen25_14b" and (path.parent / "heads.safetensors").is_file()
    ]
    assert len(matches) == 1, matches
    return matches[0]


q25_start = time.time()
q25_bundle_dir = locate_q25_bundle()
q25_meta = json.loads((q25_bundle_dir / "meta.json").read_text())
q25_cfg = q25_meta["cfg"]
assert int(q25_cfg["keep"]) == KEEP25
Q25_LAYERS = [int(layer) for layer in q25_cfg["layers"]]
assert Q25_LAYERS == [3, 5, 8, 11, 16, 21, 27, 32]
assert int(q25_cfg["r"]) == 16 and int(q25_cfg["alpha"]) == 32
assert q25_cfg["model_dir"] == "Qwen/Qwen2.5-14B"
Q25_MAX_LEN = int(q25_cfg["max_len"])
Q25_EOS = int(q25_cfg["eos_id"])
if not NTL_SMOKE:
    assert Q25_MAX_LEN == 447
    assert Q25_EOS == 151643
else:
    assert Q25_MAX_LEN > 1 and Q25_EOS >= 0
q25_meta_labels = [str(label) for label in q25_meta["labels"]]
assert sorted(q25_meta_labels) == classes.tolist()


def remap_q25_rows(weight, bias, meta_labels):
    permutation = [meta_labels.index(label) for label in classes]
    return weight[permutation], bias[permutation]


def load_q25_head(bundle_dir, hidden_size, meta_labels):
    heads = load_file(str(bundle_dir / "heads.safetensors"), device="cpu")
    head_hidden_size = int(heads["norms.0.weight"].numel())
    assert head_hidden_size == hidden_size, (head_hidden_size, hidden_size)
    if not NTL_SMOKE:
        assert hidden_size == 5120, hidden_size
    assert heads["plain.weight"].shape == (len(classes), hidden_size * len(Q25_LAYERS))
    assert heads["plain.bias"].shape == (len(classes),)
    norms = []
    for layer_index in range(len(Q25_LAYERS)):
        norm = torch.nn.LayerNorm(hidden_size)
        norm.load_state_dict(
            {
                "weight": heads[f"norms.{layer_index}.weight"],
                "bias": heads[f"norms.{layer_index}.bias"],
            }
        )
        norm.eval()
        norms.append(norm)
    weight, bias = remap_q25_rows(heads["plain.weight"], heads["plain.bias"], meta_labels)
    return norms, weight.float().numpy(), bias.float().numpy()


if NTL_SMOKE:
    assert q25_meta_labels != classes.tolist()
    one_hot_features = np.eye(len(classes), dtype=np.float32)
    one_hot_weight = torch.eye(len(classes), dtype=torch.float32)
    one_hot_bias = torch.zeros(len(classes), dtype=torch.float32)
    remapped_weight, remapped_bias = remap_q25_rows(one_hot_weight, one_hot_bias, q25_meta_labels)
    remapped_logits = one_hot_features @ remapped_weight.numpy().T + remapped_bias.numpy()
    assert classes[remapped_logits.argmax(axis=1)].tolist() == q25_meta_labels
    print("NTL_SMOKE Q25 one-hot label remap passed for 18 labels", flush=True)

tokenizer25 = AutoTokenizer.from_pretrained(MODEL25_DIR)
if tokenizer25.pad_token is None:
    tokenizer25.pad_token = tokenizer25.eos_token
model25, config25 = load_truncated_model(MODEL25_DIR, KEEP25)
device25 = next(model25.parameters()).device
hidden_size25 = config25.hidden_size
assert hasattr(model25, "norm")
assert "rmsnorm" in type(model25.norm).__name__.lower(), type(model25.norm).__name__
model25.norm = torch.nn.Identity()

q25_encoded_raw = tokenizer25(
    test.text.tolist(),
    truncation=True,
    max_length=Q25_MAX_LEN - 1,
)["input_ids"]
q25_encoded = [
    (list(tokens) if text.strip() else []) + [Q25_EOS]
    for text, tokens in zip(test.text.tolist(), q25_encoded_raw)
]
assert all(tokens[-1] == Q25_EOS and len(tokens) <= Q25_MAX_LEN for tokens in q25_encoded)
if NTL_SMOKE:
    blank_indices = [index for index, text in enumerate(test.text.tolist()) if not text.strip()]
    assert blank_indices and all(q25_encoded[index] == [Q25_EOS] for index in blank_indices)
    max_q25_length = max(map(len, q25_encoded))
    assert max_q25_length == Q25_MAX_LEN
    print(f"NTL_SMOKE Q25 EOS/max-length checks passed: max={max_q25_length}", flush=True)

q25_adapter_config_path = q25_bundle_dir / "adapter" / "adapter_config.json"
with q25_adapter_config_path.open() as file:
    q25_adapter_config = json.load(file)
assert int(q25_adapter_config["r"]) == int(q25_cfg["r"])
assert int(q25_adapter_config["lora_alpha"]) == int(q25_cfg["alpha"])
q25_bundle, q25_deltas, q25_scale, q25_merged, q25_original_weights = merge_adapter(
    q25_bundle_dir, model25, KEEP25
)
assert q25_merged == 224, q25_merged
print(f"[q25_14] merged {q25_merged} modules {time.time() - T0:.0f}s", flush=True)
q25_norms, q25_weight, q25_bias = load_q25_head(q25_bundle, hidden_size25, q25_meta_labels)
Hq25, q25_bf16_redo_count = featurize_q14(
    q25_encoded,
    "q25_14 test",
    model25,
    tokenizer25,
    q25_norms,
    hidden_size25,
    MODEL_DTYPE,
    device25,
    Q25_LAYERS,
)
q25_logits = Hq25 @ q25_weight.T + q25_bias
assert q25_logits.shape == (len(test), len(classes)) and np.isfinite(q25_logits).all()
logPq25 = log_softmax(q25_logits, axis=1)
assert logPq25.shape == (len(test), len(classes)) and np.isfinite(logPq25).all()
unmerge_adapter(q25_deltas, q25_scale, q25_original_weights)
if NTL_SMOKE:
    assert all(
        torch.equal(weight, original)
        for (weight, _, _), original in zip(q25_deltas, q25_original_weights)
    )
    assert np.all(classes[logPq25.argmax(axis=1)] == q25_meta_labels[0])
    assert np.array_equal(logP14, q14_logp_snapshot)
    print("NTL_SMOKE Q25 merge/unmerge restored weights bit-exactly", flush=True)
    print(f"NTL_SMOKE Q25 one-hot head predicts {q25_meta_labels[0]} after remap", flush=True)
    print("NTL_SMOKE q14 probabilities unchanged by q25 inference", flush=True)
print(
    f"elapsed Q25 {time.time() - q25_start:.1f}s; bf16 redo batches {q25_bf16_redo_count}",
    flush=True,
)
del q25_deltas, q25_original_weights, q25_norms, q25_weight, q25_bias, Hq25, q25_logits
gc.collect()

del model25, tokenizer25
gc.collect()
if torch.cuda.is_available():
    for device_index in range(torch.cuda.device_count()):
        torch.cuda.synchronize(device_index)
    torch.cuda.empty_cache()


def locate_deberta_bundle():
    matches = [
        path.parent
        for path in Path(ADAPTER_ROOT).rglob("meta.json")
        if path.parent.name == "deberta_v3_large" and (path.parent / "head.safetensors").is_file()
    ]
    assert len(matches) == 1, matches
    return matches[0]


deb_start = time.time()
deb_source_dir = locate_deberta_bundle()
deb_working_root = Path(os.environ.get("NTL_DEB_WORKING_DIR", "/kaggle/working"))
deb_working_root.mkdir(parents=True, exist_ok=True)
deb_bundle_dir = deb_working_root / "deberta_v3_large"
shutil.copytree(deb_source_dir, deb_bundle_dir, dirs_exist_ok=True)
deb_meta = json.loads((deb_bundle_dir / "meta.json").read_text())
assert int(deb_meta["max_len"]) == 384
assert deb_meta["pool"] == "mean_last"
deb_meta_labels = [str(label) for label in deb_meta["labels"]]
assert sorted(deb_meta_labels) == classes.tolist()
deb_encoder_dir = deb_bundle_dir / "encoder"
assert (deb_encoder_dir / "tokenizer.json").is_file()
deb_config = AutoConfig.from_pretrained(deb_encoder_dir)
assert deb_config.model_type == "deberta-v2"
deb_hidden_size = int(deb_config.hidden_size)
if not NTL_SMOKE:
    assert deb_hidden_size == 1024, deb_hidden_size
deb_head = load_file(str(deb_bundle_dir / "head.safetensors"), device="cpu")
if not NTL_SMOKE:
    assert deb_head["weight"].shape == (18, 1024)
    assert deb_head["bias"].shape == (18,)
assert deb_head["weight"].shape == (len(classes), deb_hidden_size)
assert deb_head["bias"].shape == (len(classes),)
deb_permutation = [deb_meta_labels.index(label) for label in classes]
Wdeb = deb_head["weight"].float().numpy()[deb_permutation]
bdeb = deb_head["bias"].float().numpy()[deb_permutation]
deb_tokenizer_path = "native"
try:
    deb_tokenizer = AutoTokenizer.from_pretrained(deb_encoder_dir, use_fast=True)
except Exception:
    deb_tokenizer = None
if deb_tokenizer is None or not deb_tokenizer.is_fast:
    deb_tokenizer_config_path = deb_encoder_dir / "tokenizer_config.json"
    deb_tokenizer_config = json.loads(deb_tokenizer_config_path.read_text())
    deb_tokenizer_config["tokenizer_class"] = "PreTrainedTokenizerFast"
    deb_tokenizer_config_path.write_text(json.dumps(deb_tokenizer_config, indent=2))
    deb_tokenizer = AutoTokenizer.from_pretrained(deb_encoder_dir, use_fast=True)
    deb_tokenizer_path = "fallback"
assert deb_tokenizer.is_fast
assert deb_config.pad_token_id == 0, deb_config.pad_token_id
assert deb_tokenizer.pad_token_id == deb_config.pad_token_id, deb_tokenizer.pad_token_id
deb_probe_ids = deb_tokenizer("Hello world.")["input_ids"]
assert deb_probe_ids[0] == deb_tokenizer.cls_token_id
assert deb_probe_ids[-1] == deb_tokenizer.sep_token_id
print(
    f"DEB tokenizer path={deb_tokenizer_path} class={type(deb_tokenizer).__name__} "
    f"transformers={transformers.__version__}",
    flush=True,
)
if NTL_SMOKE:
    assert not list(deb_encoder_dir.rglob("spm.model"))
    assert deb_meta_labels != classes.tolist()
    one_hot_features = np.eye(len(classes), deb_hidden_size, dtype=np.float32)
    one_hot_weight = torch.zeros((len(classes), deb_hidden_size))
    for meta_index, label in enumerate(deb_meta_labels):
        one_hot_weight[meta_index, int(np.where(classes == label)[0][0])] = 1
    one_hot_remapped = one_hot_weight[deb_permutation].numpy()
    one_hot_predictions = classes[(one_hot_features @ one_hot_remapped.T).argmax(axis=1)]
    assert one_hot_predictions.tolist() == classes.tolist()
    print("NTL_SMOKE DeBERTa one-hot head remap passed for 18 labels", flush=True)
deb_device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
deb_encoder = AutoModel.from_pretrained(deb_encoder_dir, dtype=torch.float32).to(deb_device).eval()
assert next(deb_encoder.parameters()).dtype == torch.float32
deb_encoded = deb_tokenizer(
    test.text.tolist(),
    truncation=True,
    max_length=int(deb_meta["max_len"]),
)["input_ids"]
assert all(len(tokens) <= int(deb_meta["max_len"]) for tokens in deb_encoded)


@torch.inference_mode()
def featurize_deberta(encoded, tag, encoder, tokenizer, device):
    order = np.argsort([len(tokens) for tokens in encoded], kind="stable")
    features = np.zeros((len(encoded), deb_hidden_size), dtype=np.float32)
    for start in range(0, len(order), 32):
        batch_indices = order[start : start + 32]
        batch = tokenizer.pad(
            {"input_ids": [encoded[index] for index in batch_indices]},
            return_tensors="pt",
        ).to(device)
        mask = batch["attention_mask"].unsqueeze(-1).float()
        hidden = encoder(**batch).last_hidden_state
        assert torch.isfinite(hidden).all(), f"{tag} hidden states are non-finite"
        pooled = (hidden * mask).sum(1) / mask.sum(1).clamp(min=1)
        features[batch_indices] = pooled.cpu().numpy()
    assert np.isfinite(features).all(), tag
    return features


if NTL_SMOKE:
    deb_lengths = [len(tokens) for tokens in deb_encoded]
    assert max(deb_lengths) == int(deb_meta["max_len"])
    assert all(tokens[0] == deb_tokenizer.cls_token_id for tokens in deb_encoded)
    assert all(tokens[-1] == deb_tokenizer.sep_token_id for tokens in deb_encoded)
    blank_indices = [index for index, text in enumerate(test.text.tolist()) if not text.strip()]
    assert blank_indices and all(
        deb_encoded[index] == [deb_tokenizer.cls_token_id, deb_tokenizer.sep_token_id]
        for index in blank_indices
    )
    short_index = next(
        index
        for index, text in enumerate(test.text.tolist())
        if text.strip() and deb_lengths[index] < int(deb_meta["max_len"])
    )
    long_index = int(np.argmax(deb_lengths))
    alone_features = featurize_deberta(
        [deb_encoded[short_index]], "deb smoke alone", deb_encoder, deb_tokenizer, deb_device
    )
    padded_features = featurize_deberta(
        [deb_encoded[short_index], deb_encoded[long_index]],
        "deb smoke padded batch",
        deb_encoder,
        deb_tokenizer,
        deb_device,
    )
    assert np.allclose(alone_features[0], padded_features[0], rtol=1e-5, atol=1e-5)
    print(
        f"NTL_SMOKE DeBERTa truncation/special tokens and padding mask passed: max={max(deb_lengths)}",
        flush=True,
    )
    print("NTL_SMOKE DeBERTa empty, whitespace, and short rows tokenized", flush=True)
deb_features = featurize_deberta(
    deb_encoded, "deberta_v3_large test", deb_encoder, deb_tokenizer, deb_device
)
deb_logits = deb_features @ Wdeb.T + bdeb
assert deb_logits.shape == (len(test), len(classes)) and np.isfinite(deb_logits).all()
logPdeb = log_softmax(deb_logits, axis=1)
assert logPdeb.shape == (len(test), len(classes)) and np.isfinite(logPdeb).all()
if NTL_SMOKE:
    expected_deb_class = deb_meta_labels[0]
    assert np.all(classes[logPdeb.argmax(axis=1)] == expected_deb_class)
    print(f"NTL_SMOKE DeBERTa one-hot head predicts {expected_deb_class}", flush=True)
deb_elapsed = time.time() - deb_start
print(f"elapsed DeBERTa {deb_elapsed:.1f}s", flush=True)
del deb_encoder, deb_tokenizer, deb_head, deb_features, deb_logits, Wdeb, bdeb
gc.collect()
if torch.cuda.is_available():
    for device_index in range(torch.cuda.device_count()):
        torch.cuda.synchronize(device_index)
    torch.cuda.empty_cache()

gpu_peak_gb = (
    max((torch.cuda.max_memory_allocated(i) for i in range(torch.cuda.device_count())), default=0)
    / 1e9
    if torch.cuda.is_available()
    else 0.0
)
print(f"GPU peak allocated GB {gpu_peak_gb:.2f}", flush=True)

# Fixed E16 blend with a 0.4 geometric share for the native expert.
z, zf = blend_e17(logP6, logP14, logPq25, logPdeb, logPnat)
assert z.shape == (len(test), len(classes)) and np.isfinite(z).all()
assert zf.shape == (len(test), len(classes)) and np.isfinite(zf).all()
pred = classes[zf.argmax(1)]
sub = pd.DataFrame({"id": test.id.to_numpy(), "model": pred})
output_dir = os.path.dirname(OUT) or "."
os.makedirs(output_dir, exist_ok=True)
sub.to_parquet(OUT, index=False)
chk = pd.read_parquet(OUT)
assert chk.columns.tolist() == ["id", "model"] and len(chk) == len(test)
assert chk.id.tolist() == test.id.tolist() and chk.id.is_unique
assert chk.model.notna().all() and chk.model.isin(LABELS).all()
if NTL_SMOKE:
    assert np.array_equal(pred, classes[zf.argmax(axis=1)])
    assert np.array_equal(chk.model.to_numpy(), pred)
    print(
        "SMOKE invariants "
        f"rows={len(chk)} ids_in_order=True unique=True nulls=0 valid=True "
        f"classes_used={chk.model.nunique()} prob_shapes="
        f"{logP6.shape},{logP14.shape},{logPq25.shape},{logPdeb.shape},"
        f"{logPnat.shape},{nat_plain_folds.shape} blend_argmax=True",
        flush=True,
    )
np.savez(
    os.path.join(output_dir, "e17_probs.npz"),
    classes=classes,
    logP6=logP6,
    logP14=logP14,
    logPq25=logPq25,
    logPdeb=logPdeb,
    z=z,
    logPnat=logPnat,
    nat_plain_folds=nat_plain_folds,
)
print(
    f"wrote {OUT}: {len(chk)} rows, {chk.model.nunique()} labels used, total {time.time() - T0:.1f}s",
    flush=True,
)
print(sub.head(3).to_string(index=False))
