# Adapted from CHEN Han / lightningv08, licensed under Apache-2.0.
"""CHEN Han's trained eight-depth Qwen3-Embedding expert, with covering windows.

The full backbone is split over two T4s. Task LoRA/head values stay FP32;
the original BF16 backbone values are loaded as FP16. Only one fold's
trainable state is resident at a time; the backbone is loaded once.
"""

import argparse
import gc
import json
import os
from pathlib import Path
import shutil
import time

os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
import numpy as np
import pandas as pd
import torch
from torch import nn
from torch.nn import functional as F
from safetensors.torch import load_file
from transformers import AutoModel, AutoTokenizer

INSTRUCTION = "Instruct: Identify the language model that generated the text based on its writing style.\nQuery: "
DEPTHS = (3, 6, 9, 12, 18, 24, 30, 36)
SUFFIXES = ("q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj")


class TaskLoRA(nn.Module):
    def __init__(self, base):
        super().__init__()
        self.base_layer = base
        device = base.weight.device
        self.a = nn.Linear(base.in_features, 16, bias=False, device=device, dtype=torch.float32)
        self.b = nn.Linear(16, base.out_features, bias=False, device=device, dtype=torch.float32)

    def forward(self, x):
        result = self.base_layer(x)
        delta = self.b(self.a(x.float())) * 2.0
        return (result + delta).to(result.dtype)


class NativeQwen:
    def __init__(self, base, assets, work):
        torch.set_num_threads(2)
        torch.use_deterministic_algorithms(True)
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.allow_tf32 = False
        for key in (
            "allow_tf32",
            "allow_fp16_reduced_precision_reduction",
            "allow_bf16_reduced_precision_reduction",
        ):
            setattr(torch.backends.cuda.matmul, key, False)
        assert torch.cuda.device_count() == 2, "The native expert uses both T4 devices"
        self.assets = Path(assets)
        self.labels = json.loads((self.assets / "manifest.json").read_text())["classes"]
        origin = Path(base)
        index = json.loads((origin / "model.safetensors.index.json").read_text())
        shards = sorted(set(index["weight_map"].values()))
        names = [
            "config.json",
            "model.safetensors.index.json",
            "tokenizer.json",
            "tokenizer_config.json",
            "merges.txt",
            "vocab.json",
        ] + shards
        # Use the original training tokenizer, not a same-named Kaggle revision.
        sources = {
            name: (
                origin / name
                if name in shards or name == "model.safetensors.index.json"
                else self.assets / "base" / name
            )
            for name in names
        }
        for name in names:
            assert sources[name].is_file(), str(sources[name])
        for fold in range(5):
            assert (self.assets / f"fold{fold}/trainables.safetensors").is_file()
        self.staged = Path(work) / "native-q8-base"
        self.staged.mkdir(parents=True, exist_ok=True)
        for name in names:
            destination = self.staged / name
            source = sources[name]
            if name in shards or name == "model.safetensors.index.json":
                if destination.is_symlink() and destination.resolve() != source.resolve():
                    destination.unlink()
                if destination.exists() and not destination.is_symlink():
                    destination.unlink()
                if not destination.exists() and not destination.is_symlink():
                    destination.symlink_to(source)
            else:
                if destination.is_symlink():
                    destination.unlink()
                if not destination.exists():
                    shutil.copyfile(source, destination)
            assert destination.stat().st_size == source.stat().st_size
        cfg = json.loads((self.staged / "config.json").read_text())
        assert cfg["num_hidden_layers"] == 36 and cfg["hidden_size"] == 4096
        self.tokenizer = AutoTokenizer.from_pretrained(self.staged, local_files_only=True)
        self.tokenizer.padding_side = "right"
        if self.tokenizer.pad_token_id is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        placement = {"embed_tokens": 0, "norm": 1, "rotary_emb": 1}
        placement.update({f"layers.{i}": 0 if i < 16 else 1 for i in range(36)})
        model_kwargs = dict(
            local_files_only=True,
            device_map=placement,
            attn_implementation="sdpa",
            output_loading_info=True,
        )
        try:
            self.model, info = AutoModel.from_pretrained(
                self.staged, dtype=torch.float16, **model_kwargs
            )
        except TypeError:
            self.model, info = AutoModel.from_pretrained(
                self.staged, torch_dtype=torch.float16, **model_kwargs
            )
        assert not any(info.get(k) for k in ("missing_keys", "mismatched_keys", "error_msgs")), info
        assert not info.get("unexpected_keys"), info
        self.model.config.use_cache = False
        self.modules = {}
        for name, layer in list(self.model.named_modules()):
            if isinstance(layer, nn.Linear) and name.rsplit(".", 1)[-1] in SUFFIXES:
                wrapped = TaskLoRA(layer)
                parent, attr = name.rsplit(".", 1)
                setattr(self.model.get_submodule(parent), attr, wrapped)
                self.modules[name] = wrapped
        assert len(self.modules) == 252, len(self.modules)
        self.norms = nn.ModuleList(
            [nn.LayerNorm(4096, device="cuda:0", dtype=torch.float32) for _ in DEPTHS]
        )
        self.plain = nn.Linear(32768, 18, device="cuda:0", dtype=torch.float32)
        self.centered = nn.Linear(32768, 18, device="cuda:0", dtype=torch.float32)
        self.model.requires_grad_(False).eval()
        for module in (self.norms, self.plain, self.centered):
            module.requires_grad_(False).eval()
        self.current = None
        self.expected = {}
        for name, module in self.modules.items():
            for suffix, parameter in [
                ("lora_A.default.weight", module.a.weight),
                ("lora_B.default.weight", module.b.weight),
            ]:
                self.expected["encoder.backbone.base_model.model." + name + "." + suffix] = (
                    parameter
                )
        for prefix, module in [
            ("norms", self.norms),
            ("plain", self.plain),
            ("centered", self.centered),
        ]:
            self.expected.update(
                {prefix + "." + name: parameter for name, parameter in module.named_parameters()}
            )
        assert len(self.expected) == 524

    @torch.inference_mode()
    def load_fold(self, fold):
        if self.current == fold:
            return
        state = load_file(str(self.assets / f"fold{fold}/trainables.safetensors"))
        assert set(state) == set(self.expected), (
            set(state) - set(self.expected),
            set(self.expected) - set(state),
        )
        for name, parameter in self.expected.items():
            assert state[name].shape == parameter.shape and state[name].dtype == torch.float32, name
            assert torch.isfinite(state[name]).all(), name
            parameter.copy_(state[name].to(parameter.device))
        del state
        gc.collect()
        self.current = fold

    def windows(self, texts):
        prefix = self.tokenizer(INSTRUCTION, add_special_tokens=False)["input_ids"]
        terminal = self.tokenizer("", add_special_tokens=True)["input_ids"]
        assert terminal == [151643], "The training tokenizer appends its native end-of-text token"
        capacity = 512 - len(prefix) - len(terminal)
        assert capacity > 0
        full = self.tokenizer(
            [INSTRUCTION + str(t) for t in texts], truncation=False, add_special_tokens=False
        )["input_ids"]
        windows = []
        owners = []
        weights = []
        lengths = []
        for row, tokens in enumerate(full):
            assert len(tokens) > 0
            first_capacity = 512 - len(terminal)
            windows.append(tokens[:first_capacity] + terminal)
            owners.append(row)
            weights.append(max(1, min(len(tokens), first_capacity) - len(prefix)))
            position = first_capacity
            while position < len(tokens):
                content = tokens[position : position + capacity]
                windows.append(prefix + content + terminal)
                owners.append(row)
                weights.append(len(content))
                position += len(content)
            lengths.append(len(tokens) + len(terminal))
            assert position >= len(tokens)
        return windows, np.asarray(owners), np.asarray(weights, dtype=np.float64), lengths

    @torch.inference_mode()
    def predict(self, texts, folds=range(5), batch_size=4):
        folds = list(folds)
        windows, owners, weights, lengths = self.windows(texts)
        # A token sequence provides a deterministic tie-break independent of row order.
        order = sorted(range(len(windows)), key=lambda i: (len(windows[i]), windows[i]))
        outputs = {"plain": []}
        timings = []
        for fold in folds:
            self.load_fold(fold)
            began = time.monotonic()
            values = {"plain": np.zeros((len(texts), 18), dtype=np.float64)}
            for start in range(0, len(order), batch_size):
                chosen = order[start : start + batch_size]
                batch = self.tokenizer.pad(
                    {"input_ids": [windows[i] for i in chosen]}, padding=True, return_tensors="pt"
                ).to("cuda:0")
                output = self.model(
                    **batch, output_hidden_states=True, use_cache=False, return_dict=True
                )
                assert len(output.hidden_states) == 37
                positions = batch["attention_mask"].sum(1) - 1
                parts = []
                for depth, norm in zip(DEPTHS, self.norms):
                    hidden = (
                        output.last_hidden_state if depth == 36 else output.hidden_states[depth]
                    )
                    pooled = hidden[
                        torch.arange(len(chosen), device=hidden.device), positions.to(hidden.device)
                    ].float()
                    if depth == 36:
                        pooled = F.normalize(pooled, dim=-1)
                    parts.append(norm(pooled.to("cuda:0")))
                features = torch.cat(parts, dim=-1)
                scores = self.plain(features).cpu().numpy().astype(np.float64)
                np.add.at(values["plain"], owners[chosen], scores * weights[chosen, None])
                del output, features, parts, batch
            denominator = np.bincount(owners, weights=weights, minlength=len(texts))[:, None]
            for name in values:
                outputs[name].append(values[name] / denominator)
            torch.cuda.synchronize(0)
            torch.cuda.synchronize(1)
            timings.append(dict(fold=int(fold), seconds=time.monotonic() - began))
            print(
                json.dumps(
                    dict(
                        completed_fold=int(fold),
                        documents=len(texts),
                        windows=len(windows),
                        seconds=timings[-1]["seconds"],
                    )
                ),
                flush=True,
            )
        for key in outputs:
            outputs[key] = np.stack(outputs[key])
            assert np.isfinite(outputs[key]).all(), key
        return outputs, dict(
            documents=len(texts),
            windows=len(windows),
            discarded_tokens=0,
            formatted_token_lengths=lengths,
            fold_timings=timings,
            peak_reserved=[torch.cuda.max_memory_reserved(i) for i in range(2)],
        )


def main():
    parser = argparse.ArgumentParser()
    for name in ("base", "assets", "query", "out", "work"):
        parser.add_argument("--" + name, type=Path, required=True)
    parser.add_argument("--folds", default="0,1,2,3,4")
    args = parser.parse_args()
    folds = [int(value.strip()) for value in args.folds.split(",") if value.strip()]
    assert folds and len(folds) == len(set(folds)) and all(0 <= fold < 5 for fold in folds), folds
    frame = pd.read_csv(args.query, dtype={"id": str})
    assert frame.id.is_unique and len(frame)
    began = time.monotonic()
    model = NativeQwen(args.base, args.assets, args.work)
    values, report = model.predict(frame.text.fillna("").astype(str).tolist(), folds=folds)
    np.savez(
        args.out,
        ids=frame.id.to_numpy(dtype=str),
        labels=np.asarray(model.labels),
        plain=values["plain"],
    )
    report["seconds_including_load_and_staging"] = time.monotonic() - began
    args.out.with_suffix(".json").write_text(json.dumps(report, indent=2))
    print(json.dumps(report), flush=True)


if __name__ == "__main__":
    main()
