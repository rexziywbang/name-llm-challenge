"""H1 measurement: context-ablated loss at fixed anchors under a frozen small causal LM.

For each passage, tokens are indexed deterministically; anchors are every `stride` tokens starting at offset 32
(so every anchor has >= 32 preceding tokens). For each anchor, a span of `span` target tokens is scored with exactly
c in CONTEXTS preceding tokens. Features aggregate loss(c=8) - loss(c=32) etc. The scorer only measures properties of
the observed text; it cannot attribute them to generator architecture."""

import numpy as np
import torch

CONTEXTS = (8, 16, 32)
FEATURE_NAMES = [
    "n_anchors",
    "loss_c8",
    "loss_c16",
    "loss_c32",
    "gain_8_32",
    "gain_8_16",
    "gain_16_32",
    "gain_8_32_std",
    "gain_short_minus_long",
    "frac_anchor_gain_pos",
]


class ContextGainScorer:
    def __init__(
        self,
        model_name,
        device=None,
        max_anchors=12,
        stride=16,
        span=4,
        offset=32,
        batch_size=64,
        dtype=None,
    ):
        from transformers import AutoModelForCausalLM, AutoTokenizer

        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.tok = AutoTokenizer.from_pretrained(model_name)
        self.model = (
            AutoModelForCausalLM.from_pretrained(model_name, dtype=dtype or torch.float32)
            .to(self.device)
            .eval()
        )
        self.max_anchors, self.stride, self.span, self.offset, self.bs = (
            max_anchors,
            stride,
            span,
            offset,
            batch_size,
        )
        self.bos = self.tok.bos_token_id

    def anchors(self, ids):
        n = len(ids)
        a = list(range(self.offset, n - self.span + 1, self.stride))
        if len(a) > self.max_anchors:
            idx = np.linspace(0, len(a) - 1, self.max_anchors).round().astype(int)
            a = [a[i] for i in idx]
        return a

    @torch.no_grad()
    def _span_losses(self, windows, n_targets):
        """windows: list of token-id lists (context + span). Returns mean NLL over the last n_targets tokens of each."""
        out = np.zeros(len(windows), dtype=np.float32)
        for s in range(0, len(windows), self.bs):
            chunk = windows[s : s + self.bs]
            L = max(len(w) for w in chunk)
            pad = self.tok.pad_token_id if self.tok.pad_token_id is not None else 0
            x = torch.full((len(chunk), L), pad, dtype=torch.long)
            att = torch.zeros((len(chunk), L), dtype=torch.long)
            for i, w in enumerate(chunk):  # left-pad so targets align at the end
                x[i, L - len(w) :] = torch.tensor(w)
                att[i, L - len(w) :] = 1
            x, att = x.to(self.device), att.to(self.device)
            logits = self.model(input_ids=x, attention_mask=att).logits[:, :-1].float()
            lp = torch.log_softmax(logits, -1).gather(-1, x[:, 1:, None])[..., 0]
            out[s : s + len(chunk)] = (-lp[:, -n_targets:]).mean(1).cpu().numpy()
        return out

    def score(self, text):
        ids = self.tok(text, add_special_tokens=False)["input_ids"]
        anc = self.anchors(ids)
        if not anc:
            return np.array([0] + [np.nan] * (len(FEATURE_NAMES) - 1), dtype=np.float32)
        windows = []
        for a in anc:
            for c in CONTEXTS:
                w = ids[a - c : a + self.span]
                if self.bos is not None:
                    w = [self.bos] + w
                windows.append(w)
        losses = self._span_losses(windows, self.span).reshape(len(anc), len(CONTEXTS))
        l8, l16, l32 = losses[:, 0], losses[:, 1], losses[:, 2]
        g = l8 - l32
        return np.array(
            [
                len(anc),
                l8.mean(),
                l16.mean(),
                l32.mean(),
                g.mean(),
                (l8 - l16).mean(),
                (l16 - l32).mean(),
                g.std(),
                (l8 - l16).mean() - (l16 - l32).mean(),
                (g > 0).mean(),
            ],
            dtype=np.float32,
        )

    def score_many(self, texts, log_every=200):
        import time

        t0 = time.time()
        out = []
        for i, t in enumerate(texts):
            out.append(self.score(t))
            if log_every and (i + 1) % log_every == 0:
                print(f"  scored {i + 1}/{len(texts)} ({time.time() - t0:.0f}s)", flush=True)
        return np.vstack(out)
