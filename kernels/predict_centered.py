# %% [markdown]
# # Name That LLM — baseline + frozen hidden-state probe with test-time prompt-cluster centring (E4; offline)
# v2 recipe (char/word TF-IDF + LinearSVC, stylometric LR, linear probe on frozen Qwen2.5-0.5B hidden states) plus E4: test
# passages are clustered into prompt groups (word 1-2gram TF-IDF, average-link cosine, threshold 0.95); inside clusters of
# size >= MIN_CLUSTER the standardised hidden features are centred (group mean removed) and scored by a probe trained on
# prompt-centred train features; smaller clusters fall back to the uncentred E3 probe. Everything fitted in-kernel from
# train.csv; blend weights frozen from prompt-grouped dev validation. Model asset: Kaggle model qwen-lm/qwen2.5/transformers/0.5b/1
# (Apache-2.0), no internet needed. Prior art: test-time group centring appears in public notebooks.
# Writes /kaggle/working/submission.parquet. See EXPERIMENTS.md for the sibling-count control.
# %%
import glob
import os
import re
import time
import unicodedata
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
import numpy as np
import pandas as pd
from scipy.sparse import hstack
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.svm import LinearSVC
from sklearn.cluster import AgglomerativeClustering

T0 = time.time()
IN = os.environ.get("NTL_INPUT_DIR", "/kaggle/input/competitions/name-that-llm")
OUT = os.environ.get("NTL_OUTPUT_PATH", "/kaggle/working/submission.parquet")
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
W_TFIDF, W_STYLE, C_SVC, C_STYLE = 1.0, 0.5, 0.5, 0.3
W_HIDDEN, C_HIDDEN, LAYERS, MAX_LEN, BATCH = (
    1.75,
    0.002,
    (6, 12, 18, 24),
    384,
    16,
)  # frozen from artifacts/e3/qwen05_decision.json
W_HIDDEN_C, TAU, CLUSTER_THR, MIN_CLUSTER = (
    2.0,
    0.0,
    0.95,
    3,
)  # frozen from artifacts/e4/qwen05_decision.json
MODEL_DIR = (
    os.environ.get("NTL_MODEL_DIR")
    or next(
        p for p in glob.glob("/kaggle/input/**/config.json", recursive=True) if "qwen2" in p.lower()
    ).rsplit("/", 1)[0]
)

# %% --- feature definitions (identical to src/ntl/text.py and src/ntl/style.py in the repo) ---
_WS = re.compile(r"\s+")


def normalize(text):
    t = unicodedata.normalize("NFKC", text)
    t = (
        t.replace("\u2019", "'")
        .replace("\u2018", "'")
        .replace("\u201c", '"')
        .replace("\u201d", '"')
    )
    t = t.replace("\u2014", " - ").replace("\u2013", " - ")
    return _WS.sub(" ", t).strip()


FUNCTION_WORDS = (
    "the of and to a in that is it for as with was on be by at this from or an are not but have "
    "which you we they their there can one all if more when so no than its also into about would what "
    "will may such these those only other some any each between however because while where through "
    "both most many much very just even then here how could should might must"
).split()
_SENT = re.compile(r"[.!?]+(?:\s|$)")
_WORD = re.compile(r"[A-Za-z']+")


def _frac(n, d):
    return n / d if d else 0.0


def style_features(text):
    chars = len(text) or 1
    words = _WORD.findall(text)
    nw = len(words) or 1
    low = [w.lower() for w in words]
    sents = [s for s in _SENT.split(text) if s.strip()]
    sl = np.array([len(_WORD.findall(s)) for s in sents] or [nw], dtype=float)
    wl = np.array([len(w) for w in words] or [0], dtype=float)
    uniq = len(set(low))
    bigrams = list(zip(low, low[1:]))
    rep_bi = _frac(len(bigrams) - len(set(bigrams)), len(bigrams))
    lines = text.split("\n")
    f = [
        np.log1p(chars),
        np.log1p(nw),
        len(sents),
        sl.mean(),
        sl.std(),
        sl.max(),
        sl.min(),
        wl.mean(),
        wl.std(),
        _frac(uniq, nw),
        rep_bi,
        _frac(sum(len(w) > 6 for w in words), nw),
        _frac(text.count(","), nw),
        _frac(text.count(";"), nw),
        _frac(text.count(":"), nw),
        _frac(text.count("—"), nw),
        _frac(text.count(" — "), nw),
        _frac(text.count("–"), nw),
        _frac(text.count("-"), nw),
        _frac(text.count("’"), nw),
        _frac(text.count("'"), nw),
        _frac(text.count("“") + text.count("”"), nw),
        _frac(text.count('"'), nw),
        _frac(text.count("("), nw),
        _frac(text.count("?"), nw),
        _frac(text.count("!"), nw),
        _frac(text.count("..."), nw) + _frac(text.count("…"), nw),
        _frac(sum(c.isdigit() for c in text), chars),
        _frac(sum(c.isupper() for c in text), chars),
        _frac(sum(ord(c) > 127 for c in text), chars),
        len(lines) - 1,
        _frac(sum(1 for ln in lines if ln.lstrip().startswith(("-", "*", "•"))), len(lines)),
        _frac(sum(1 for ln in lines if re.match(r"\s*\d+[.)]", ln)), len(lines)),
        text.count("**"),
        text.count("#"),
        text.count("`"),
        float(text.rstrip().endswith((".", "!", "?", '"', "”"))),
        float(text.strip()[:1].islower() if text.strip() else 0),
        _frac(sum(1 for w in low if w in ("i", "my", "me")), nw),
        _frac(sum(1 for w in low if w in ("you", "your")), nw),
        _frac(sum(1 for w in low if w in ("we", "our", "us")), nw),
    ]
    return np.array(f + [_frac(low.count(w), nw) for w in FUNCTION_WORDS], dtype=np.float32)


def style_matrix(texts):
    return np.vstack([style_features(t) for t in texts])


def softmax(z):
    z = z - z.max(1, keepdims=True)
    e = np.exp(z)
    return e / e.sum(1, keepdims=True)


# %% --- load ---
train = pd.read_csv(f"{IN}/train.csv")
test = pd.read_csv(f"{IN}/test.csv")
if os.environ.get("NTL_TRAIN_LIMIT"):  # local smoke tests only (never set on Kaggle)
    train = train.sample(int(os.environ["NTL_TRAIN_LIMIT"]), random_state=0).reset_index(drop=True)
test["text"] = test["text"].fillna("").astype(str)
y = train.model.map({lab: i for i, lab in enumerate(LABELS)}).to_numpy()
assert not np.isnan(y.astype(float)).any()
print(f"train {train.shape} test {test.shape} load {time.time() - T0:.1f}s", flush=True)


# %% --- TF-IDF + LinearSVC with temperature from an inner prompt-grouped split (same recipe as repo TfidfLinear) ---
def tfidf_feats(cv, wv, texts, fit=False):
    norm = [normalize(t) for t in texts]
    return (
        hstack([cv.fit_transform(texts), wv.fit_transform(norm)])
        if fit
        else hstack([cv.transform(texts), wv.transform(norm)])
    ).tocsr()


def new_vecs():
    return (
        TfidfVectorizer(
            analyzer="char",
            ngram_range=(2, 5),
            sublinear_tf=True,
            min_df=2,
            max_features=300_000,
            lowercase=False,
        ),
        TfidfVectorizer(
            analyzer="word",
            ngram_range=(1, 2),
            sublinear_tf=True,
            min_df=2,
            max_features=150_000,
            token_pattern=r"(?u)\b\w+\b|[^\w\s]",
        ),
    )


rng = np.random.default_rng(20261027)
prompts = train.prompt_id.unique()
rng.shuffle(prompts)
cal_prompts = set(prompts[: len(prompts) // 5])
cal = train.prompt_id.isin(cal_prompts).to_numpy()
cv, wv = new_vecs()
Xtr = tfidf_feats(cv, wv, train.text[~cal].tolist(), fit=True)
d = (
    LinearSVC(C=C_SVC, class_weight="balanced")
    .fit(Xtr, y[~cal])
    .decision_function(tfidf_feats(cv, wv, train.text[cal].tolist()))
)
best, T = 1e9, 1.0
for t in np.geomspace(0.05, 5, 60):
    nll = -np.log(softmax(d / t)[np.arange(cal.sum()), y[cal]] + 1e-12).mean()
    if nll < best:
        best, T = nll, t
cv, wv = new_vecs()
X_all = tfidf_feats(cv, wv, train.text.tolist(), fit=True)
svc = LinearSVC(C=C_SVC, class_weight="balanced").fit(X_all, y)
print(f"tfidf fitted (T={T:.3f}) {time.time() - T0:.1f}s", flush=True)

# %% --- style LR ---
S = style_matrix(train.text.tolist())
sc = StandardScaler().fit(S)
lr = LogisticRegression(C=C_STYLE, max_iter=3000, class_weight="balanced").fit(sc.transform(S), y)
print(f"style fitted {time.time() - T0:.1f}s", flush=True)

# %% --- frozen hidden-state features (identical to src/ntl/hidden.py) + LR probe ---
DEV = "cuda" if torch.cuda.is_available() else "cpu"
torch.set_num_threads(os.cpu_count() or 4)
tok = AutoTokenizer.from_pretrained(MODEL_DIR)
tok.padding_side = "right"
if tok.pad_token is None:
    tok.pad_token = tok.eos_token
lm = AutoModelForCausalLM.from_pretrained(MODEL_DIR, dtype=torch.float32).to(DEV).eval()


@torch.no_grad()
def hidden_matrix(texts, log_every=0):
    out = []
    texts = [
        t if t.strip() else tok.eos_token for t in texts
    ]  # empty passage -> one token, avoids 0/0
    for s in range(0, len(texts), BATCH):
        enc = tok(
            texts[s : s + BATCH],
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=MAX_LEN,
        ).to(DEV)
        hs = lm(**enc, output_hidden_states=True).hidden_states
        mask = enc["attention_mask"].unsqueeze(-1).float()
        last = (enc["attention_mask"].sum(1) - 1).clamp(min=0)
        feats = []
        for L in LAYERS:
            feats.append(((hs[L] * mask).sum(1) / mask.sum(1).clamp(min=1)).float())
            feats.append(hs[L][torch.arange(hs[L].shape[0]), last].float())
        out.append(torch.cat(feats, 1).cpu().numpy().astype(np.float32))
        if log_every and (s // BATCH) % log_every == 0:
            print(
                f"  hidden {s + len(enc['input_ids'])}/{len(texts)} {time.time() - T0:.0f}s",
                flush=True,
            )
    return (
        np.vstack(out)
        if out
        else np.zeros((0, 2 * len(LAYERS) * lm.config.hidden_size), np.float32)
    )


t1 = time.time()
H = hidden_matrix(train.text.tolist(), log_every=25)
print(
    f"train hidden {H.shape} in {time.time() - t1:.0f}s ({(time.time() - t1) / len(H):.3f}s/passage) -> projected 4000 rows {(time.time() - t1) / len(H) * 4000 / 60:.1f} min",
    flush=True,
)
hsc = StandardScaler().fit(H)
hlr = LogisticRegression(C=C_HIDDEN, max_iter=2000, class_weight="balanced").fit(
    hsc.transform(H), y
)
print(f"hidden probe fitted {time.time() - T0:.1f}s", flush=True)


def centre(Z, groups, tau=TAU):
    Z = Z.copy()
    for c in np.unique(groups):
        m = groups == c
        n = m.sum()
        Z[m] -= Z[m].mean(0) * (n / (n + tau))
    return Z


hlr_c = LogisticRegression(C=C_HIDDEN, max_iter=2000, class_weight="balanced").fit(
    centre(hsc.transform(H), train.prompt_id.to_numpy()), y
)
print(f"centred hidden probe fitted {time.time() - T0:.1f}s", flush=True)


def cluster_texts(texts):
    if len(texts) < 2:
        return np.zeros(len(texts), int)
    X = TfidfVectorizer(ngram_range=(1, 2), sublinear_tf=True).fit_transform(
        [normalize(t) for t in texts]
    )
    D = np.clip(1.0 - (X @ X.T).toarray().astype(np.float32), 0.0, 2.0)
    np.fill_diagonal(D, 0.0)
    return AgglomerativeClustering(
        n_clusters=None, distance_threshold=CLUSTER_THR, metric="precomputed", linkage="average"
    ).fit_predict(D)


groups = cluster_texts(test.text.tolist())
_, inv, cnt = np.unique(groups, return_inverse=True, return_counts=True)
size = cnt[inv]
print(
    f"test clustered: {len(cnt)} clusters, sizes min/median/max {cnt.min()}/{int(np.median(cnt))}/{cnt.max()}, "
    f"{(size < MIN_CLUSTER).sum()} rows gated to the uncentred probe; {time.time() - T0:.1f}s",
    flush=True,
)

# %% --- predict (order preserved by construction) ---
Ht = (
    hsc.transform(hidden_matrix(test.text.tolist(), log_every=25))
    if len(test)
    else np.zeros((0, H.shape[1]), np.float32)
)
print(f"test hidden {Ht.shape} {time.time() - T0:.0f}s", flush=True)
p3 = hlr.predict_proba(Ht) if len(test) else np.zeros((0, len(LABELS)))
use_c = size >= MIN_CLUSTER
if use_c.any():
    p3c = hlr_c.predict_proba(centre(Ht, groups))
preds = []
for s in range(0, len(test), 512):
    chunk = test.text.iloc[s : s + 512].tolist()
    sl = slice(s, s + 512)
    print(f"predict {s}/{len(test)} {time.time() - T0:.0f}s", flush=True)
    p1 = softmax(svc.decision_function(tfidf_feats(cv, wv, chunk)) / T)
    p2 = lr.predict_proba(sc.transform(style_matrix(chunk)))
    z = W_TFIDF * np.log(p1 + 1e-9) + W_STYLE * np.log(p2 + 1e-9) + W_HIDDEN * np.log(p3[sl] + 1e-9)
    if use_c[sl].any():
        zc = (
            W_TFIDF * np.log(p1 + 1e-9)
            + W_STYLE * np.log(p2 + 1e-9)
            + W_HIDDEN_C * np.log(p3c[sl] + 1e-9)
        )
        z[use_c[sl]] = zc[use_c[sl]]
    preds.append(z.argmax(1))
pred = np.concatenate(preds) if preds else np.array([], dtype=int)
sub = pd.DataFrame({"id": test["id"].to_numpy(), "model": [LABELS[i] for i in pred]})
os.makedirs(os.path.dirname(OUT), exist_ok=True)
sub.to_parquet(OUT, index=False)

# %% --- verification ---
chk = pd.read_parquet(OUT)
assert list(chk.columns) == ["id", "model"], chk.columns
assert (
    len(chk) == len(test)
    and chk["id"].is_unique
    and (chk["id"].to_numpy() == test["id"].to_numpy()).all()
)
assert chk["model"].notna().all() and chk["model"].isin(LABELS).all()
print(
    f"wrote {OUT}: {len(chk)} rows, {chk.model.nunique()} labels used, total {time.time() - T0:.1f}s"
)
print(chk.model.value_counts().head(18))
