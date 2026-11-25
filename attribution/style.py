"""Small fixed stylometric feature set computed on RAW text (typography is signal here)."""

import re

import numpy as np

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


def style_features(text: str) -> np.ndarray:
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
    fw = [_frac(low.count(w), nw) for w in FUNCTION_WORDS]
    return np.array(f + fw, dtype=np.float32)


def style_matrix(texts):
    return np.vstack([style_features(t) for t in texts])
