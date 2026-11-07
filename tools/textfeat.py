"""Hashed text features of an item (Python standard library only).

Shared by the offline trainer (tools/text_prior.py) and model.py, so training and scoring
compute exactly the same features.
"""
import math
import re
import zlib

DIM = 1 << 18
N_DENSE = 8
_TOK = re.compile(r"[a-z]{2,}|\d+|[^\sa-z\d]")
_OPT = re.compile(r"(?m)^\s*\(?[A-Ja-j][\).:]\s")


def _h(tok):
    return N_DENSE + zlib.crc32(tok.encode("utf-8")) % (DIM - N_DENSE)


def features(content, item_features="", interactors=""):
    """Return {index: value}; hashed unigrams/bigrams (L2-normalised) plus dense stats."""
    content = str(content or "")
    text = (content[:3000] + " " + str(item_features or "")[:500] + " " + str(interactors or "")[:200]).lower()
    toks = ["<n>" if t.isdigit() else t for t in _TOK.findall(text)]
    f = {}
    for t in toks:
        k = _h("u:" + t)
        f[k] = 1.0
    for a, b in zip(toks, toks[1:]):
        k = _h("b:" + a + " " + b)
        f[k] = 1.0
    if f:
        norm = 1.0 / math.sqrt(len(f))
        for k in f:
            f[k] = norm
    n = max(len(content), 1)
    f[0] = math.log1p(len(content)) / 10.0
    f[1] = sum(ch.isdigit() for ch in content[:3000]) / min(n, 3000)
    f[2] = sum(ord(ch) > 127 for ch in content[:3000]) / min(n, 3000)
    f[3] = min(len(_OPT.findall(content[:3000])), 10) / 10.0
    f[4] = 1.0 if "```" in content or "def " in content else 0.0
    f[5] = math.log1p(content.count("\n")) / 5.0
    f[6] = 1.0 if str(interactors or "").strip() else 0.0
    f[7] = 1.0 if str(item_features or "").strip() not in ("", "nan", "{}") else 0.0
    return f
