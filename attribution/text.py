import re
import unicodedata

_WS = re.compile(r"\s+")


def normalize(text: str) -> str:
    """Normalized view used for word/char n-grams. Raw text is kept separately for style features."""
    t = unicodedata.normalize("NFKC", text)
    t = (
        t.replace("\u2019", "'")
        .replace("\u2018", "'")
        .replace("\u201c", '"')
        .replace("\u201d", '"')
    )
    t = t.replace("\u2014", " - ").replace("\u2013", " - ")
    return _WS.sub(" ", t).strip()
