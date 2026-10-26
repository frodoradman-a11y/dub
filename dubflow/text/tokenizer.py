"""Character-level tokenizer for Persian, English and pinyin (Chinese) text.

Design notes
- Persian is written without short vowels; the external translator is expected
  to add minimal diacritics (ezafe kasre U+0650 and ambiguous short vowels).
  These diacritics are first-class tokens here.
- Chinese text must be given as tone-numbered pinyin (e.g. "ni3 hao3"); use
  `pypinyin` in preprocessing. Pinyin reuses the Latin letters + digits.
- Inline tags such as <emph>...</emph> or <laugh> are single tokens.
"""

from __future__ import annotations

import re
import unicodedata

LANGS = {"en": 0, "fa": 1, "zh": 2, "other": 3}

SPECIALS = ["<pad>", "<unk>", "<bos>", "<eos>", " "]
TAGS = [
    "<emph>", "</emph>", "<whisper>", "</whisper>", "<shout>", "</shout>",
    "<laugh>", "<breath>", "<sigh>", "<cry>", "<cough>", "<gasp>", "<scream>",
    "<pause>", "<hum>",
]
LATIN = list("abcdefghijklmnopqrstuvwxyz")
DIGITS = list("0123456789")
PUNCT = list(".,!?;:'\"-()…«»؟،؛")
PERSIAN_LETTERS = list("آابپتثجچحخدذرزژسشصضطظعغفقکگلمنوهیءأؤئإةۀ")
PERSIAN_DIACRITICS = ["ً", "ٌ", "ٍ", "َ", "ُ", "ِ", "ّ", "ْ", "ٰ"]
ZWNJ = ["‌"]

_ARABIC_TO_PERSIAN = {
    "ي": "ی", "ى": "ی", "ك": "ک", "ە": "ه", "ہ": "ه", "ھ": "ه",
    "٠": "0", "١": "1", "٢": "2", "٣": "3", "٤": "4", "٥": "5", "٦": "6", "٧": "7", "٨": "8", "٩": "9",
    "۰": "0", "۱": "1", "۲": "2", "۳": "3", "۴": "4", "۵": "5", "۶": "6", "۷": "7", "۸": "8", "۹": "9",
    "ـ": "",  # tatweel
    "‍": "",  # ZWJ
}
_TAG_RE = re.compile(r"(</?[a-z]+>)")


def normalize_text(text: str, lang: str) -> str:
    """Unicode / orthographic normalization (no number expansion: the
    translator should spell out numbers in the target language)."""
    text = unicodedata.normalize("NFC", text)
    text = "".join(_ARABIC_TO_PERSIAN.get(c, c) for c in text)
    if lang in ("en", "zh", "other"):
        text = text.lower()
    text = re.sub(r"[ \t\r\n]+", " ", text).strip()
    return text


class TextTokenizer:
    def __init__(self) -> None:
        vocab = SPECIALS + TAGS + LATIN + DIGITS + PUNCT + PERSIAN_LETTERS + PERSIAN_DIACRITICS + ZWNJ
        seen: dict[str, int] = {}
        for tok in vocab:
            if tok not in seen:
                seen[tok] = len(seen)
        self.stoi = seen
        self.itos = {i: s for s, i in seen.items()}
        self.pad_id = self.stoi["<pad>"]
        self.unk_id = self.stoi["<unk>"]
        self.space_id = self.stoi[" "]

    @property
    def vocab_size(self) -> int:
        return len(self.stoi)

    def encode(self, text: str, lang: str) -> list[int]:
        text = normalize_text(text, lang)
        ids: list[int] = []
        for piece in _TAG_RE.split(text):
            if not piece:
                continue
            if piece in self.stoi and piece.startswith("<"):
                ids.append(self.stoi[piece])
                continue
            for ch in piece:
                ids.append(self.stoi.get(ch, self.unk_id))
        return ids

    def decode(self, ids: list[int]) -> str:
        return "".join(self.itos.get(i, "?") for i in ids if i != self.pad_id)
