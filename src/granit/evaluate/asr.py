"""Speech recognition quality: word error rate after normalization (PLAN.md §4.9).

Normalization makes formatting choices invisible: lowercase, no punctuation, common contractions expanded, and number
words turned into digits ("twelve percent" and TurboCTC's "12%" both become "12"). ``jiwer`` (Apache-2.0) is a dev
dependency, imported lazily: WER is computed on the Mac (golden tests, ``granit eval``), never in the app.
"""

from __future__ import annotations

import re

_NUMBER_SEPARATOR = re.compile(r"(?<=\d)[,_](?=\d{3}\b)")
_NOT_WORD = re.compile(r"[^\w\s']|_")

CONTRACTIONS = {
    "let's": "let us",
    "it's": "it is",
    "that's": "that is",
    "what's": "what is",
    "there's": "there is",
    "i'm": "i am",
    "we're": "we are",
    "you're": "you are",
    "they're": "they are",
    "i'll": "i will",
    "we'll": "we will",
    "you'll": "you will",
    "they'll": "they will",
    "i've": "i have",
    "we've": "we have",
    "don't": "do not",
    "doesn't": "does not",
    "didn't": "did not",
    "can't": "cannot",
    "won't": "will not",
    "isn't": "is not",
    "aren't": "are not",
    "wasn't": "was not",
    "shouldn't": "should not",
}

_UNITS: dict[str, int] = {
    w: i
    for i, w in enumerate(
        [
            "zero",
            "one",
            "two",
            "three",
            "four",
            "five",
            "six",
            "seven",
            "eight",
            "nine",
            "ten",
            "eleven",
            "twelve",
            "thirteen",
            "fourteen",
            "fifteen",
            "sixteen",
            "seventeen",
            "eighteen",
            "nineteen",
        ]
    )
}
_TENS: dict[str, int] = {
    w: 10 * i
    for i, w in enumerate(
        ["twenty", "thirty", "forty", "fifty", "sixty", "seventy", "eighty", "ninety"], start=2
    )
}
_SCALES = {"hundred": 100, "thousand": 1_000, "million": 1_000_000}


def _kind(word: str) -> str | None:
    if word in _UNITS:
        return (
            "unit" if 1 <= _UNITS[word] <= 9 else "teen"
        )  # zero and ten–nineteen can't take a unit after them
    if word in _TENS:
        return "tens"
    if word == "hundred":
        return "hundred"
    if word in _SCALES:
        return "scale"
    return None


# Which kind of word may follow which inside one number ("twenty five" yes; "five twenty" is two numbers).
_FOLLOWS: dict[str | None, set[str]] = {
    None: {"unit", "teen", "tens"},
    "unit": {"hundred", "scale"},
    "teen": {"hundred", "scale"},
    "tens": {"unit", "scale"},
    "hundred": {"unit", "teen", "tens", "scale"},
    "scale": {"unit", "teen", "tens"},
}


def words_to_numbers(words: list[str]) -> list[str]:
    """Replace runs of English number words with digits: ``one hundred and five`` → ``105``.

    A new number starts whenever the next word can't extend the current one, so ``twenty twenty six`` → ``20 26``.
    """
    out: list[str] = []
    i = 0
    while i < len(words):
        if _kind(words[i]) not in _FOLLOWS[None]:
            out.append(words[i])
            i += 1
            continue
        total = current = 0
        last: str | None = None
        while i < len(words):
            word, kind = words[i], _kind(words[i])
            if (
                word == "and"
                and last in {"hundred", "scale"}
                and i + 1 < len(words)
                and _kind(words[i + 1]) in _FOLLOWS[last]
            ):  # "one hundred and five"
                i += 1
                continue
            if kind is None or kind not in _FOLLOWS[last]:
                break
            if kind in {"unit", "teen"}:
                current += _UNITS[word]
            elif kind == "tens":
                current += _TENS[word]
            elif kind == "hundred":
                current *= 100
            else:
                total += current * _SCALES[word]
                current = 0
            last = kind
            i += 1
        out.append(str(total + current))
    return out


def normalize(text: str) -> str:
    text = _NUMBER_SEPARATOR.sub("", text.lower().replace("’", "'").replace("%", " percent"))
    text = _NOT_WORD.sub(" ", text)
    words = []
    for word in text.split():
        words.extend(CONTRACTIONS.get(word, word.replace("'", "")).split())
    return " ".join(w for w in words_to_numbers(words) if w)


def wer(reference: str, hypothesis: str) -> float:
    import jiwer

    ref, hyp = normalize(reference), normalize(hypothesis)
    if not ref:
        return 0.0 if not hyp else 1.0
    return float(jiwer.wer(ref, hyp))
