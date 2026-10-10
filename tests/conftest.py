import random

import pytest

URDU_WORDS = [
    "پاکستان",
    "کراچی",
    "لاہور",
    "اسلام",
    "آباد",
    "حکومت",
    "وزیر",
    "اعظم",
    "نے",
    "آج",
    "کہا",
    "کہ",
    "ملک",
    "میں",
    "معاشی",
    "صورتحال",
    "بہتر",
    "ہو",
    "رہی",
    "ہے",
    "اور",
    "عوام",
    "کو",
    "ریلیف",
    "دیا",
    "جائے",
    "گا",
    "کتاب",
    "اردو",
    "زبان",
    "ادب",
    "شاعری",
    "غزل",
    "نظم",
    "افسانہ",
    "ناول",
    "کھیل",
    "کرکٹ",
    "ٹیم",
    "میچ",
    "جیت",
    "فتح",
    "شکست",
    "کپتان",
    "بلے",
    "باز",
    "گیند",
    "باز",
    "اسٹیڈیم",
    "تماشائی",
    "سردی",
    "گرمی",
    "بارش",
    "موسم",
    "سیلاب",
    "زلزلہ",
    "تعلیم",
    "صحت",
    "روزگار",
    "مہنگائی",
    "بجلی",
    "گیس",
    "پانی",
    "سڑک",
    "پل",
    "اسکول",
    "ہسپتال",
    "یونیورسٹی",
    "طالب",
    "علم",
    "استاد",
    "والدین",
    "بچے",
    "خواتین",
    "مرد",
    "بزرگ",
]
ENGLISH_WORDS = ["iPhone", "Google", "Facebook", "cricket", "news", "update", "ok", "team"]


def make_text(seed: int, n_docs: int, words_per_doc: int = 120, eot: str = "<|endoftext|>") -> str:
    """Synthetic, reproducible Urdu-looking corpus: Zipf-ish words, numbers, punctuation, English."""
    rng = random.Random(seed)
    vocab = URDU_WORDS + [w + "ی" for w in URDU_WORDS[:30]] + ENGLISH_WORDS
    weights = [1 / (i + 1) for i in range(len(vocab))]
    docs = []
    for _ in range(n_docs):
        lines, line = [], []
        for _ in range(words_per_doc):
            token = rng.choices(vocab, weights)[0]
            if rng.random() < 0.05:
                token = str(rng.randrange(1990, 2030))
            line.append(token)
            if rng.random() < 0.1:
                lines.append(" ".join(line) + rng.choice(["۔", "،", "؟", "!"]))
                line = []
        if line:
            lines.append(" ".join(line) + "۔")
        docs.append("\n".join(lines) + eot)
    return "".join(docs)


@pytest.fixture
def corpus_text() -> str:
    return make_text(seed=1, n_docs=40)
