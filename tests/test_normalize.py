import random

from urdu_bpe.normalize import RULES, normalize, rule_hits


def test_letter_variants_map_to_urdu_forms():
    assert normalize("كتاب") == "کتاب"  # Arabic kaf
    assert normalize("پاكستاني") == "پاکستانی"  # Arabic kaf + Arabic yeh
    assert normalize("على") == "علی"  # alef maksura
    assert normalize("هم") == "ہم"  # Arabic heh -> gol heh
    assert normalize("ھ") == "ھ"  # do-chashmi heh stays distinct
    assert normalize("ہ") == "ہ"
    assert normalize("رحمة") == "رحمۃ"  # teh marbuta


def test_tatweel_zwnj_and_invisibles_removed():
    assert normalize("ســـلام") == "سلام"
    assert normalize("ہو‌گئی") == "ہوگئی"
    assert normalize("﻿اردو‎‏​‍") == "اردو"


def test_digits_become_ascii():
    assert normalize("٢٠٢٤ ۲۰۲۴ 2024") == "2024 2024 2024"
    assert normalize("٣٫١٤") == "3.14"


def test_whitespace_and_line_breaks():
    assert normalize("a  b  c\r\n\r\n\nd \t e ") == "a b c\nd e"
    assert normalize("  \n\nالف \n ب\n") == "الف\nب"
    assert normalize("") == ""


def test_presentation_forms_and_composition():
    assert normalize("ﷲ") == "اللہ"
    assert normalize("ﺍﻣ") == "ام"  # isolated alef, isolated meem
    assert normalize("آ") == "آ"  # alef + maddah composes
    assert normalize("ＡＢＣ１") == "ABC1"  # fullwidth ASCII
    assert normalize("اـٓ") == "آ"  # deleting the tatweel exposes a composable pair


def test_diacritics_kept_quranic_marks_dropped_duplicates_collapsed():
    assert normalize("اُردُو") == "اُردُو"
    assert normalize("کتابَّ") == "کتابَّ"
    assert normalize("بََ") == "بَ"
    assert normalize("بِسۡمِ") == "بِسمِ"  # U+06E1 small high dotless head of khah removed


def test_idempotent_on_adversarial_strings():
    alphabet = list("ايكهةىـ‌‍‎ھہًَٓا آ\n\t\r٠١٢۰۱۲abc 😀ﷲﺍۖ.،")
    rng = random.Random(0)
    for _ in range(500):
        text = "".join(rng.choice(alphabet) for _ in range(rng.randrange(0, 40)))
        once = normalize(text)
        assert normalize(once) == once


def test_rule_hits_counts_rewritten_characters():
    hits = rule_hits("كتابـ ٢٠ ‌ﷲ")
    assert hits["arabic_kaf"] == 1 and hits["tatweel"] == 1 and hits["arabic_indic_digits"] == 2
    assert hits["zwnj"] == 1 and hits["presentation_forms"] == 1
    assert set(hits) <= set(RULES) | {"presentation_forms"}
