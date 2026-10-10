from urdu_bpe.collect.textutil import html_to_text, tidy_lines, urdu_ratio


def test_urdu_ratio():
    assert urdu_ratio("یہ اردو ہے") == 1.0
    assert urdu_ratio("hello world") == 0.0
    assert 0.4 < urdu_ratio("یہ اردو ہے hello world") < 0.7
    assert urdu_ratio("123 ۔") == 0.0  # no letters


def test_html_to_text_paragraphs_and_junk():
    html = (
        "<div><p>پہلا <b>پیرا</b></p><script>x()</script><p>دوسرا<br>تیسرا</p>"
        "<figure>تصویر</figure><ul><li>ایک</li><li>دو</li></ul></div>"
    )
    assert html_to_text(html).split("\n") == ["پہلا پیرا", "دوسرا", "تیسرا", "ایک", "دو"]


def test_inline_tags_do_not_split_words():
    assert html_to_text("<p>ان<span>سان</span> <a href='#'>لنک</a></p>") == "انسان لنک"


def test_tidy_lines_keeps_content_untouched():
    assert tidy_lines("  الف \n\n \n ب\u200c ج ") == "الف\nب\u200c ج"  # ZWNJ is not stripped here


def test_ocr_noise_is_rejected_but_prose_and_poetry_pass():
    from urdu_bpe.collect.textutil import looks_like_clean_urdu

    prose = "پاکستان میں یہ پہلی بار ہے کہ ایک اسکول کا قیام کیا گیا اور اس کی تیاری جاری ہے۔ " * 5
    poetry = (
        "ہزاروں خواہشیں ایسی کہ ہر خواہش پہ دم نکلے\nبہت نکلے مرے ارمان لیکن پھر بھی کم نکلے\n" * 3
    )
    noise = "ھمر ) سے )< ھٹگ ٭ گا ک رر ٢5ں 48 11 ۲ 8 ۳ 5اک ۱ کے ےد ۱ ا ر0 ۰ 5 <۰ " * 5
    assert looks_like_clean_urdu(prose) and looks_like_clean_urdu(poetry)
    assert not looks_like_clean_urdu(noise)
