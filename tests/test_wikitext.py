from urdu_bpe.collect.wikitext import strip_wikitext


def test_prose_links_and_markup():
    src = "'''اردو''' ایک [[زبان]] ہے، جو [[پاکستان|ملک]] میں بولی جاتی ہے۔<ref>حوالہ</ref>"
    assert strip_wikitext(src) == "اردو ایک زبان ہے، جو ملک میں بولی جاتی ہے۔"


def test_nested_templates_tables_files_and_categories():
    src = (
        "{{معلومات|نام={{ٹیمپلیٹ|x}}}}\n"
        "{| class=wikitable\n|-\n| a || b\n|}\n"
        "[[فائل:Pic.jpg|thumb|کیپشن [[لنک]]]]\n"
        "متن باقی رہے گا۔\n[[زمرہ:مثال]]\n[[en:Urdu]]"
    )
    assert strip_wikitext(src) == "متن باقی رہے گا۔"


def test_headings_lists_external_links_and_tail_sections():
    src = (
        "== تاریخ ==\n* پہلی بات\n# دوسری بات\n"
        "[http://example.com مثال] اور [http://x.org]\n"
        "== حوالہ جات ==\n<references/>\nیہ حصہ حذف ہو۔"
    )
    out = strip_wikitext(src)
    assert out == "تاریخ\nپہلی بات\nدوسری بات\nمثال اور"
    assert "حذف" not in out


def test_html_entities_and_comments():
    assert strip_wikitext("الف&nbsp;ب <!-- پوشیدہ --> پ") == "الف\xa0ب  پ"


def test_file_caption_with_stray_brackets_and_arabic_namespace():
    src = "متن [[فائل:A.jpg|تصغیر|250px|[کیپشن]]] باقی [[ملف:B.png|x [[لنک]] y]] آخر"
    assert strip_wikitext(src) == "متن  باقی  آخر"
