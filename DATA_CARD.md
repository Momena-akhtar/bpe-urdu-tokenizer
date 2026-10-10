# Data card: Urdu BPE corpus (2026-10-10)

Cleaned Urdu text for training a byte-level BPE tokenizer and, in the next assignment, a GPT-style model. Built by `usage/01_collect.py` then `usage/02_clean.py` (fixed split seed 2026). Documents are separated by `<|endoftext|>`; the split is by document, stratified by source, 98/1/1 by bytes.

## Sources

| source | genre | URL | licence | collected | raw MB | cleaned MB |
|---|---|---|---|---|---|---|
| ARY News Urdu ★ | news | https://urdu.arynews.tv/ | All rights reserved (publisher) | 2026-10-10 | 50.1 | 47.7 |
| Hum News Urdu ★ | news | https://humnews.pk/ | All rights reserved (publisher) | 2026-10-10 | 40.1 | 38.0 |
| Daily Ummat ★ | news | https://ummat.net/ | All rights reserved (publisher) | 2026-10-10 | 40.1 | 39.2 |
| Daily Qudrat ★ | news | https://dailyqudrat.pk/ | All rights reserved (publisher) | 2026-10-10 | 40.1 | 35.5 |
| Daily Ausaf ★ | news | https://dailyausaf.com/ | All rights reserved (publisher) | 2026-10-10 | 50.0 | 48.9 |
| Mashriq TV / Daily Mashriq ★ | news | https://mashriqtv.pk/ | All rights reserved (publisher) | 2026-10-10 | 45.1 | 44.9 |
| Aik Rozan (Urdu blog) ★ | blog | https://www.aikrozan.com/ | All rights reserved (authors) | 2026-10-10 | 23.8 | 23.7 |
| Express News Urdu ★ | news | https://www.express.pk/ | All rights reserved (publisher) | 2026-10-10 | 15.0 | 14.8 |
| Internet Archive: public-domain Urdu books ★ | books | https://archive.org/ | Public domain (per item metadata) | 2026-10-10 | 40.6 | excluded (OCR noise) |
| Urdu Wikipedia (dump) | encyclopedia | https://ur.wikipedia.org/ | CC BY-SA 4.0 (and GFDL) | 2026-10-10 | 475.8 | 454.0 |
| Urdu Wikisource (dump) | books | https://ur.wikisource.org/ | CC BY-SA 4.0 (and GFDL) | 2026-10-10 | 44.7 | 44.2 |
| Urdu Wikibooks (dump) | books | https://ur.wikibooks.org/ | CC BY-SA 4.0 (and GFDL) | 2026-10-10 | 5.7 | 5.0 |
| CC-100 Urdu (Common Crawl) | web | https://data.statmt.org/cc-100/ | Common Crawl Terms of Use; no claim on page content | 2026-10-10 | 450.0 | 419.1 |

★ = self-collected by scraping or public APIs (robots.txt obeyed; request logs and robots.txt snapshots in `evidence/` of each source): **344.9 MB** of extracted text.

## Corpus totals

| split | MB | documents | words |
|---|---|---|---|
| train | 1,196.0 (97.99%) | 420,656 | 142,161,435 |
| val | 12.3 (1.01%) | 4,450 | 1,455,837 |
| test | 12.3 (1.00%) | 4,415 | 1,461,558 |
| **total** | **1,220.5** | **429,521** | **145,078,830** |

Unique words: 1,169,649 overall, 1,155,525 in train. 0.56% of test word tokens never occur in train. A word is a maximal run of letters, combining marks or digits.

## Cleaning steps (in order; MB removed from the extracted text)

| step | removed MB | docs / lines |
|---|---|---|
| extract (collectors) | 0.00 |  |
| normalise (normalize.py) | 0.82 |  |
| line filter: mostly_not_urdu | 1.14 | 13,285 lines |
| line filter: no_urdu_letters | 2.08 | 58,435 lines |
| line filter: replacement_character | 0.90 | 1,676 lines |
| line filter: too_few_urdu_letters | 0.29 | 39,376 lines |
| line filter: url_or_email_removed | 0.50 | 8,047 lines |
| dedup/filter: duplicate_doc | 0.99 | 349 docs |
| dedup/filter: boilerplate_line | 34.45 | 590,322 lines |
| dedup/filter: duplicate_paragraph | 18.85 | 46,405 lines |
| dedup/filter: repetitive_doc | 0.10 | 27 docs |
| dedup/filter: too_short_doc | 4.76 | 17,755 docs |

## Known gaps and biases

- Genre mix of train by bytes: encyclopedia 37.2%, web 34.3%, news 19.6%, books 4.0%, opinion 2.3%, blog 2.1%.
- Largest single source: urdu_wikipedia at 37.2% of train.
- Time: 5 of the 8 dated scraped sources (ary_urdu, daily_ausaf, daily_qudrat, hum_news, ummat) only reach back 2025-07 to 2026-10 because the sites were walked newest-first, so the news reflects one recent year of events. Sources spanning years: aik_rozan 2017-07 to 2026-09 (median 2018-08); express_urdu 2013-08 to 2026-02 (median 2021-01); mashriq_tv 2021-07 to 2026-10 (median 2026-02).
- Register: written, formal Urdu (news, essays, encyclopedia). Conversational text, social media, SMS-style writing and Roman Urdu are almost absent.
- Place and topics: the scraped outlets are Pakistani (Karachi, Islamabad, Peshawar, Quetta), so politics, crime, cricket, prices and the Iran-Israel conflict are over-represented; Indian Urdu, older literary Urdu and dialects are under-represented.
- CC-100 is only the first slice of its file, so it reflects whichever crawl shards come first, and it contains pages that also appear in the scraped news (removed only when identical).
- Deduplication is exact (documents, lines, long paragraphs). Near-duplicates, such as the same wire story re-worded by two outlets, remain, so val/test are not guaranteed leak-free against train.
- Normalisation choices alter the text: ZWNJ is deleted, digits become ASCII, Quranic annotation marks are dropped, so the tokenizer never sees those variants.
- Books come from Wikisource and Wikibooks only: the Internet Archive OCR books (40 MB collected) were excluded because even after line-level OCR filtering the surviving text stayed noisy, so long-form literary prose and Urdu poetry are thin.
- Scraped publisher text is copyrighted; it is used for coursework only and is not redistributed.

## Considered and rejected

**bbc_urdu**: robots.txt/terms forbid scraping, dataset creation and LLM training use; **dw_urdu**: robots.txt states automated data mining is prohibited without permission; **urdupoint, roznama92, jasarat, urdu.dawn.com, hamariweb**: Cloudflare challenge: automated access blocked; **independent_urdu**: Crawl-delay 10 s makes the 50 MB target impractical; skipped rather than ignored; **humsub**: Crawl-delay 20 s, same reason; **ufal_urdu_monolingual**: shuffled, pre-tokenised sentences with no document boundaries: cannot be split by document; **makhzan**: small (6.26M words) and its text may not be republished (assignment table); **urdu_news_dataset (kaggle)**: hosted behind a Kaggle login, not fetchable by a polite crawler.
