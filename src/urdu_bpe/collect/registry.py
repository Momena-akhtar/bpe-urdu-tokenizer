"""Every source the corpus is built from, with its licence/terms and default byte target.

Targets are megabytes of *extracted* text before cleaning (0 = take everything the source has).
The self-collected sources add up to well over the 50 MB the assignment requires; the published
corpora (Wikipedia dumps, CC-100) supply the bulk.
"""

from __future__ import annotations

from collections.abc import Callable
from pathlib import Path

from urdu_bpe.collect.base import Collector
from urdu_bpe.collect.record import SourceSpec
from urdu_bpe.collect.sources.archive_org import ArchiveOrgBooks
from urdu_bpe.collect.sources.cc100 import CC100
from urdu_bpe.collect.sources.express import ExpressScraper
from urdu_bpe.collect.sources.wikimedia import WikimediaDump
from urdu_bpe.collect.sources.wordpress import Section, WordPressCollector

CC_BY_SA = "CC BY-SA 4.0 (and GFDL)"
CC_BY_SA_URL = "https://creativecommons.org/licenses/by-sa/4.0/"
PUBLISHER_TERMS = (
    "Publisher copyright, all rights reserved. robots.txt allows automated access to the paths "
    "used; no clause on automated access was found in the site's footer legal/privacy pages "
    "(checked 2026-10-10). Text is kept private for coursework and is not redistributed."
)


def _spec(
    name: str,
    title: str,
    genre: str,
    method: str,
    homepage: str,
    licence: str,
    licence_url: str,
    terms: str,
    self_collected: bool,
    mb: float,
    desc: str = "",
) -> SourceSpec:
    return SourceSpec(
        name, title, genre, method, homepage, licence, licence_url, terms, self_collected, mb, desc
    )


def _news(
    name: str, title: str, home: str, mb: float, desc: str = "", method: str = "api"
) -> SourceSpec:
    return _spec(
        name,
        title,
        "news",
        method,
        home,
        "All rights reserved (publisher)",
        home + "robots.txt",
        PUBLISHER_TERMS,
        True,
        mb,
        desc,
    )


Factory = Callable[..., Collector]


def _wp(spec: SourceSpec, base_url: str, *sections: Section) -> Factory:
    return lambda root, **kw: WordPressCollector(
        spec, root, base_url=base_url, sections=sections, **kw
    )


def _wiki(spec: SourceSpec, wiki: str, min_chars: int = 300) -> Factory:
    return lambda root, **kw: WikimediaDump(spec, root, wiki=wiki, min_chars=min_chars, **kw)


WIKI_TERMS = "Wikimedia dumps are free to reuse under CC BY-SA with attribution (https://dumps.wikimedia.org/legal.html)."

ARY = _news("ary_urdu", "ARY News Urdu", "https://urdu.arynews.tv/", 50)
AUSAF = _news("daily_ausaf", "Daily Ausaf", "https://dailyausaf.com/", 50)
MASHRIQ = _news("mashriq_tv", "Mashriq TV / Daily Mashriq", "https://mashriqtv.pk/", 45)
HUM = _news("hum_news", "Hum News Urdu", "https://humnews.pk/", 40)
UMMAT = _news("ummat", "Daily Ummat", "https://ummat.net/", 40)
QUDRAT = _news("daily_qudrat", "Daily Qudrat", "https://dailyqudrat.pk/", 40)
AIKROZAN = _spec(
    "aik_rozan",
    "Aik Rozan (Urdu blog)",
    "blog",
    "api",
    "https://www.aikrozan.com/",
    "All rights reserved (authors)",
    "https://www.aikrozan.com/robots.txt",
    PUBLISHER_TERMS.replace("footer legal/privacy pages", "site pages (none found)"),
    True,
    40,
    "Urdu blogging site: essays, politics, history, literature",
)

FACTORIES: dict[str, tuple[SourceSpec, Factory]] = {}


def _register(spec: SourceSpec, factory: Factory) -> None:
    FACTORIES[spec.name] = (spec, factory)


_register(ARY, _wp(ARY, "https://urdu.arynews.tv", Section("all", "news")))
_register(HUM, _wp(HUM, "https://humnews.pk", Section("all", "news")))
_register(UMMAT, _wp(UMMAT, "https://ummat.net", Section("all", "news")))
_register(QUDRAT, _wp(QUDRAT, "https://dailyqudrat.pk", Section("all", "news")))
_register(
    AUSAF,
    _wp(
        AUSAF,
        "https://dailyausaf.com",
        Section("columns", "opinion", categories=("columns",), max_mb=20),
        Section("news", "news", exclude=("columns",)),
    ),
)
_register(
    MASHRIQ,
    _wp(
        MASHRIQ,
        "https://mashriqtv.pk",
        Section("blog", "blog", categories=("urdu-blog",)),
        Section("editorial", "opinion", categories=("idariya",), max_mb=8),
        Section("news", "news", exclude=("urdu-blog", "idariya")),
    ),
)
_register(AIKROZAN, _wp(AIKROZAN, "https://www.aikrozan.com", Section("all", "blog")))

EXPRESS = _news(
    "express_urdu",
    "Express News Urdu",
    "https://www.express.pk/",
    15,
    "HTML scraper over the site's sitemaps (the only source here without an open API)",
    method="scrape",
)
_register(EXPRESS, lambda root, **kw: ExpressScraper(EXPRESS, root, **kw))

ARCHIVE = _spec(
    "archive_org_books",
    "Internet Archive: public-domain Urdu books",
    "books",
    "api",
    "https://archive.org/",
    "Public domain (per item metadata)",
    "https://archive.org/about/terms.php",
    "Only items whose metadata declares a public-domain licence or NOT_IN_COPYRIGHT. OCR text, "
    "filtered for Urdu-script ratio.",
    True,
    40,
)
_register(ARCHIVE, lambda root, **kw: ArchiveOrgBooks(ARCHIVE, root, **kw))

URWIKI = _spec(
    "urdu_wikipedia",
    "Urdu Wikipedia (dump)",
    "encyclopedia",
    "dump",
    "https://ur.wikipedia.org/",
    CC_BY_SA,
    CC_BY_SA_URL,
    WIKI_TERMS,
    False,
    0,
    "pages-articles XML dump, wiki markup stripped by urdu_bpe.collect.wikitext",
)
_register(URWIKI, _wiki(URWIKI, "urwiki"))
URWS = _spec(
    "urdu_wikisource",
    "Urdu Wikisource (dump)",
    "books",
    "dump",
    "https://ur.wikisource.org/",
    CC_BY_SA,
    CC_BY_SA_URL,
    WIKI_TERMS + " Hosted texts are public domain or freely licensed.",
    False,
    0,
    "Digitised literature: poetry, prose, classics",
)
_register(URWS, _wiki(URWS, "urwikisource", min_chars=100))
URWB = _spec(
    "urdu_wikibooks",
    "Urdu Wikibooks (dump)",
    "books",
    "dump",
    "https://ur.wikibooks.org/",
    CC_BY_SA,
    CC_BY_SA_URL,
    WIKI_TERMS,
    False,
    0,
    "Open textbooks",
)
_register(URWB, _wiki(URWB, "urwikibooks"))

CC100_SPEC = _spec(
    "cc100_urdu",
    "CC-100 Urdu (Common Crawl)",
    "web",
    "published-download",
    "https://data.statmt.org/cc-100/",
    "Common Crawl Terms of Use; no claim on page content",
    "https://commoncrawl.org/terms-of-use",
    "Corpus authors claim no rights over the underlying pages; used for research only and "
    "not redistributed. Noisy: needs strong cleaning and deduplication.",
    False,
    450,
    "Web crawl, CC-Net processed (Conneau et al. 2020, Wenzek et al. 2020); first slice of "
    "the file is used, see data card for the sampling caveat",
)
_register(CC100_SPEC, lambda root, **kw: CC100(CC100_SPEC, root, **kw))

SELF_COLLECTED = [n for n, (s, _) in FACTORIES.items() if s.self_collected]
PUBLISHED = [n for n, (s, _) in FACTORIES.items() if not s.self_collected]
# Considered and rejected (kept for the data card):
REJECTED = {
    "bbc_urdu": "robots.txt/terms forbid scraping, dataset creation and LLM training use",
    "dw_urdu": "robots.txt states automated data mining is prohibited without permission",
    "urdupoint, roznama92, jasarat, urdu.dawn.com, hamariweb": "Cloudflare challenge: automated access blocked",
    "independent_urdu": "Crawl-delay 10 s makes the 50 MB target impractical; skipped rather than ignored",
    "humsub": "Crawl-delay 20 s, same reason",
    "ufal_urdu_monolingual": "shuffled, pre-tokenised sentences with no document boundaries: cannot be split by document",
    "makhzan": "small (6.26M words) and its text may not be republished (assignment table)",
    "urdu_news_dataset (kaggle)": "hosted behind a Kaggle login, not fetchable by a polite crawler",
}


def build(name: str, root: Path, **kwargs: object) -> Collector:
    _, factory = FACTORIES[name]
    return factory(root, **kwargs)
