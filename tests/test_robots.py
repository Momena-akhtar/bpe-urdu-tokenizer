from urdu_bpe.collect.robots import RobotsTxt

UA = "UrduBPE-CourseCrawler/1.0"


def test_wildcards_and_longest_match():
    r = RobotsTxt("User-agent: *\nDisallow: /search\nDisallow: /*?s=\nAllow: /search/public\n")
    assert not r.can_fetch(UA, "/search/foo")
    assert r.can_fetch(UA, "/search/public/x")  # longer Allow beats shorter Disallow
    assert not r.can_fetch(UA, "/a/b?s=term")
    assert r.can_fetch(UA, "/a/b?page=2")


def test_dollar_anchor_and_empty_disallow():
    assert not RobotsTxt("User-agent: *\nDisallow: /*.pdf$").can_fetch(UA, "/x/y.pdf")
    assert RobotsTxt("User-agent: *\nDisallow: /*.pdf$").can_fetch(UA, "/x/y.pdf?v=1")
    assert RobotsTxt("User-agent: *\nDisallow:\n").can_fetch(UA, "/anything")


def test_specific_agent_group_wins_and_crawl_delay():
    text = "User-agent: *\nDisallow: /\n\nUser-agent: UrduBPE-CourseCrawler\nCrawl-delay: 7\nAllow: /\n"
    r = RobotsTxt(text)
    assert r.can_fetch(UA, "/page")
    assert r.crawl_delay(UA) == 7
    assert not r.can_fetch("OtherBot", "/page")


def test_status_semantics_and_sitemaps():
    assert RobotsTxt.from_status(404).can_fetch(UA, "/x")  # 4xx: no restrictions
    assert not RobotsTxt.from_status(503).can_fetch(UA, "/x")  # 5xx: assume disallowed
    r = RobotsTxt("Sitemap: https://e.com/s.xml\\\nUser-agent: *\nAllow: /")
    assert r.sitemaps == ["https://e.com/s.xml"]
