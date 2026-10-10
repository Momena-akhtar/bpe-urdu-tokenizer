"""A small robots.txt parser (RFC 9309 semantics) with ``*``/``$`` wildcards and Crawl-delay.

``urllib.robotparser`` ignores wildcards, which several Urdu news sites rely on, so we parse the
file ourselves: longest matching rule wins, ``Allow`` beats ``Disallow`` on a tie.
"""

from __future__ import annotations

import contextlib
import re
from dataclasses import dataclass, field


@dataclass
class _Group:
    agents: list[str] = field(default_factory=list)
    rules: list[tuple[bool, str]] = field(default_factory=list)  # (allow, path pattern)
    crawl_delay: float | None = None


def _pattern(path: str) -> re.Pattern[str]:
    anchored = path.endswith("$")
    body = re.escape(path[:-1] if anchored else path).replace(r"\*", ".*")
    return re.compile(body + ("$" if anchored else ""))


class RobotsTxt:
    def __init__(self, text: str = "", *, allow_all: bool = False, disallow_all: bool = False):
        self.allow_all = allow_all
        self.disallow_all = disallow_all
        self.sitemaps: list[str] = []
        self.groups: list[_Group] = []
        self._parse(text)

    @classmethod
    def from_status(cls, status: int | None, text: str = "") -> RobotsTxt:
        """RFC 9309: 2xx parse; 4xx = no restrictions; 5xx/unreachable = assume disallowed."""
        if status is not None and 200 <= status < 300:
            return cls(text)
        if status is not None and 400 <= status < 500:
            return cls(allow_all=True)
        return cls(disallow_all=True)

    def _parse(self, text: str) -> None:
        group: _Group | None = None
        collecting_agents = False
        for raw in text.splitlines():
            line = raw.split("#", 1)[0].strip()
            if ":" not in line:
                continue
            key, value = (p.strip() for p in line.split(":", 1))
            key = key.lower()
            if key == "sitemap":
                self.sitemaps.append(value.rstrip("\\"))
            elif key == "user-agent":
                if not collecting_agents:
                    group = _Group()
                    self.groups.append(group)
                    collecting_agents = True
                assert group is not None
                group.agents.append(value.lower())
            elif key in ("allow", "disallow") and group is not None:
                collecting_agents = False
                if value:  # an empty Disallow means "allow everything"
                    group.rules.append((key == "allow", value))
            elif key == "crawl-delay" and group is not None:
                collecting_agents = False
                with contextlib.suppress(ValueError):
                    group.crawl_delay = float(value)
            else:
                collecting_agents = False

    def _group_for(self, agent: str) -> _Group | None:
        agent = agent.lower()
        best: tuple[int, _Group] | None = None
        for g in self.groups:
            for a in g.agents:
                if a != "*" and a in agent and (best is None or len(a) > best[0]):
                    best = (len(a), g)
        if best:
            return best[1]
        return next((g for g in self.groups if "*" in g.agents), None)

    def can_fetch(self, agent: str, path_and_query: str) -> bool:
        if self.disallow_all:
            return False
        if self.allow_all:
            return True
        group = self._group_for(agent)
        if group is None:
            return True
        verdict, best_len = True, -1
        for allow, pattern in group.rules:
            if _pattern(pattern).match(path_and_query):
                n = len(pattern)
                if n > best_len or (n == best_len and allow):
                    verdict, best_len = allow, n
        return verdict

    def crawl_delay(self, agent: str) -> float | None:
        group = self._group_for(agent)
        return group.crawl_delay if group else None
