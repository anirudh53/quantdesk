"""Published-narrative sentiment from headlines. Context only: a labelled HYPOTHESIS, never evidence of price cause.
Social media is not used. No headlines = unknown, never neutral."""
from __future__ import annotations
from .data import now_utc

def score_headlines(headlines: list) -> dict:
    """headlines: [{title, link, published}]"""
    if not headlines:
        return {"status": "no_headlines", "n": 0, "retrieved": now_utc()}
    from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
    a = SentimentIntensityAnalyzer()
    scored = [dict(h, compound=a.polarity_scores(h["title"])["compound"]) for h in headlines]
    mean = sum(s["compound"] for s in scored) / len(scored)
    return {"status": "ok", "n": len(scored), "mean_compound": round(mean, 3), "items": scored[:10],
            "retrieved": now_utc(), "label": "hypothesis (press tone, not causation)"}

def fetch_rss(query: str, feeds: list, max_items: int = 20) -> list:
    """feeds: URL templates containing {q}. Keep title, link and published time for audit."""
    import feedparser
    out = []
    for url in feeds:
        for e in feedparser.parse(url.format(q=query)).entries[:max_items]:
            out.append({"title": e.get("title", ""), "link": e.get("link", ""), "published": e.get("published", "")})
    return out
