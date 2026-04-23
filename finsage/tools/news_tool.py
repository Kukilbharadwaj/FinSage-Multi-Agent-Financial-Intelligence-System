# tools/news_tool.py
# Fetches news from Google News RSS and MoneyControl RSS.
# No API key required.

import feedparser
from urllib.parse import quote_plus
from typing import List


def get_news(query: str, limit: int = 10) -> List[dict]:
    """
    Fetch news articles from Google News RSS and MoneyControl RSS.

    Args:
        query: Search query string (e.g., "Reliance Industries")
        limit: Maximum number of articles to return

    Returns:
        List of dicts with keys: title, source, link
    """
    articles = []
    seen_titles = set()

    # Source 1: Google News RSS
    try:
        encoded_query = quote_plus(f"{query} India stock market")
        google_url = (
            f"https://news.google.com/rss/search?"
            f"q={encoded_query}&hl=en-IN&gl=IN&ceid=IN:en"
        )
        feed = feedparser.parse(google_url)

        for entry in feed.entries:
            title = entry.get("title", "").strip()
            if title and title not in seen_titles:
                articles.append({
                    "title": title,
                    "source": "Google News",
                    "link": entry.get("link", ""),
                })
                seen_titles.add(title)

    except Exception:
        pass  # Skip silently if Google News fails

    # Source 2: MoneyControl Market Reports RSS
    try:
        mc_url = "https://www.moneycontrol.com/rss/marketreports.xml"
        feed = feedparser.parse(mc_url)

        for entry in feed.entries:
            title = entry.get("title", "").strip()
            if title and title not in seen_titles:
                articles.append({
                    "title": title,
                    "source": "MoneyControl",
                    "link": entry.get("link", ""),
                })
                seen_titles.add(title)

    except Exception:
        pass  # Skip silently if MoneyControl fails

    return articles[:limit]
