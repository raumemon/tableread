"""Fetch dining discourse from r/Spokane + r/CoeurdAlene via Arctic Shift.

Same client pattern as site-scout/scripts/fetch_reddit.py, but the search
terms target restaurant/food conversation instead of neighborhoods: this
corpus teaches personas the local dining vocabulary, complaints, price
sensitivity, and competitor reference points.

Caches to data/corpus/<slug>.json; delete files to refresh.
"""
import json
import os
import time

import requests

API = "https://arctic-shift.photon-reddit.com/api"
UA = {"User-Agent": "tableread-research/0.1 (restaurant concept testing, Spokane/CDA)"}
DELAY = 12.0
AFTER_UTC = 1672531200  # 2023-01-01: dining opinions go stale faster than neighborhoods

SUBS = ["Spokane", "CoeurdAlene"]

# slug -> (post-search terms, comment-search terms)
# Post search matches title+selftext via `query`; comment search matches `body`.
TOPICS = {
    "recs-general": (["restaurant recommendations", "where to eat", "best restaurant"],
                     ["best restaurant in"]),
    "new-openings": (["new restaurant", "just opened", "coming soon restaurant"], []),
    "gyro-mediterranean": (["gyro", "shawarma", "mediterranean food", "greek food", "falafel"],
                           ["gyro", "shawarma"]),
    "fast-casual": (["fast casual", "quick lunch", "takeout", "food truck"],
                    ["worth the price food"]),
    "value-complaints": (["overpriced restaurant", "restaurant prices", "cheap eats"],
                         ["overpriced"]),
    "vibes-branding": (["restaurant vibe", "atmosphere restaurant", "date night restaurant"], []),
}


def get(path, **params):
    time.sleep(DELAY)
    for attempt in range(3):
        r = requests.get(f"{API}/{path}", params=params, headers=UA, timeout=90)
        if r.status_code in (422, 429):
            time.sleep(15 * (attempt + 1))
            continue
        r.raise_for_status()
        return r.json().get("data", [])
    return []


def slim_post(p):
    return {
        "id": p.get("id"),
        "title": p.get("title", ""),
        "selftext": (p.get("selftext") or "")[:2000],
        "score": p.get("score", 0),
        "num_comments": p.get("num_comments", 0),
        "created_utc": p.get("created_utc", 0),
    }


def slim_comment(c):
    return {
        "id": c.get("id"),
        "body": (c.get("body") or "")[:2000],
        "score": c.get("score", 0),
        "created_utc": c.get("created_utc", 0),
    }


def main():
    os.makedirs("data/corpus", exist_ok=True)
    for topic, (post_terms, comment_terms) in TOPICS.items():
        for sub in SUBS:
            slug = f"{topic}--{sub.lower()}"
            path = f"data/corpus/{slug}.json"
            if os.path.exists(path):
                print(f"skip {slug} (cached)")
                continue
            posts, comments = {}, {}
            for term in post_terms:
                for p in get("posts/search", subreddit=sub, query=term, limit=100):
                    if (p.get("created_utc") or 0) >= AFTER_UTC and p.get("selftext") != "[removed]":
                        posts[p["id"]] = slim_post(p)
            for term in comment_terms:
                for c in get("comments/search", subreddit=sub, body=term, limit=100):
                    if (c.get("created_utc") or 0) >= AFTER_UTC and c.get("body") not in ("[removed]", "[deleted]"):
                        comments[c["id"]] = slim_comment(c)
            # Pull comment threads for the most-discussed posts: that's where
            # the real opinions live, not in the question posts themselves.
            top_posts = sorted(posts.values(), key=lambda p: -p["num_comments"])[:8]
            for p in top_posts:
                if p["num_comments"] == 0:
                    continue
                for c in get("comments/search", subreddit=sub, link_id=p["id"], limit=100):
                    if c.get("body") not in ("[removed]", "[deleted]", None):
                        comments[c["id"]] = slim_comment(c)
            out = {"slug": slug, "subreddit": sub, "topic": topic,
                   "posts": list(posts.values()), "comments": list(comments.values())}
            with open(path, "w") as f:
                json.dump(out, f)
            print(f"{slug}: {len(posts)} posts, {len(comments)} comments")


if __name__ == "__main__":
    main()
