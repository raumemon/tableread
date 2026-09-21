"""Fetch YouTube comments on Spokane/CDA food and local-life videos.

YouTube commenters skew older and more working-class than Reddit — the
demographic hole in the corpus. Official Data API, free quota, ToS-clean.
Search costs 100 quota units/query, comments 1/page; this stays well under
the 10k/day free quota.

Output: data/corpus_reviews/youtube-comments.json
"""
import json
import os
import time

import requests

import sys
sys.path.insert(0, os.path.dirname(__file__))

KEY_PATH = os.path.expanduser("~/projects/site-scout/data/google_api_key.txt")
OUT = "data/corpus_reviews/youtube-comments.json"

QUERIES = [
    "Spokane restaurants", "Spokane food review", "best food Spokane",
    "Coeur d'Alene restaurants", "Spokane Washington living", "moving to Spokane",
    "downtown Spokane", "Spokane cost of living", "Spokane Valley food",
    # targeted channels (agent-verified local comment ecosystems)
    "Katina Eats Kilos Spokane", "Katina Eats Kilos challenge Post Falls",
    "KREM 2 Spokane restaurant", "KXLY Spokane food", "inlandNWeats",
]
VIDEOS_PER_QUERY = 6
COMMENTS_PER_VIDEO = 60


def get(url, **params):
    r = requests.get(url, params=params, timeout=30)
    r.raise_for_status()
    return r.json()


def main():
    from llm import load_env
    load_env()
    key = os.environ.get("GOOGLE_YT_KEY") or open(KEY_PATH).read().strip()
    videos = {}
    for q in QUERIES:
        try:
            j = get("https://www.googleapis.com/youtube/v3/search", key=key, q=q,
                    part="snippet", maxResults=VIDEOS_PER_QUERY, type="video",
                    relevanceLanguage="en", order="relevance")
            for it in j.get("items", []):
                videos[it["id"]["videoId"]] = {"title": it["snippet"]["title"], "query": q}
        except Exception as e:
            print(f"search FAIL {q!r}: {e}")
        time.sleep(0.2)
    print(f"{len(videos)} videos found")

    out = []
    for vid, meta in videos.items():
        try:
            j = get("https://www.googleapis.com/youtube/v3/commentThreads", key=key,
                    part="snippet", videoId=vid, maxResults=min(COMMENTS_PER_VIDEO, 100),
                    order="relevance", textFormat="plainText")
            comments = []
            for it in j.get("items", []):
                s = it["snippet"]["topLevelComment"]["snippet"]
                text = s.get("textDisplay", "")
                if len(text) > 40:
                    comments.append({"text": text[:1200], "likes": s.get("likeCount", 0)})
            if comments:
                out.append({"video_id": vid, "title": meta["title"], "query": meta["query"],
                            "comments": comments})
        except requests.HTTPError as e:
            # comments disabled on many videos; skip quietly
            if "403" not in str(e):
                print(f"comments FAIL {meta['title'][:40]!r}: {e}")
        time.sleep(0.15)

    os.makedirs("data/corpus_reviews", exist_ok=True)
    with open(OUT, "w") as f:
        json.dump({"source": "youtube", "videos": out}, f)
    total = sum(len(v["comments"]) for v in out)
    print(f"done: {len(out)} videos with comments, {total} comment texts -> {OUT}")


if __name__ == "__main__":
    main()
