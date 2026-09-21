"""Fetch Google Maps review text for corridor restaurants via Places API (New).

Reuses site-scout's place inventory (gid = place id) and Google key. Up to 5
reviews per place. Review writers span far broader demographics than Reddit,
so this becomes the PRIMARY voice source; Reddit stays as one tagged source
among several.

Output: data/corpus_reviews/google-reviews.json
"""
import json
import os
import time

import requests

PLACES = os.path.expanduser("~/projects/site-scout/data/places_google.json")
KEY_PATH = os.path.expanduser("~/projects/site-scout/data/google_api_key.txt")
OUT = "data/corpus_reviews/google-reviews.json"
TOP_N = 600


def main():
    key = open(KEY_PATH).read().strip()
    d = json.load(open(PLACES))
    food = [p for p in d if "restaurant" in (p.get("types") or [])
            or p.get("primary") in ("meal_takeaway", "sandwich_shop", "fast_food_restaurant", "cafe")]
    food.sort(key=lambda p: -(p.get("reviews") or 0))
    targets = food[:TOP_N]
    print(f"fetching reviews for {len(targets)} of {len(food)} food places")

    os.makedirs("data/corpus_reviews", exist_ok=True)
    done = {}
    if os.path.exists(OUT):
        done = {x["place_id"]: x for x in json.load(open(OUT))["places"]}
        print(f"resuming: {len(done)} cached")

    for i, p in enumerate(targets):
        if p["gid"] in done:
            continue
        try:
            r = requests.get(
                f"https://places.googleapis.com/v1/places/{p['gid']}",
                headers={"X-Goog-Api-Key": key, "X-Goog-FieldMask": "displayName,reviews"},
                timeout=30)
            r.raise_for_status()
            revs = []
            for rev in (r.json().get("reviews") or []):
                text = (rev.get("text") or {}).get("text", "")
                if len(text) > 40:
                    revs.append({"rating": rev.get("rating"), "text": text[:1200],
                                 "time": rev.get("publishTime", "")[:10]})
            done[p["gid"]] = {"place_id": p["gid"], "name": p["name"],
                              "primary": p.get("primary"), "address": p.get("address"),
                              "rating": p.get("rating"), "review_count": p.get("reviews"),
                              "reviews": revs}
        except Exception as e:
            print(f"FAIL {p['name']}: {e}")
        if (i + 1) % 100 == 0:
            print(f"{i + 1}/{len(targets)}")
            with open(OUT, "w") as f:
                json.dump({"source": "google_reviews", "places": list(done.values())}, f)
        time.sleep(0.12)

    with open(OUT, "w") as f:
        json.dump({"source": "google_reviews", "places": list(done.values())}, f)
    total = sum(len(x["reviews"]) for x in done.values())
    print(f"done: {len(done)} places, {total} review texts -> {OUT}")


if __name__ == "__main__":
    main()
