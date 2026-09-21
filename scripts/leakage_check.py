"""Knowledge-leakage detector: find business/place references in survey
verbatims that aren't part of the local universe.

Personas share the model's world knowledge, so out-of-region businesses leak
into 'local' reactions (measured: 19% of K Bob reactions referenced K-Bob's
Steakhouse, a NM/TX chain). This scans verbatims for capitalized entities,
whitelists the local place universe (site-scout's Google Places inventory) and
common national chains, and reports what's left for human review.

Usage: leakage_check.py data/results/<run>.json
"""
import argparse
import json
import os
import re
from collections import Counter

PLACES = os.path.expanduser("~/projects/site-scout/data/places_google.json")

NATIONAL = {w.lower() for w in [
    "McDonald's", "McDonalds", "Subway", "Starbucks", "Chipotle", "Taco Bell", "Wendy's",
    "Burger King", "KFC", "Panda Express", "Olive Garden", "Applebee's", "Denny's", "IHOP",
    "Domino's", "Pizza Hut", "Papa John's", "Costco", "Walmart", "Target", "Amazon",
    "Chick-fil-A", "Five Guys", "Panera", "Jimmy John's", "Qdoba", "Red Robin",
    "Texas Roadhouse", "Buffalo Wild Wings", "Dutch Bros", "DoorDash", "Uber Eats",
    "Google", "Yelp", "Facebook", "Instagram", "TikTok", "Reddit", "YouTube",
]}
STOP = {w.lower() for w in [
    "I'd", "I'm", "I've", "I'll", "It's", "That's", "There's", "What's", "Don't", "Can't",
    "Won't", "Isn't", "Doesn't", "Wouldn't", "Couldn't", "She's", "He's", "We're", "They're",
    "You're", "Ooo", "Oooh", "Ooooh", "Ooooooh", "Hmm", "Nope", "Yeah", "Okay", "Eh", "Ugh",
    "Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday",
    "January", "February", "March", "April", "May", "June", "July", "August", "September",
    "October", "November", "December", "Painfully", "Honestly",
]}
GEO_OK = {w.lower() for w in [
    "Spokane", "Coeur", "Alene", "CDA", "Post Falls", "Hayden", "Kendall Yards", "Gonzaga",
    "Fairchild", "Sacred Heart", "Perry", "Browne", "Hillyard", "Garland", "Wandermere",
    "Airway Heights", "Liberty Lake", "Cheney", "Millwood", "Silverwood", "Bloomsday",
    "Division", "Sprague", "Monroe", "Sherman", "Riverfront", "Northtown", "NorthTown",
    "Idaho", "Washington", "Seattle", "Portland", "Greek", "Mediterranean", "Middle",
    "Eastern", "Lebanese", "Turkish", "Halal", "Gyro", "Kebab", "Shawarma", "Sumak",
    "Fred Meyer", "Rosauers", "Safeway", "WinCo", "Trader Joe", "Valley", "Inlander",
    "Balkan", "Zips", "Dick's", "Frank's Diner", "De Leon",
]}


def entities(text, common):
    """Proper-noun candidates: capitalized runs whose words never appear
    lowercase anywhere in the corpus (filters sentence-initial words)."""
    for m in re.finditer(r"\b([A-Z][A-Za-z'&\-]+(?:[\s\-]+[A-Z][A-Za-z'&\-]+){0,3})", text):
        e = m.group(1).strip()
        words = re.findall(r"[A-Za-z'&\-]+", e)
        if all(w.lower() not in common for w in words):
            yield e


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("run")
    ap.add_argument("--top", type=int, default=30)
    args = ap.parse_args()

    local = set()
    if os.path.exists(PLACES):
        for pl in json.load(open(PLACES)):
            for w in re.findall(r"[A-Za-z'&\-]+", pl["name"]):
                if len(w) > 2:
                    local.add(w.lower())

    d = json.load(open(args.run))
    concept_words = {w.lower() for w in re.findall(r"[A-Za-z'\-]+", json.dumps(d.get("concept", {})))}
    all_text = " ".join(json.dumps(r["answers"]) for r in d["results"])
    common = set(re.findall(r"(?<![A-Za-z])[a-z][a-z'&\-]+", all_text))
    counts = Counter()
    samples = {}
    for r in d["results"]:
        blob = json.dumps(r["answers"])
        for e in entities(blob, common):
            words = [w.lower() for w in re.findall(r"[A-Za-z'&\-]+", e)]
            if all(w in local or w in NATIONAL or w in GEO_OK or w in concept_words
                   or w in STOP or len(w) <= 2 for w in words):
                continue
            counts[e] += 1
            samples.setdefault(e, f"{r['persona']['age']}yo {r['persona']['area']}")

    print(f"POSSIBLE KNOWLEDGE LEAKAGE — entities outside the local universe "
          f"(review by hand; some are false positives):")
    for e, n in counts.most_common(args.top):
        print(f"  {n:4d}x  {e}   (e.g. {samples[e]})")
    flagged = sum(counts.values())
    print(f"\n{flagged} flagged mentions across {len(d['results'])} respondents")


if __name__ == "__main__":
    main()
