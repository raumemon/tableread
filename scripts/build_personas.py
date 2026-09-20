"""Build a synthetic local panel: ACS demographics x corpus-mined dining archetypes.

1. Sample N personas from block groups (population-weighted, seeded/reproducible).
   Block groups come from site-scout's parsed ACS data (Spokane WA 53063 +
   Kootenai ID 16055).
2. One Sonnet call mines dining archetypes from the Reddit corpus, each grounded
   in real quotes, with prevalence + demographic skew for assignment.
3. Merge: each persona = demographics + archetype + verbatim local quotes as
   voice anchors.

Cost: one API call, roughly $0.05-0.10. Output: data/personas.json
"""
import argparse
import json
import os
import random
import re

from llm import LLM, MODEL

ACS_PATH = os.path.expanduser("~/projects/site-scout/data/acs_blockgroups.json")
CORPUS_DIR = "data/corpus"
OUT_PATH = "data/personas.json"
SEED = 47

COUNTY_AREA = {"53063": "Spokane", "16055": "Coeur d'Alene / Post Falls"}


WORK = [("full-time, office/desk job", 32), ("full-time, on your feet (retail/healthcare/trades)", 30),
        ("shift work, irregular hours", 10), ("self-employed / small business", 8),
        ("student (part-time work)", 6), ("retired", 9), ("between jobs / gig work", 5)]
DIET = [("none", 78), ("vegetarian-ish, flexible", 6), ("gluten-free household member", 5),
        ("watching cholesterol/sodium", 7), ("halal-preferring", 1), ("low-carb most of the time", 3)]


def household_for(age, renter, rng):
    if age < 30:
        opts = [("lives with roommates", 30), ("lives alone", 25), ("couple, no kids", 30),
                ("young kids at home", 15)]
    elif age < 50:
        opts = [("kids at home who veto restaurants", 38), ("couple, no kids", 27),
                ("lives alone", 20), ("teenagers at home", 15)]
    else:
        opts = [("empty nester couple", 45), ("lives alone", 30),
                ("adult kid back home", 10), ("grandkids visit on weekends", 15)]
    return rng.choices([o for o, _ in opts], weights=[w for _, w in opts], k=1)[0]


def sample_demographics(n, rng):
    acs = json.load(open(ACS_PATH))
    bgs = [(geoid, d) for geoid, d in acs.items() if (d.get("population") or 0) > 0]
    weights = [d["population"] for _, d in bgs]
    people = []
    for i in range(n):
        geoid, d = rng.choices(bgs, weights=weights, k=1)[0]
        median_age = d.get("median_age") or 38.0
        # Adults only; jitter around the block group's median.
        age = max(21, min(79, int(rng.gauss(median_age, 11))))
        income = d.get("median_hh_income") or 65000
        income = max(15000, int(rng.gauss(income, income * 0.35)))
        renter_share = (d.get("renter_households") or 0) / max(d.get("total_households") or 1, 1)
        renter = rng.random() < renter_share
        work = "retired" if age >= 68 else rng.choices(
            [w for w, _ in WORK], weights=[wt for _, wt in WORK], k=1)[0]
        people.append({
            "id": f"p{i:03d}",
            "block_group": geoid,
            "area": COUNTY_AREA.get(geoid[:5], "Spokane"),
            "age": age,
            "household_income": income,
            "renter": renter,
            "household": household_for(age, renter, rng),
            "work": work,
            "diet": rng.choices([dd for dd, _ in DIET], weights=[w for _, w in DIET], k=1)[0],
            # 0-2: never had the cuisine being tested; 3-5: occasional; 6-9: knows it well
            "cuisine_familiarity": rng.choices(range(10), weights=[8, 8, 10, 14, 14, 12, 12, 10, 7, 5], k=1)[0],
        })
    return people


SUPPLEMENT_DIR = os.path.expanduser("~/projects/site-scout/data/reddit")


def load_corpus_sample(max_chars=45000):
    """High-signal posts and comments across all topics, score-weighted.

    Blends the dining-topic corpus with site-scout's neighborhood corpus
    (same file shape); dining files win ties via a small score boost.
    """
    paths = [os.path.join(CORPUS_DIR, fn) for fn in sorted(os.listdir(CORPUS_DIR))]
    if os.path.isdir(SUPPLEMENT_DIR):
        paths += [os.path.join(SUPPLEMENT_DIR, fn) for fn in sorted(os.listdir(SUPPLEMENT_DIR))]
    items = []
    for path in paths:
        d = json.load(open(path))
        boost = 3 if path.startswith(CORPUS_DIR) else 0
        for p in d.get("posts", []):
            text = (p["title"] + ". " + p.get("selftext", "")).strip()
            if len(text) > 40:
                items.append({"text": text[:900], "score": p.get("score", 0) + boost, "sub": d["subreddit"]})
        for c in d.get("comments", []):
            if len(c.get("body", "")) > 40:
                items.append({"text": c["body"][:900], "score": c.get("score", 0) + boost, "sub": d["subreddit"]})
    items.sort(key=lambda x: -x["score"])
    out, total = [], 0
    for it in items:
        if total + len(it["text"]) > max_chars:
            break
        out.append(it)
        total += len(it["text"])
    return out


ARCHETYPE_SCHEMA = {
    "type": "object",
    "properties": {
        "archetypes": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "key": {"type": "string"},
                    "name": {"type": "string"},
                    "description": {"type": "string"},
                    "dining_habits": {"type": "string"},
                    "price_sensitivity": {"type": "string", "enum": ["low", "medium", "high"]},
                    "prevalence_pct": {"type": "number"},
                    "age_skew": {"type": "string", "enum": ["young", "middle", "older", "none"]},
                    "income_skew": {"type": "string", "enum": ["lower", "middle", "higher", "none"]},
                    "voice_notes": {"type": "string"},
                    "grounding_quotes": {"type": "array", "items": {"type": "string"}},
                },
                "required": ["key", "name", "description", "dining_habits", "price_sensitivity",
                             "prevalence_pct", "age_skew", "income_skew", "voice_notes", "grounding_quotes"],
                "additionalProperties": False,
            },
        },
        "life_details": {
            "type": "array",
            "items": {"type": "string"},
            "description": "40-60 short, concrete, corpus-grounded life details a local might have",
        },
    },
    "required": ["archetypes", "life_details"],
    "additionalProperties": False,
}


def mine_archetypes(client, corpus):
    corpus_text = "\n---\n".join(f"[r/{it['sub']}, score {it['score']}] {it['text']}" for it in corpus)
    system = (
            "You are a consumer insights researcher segmenting the dining public of the "
            "Spokane WA / Coeur d'Alene ID corridor. You are given real Reddit posts and "
            "comments from locals discussing restaurants, food, prices, and vibes. Derive "
            "6-8 dining psychographic archetypes that together cover the local restaurant-"
            "going population (prevalence_pct sums to ~100). Ground every archetype in the "
            "actual corpus: real attitudes, real complaints, real vocabulary. "
            "grounding_quotes must be verbatim excerpts (or near-verbatim trims) from the "
            "corpus, 2-4 per archetype. voice_notes describes how this person writes and "
            "talks (register, slang, reference points). Include unglamorous segments "
            "(chain loyalists, rarely-eats-out) at honest prevalence, not just foodies. "
            "Also produce life_details: 40-60 short concrete life circumstances grounded in "
            "the corpus and local geography (e.g. 'commutes to Fairchild AFB', 'kid plays "
            "club soccer, eats dinner in the car twice a week', 'moved here from Seattle in "
            "2021', 'worked in restaurants through college'). Specific and mundane beats "
            "colorful; these get randomly attached to survey personas to make reactions "
            "individual, so they must be things that would plausibly color how someone "
            "judges a new restaurant."
    )
    data = client.complete_json(system, f"Local corpus:\n\n{corpus_text}",
                                ARCHETYPE_SCHEMA, max_tokens=8000)
    print(f"archetype mining usage: {client.usage}")
    return data["archetypes"], data["life_details"]


def assign_archetype(person, archetypes, rng):
    def skew_mult(arch):
        m = 1.0
        a, inc = person["age"], person["household_income"]
        m *= {"young": 2.0 if a < 35 else 0.5, "middle": 2.0 if 35 <= a < 55 else 0.6,
              "older": 2.0 if a >= 55 else 0.5, "none": 1.0}[arch["age_skew"]]
        m *= {"lower": 2.0 if inc < 50000 else 0.5, "middle": 2.0 if 50000 <= inc < 100000 else 0.6,
              "higher": 2.0 if inc >= 100000 else 0.5, "none": 1.0}[arch["income_skew"]]
        return m

    weights = [max(a["prevalence_pct"], 1) * skew_mult(a) for a in archetypes]
    return rng.choices(archetypes, weights=weights, k=1)[0]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("-n", type=int, default=300, help="panel size")
    args = ap.parse_args()

    rng = random.Random(SEED)
    client = LLM()
    print(f"provider: {client.provider}, model: {MODEL}")

    corpus = load_corpus_sample()
    print(f"corpus sample: {len(corpus)} items")
    archetypes, life_details = mine_archetypes(client, corpus)
    print(f"archetypes: {[a['name'] for a in archetypes]}; {len(life_details)} life details")

    quote_pool = [it["text"] for it in corpus if it["score"] >= 3]
    people = sample_demographics(args.n, rng)
    for p in people:
        arch = assign_archetype(p, archetypes, rng)
        p["archetype"] = arch["key"]
        p["voice_quotes"] = rng.sample(quote_pool, k=min(3, len(quote_pool)))
        p["life_details"] = rng.sample(life_details, k=min(2, len(life_details)))

    out = {"archetypes": archetypes, "personas": people, "seed": SEED}
    os.makedirs("data", exist_ok=True)
    with open(OUT_PATH, "w") as f:
        json.dump(out, f, indent=1)
    print(f"wrote {len(people)} personas -> {OUT_PATH}")


if __name__ == "__main__":
    main()
