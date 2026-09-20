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

import anthropic

ACS_PATH = os.path.expanduser("~/projects/site-scout/data/acs_blockgroups.json")
CORPUS_DIR = "data/corpus"
OUT_PATH = "data/personas.json"
MODEL = "claude-sonnet-5"
SEED = 47

COUNTY_AREA = {"53063": "Spokane", "16055": "Coeur d'Alene / Post Falls"}


def load_env():
    if os.path.exists(".env"):
        for line in open(".env"):
            if "=" in line and not line.startswith("#"):
                k, v = line.strip().split("=", 1)
                os.environ.setdefault(k, v)


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
        people.append({
            "id": f"p{i:03d}",
            "block_group": geoid,
            "area": COUNTY_AREA.get(geoid[:5], "Spokane"),
            "age": age,
            "household_income": income,
            "renter": rng.random() < renter_share,
        })
    return people


def load_corpus_sample(max_chars=45000):
    """High-signal posts and comments across all topics, score-weighted."""
    items = []
    for fn in sorted(os.listdir(CORPUS_DIR)):
        d = json.load(open(os.path.join(CORPUS_DIR, fn)))
        for p in d.get("posts", []):
            text = (p["title"] + ". " + p.get("selftext", "")).strip()
            if len(text) > 40:
                items.append({"text": text[:900], "score": p.get("score", 0), "sub": d["subreddit"]})
        for c in d.get("comments", []):
            if len(c.get("body", "")) > 40:
                items.append({"text": c["body"][:900], "score": c.get("score", 0), "sub": d["subreddit"]})
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
        }
    },
    "required": ["archetypes"],
    "additionalProperties": False,
}


def mine_archetypes(client, corpus):
    corpus_text = "\n---\n".join(f"[r/{it['sub']}, score {it['score']}] {it['text']}" for it in corpus)
    response = client.messages.create(
        model=MODEL,
        max_tokens=8000,
        system=(
            "You are a consumer insights researcher segmenting the dining public of the "
            "Spokane WA / Coeur d'Alene ID corridor. You are given real Reddit posts and "
            "comments from locals discussing restaurants, food, prices, and vibes. Derive "
            "6-8 dining psychographic archetypes that together cover the local restaurant-"
            "going population (prevalence_pct sums to ~100). Ground every archetype in the "
            "actual corpus: real attitudes, real complaints, real vocabulary. "
            "grounding_quotes must be verbatim excerpts (or near-verbatim trims) from the "
            "corpus, 2-4 per archetype. voice_notes describes how this person writes and "
            "talks (register, slang, reference points). Include unglamorous segments "
            "(chain loyalists, rarely-eats-out) at honest prevalence, not just foodies."
        ),
        messages=[{"role": "user", "content": f"Local corpus:\n\n{corpus_text}"}],
        output_config={"format": {"type": "json_schema", "schema": ARCHETYPE_SCHEMA}},
    )
    text = next(b.text for b in response.content if b.type == "text")
    usage = response.usage
    print(f"archetype mining: {usage.input_tokens} in / {usage.output_tokens} out tokens")
    return json.loads(text)["archetypes"]


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
    ap.add_argument("-n", type=int, default=100, help="panel size")
    args = ap.parse_args()

    load_env()
    rng = random.Random(SEED)
    client = anthropic.Anthropic()

    corpus = load_corpus_sample()
    print(f"corpus sample: {len(corpus)} items")
    archetypes = mine_archetypes(client, corpus)
    print(f"archetypes: {[a['name'] for a in archetypes]}")

    quote_pool = [it["text"] for it in corpus if it["score"] >= 3]
    people = sample_demographics(args.n, rng)
    for p in people:
        arch = assign_archetype(p, archetypes, rng)
        p["archetype"] = arch["key"]
        p["voice_quotes"] = rng.sample(quote_pool, k=min(3, len(quote_pool)))

    out = {"archetypes": archetypes, "personas": people, "seed": SEED}
    os.makedirs("data", exist_ok=True)
    with open(OUT_PATH, "w") as f:
        json.dump(out, f, indent=1)
    print(f"wrote {len(people)} personas -> {OUT_PATH}")


if __name__ == "__main__":
    main()
