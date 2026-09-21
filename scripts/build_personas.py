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
    if age < 25:
        opts = [("lives with roommates", 38), ("lives alone", 22), ("couple, no kids", 24),
                ("lives with parents", 8), ("young kids at home", 8)]
    elif age < 30:
        opts = [("lives with roommates", 22), ("lives alone", 24), ("couple, no kids", 36),
                ("young kids at home", 18)]
    elif age < 50:
        opts = [("kids at home who veto restaurants", 38), ("couple, no kids", 27),
                ("lives alone", 20), ("teenagers at home", 15)]
    else:
        opts = [("empty nester couple", 45), ("lives alone", 30),
                ("adult kid back home", 10), ("grandkids visit on weekends", 15)]
    return rng.choices([o for o, _ in opts], weights=[w for _, w in opts], k=1)[0]


PUMS_PATH = "data/reference/pums_records.json"

HHT2_TEXT = {
    "01": "married couple with kids at home", "02": "married couple, no kids at home",
    "03": "cohabiting couple with kids at home", "04": "cohabiting couple, no kids at home",
    "05": "lives alone", "06": "single parent with kids at home",
    "07": "lives with relatives", "08": "lives with roommates",
    "09": "lives alone", "10": "single parent with kids at home",
    "11": "lives with relatives", "12": "lives with roommates",
}


def emp_status(rec):
    e = rec["employment"]
    if e in ("1", "2"):
        return "employed"
    if e == "3":
        return "unemployed, looking"
    if e in ("4", "5"):
        return "military"
    if rec["age"] >= 60:
        return "retired"
    return "not in the labor force"


def edu_band(schl):
    try:
        v = int(schl)
    except ValueError:
        return "unknown"
    if v >= 21:
        return "bachelor's or higher"
    if v >= 18:
        return "some college / associate's"
    return "high school or less"


def sample_demographics_pums(n, rng):
    """Sample REAL Census respondents (ACS PUMS): actual joint distributions,
    so implausible attribute combos can't occur at the demographic layer."""
    recs = json.load(open(PUMS_PATH))
    weights = [r["weight"] for r in recs]
    people = []
    for i in range(n):
        rec = rng.choices(recs, weights=weights, k=1)[0]
        hh = HHT2_TEXT.get(rec["household_type"], "lives alone")
        if rec["own_children"] == 0 and "kids at home" in hh:
            hh = hh.replace(" with kids at home", ", no kids at home")
        people.append({
            "id": f"p{i:03d}",
            "pums": True, "puma": rec["puma"],
            "area": rec["area"],
            "age": rec["age"],
            "household_income": max(rec["household_income"], 5000),
            "renter": rec["renter"],
            "household": hh + (f" ({rec['own_children']} kids)" if rec["own_children"] else ""),
            "employment_status": emp_status(rec),
            "student": rec["school_enrolled"],
            "education": edu_band(rec["education"]),
            "household_size": rec["household_size"],
            "diet": rng.choices([dd for dd, _ in DIET], weights=[w for _, w in DIET], k=1)[0],
            "cuisine_familiarity": rng.choices(range(10), weights=[8, 8, 10, 14, 14, 12, 12, 10, 7, 5], k=1)[0],
        })
    return people


def sample_demographics(n, rng):
    if os.path.exists(PUMS_PATH):
        return sample_demographics_pums(n, rng)
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
        if age >= 68:
            work = "retired"
        else:
            # "retired" and "student" only at plausible ages
            opts = [(w, wt) for w, wt in WORK
                    if not (w == "retired" and age < 55) and not (w.startswith("student") and age > 30)]
            work = rng.choices([w for w, _ in opts], weights=[wt for _, wt in opts], k=1)[0]
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
REVIEWS_PATH = "data/corpus_reviews/google-reviews.json"
MANUAL_DIR = "data/corpus_manual"

# Reddit skews young/male/extremely-online; reviews span the broadest local
# demographics; manual drops (Facebook groups, Nextdoor) skew older/family.
SOURCE_MIX = {"google_reviews": 0.45, "reddit": 0.20, "youtube": 0.15, "manual": 0.20}


def _reddit_items():
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
                items.append({"text": text[:900], "score": p.get("score", 0) + boost,
                              "sub": d["subreddit"], "source": "reddit"})
        for c in d.get("comments", []):
            if len(c.get("body", "")) > 40:
                items.append({"text": c["body"][:900], "score": c.get("score", 0) + boost,
                              "sub": d["subreddit"], "source": "reddit"})
    return items


def _review_items():
    if not os.path.exists(REVIEWS_PATH):
        return []
    d = json.load(open(REVIEWS_PATH))
    items = []
    for place in d["places"]:
        for rev in place["reviews"]:
            # Longer, opinionated reviews carry the most voice signal.
            score = len(rev["text"]) // 120 + (2 if rev.get("rating") in (1, 2, 5) else 0)
            items.append({"text": f"[{place['name']}, {rev.get('rating')}*] {rev['text'][:900]}",
                          "score": score, "sub": "google", "source": "google_reviews"})
    return items


def _youtube_items():
    path = "data/corpus_reviews/youtube-comments.json"
    if not os.path.exists(path):
        return []
    d = json.load(open(path))
    items = []
    for v in d.get("videos", []):
        for c in v["comments"]:
            items.append({"text": f"[on: {v['title'][:60]}] {c['text'][:900]}",
                          "score": c.get("likes", 0), "sub": "youtube", "source": "youtube"})
    return items


def _manual_items():
    """Drop .txt files (Facebook threads, Nextdoor posts, anything pasted) into
    data/corpus_manual/ — one item per blank-line-separated block."""
    if not os.path.isdir(MANUAL_DIR):
        return []
    items = []
    for fn in sorted(os.listdir(MANUAL_DIR)):
        if not fn.endswith(".txt"):
            continue
        blocks = open(os.path.join(MANUAL_DIR, fn)).read().split("\n\n")
        for b in blocks:
            b = b.strip()
            if len(b) > 40:
                items.append({"text": b[:900], "score": 5, "sub": fn[:-4], "source": "manual"})
    return items


def load_corpus_sample(max_chars=45000):
    """Score-weighted sample, mixed across sources by SOURCE_MIX quotas.

    A source with no material yields its quota to the others.
    """
    pools = {"reddit": _reddit_items(), "google_reviews": _review_items(), "youtube": _youtube_items(), "manual": _manual_items()}
    for pool in pools.values():
        pool.sort(key=lambda x: -x["score"])
    available = {k: v for k, v in pools.items() if v}
    weight_total = sum(SOURCE_MIX[k] for k in available)
    out = []
    for src, pool in available.items():
        budget = max_chars * SOURCE_MIX[src] / weight_total
        total = 0
        for it in pool:
            if total + len(it["text"]) > budget:
                break
            out.append(it)
            total += len(it["text"])
    counts = {}
    for it in out:
        counts[it["source"]] = counts.get(it["source"], 0) + 1
    print(f"corpus mix: {counts}")
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


def load_inlander():
    path = "data/reference/inlander_best_of.json"
    if not os.path.exists(path):
        return ""
    d = json.load(open(path))
    lines = [f"{e['category']}: {e['winner']}" for sec in d.values() for e in sec]
    return ("\n\nACTUAL LOCAL PREFERENCE DATA — Inlander 'Best of the Inland Northwest' "
            "reader-poll winners (thousands of local votes; treat as revealed preference "
            "about what this market already rewards):\n" + "\n".join(lines))


def mine_archetypes(client, corpus):
    corpus_text = "\n---\n".join(f"[{it['source']}/{it['sub']}, score {it['score']}] {it['text']}" for it in corpus)
    corpus_text += load_inlander()
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


COHERENCE_SCHEMA = {
    "type": "object",
    "properties": {
        "personas": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "id": {"type": "string"},
                    "occupation": {"type": "string",
                                   "description": "Specific and realistic, MUST match the record's employment_status/education/age/income, e.g. 'CNA on nights at Sacred Heart', 'retired, formerly a Kaiser aluminum worker', 'drywall contractor'. Mundane majority."},
                    "life_details": {"type": "array", "items": {"type": "string"},
                                     "description": "Exactly 2, adapted from the pool (or lightly invented in the same spirit) so they FIT this person's age, income, schedule, and area"},
                },
                "required": ["id", "occupation", "life_details"],
                "additionalProperties": False,
            },
        }
    },
    "required": ["personas"],
    "additionalProperties": False,
}


def coherence_pass(client, people, life_pool, batch=50):
    """Resolve independently-sampled attributes into internally consistent people.

    Hard anchors (never changed): age, area, income, renter, diet, familiarity.
    The model assigns occupation/household/life details that make sense JOINTLY.
    """
    system = (
        "You make survey panel personas internally coherent. Each persona is anchored to a "
        "REAL anonymized Census respondent: age, area, household income and size, household "
        "type, renter status, employment status, education, and student status are FIXED — "
        "never contradict them. Produce: (1) an occupation that fits those anchors exactly "
        "(employed -> a specific plausible Spokane/CDA job matching education and income; "
        "'retired' -> 'retired, formerly X'; 'unemployed, looking' -> last job + looking; "
        "'not in the labor force' -> a plausible reason; 'military' -> plausibly Fairchild "
        "AFB); keep the boring majority boring — warehouse, retail, healthcare support, "
        "trades, office admin, food service dominate real local employment. (2) two life "
        "details adapted from the pool (rewrite freely) that fit this exact person's age, "
        "schedule, income, and area; never give someone a detail their anchors contradict."
        f"\nLIFE DETAIL POOL: {json.dumps(life_pool)}"
    )
    out = {}
    for i in range(0, len(people), batch):
        chunk = people[i:i + batch]
        skeletons = [{"id": p["id"], "age": p["age"], "area": p["area"],
                      "household_income": p["household_income"], "renter": p["renter"],
                      "household": p.get("household", ""),
                      "employment_status": p.get("employment_status", ""),
                      "education": p.get("education", ""), "student": p.get("student", False)}
                     for p in chunk]
        data = client.complete_json(system, json.dumps(skeletons), COHERENCE_SCHEMA, max_tokens=8000)
        for fixed in data["personas"]:
            out[fixed["id"]] = fixed
        print(f"coherence: {min(i + batch, len(people))}/{len(people)}")
    for p in people:
        f = out.get(p["id"])
        if f:
            p["work"] = f["occupation"]
            p.setdefault("household", "")
            p["life_details"] = f["life_details"][:2]
    return people


def lint_panel(people):
    """Flag residual implausible combos; returns count."""
    bad = 0
    for p in people:
        w = p["work"].lower()
        issues = []
        if p["age"] < 22 and ("nurse" in w and "cna" not in w and "aide" not in w):
            issues.append("underage RN")
        if p["age"] < 55 and "retired" in w:
            issues.append("early retiree")
        if p["age"] > 32 and "student" in w and "grad" not in w and not p.get("student"):
            issues.append("implausible student")
        # PUMS-derived households are real records; only synthetic ones get this check
        if not p.get("pums") and p["age"] < 24 and "with kids at home" in p.get("household", ""):
            issues.append("teen parent of older kids")
        if issues:
            bad += 1
            print(f"  LINT {p['id']}: {', '.join(issues)} — {p['age']}yo, {p['work']!r}, {p['household']!r}")
    return bad


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
    people = coherence_pass(client, people, life_details)
    print("lint:")
    bad = lint_panel(people)
    print(f"lint flagged {bad}/{len(people)}")

    out = {"archetypes": archetypes, "personas": people, "seed": SEED}
    os.makedirs("data", exist_ok=True)
    with open(OUT_PATH, "w") as f:
        json.dump(out, f, indent=1)
    print(f"wrote {len(people)} personas -> {OUT_PATH}")


if __name__ == "__main__":
    main()
