"""Targeted follow-up probe: visible spit vs. fresh-carved-in-back-kitchen.

Panel subset: every persona whose round-1 verbatims hit authenticity-skeptic
language (heat lamp / frozen / cone / fresh-carved) + a random 50 for base
rates. Output: data/results/probe-spit.json + printed aggregate.
"""
import json
import random
import re
from collections import Counter
from concurrent.futures import ThreadPoolExecutor, as_completed

from llm import LLM, MODEL

RUN = "data/results/gyro-shop-20260920-181023.json"
SKEPTIC_RE = re.compile(r"heat lamp|frozen|cone|fresh.?carved|pressed meat", re.I)

SCHEMA = {
    "type": "object",
    "properties": {
        "visible_spit_importance": {
            "type": "integer",
            "description": "1-10: how much does physically SEEING the spit/carving matter to you, 1=irrelevant, 10=essential"},
        "back_kitchen_verdict": {
            "type": "string",
            "enum": ["fine_no_proof_needed", "fine_with_other_proof", "meaningful_downgrade", "dealbreaker"],
            "description": "If the meat IS fresh-carved off a real spit, but the kitchen is closed so you never see it"},
        "trust_substitutes": {
            "type": "array", "maxItems": 3,
            "items": {"type": "string", "enum": [
                "carved_to_order_language_on_menu",
                "pass_window_or_glimpse_of_kitchen",
                "photos_or_video_screen_of_the_spit",
                "watching_my_wrap_assembled_fresh",
                "taste_and_food_quality_itself",
                "reviews_and_word_of_mouth",
                "owner_story_and_transparency",
                "nothing_substitutes_for_seeing_it"]},
            "description": "What would most make you believe the fresh-carved claim without seeing the spit (up to 3, most important first)"},
        "comment": {"type": "string", "description": "1-2 sentences in your own voice"},
    },
    "required": ["visible_spit_importance", "back_kitchen_verdict", "trust_substitutes", "comment"],
    "additionalProperties": False,
}


def main():
    client = LLM()
    print(f"provider: {client.provider}, model: {MODEL}")
    run = json.load(open(RUN))
    panel = json.load(open("data/personas.json"))
    arch = {a["key"]: a for a in panel["archetypes"]}

    skeptics, others = [], []
    for r in run["results"]:
        text = json.dumps(r["answers"])
        (skeptics if SKEPTIC_RE.search(text) else others).append(r)
    rng = random.Random(7)
    sample = skeptics + rng.sample(others, min(50, len(others)))
    print(f"probing {len(sample)} personas ({len(skeptics)} authenticity-skeptics + {len(sample) - len(skeptics)} baseline)")

    system = (
        "You are role-playing a specific Spokane/CDA resident who already took a survey about "
        "a new fast-casual gyro/kebab shop (fresh-carved meat, house sauces, pitas made daily, "
        "open late weekends, $13-16 combos). Answer this follow-up in character, consistent "
        "with your survey stance (provided). Be honest: if you would not notice or care, say so.\n\n"
        "SCENARIO: The location under consideration does NOT have an open kitchen. The meat is "
        "genuinely carved to order off a real vertical spit, but the spit is in the back — "
        "customers never see it."
    )

    def probe(r):
        p = r["persona"]
        a = arch[p["archetype"]]
        stance = json.dumps({k: r["answers"].get(k) for k in ("first_impression", "biggest_turnoff")})
        user = (f"YOU: {p['age']}yo in {p['area']}, ~${p['household_income']:,}, {p.get('household')}, "
                f"{p.get('work')}. Archetype: {a['name']} — {a['description']}\n"
                f"Your earlier survey answers: {stance}\n\nAnswer the follow-up questions.")
        ans = client.complete_json(system, user, SCHEMA, max_tokens=600)
        return {"persona_id": p["id"], "persona": p, "answers": ans}

    results = []
    with ThreadPoolExecutor(max_workers=3) as ex:
        futs = [ex.submit(probe, r) for r in sample]
        for f in as_completed(futs):
            try:
                results.append(f.result())
            except Exception as e:
                print("FAIL:", e)

    imp = [r["answers"]["visible_spit_importance"] for r in results]
    verdicts = Counter(r["answers"]["back_kitchen_verdict"] for r in results)
    subs = Counter()
    for r in results:
        for s in r["answers"]["trust_substitutes"]:
            subs[s] += 1
    sk_ids = {r["persona"]["id"] for r in sample[: len(skeptics)]}
    sk_verdicts = Counter(r["answers"]["back_kitchen_verdict"] for r in results if r["persona_id"] in sk_ids)

    import statistics as st
    print(f"\nvisible-spit importance: mean {st.mean(imp):.1f}, median {st.median(imp)}")
    print("back-kitchen verdict (all):", dict(verdicts))
    print("back-kitchen verdict (skeptics only):", dict(sk_verdicts))
    print("top trust substitutes:")
    for s, n in subs.most_common():
        print(f"  {n:3d}  {s}")
    print(f"\ncost: ${client.cost():.2f}")

    with open("data/results/probe-spit.json", "w") as f:
        json.dump({"results": results, "skeptic_ids": sorted(sk_ids)}, f, indent=1)
    print("wrote data/results/probe-spit.json")


if __name__ == "__main__":
    main()
