"""Run a concept test across the synthetic panel.

Each persona answers a structured survey about the concept's branding variants
(names, slogans, packaging) plus any concept-specific questions:
  importance_questions:  list of features rated 1-10 for importance, with comment
  habit_price_questions: asks the price at which they'd eat here >1x/week and >1x/month

Usage:
  run_panel.py concepts/gyro-shop.yaml            # full panel
  run_panel.py concepts/gyro-shop.yaml --limit 3  # smoke test
"""
import argparse
import json
import os
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed

import yaml

from llm import LLM, MODEL

PERSONAS_PATH = "data/personas.json"


def survey_schema(concept):
    n_names = len(concept["name_options"])
    variant_rating = {
        "type": "object",
        "properties": {
            "option": {"type": "string"},
            "score": {"type": "integer", "minimum": 1, "maximum": 10},
            "reaction": {"type": "string"},
        },
        "required": ["option", "score", "reaction"],
        "additionalProperties": False,
    }
    props = {
        "first_impression": {"type": "string"},
        "name_ratings": {"type": "array", "items": variant_rating,
                         "minItems": n_names, "maxItems": n_names},
        "name_ranking": {"type": "array", "items": {"type": "string"}},
        "would_try_within_month": {"type": "string",
                                   "enum": ["definitely", "probably", "maybe", "unlikely", "no"]},
        "expected_price": {"type": "number",
                           "description": f"What you'd expect to pay for: {concept.get('price_probe', 'a typical order')}"},
        "biggest_turnoff": {"type": "string"},
        "one_change_suggestion": {"type": "string"},
    }
    if concept.get("slogan_options"):
        props["slogan_ratings"] = {"type": "array", "items": variant_rating}
    if concept.get("packaging_options"):
        props["packaging_ratings"] = {"type": "array", "items": variant_rating}
    if concept.get("importance_questions"):
        props["importance_ratings"] = {
            "type": "array",
            "description": "One entry per listed feature: how much it matters TO YOU (1=don't care, 10=dealbreaker), with a comment in your voice",
            "items": {
                "type": "object",
                "properties": {
                    "item": {"type": "string"},
                    "score": {"type": "integer", "minimum": 1, "maximum": 10},
                    "comment": {"type": "string"},
                },
                "required": ["item", "score", "comment"],
                "additionalProperties": False,
            },
        }
    if concept.get("habit_price_questions"):
        props["price_for_weekly_habit"] = {
            "type": "number",
            "description": ("The per-order price at which you would REALISTICALLY eat here more "
                            "than once a week, given your actual budget and habits. If no price "
                            "would make you a weekly regular, answer 0.")}
        props["price_for_monthly_habit"] = {
            "type": "number",
            "description": ("The per-order price at which you'd eat here more than once a month. "
                            "If you'd basically never come back regardless of price, answer 0.")}
    return {"type": "object", "properties": props, "required": list(props),
            "additionalProperties": False}


def concept_system(concept):
    return "\n".join([
        "You are role-playing a specific real resident of the Spokane WA / Coeur d'Alene ID area",
        "responding to a local market research survey about a new restaurant concept. Stay fully",
        "in character: react with this person's tastes, budget, skepticism, and local reference",
        "points. Not every persona is excited about new restaurants; honest indifference and",
        "criticism are more valuable than politeness. Judge names/slogans/packaging like a real",
        "person scrolling past them, not like a marketer. Importance and price answers must",
        "reflect the persona's actual budget, household, and eating habits — a parent of three",
        "on $45k answers differently than a single foodie on $95k. Use the persona's own voice",
        "in free-text answers, informed by how locals actually write (voice quotes provided).",
        "",
        "THE CONCEPT BEING TESTED:",
        json.dumps({k: v for k, v in concept.items() if k != "notes"}, indent=1),
    ])


def persona_prompt(p, arch):
    fam = p.get("cuisine_familiarity", 5)
    fam_desc = ("you've basically never eaten this kind of food" if fam <= 2
                else "you've had this kind of food occasionally" if fam <= 5
                else "you know this kind of food well and have opinions")
    return (
        f"YOUR PERSONA:\n"
        f"- Age {p['age']}, household income ~${p['household_income']:,}, "
        f"{'renter' if p['renter'] else 'homeowner'}, lives in {p['area']}\n"
        f"- Household: {p.get('household', 'n/a')}. Work: {p.get('work', 'n/a')}.\n"
        f"- Diet: {p.get('diet', 'none')}. Familiarity with this concept's cuisine: {fam_desc}.\n"
        f"- Life details that color your reactions: {'; '.join(p.get('life_details', []))}\n"
        f"- Dining archetype: {arch['name']} — {arch['description']} "
        f"Habits: {arch['dining_habits']} Price sensitivity: {arch['price_sensitivity']}.\n"
        f"- Voice notes: {arch['voice_notes']}\n"
        f"- How locals like you actually write (real excerpts):\n"
        + "\n".join(f"  > {q[:300]}" for q in p["voice_quotes"])
        + "\n\nComplete the survey about the concept described in your instructions. "
          "Rate every listed option and every listed feature. Be specific about WHY in each "
          "reaction — and let your own circumstances (household, schedule, diet, familiarity) "
          "drive the answer, not a generic consumer's."
    )


def run_one(client, system, schema, p, arch):
    answers = client.complete_json(system, persona_prompt(p, arch), schema, max_tokens=3000)
    return {"persona_id": p["id"], "persona": p, "answers": answers}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("concept", help="path to concept yaml")
    ap.add_argument("--limit", type=int, default=None, help="only run first N personas (smoke test)")
    ap.add_argument("--workers", type=int, default=3)
    ap.add_argument("--resume", default=None,
                    help="previous results json; reuses its completed personas, runs only the rest")
    args = ap.parse_args()

    client = LLM()
    print(f"provider: {client.provider}, model: {MODEL}")
    concept = yaml.safe_load(open(args.concept))
    panel = json.load(open(PERSONAS_PATH))
    archetypes = {a["key"]: a for a in panel["archetypes"]}
    personas = panel["personas"][: args.limit] if args.limit else panel["personas"]

    system = concept_system(concept)
    schema = survey_schema(concept)

    slug = os.path.splitext(os.path.basename(args.concept))[0]
    ckpt_path = f"data/results/.checkpoint-{slug}.jsonl"
    ckpt_lock = threading.Lock()

    done = {}
    if args.resume:
        prev = json.load(open(args.resume))
        done.update({r["persona_id"]: r for r in prev["results"]})
    # Checkpoint survives crashes/kills: every completed survey is a line here.
    if os.path.exists(ckpt_path):
        for line in open(ckpt_path):
            r = json.loads(line)
            done[r["persona_id"]] = r
    results, errors = list(done.values()), []
    if done:
        personas = [p for p in personas if p["id"] not in done]
        print(f"resuming: {len(results)} already done, {len(personas)} to run", flush=True)

    def checkpoint(r):
        with ckpt_lock:
            with open(ckpt_path, "a") as f:
                f.write(json.dumps(r) + "\n")
    with ThreadPoolExecutor(max_workers=args.workers) as ex:
        futs = {ex.submit(run_one, client, system, schema, p, archetypes[p["archetype"]]): p
                for p in personas}
        total = len(done) + len(personas)
        for fut in as_completed(futs):
            p = futs[fut]
            try:
                r = fut.result()
                checkpoint(r)
                results.append(r)
                if len(results) % 10 == 0 or len(results) == total:
                    print(f"{len(results)}/{total}", flush=True)
            except Exception as e:
                errors.append({"persona_id": p["id"], "error": str(e)})
                print(f"FAIL {p['id']}: {e}", flush=True)

    results.sort(key=lambda r: r["persona_id"])
    cost = client.cost()
    print(f"\nusage: {client.usage} — est ${cost:.2f}")

    os.makedirs("data/results", exist_ok=True)
    out_path = f"data/results/{slug}-{time.strftime('%Y%m%d-%H%M%S')}.json"
    with open(out_path, "w") as f:
        json.dump({"concept": concept, "results": results, "errors": errors,
                   "model": MODEL, "est_cost_usd": round(cost, 2)}, f, indent=1)
    print(f"wrote {out_path}")
    if not errors and os.path.exists(ckpt_path):
        os.remove(ckpt_path)


if __name__ == "__main__":
    main()
