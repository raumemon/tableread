"""Run a concept test across the synthetic panel.

v3 instruments:
- Option ORDER is randomized per persona (seeded by persona id + rep), and the
  presentation order is recorded with each result. The concept description
  stays in the cached system prompt; the shuffled option lists travel in the
  persona message.
- --rep N tags a replication run: same personas, different shuffle seeds and
  fresh sampling. Aggregate reps with stability.py; report rank stability
  across reps, not fake margins of error.
- choice_scenarios in the concept yaml adds forced-choice situations
  ("It's 12:20pm, 30 minutes, here are your options... or don't eat out").
- maxdiff_attributes adds best/worst tradeoff tasks (5 tasks x 4 attributes
  per persona, seeded), scored with stability.py.

Usage:
  run_panel.py concepts/gyro-shop.yaml --limit 3        # smoke test
  run_panel.py concepts/gyro-shop.yaml --rep 1          # replication 1 of N
  run_panel.py concepts/gyro-shop.yaml --dry-run        # build prompts, no API
"""
import argparse
import json
import os
import random
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed

import yaml

from llm import LLM, MODEL

PERSONAS_PATH = "data/personas.json"
MAXDIFF_TASKS = 5
MAXDIFF_SET_SIZE = 4


def shuffled(options, rng):
    out = list(options)
    rng.shuffle(out)
    return out


def maxdiff_tasks(attributes, rng):
    """Seeded per persona: 5 sets of 4, biased toward least-shown attributes
    so coverage stays balanced."""
    shown = {a: 0 for a in attributes}
    tasks = []
    for t in range(MAXDIFF_TASKS):
        pool = sorted(attributes, key=lambda a: (shown[a], rng.random()))
        subset = pool[:MAXDIFF_SET_SIZE]
        rng.shuffle(subset)
        for a in subset:
            shown[a] += 1
        tasks.append(subset)
    return tasks


def survey_schema(concept, plan):
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
                         "minItems": len(plan["names"]), "maxItems": len(plan["names"])},
        "name_ranking": {"type": "array", "items": {"type": "string"}},
        "would_try_within_month": {"type": "string",
                                   "enum": ["definitely", "probably", "maybe", "unlikely", "no"]},
        "expected_price": {"type": "number",
                           "description": f"What you'd expect to pay for: {concept.get('price_probe', 'a typical order')}"},
        "biggest_turnoff": {"type": "string"},
        "one_change_suggestion": {"type": "string"},
    }
    if plan["slogans"]:
        props["slogan_ratings"] = {"type": "array", "items": variant_rating}
    if plan["packaging"]:
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
    if plan["choices"]:
        props["choice_answers"] = {
            "type": "array",
            "description": "One entry per scenario, in the order presented",
            "items": {
                "type": "object",
                "properties": {
                    "scenario_id": {"type": "string"},
                    "choice": {"type": "string",
                               "description": "EXACTLY one of that scenario's listed options"},
                    "why": {"type": "string"},
                },
                "required": ["scenario_id", "choice", "why"],
                "additionalProperties": False,
            },
        }
    if plan["maxdiff"]:
        props["maxdiff_answers"] = {
            "type": "array",
            "description": "One entry per task: pick the MOST and LEAST important attribute from that task's set only",
            "items": {
                "type": "object",
                "properties": {
                    "task_id": {"type": "string"},
                    "most_important": {"type": "string"},
                    "least_important": {"type": "string"},
                },
                "required": ["task_id", "most_important", "least_important"],
                "additionalProperties": False,
            },
        }
    return {"type": "object", "properties": props, "required": list(props),
            "additionalProperties": False}


def concept_system(concept):
    """Stable across all personas and reps -> prompt-cache friendly.
    Option lists are NOT here; they arrive shuffled per persona."""
    stable = {k: v for k, v in concept.items()
              if k not in ("notes", "name_options", "slogan_options", "packaging_options",
                           "choice_scenarios", "maxdiff_attributes")}
    return "\n".join([
        "You are role-playing a specific real resident of the Spokane WA / Coeur d'Alene ID area",
        "responding to a local market research survey about a new restaurant concept. Stay fully",
        "in character: react with this person's tastes, budget, skepticism, and local reference",
        "points. Not every persona is excited about new restaurants; honest indifference and",
        "criticism are more valuable than politeness. Judge names/slogans/packaging like a real",
        "person scrolling past them, not like a marketer. Importance and price answers must",
        "reflect the persona's actual budget, household, and eating habits. In forced-choice",
        "scenarios, choose like a real person under real constraints — 'none of these' is a",
        "legitimate answer when that's what this person would do. In most/least important",
        "tasks you MUST pick from that task's listed set only: tradeoffs, not diplomacy.",
        "Use the persona's own voice in free-text answers.",
        "",
        "THE CONCEPT BEING TESTED:",
        json.dumps(stable, indent=1),
    ])


def presentation_plan(concept, persona_id, rep):
    """Everything order-dependent, seeded by persona and rep."""
    rng = random.Random(f"{persona_id}|rep{rep}")
    plan = {
        "names": shuffled(concept["name_options"], rng),
        "slogans": shuffled(concept.get("slogan_options") or [], rng),
        "packaging": shuffled(concept.get("packaging_options") or [], rng),
        "choices": [],
        "maxdiff": [],
    }
    for i, sc in enumerate(shuffled(concept.get("choice_scenarios") or [], rng)):
        plan["choices"].append({"scenario_id": f"s{i + 1}", "prompt": sc["prompt"],
                                "options": shuffled(sc["options"], rng)})
    if concept.get("maxdiff_attributes"):
        plan["maxdiff"] = maxdiff_tasks(concept["maxdiff_attributes"], rng)
    return plan


def persona_prompt(p, arch, plan):
    fam = p.get("cuisine_familiarity", 5)
    fam_desc = ("you've basically never eaten this kind of food" if fam <= 2
                else "you've had this kind of food occasionally" if fam <= 5
                else "you know this kind of food well and have opinions")
    parts = [
        "YOUR PERSONA:",
        f"- Age {p['age']}, household income ~${p['household_income']:,}, "
        f"{'renter' if p['renter'] else 'homeowner'}, lives in {p['area']}",
        f"- Household: {p.get('household', 'n/a')}. Work: {p.get('work', 'n/a')}.",
        f"- Diet: {p.get('diet', 'none')}. Familiarity with this concept's cuisine: {fam_desc}.",
        f"- Life details that color your reactions: {'; '.join(p.get('life_details', []))}",
        f"- Dining archetype: {arch['name']} — {arch['description']} "
        f"Habits: {arch['dining_habits']} Price sensitivity: {arch['price_sensitivity']}.",
        f"- Voice notes: {arch['voice_notes']}",
        "- How locals like you actually write (real excerpts):",
    ]
    parts += [f"  > {q[:300]}" for q in p["voice_quotes"]]
    parts += ["", "NAME OPTIONS to rate (rate every one):"]
    parts += [f"  {i + 1}. {n}" for i, n in enumerate(plan["names"])]
    if plan["slogans"]:
        parts += ["SLOGAN OPTIONS to rate:"] + [f"  - {x}" for x in plan["slogans"]]
    if plan["packaging"]:
        parts += ["PACKAGING OPTIONS to rate:"] + [f"  - {x}" for x in plan["packaging"]]
    for sc in plan["choices"]:
        parts += [f"CHOICE SCENARIO {sc['scenario_id']}: {sc['prompt']}", "Your options:"]
        parts += [f"  - {o}" for o in sc["options"]]
    for t, subset in enumerate(plan["maxdiff"]):
        parts += [f"TRADEOFF TASK t{t + 1} — from ONLY these, pick the MOST important and the "
                  f"LEAST important to you personally when deciding whether a $10 visit is worth it:"]
        parts += [f"  - {a}" for a in subset]
    parts += ["", "Complete the survey. Rate every listed option and answer every task. Be "
              "specific about WHY — and let your own circumstances (household, schedule, diet, "
              "budget, familiarity) drive every answer, not a generic consumer's."]
    return "\n".join(parts)


def run_one(client, system, concept, p, arch, rep):
    plan = presentation_plan(concept, p["id"], rep)
    schema = survey_schema(concept, plan)
    answers = client.complete_json(system, persona_prompt(p, arch, plan), schema, max_tokens=3500)
    return {"persona_id": p["id"], "persona": p, "rep": rep,
            "presentation": {"names": plan["names"],
                             "maxdiff": [list(t) for t in plan["maxdiff"]],
                             "choices": [{"scenario_id": c["scenario_id"], "options": c["options"]}
                                         for c in plan["choices"]]},
            "answers": answers}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("concept", help="path to concept yaml")
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--workers", type=int, default=3)
    ap.add_argument("--rep", type=int, default=1, help="replication number (changes shuffle seeds)")
    ap.add_argument("--resume", default=None, help="previous results json to continue")
    ap.add_argument("--dry-run", action="store_true",
                    help="print one persona's constructed prompt + schema and exit (no API)")
    args = ap.parse_args()

    concept = yaml.safe_load(open(args.concept))
    panel = json.load(open(PERSONAS_PATH))
    archetypes = {a["key"]: a for a in panel["archetypes"]}
    personas = panel["personas"][: args.limit] if args.limit else panel["personas"]
    system = concept_system(concept)

    if args.dry_run:
        p = personas[0]
        plan = presentation_plan(concept, p["id"], args.rep)
        print("=== SYSTEM ===\n" + system)
        print("\n=== PERSONA PROMPT ===\n" + persona_prompt(p, archetypes[p["archetype"]], plan))
        print("\n=== SCHEMA KEYS ===", list(survey_schema(concept, plan)["properties"]))
        p2plan = presentation_plan(concept, personas[1]["id"], args.rep)
        print("\norder check p0 vs p1:", plan["names"], "|", p2plan["names"])
        return

    client = LLM()
    print(f"provider: {client.provider}, model: {MODEL}, rep {args.rep}")

    slug = os.path.splitext(os.path.basename(args.concept))[0]
    ckpt_path = f"data/results/.checkpoint-{slug}-rep{args.rep}.jsonl"
    ckpt_lock = threading.Lock()

    done = {}
    if args.resume:
        prev = json.load(open(args.resume))
        done.update({r["persona_id"]: r for r in prev["results"]})
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

    os.makedirs("data/results", exist_ok=True)
    with ThreadPoolExecutor(max_workers=args.workers) as ex:
        futs = {ex.submit(run_one, client, system, concept, p,
                          archetypes[p["archetype"]], args.rep): p for p in personas}
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
    out_path = f"data/results/{slug}-rep{args.rep}-{time.strftime('%Y%m%d-%H%M%S')}.json"
    with open(out_path, "w") as f:
        json.dump({"concept": concept, "rep": args.rep, "results": results, "errors": errors,
                   "model": MODEL, "est_cost_usd": round(cost, 2)}, f, indent=1)
    print(f"wrote {out_path}")
    if not errors and os.path.exists(ckpt_path):
        os.remove(ckpt_path)


if __name__ == "__main__":
    main()
