"""Run a concept test across the synthetic panel.

Each persona answers a structured survey about the concept's branding variants
(names, slogans, packaging). The concept + survey framing lives in the system
prompt under a cache breakpoint, so 100 calls pay for it roughly once.

Usage:
  run_panel.py concepts/gyro-shop.yaml            # full panel
  run_panel.py concepts/gyro-shop.yaml --limit 3  # smoke test
"""
import argparse
import json
import os
import time
from concurrent.futures import ThreadPoolExecutor, as_completed

import anthropic
import yaml

MODEL = "claude-sonnet-5"
PERSONAS_PATH = "data/personas.json"


def load_env():
    if os.path.exists(".env"):
        for line in open(".env"):
            if "=" in line and not line.startswith("#"):
                k, v = line.strip().split("=", 1)
                os.environ.setdefault(k, v)


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
    return {
        "type": "object",
        "properties": {
            "first_impression": {"type": "string"},
            "name_ratings": {"type": "array", "items": variant_rating,
                             "minItems": n_names, "maxItems": n_names},
            "name_ranking": {"type": "array", "items": {"type": "string"}},
            "slogan_ratings": {"type": "array", "items": variant_rating},
            "packaging_ratings": {"type": "array", "items": variant_rating},
            "would_try_within_month": {"type": "string",
                                       "enum": ["definitely", "probably", "maybe", "unlikely", "no"]},
            "expected_price": {"type": "number",
                               "description": f"What you'd expect to pay for: {concept.get('price_probe', 'a typical order')}"},
            "biggest_turnoff": {"type": "string"},
            "one_change_suggestion": {"type": "string"},
        },
        "required": ["first_impression", "name_ratings", "name_ranking", "slogan_ratings",
                     "packaging_ratings", "would_try_within_month",
                     "expected_price", "biggest_turnoff", "one_change_suggestion"],
        "additionalProperties": False,
    }


def concept_system(concept, archetypes_by_key):
    lines = [
        "You are role-playing a specific real resident of the Spokane WA / Coeur d'Alene ID area",
        "responding to a local market research survey about a new restaurant concept. Stay fully",
        "in character: react with this person's tastes, budget, skepticism, and local reference",
        "points. Not every persona is excited about new restaurants; honest indifference and",
        "criticism are more valuable than politeness. Judge names/slogans/packaging like a real",
        "person scrolling past them, not like a marketer. Use the persona's own voice in free-text",
        "answers, informed by how locals actually write (voice quotes provided).",
        "",
        "THE CONCEPT BEING TESTED:",
        json.dumps({k: v for k, v in concept.items() if k != "notes"}, indent=1),
    ]
    return [{"type": "text", "text": "\n".join(lines), "cache_control": {"type": "ephemeral"}}]


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
          "Rate every listed option. Be specific about WHY in each reaction — and let your "
          "own circumstances (household, schedule, diet, familiarity) drive the answer, not "
          "a generic consumer's."
    )


def run_one(client, system, schema, p, arch):
    response = client.messages.create(
        model=MODEL,
        max_tokens=3000,
        system=system,
        messages=[{"role": "user", "content": persona_prompt(p, arch)}],
        output_config={"format": {"type": "json_schema", "schema": schema}},
    )
    text = next(b.text for b in response.content if b.type == "text")
    return {"persona_id": p["id"], "persona": p, "answers": json.loads(text),
            "usage": {"in": response.usage.input_tokens, "out": response.usage.output_tokens,
                      "cache_read": response.usage.cache_read_input_tokens or 0,
                      "cache_write": response.usage.cache_creation_input_tokens or 0}}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("concept", help="path to concept yaml")
    ap.add_argument("--limit", type=int, default=None, help="only run first N personas (smoke test)")
    ap.add_argument("--workers", type=int, default=6)
    args = ap.parse_args()

    load_env()
    client = anthropic.Anthropic()
    concept = yaml.safe_load(open(args.concept))
    panel = json.load(open(PERSONAS_PATH))
    archetypes = {a["key"]: a for a in panel["archetypes"]}
    personas = panel["personas"][: args.limit] if args.limit else panel["personas"]

    system = concept_system(concept, archetypes)
    schema = survey_schema(concept)

    # Warm the cache with one sequential call, then fan out.
    results, errors = [], []
    first = run_one(client, system, schema, personas[0], archetypes[personas[0]["archetype"]])
    results.append(first)
    print(f"1/{len(personas)} (cache write {first['usage']['cache_write']} tokens)")

    with ThreadPoolExecutor(max_workers=args.workers) as ex:
        futs = {ex.submit(run_one, client, system, schema, p, archetypes[p["archetype"]]): p
                for p in personas[1:]}
        for fut in as_completed(futs):
            p = futs[fut]
            try:
                results.append(fut.result())
                print(f"{len(results)}/{len(personas)}")
            except Exception as e:
                errors.append({"persona_id": p["id"], "error": str(e)})
                print(f"FAIL {p['id']}: {e}")

    tot_in = sum(r["usage"]["in"] for r in results)
    tot_out = sum(r["usage"]["out"] for r in results)
    tot_cached = sum(r["usage"]["cache_read"] for r in results)
    cost = tot_in / 1e6 * 2.00 + tot_out / 1e6 * 10.00 + tot_cached / 1e6 * 0.20
    print(f"\ntokens: {tot_in} in, {tot_cached} cached-read, {tot_out} out — est ${cost:.2f}")

    os.makedirs("data/results", exist_ok=True)
    slug = os.path.splitext(os.path.basename(args.concept))[0]
    out_path = f"data/results/{slug}-{time.strftime('%Y%m%d-%H%M%S')}.json"
    with open(out_path, "w") as f:
        json.dump({"concept": concept, "results": results, "errors": errors,
                   "model": MODEL, "est_cost_usd": round(cost, 2)}, f, indent=1)
    print(f"wrote {out_path}")


if __name__ == "__main__":
    main()
