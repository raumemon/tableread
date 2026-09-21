"""The Price of Certainty — full variant battery on the panel.

A procedure guarantees a certain answer to whether God exists; the price is
permanent blindness. Baseline first, five variants shuffled per persona,
max-price ladder last. Output includes a falsifiable prediction sheet for the
human-sample comparison run.

Usage: probe_certainty.py [--limit N]
Output: data/results/certainty-<ts>.json + printed analysis + predictions
"""
import argparse
import glob
import json
import os
import random
import threading
import time
from collections import Counter, defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed

from llm import LLM, MODEL

ACCEPT = ["definitely accept", "probably accept", "probably refuse", "definitely refuse"]
PRICES = ["nothing — I wouldn't pay any real price", "a finger", "my sense of taste",
          "ten years of my life", "my eyesight", "my remaining lifespan — I'd die knowing"]

VARIANTS = {
    "one_sided": ("The procedure only works one way: IF God exists you'll know with certainty; "
                  "if God doesn't exist you learn nothing (and still go blind). Accept?"),
    "unprovable": ("Same as the original deal, but you alone will know. You can never prove it "
                   "to anyone; everyone will think your certainty is just faith or delusion. Accept?"),
    "fading": ("Same as the original deal, but the knowledge fades completely after one year — "
               "you'll remember only that you once knew. The blindness is permanent. Accept?"),
    "ten_years": ("Same certain answer about God, but the price is ten years off the end of "
                  "your life instead of your sight. Accept?"),
    "afterlife": ("The question answered is instead: 'Is there an afterlife?' — with certainty, "
                  "at the price of your sight. Accept?"),
}

BASELINE = ("You are offered a procedure with two guarantees. First, it will tell you with "
            "absolute certainty whether God exists — if God exists you will know it, if God "
            "does not exist you will know that. No deception or error is possible. Second, "
            "immediately afterward, you will permanently lose your sight. You gain nothing "
            "else: no heaven, money, power, or longer life — only the answer. Do you accept?")


def schema(variant_order):
    answer = {"type": "object", "properties": {
        "variant": {"type": "string"},
        "answer": {"type": "string", "enum": ACCEPT},
        "reasoning": {"type": "string"},
    }, "required": ["variant", "answer", "reasoning"], "additionalProperties": False}
    return {"type": "object", "properties": {
        "baseline_answer": {"type": "string", "enum": ACCEPT},
        "baseline_reasoning": {"type": "string",
                               "description": "2-4 sentences in your own voice; what actually moves your decision"},
        "current_belief": {"type": "string",
                           "enum": ["believe God exists", "lean toward existing", "genuinely unsure",
                                    "lean toward not existing", "believe God does not exist"]},
        "variant_answers": {"type": "array", "minItems": len(variant_order), "maxItems": len(variant_order),
                            "description": f"One per variant, in this order: {variant_order}",
                            "items": answer},
        "max_price": {"type": "string", "enum": PRICES,
                      "description": "The MAXIMUM price you would personally pay for the certain answer"},
        "max_price_reasoning": {"type": "string"},
    }, "required": ["baseline_answer", "baseline_reasoning", "current_belief",
                    "variant_answers", "max_price", "max_price_reasoning"],
        "additionalProperties": False}


SYSTEM = (
    "You are role-playing a specific real resident of the Spokane WA / Coeur d'Alene ID area "
    "answering a philosophical survey honestly, in character. This is not about performing "
    "depth: answer the way THIS person actually would — some people refuse instantly and "
    "think the question is silly, some can't imagine trading their sight for anything, some "
    "have wanted this answer their whole life. Let their faith or lack of it, their age, "
    "their family situation, and their practical circumstances drive the answer. Plain "
    "spoken language, not philosophy-seminar language. Knowledge boundary: reference "
    "only what this person's actual life would expose them to."
)


def prompt(p, arch, variant_order):
    lines = [
        f"YOU: age {p['age']}, {p['area']}, {p.get('household')}, {p.get('work')}. ",
        f"Religion: {p['religion']}; faith is {p['faith_importance']} in your life.",
        f"Life details: {'; '.join(p.get('life_details', []))}",
        "",
        "THE OFFER:", BASELINE,
        "",
        "Answer the baseline, then each variant below IN ORDER, then the maximum-price question.",
    ]
    for key in variant_order:
        lines.append(f"VARIANT {key}: {VARIANTS[key]}")
    lines.append("MAX PRICE: Of these — nothing, a finger, your sense of taste, ten years of "
                 "life, your eyesight, your remaining lifespan — what is the MOST you would "
                 "actually pay for the certain answer?")
    return "\n".join(lines)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--workers", type=int, default=3)
    args = ap.parse_args()

    client = LLM()
    print(f"provider: {client.provider}, model: {MODEL}")
    panel = json.load(open("data/personas.json"))
    arch = {a["key"]: a for a in panel["archetypes"]}
    personas = panel["personas"][: args.limit] if args.limit else panel["personas"]

    def run_one(p):
        rng = random.Random(f"certainty:{p['id']}")
        order = list(VARIANTS)
        rng.shuffle(order)
        ans = client.complete_json(SYSTEM, prompt(p, arch[p["archetype"]], order),
                                   schema(order), max_tokens=1800)
        return {"persona_id": p["id"], "persona": p, "variant_order": order, "answers": ans}

    # resume from newest prior run + kill-safe checkpoint
    ckpt = "data/results/.checkpoint-certainty.jsonl"
    lock = threading.Lock()
    done = {}
    for f in sorted(glob.glob("data/results/certainty-*.json")):
        for r in json.load(open(f))["results"]:
            done[r["persona_id"]] = r
    if os.path.exists(ckpt):
        for line in open(ckpt):
            r = json.loads(line)
            done[r["persona_id"]] = r
    results, errors = list(done.values()), []
    if done:
        personas = [p for p in personas if p["id"] not in done]
        print(f"resuming: {len(results)} done, {len(personas)} to run", flush=True)

    def checkpoint(r):
        with lock:
            with open(ckpt, "a") as f:
                f.write(json.dumps(r) + "\n")

    with ThreadPoolExecutor(max_workers=args.workers) as ex:
        futs = {ex.submit(run_one, p): p for p in personas}
        for fut in as_completed(futs):
            try:
                r = fut.result()
                checkpoint(r)
                results.append(r)
                if len(results) % 25 == 0:
                    print(f"{len(results)}/300", flush=True)
            except Exception as e:
                errors.append(str(e))
                print("FAIL:", e, flush=True)

    out = f"data/results/certainty-{time.strftime('%Y%m%d-%H%M%S')}.json"
    with open(out, "w") as f:
        json.dump({"results": results, "errors": errors, "model": MODEL,
                   "est_cost_usd": round(client.cost(), 2)}, f, indent=1)
    print(f"\nwrote {out} — est ${client.cost():.2f}")
    if not errors and os.path.exists(ckpt):
        os.remove(ckpt)

    # ---------- analysis ----------
    n = len(results)
    acc = lambda a: a in ("definitely accept", "probably accept")
    base = Counter(r["answers"]["baseline_answer"] for r in results)
    print(f"\nBASELINE (sight for certainty), n={n}:")
    for k in ACCEPT:
        print(f"  {k:20s} {base.get(k, 0) / n:6.1%}")
    base_acc = sum(1 for r in results if acc(r["answers"]["baseline_answer"])) / n
    print(f"  -> ACCEPT overall: {base_acc:.1%}")

    print("\nACCEPT rate by faith importance:")
    seg = defaultdict(list)
    for r in results:
        seg[r["persona"]["faith_importance"]].append(acc(r["answers"]["baseline_answer"]))
    for k, v in sorted(seg.items()):
        print(f"  {k:20s} {sum(v) / len(v):6.1%}  (n={len(v)})")
    print("ACCEPT rate by current belief:")
    seg = defaultdict(list)
    for r in results:
        seg[r["answers"]["current_belief"]].append(acc(r["answers"]["baseline_answer"]))
    for k, v in sorted(seg.items(), key=lambda kv: -len(kv[1])):
        print(f"  {k:28s} {sum(v) / len(v):6.1%}  (n={len(v)})")

    print("\nVARIANTS (accept rate vs baseline):")
    va = defaultdict(list)
    for r in results:
        for a in r["answers"]["variant_answers"]:
            va[a["variant"]].append(acc(a["answer"]))
    for k in VARIANTS:
        vals = va.get(k, [])
        if vals:
            print(f"  {k:12s} {sum(vals) / len(vals):6.1%}   (baseline {base_acc:.1%})")

    print("\nMAX PRICE distribution:")
    mp = Counter(r["answers"]["max_price"] for r in results)
    for k in PRICES:
        print(f"  {k:44s} {mp.get(k, 0) / n:6.1%}")


if __name__ == "__main__":
    main()
