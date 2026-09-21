"""Aggregate replication runs into a stability report.

Synthetic respondents share one model, so errors correlate — a single run's
margin of error is fiction. What IS meaningful: does the answer hold across
independent replications (different option orders, fresh sampling)?

Usage: stability.py data/results/gyro-shop-rep*.json
Reports, per name option: mean per rep, top-2-box range, rank per rep, and
"wins N/N reps". Also aggregates MaxDiff best-worst scores per rep when present.
"""
import argparse
import json
import statistics as st
from collections import defaultdict


def name_stats(results):
    by = defaultdict(list)
    for r in results:
        for nr in r["answers"]["name_ratings"]:
            by[nr["option"]].append(nr["score"])
    return {opt: {"mean": st.mean(sc), "t2b": sum(1 for s in sc if s >= 8) / len(sc),
                  "rej": sum(1 for s in sc if s <= 4) / len(sc), "n": len(sc)}
            for opt, sc in by.items() if len(sc) >= 20}


def maxdiff_scores(results):
    """Best-worst score: (times most - times least) / times shown."""
    shown, most, least = defaultdict(int), defaultdict(int), defaultdict(int)
    for r in results:
        tasks = r.get("presentation", {}).get("maxdiff", [])
        answers = {a["task_id"]: a for a in r["answers"].get("maxdiff_answers", [])}
        for t, subset in enumerate(tasks):
            for a in subset:
                shown[a] += 1
            ans = answers.get(f"t{t + 1}")
            if not ans:
                continue
            if ans["most_important"] in subset:
                most[ans["most_important"]] += 1
            if ans["least_important"] in subset:
                least[ans["least_important"]] += 1
    return {a: {"bw": (most[a] - least[a]) / shown[a], "shown": shown[a],
                "most": most[a], "least": least[a]}
            for a in shown}


def choice_shares(results):
    shares = defaultdict(lambda: defaultdict(int))
    for r in results:
        for a in r["answers"].get("choice_answers", []):
            shares[a["scenario_id"]][a["choice"][:60]] += 1
    return shares


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("runs", nargs="+", help="results json files (one per rep)")
    args = ap.parse_args()

    reps = []
    for path in sorted(args.runs):
        d = json.load(open(path))
        reps.append({"path": path, "rep": d.get("rep", "?"), "stats": name_stats(d["results"]),
                     "maxdiff": maxdiff_scores(d["results"]),
                     "choices": choice_shares(d["results"]), "n": len(d["results"])})
    n_reps = len(reps)
    options = sorted({o for r in reps for o in r["stats"]},
                     key=lambda o: -st.mean(r["stats"][o]["mean"] for r in reps if o in r["stats"]))

    print(f"STABILITY REPORT — {n_reps} replication(s)\n")
    print(f"{'option':22s} {'mean by rep':>{n_reps * 6}s}  {'t2b range':>12s}  {'rank by rep':>{n_reps * 3}s}  wins")
    ranks_per_rep = []
    for r in reps:
        ranked = sorted(r["stats"], key=lambda o: -r["stats"][o]["mean"])
        ranks_per_rep.append({o: i + 1 for i, o in enumerate(ranked)})
    for o in options:
        means = " ".join(f"{r['stats'][o]['mean']:5.2f}" for r in reps if o in r["stats"])
        t2bs = [r["stats"][o]["t2b"] for r in reps if o in r["stats"]]
        t2b = f"{min(t2bs):.0%}-{max(t2bs):.0%}"
        ranks = " ".join(str(rr.get(o, "-")) for rr in ranks_per_rep)
        wins = sum(1 for rr in ranks_per_rep if rr.get(o) == 1)
        print(f"{o:22s} {means:>{n_reps * 6}s}  {t2b:>12s}  {ranks:>{n_reps * 3}s}  {wins}/{n_reps}")

    if any(r["maxdiff"] for r in reps):
        print("\nMAXDIFF best-worst scores (per rep):")
        attrs = sorted({a for r in reps for a in r["maxdiff"]},
                       key=lambda a: -st.mean(r["maxdiff"][a]["bw"] for r in reps if a in r["maxdiff"]))
        for a in attrs:
            scores = " ".join(f"{r['maxdiff'][a]['bw']:+.2f}" for r in reps if a in r["maxdiff"])
            print(f"  {a[:44]:46s} {scores}")

    if any(r["choices"] for r in reps):
        print("\nCHOICE SCENARIOS (share by rep):")
        sids = sorted({s for r in reps for s in r["choices"]})
        for sid in sids:
            print(f"  {sid}:")
            opts = sorted({o for r in reps for o in r["choices"].get(sid, {})},
                          key=lambda o: -sum(r["choices"].get(sid, {}).get(o, 0) for r in reps))
            for o in opts:
                shares = " ".join(f"{r['choices'].get(sid, {}).get(o, 0) / max(r['n'], 1):5.0%}"
                                  for r in reps)
                print(f"    {o:58s} {shares}")


if __name__ == "__main__":
    main()
