"""Panel composition vs. Census — the 'is this panel representative' screen.

Compares the sampled panel's marginals against the full weighted PUMS adult
population for the corridor. Prints the table and writes
data/reference/panel_vs_census.json for report embedding.
"""
import json
from collections import defaultdict

PUMS = "data/reference/pums_records.json"
PANEL = "data/personas.json"


def band_age(a):
    return "21-34" if a < 35 else "35-54" if a < 55 else "55-64" if a < 65 else "65+"


def band_income(i):
    return "<$35k" if i < 35000 else "$35-75k" if i < 75000 else "$75-125k" if i < 125000 else "$125k+"


def emp_of_pums(rec):
    e = rec["employment"]
    if e in ("1", "2"):
        return "employed"
    if e == "3":
        return "unemployed"
    if e in ("4", "5"):
        return "military"
    return "retired/not in labor force"


def emp_of_panel(p):
    s = p.get("employment_status", "")
    if s == "employed":
        return "employed"
    if s.startswith("unemployed"):
        return "unemployed"
    if s == "military":
        return "military"
    return "retired/not in labor force"


def shares(pairs):
    tot = sum(w for _, w in pairs)
    out = defaultdict(float)
    for k, w in pairs:
        out[k] += w / tot
    return dict(out)


def main():
    pums = json.load(open(PUMS))
    panel = json.load(open(PANEL))["personas"]

    dims = {
        "age": (lambda r: band_age(r["age"]), lambda p: band_age(p["age"])),
        "household income": (lambda r: band_income(r["household_income"]),
                             lambda p: band_income(p["household_income"])),
        "area": (lambda r: r["area"], lambda p: p["area"]),
        "tenure": (lambda r: "renter" if r["renter"] else "owner",
                   lambda p: "renter" if p["renter"] else "owner"),
        "employment": (emp_of_pums, emp_of_panel),
    }

    report = {}
    for dim, (f_pums, f_panel) in dims.items():
        census = shares([(f_pums(r), r["weight"]) for r in pums])
        pan = shares([(f_panel(p), 1) for p in panel])
        keys = sorted(set(census) | set(pan))
        report[dim] = {k: {"census": round(census.get(k, 0), 3),
                           "panel": round(pan.get(k, 0), 3)} for k in keys}
        print(f"\n{dim.upper():18s} {'census':>8s} {'panel':>8s} {'diff':>7s}")
        for k in keys:
            c, p_ = census.get(k, 0), pan.get(k, 0)
            flag = "  <-- check" if abs(c - p_) > 0.05 else ""
            print(f"  {k:16s} {c:8.1%} {p_:8.1%} {p_ - c:+7.1%}{flag}")

    with open("data/reference/panel_vs_census.json", "w") as f:
        json.dump(report, f, indent=1)
    print("\nwrote data/reference/panel_vs_census.json")


if __name__ == "__main__":
    main()
