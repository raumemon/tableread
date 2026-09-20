"""Aggregate a panel run into a single HTML report.

Usage: build_report.py data/results/<run>.json   -> web/report-<slug>.html
"""
import argparse
import html
import json
import os
import statistics as stats
from collections import Counter, defaultdict

INTENT_WEIGHT = {"definitely": 1.0, "probably": 0.7, "maybe": 0.4, "unlikely": 0.1, "no": 0.0}


def agg_variants(results, field):
    by_opt = defaultdict(list)
    reactions = defaultdict(list)
    for r in results:
        for rating in r["answers"].get(field, []):
            by_opt[rating["option"]].append(rating["score"])
            reactions[rating["option"]].append((rating["score"], rating["reaction"], r["persona"]))
    rows = []
    for opt, scores in by_opt.items():
        rows.append({
            "option": opt, "mean": stats.mean(scores), "n": len(scores),
            "top2box": sum(1 for s in scores if s >= 8) / len(scores),
            "bottom": sum(1 for s in scores if s <= 4) / len(scores),
            "reactions": sorted(reactions[opt], key=lambda x: x[0]),
        })
    rows.sort(key=lambda r: -r["mean"])
    return rows


def seg_label(p):
    age = "under 35" if p["age"] < 35 else ("35-54" if p["age"] < 55 else "55+")
    inc = "<$50k" if p["household_income"] < 50000 else ("$50-100k" if p["household_income"] < 100000 else "$100k+")
    return age, inc


def segment_table(results, field):
    """Mean score of the winning option per segment."""
    seg = defaultdict(lambda: defaultdict(list))
    for r in results:
        age, inc = seg_label(r["persona"])
        arch = r["persona"]["archetype"]
        for rating in r["answers"].get(field, []):
            for key in (f"age {age}", f"income {inc}", f"type: {arch}", f"area: {r['persona']['area']}"):
                seg[key][rating["option"]].append(rating["score"])
    return {k: {o: round(stats.mean(v), 1) for o, v in opts.items()} for k, opts in seg.items()}


def bar(pct, color="#4a7c59"):
    return (f'<div style="background:#eee;border-radius:4px;height:14px;width:220px;display:inline-block;'
            f'vertical-align:middle"><div style="background:{color};height:14px;border-radius:4px;'
            f'width:{pct * 100:.0f}%"></div></div>')


def variant_section(title, rows):
    if not rows:
        return ""
    out = [f"<h2>{html.escape(title)}</h2><table><tr><th>Option</th><th>Mean /10</th>"
           "<th>Top-2-box (8+)</th><th>Weak (&le;4)</th></tr>"]
    for r in rows:
        out.append(f"<tr><td><b>{html.escape(r['option'])}</b></td>"
                   f"<td>{r['mean']:.1f} {bar(r['mean'] / 10)}</td>"
                   f"<td>{r['top2box']:.0%}</td><td>{r['bottom']:.0%}</td></tr>")
    out.append("</table>")
    winner = rows[0]
    out.append(f"<h3>Voices on the winner: {html.escape(winner['option'])}</h3><div class=quotes>")
    picks = winner["reactions"][-3:] + winner["reactions"][:2]  # 3 highest + 2 lowest
    for score, reaction, p in picks:
        out.append(f'<blockquote>&ldquo;{html.escape(reaction)}&rdquo;'
                   f'<footer>{score}/10 — {p["age"]}yo, {html.escape(p["area"])}, '
                   f'~${p["household_income"] // 1000}k, {html.escape(p["archetype"])}</footer></blockquote>')
    out.append("</div>")
    return "\n".join(out)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("run", help="path to results json")
    args = ap.parse_args()
    data = json.load(open(args.run))
    results, concept = data["results"], data["concept"]
    n = len(results)

    intent = Counter(r["answers"]["would_try_within_month"] for r in results)
    intent_score = sum(INTENT_WEIGHT[k] * v for k, v in intent.items()) / n
    prices = [r["answers"]["expected_price"] for r in results if r["answers"].get("expected_price")]
    turnoffs = [(r["answers"]["biggest_turnoff"], r["persona"]) for r in results]
    suggestions = Counter()
    for r in results:
        suggestions[r["answers"]["one_change_suggestion"][:120]] += 1

    body = [f"<h1>{html.escape(concept.get('working_title', 'Concept test'))}</h1>",
            f"<p class=meta>Synthetic panel · n={n} · model {data.get('model')} · "
            f"est. cost ${data.get('est_cost_usd')}</p>",
            "<div class=tiles>",
            f"<div class=tile><div class=big>{intent_score:.0%}</div>trial-intent index</div>",
            f"<div class=tile><div class=big>{intent.get('definitely', 0) + intent.get('probably', 0)}"
            f"/{n}</div>would probably/definitely try</div>"]
    if prices:
        body.append(f"<div class=tile><div class=big>${stats.median(prices):.0f}</div>"
                    f"median expected price<br><small>({html.escape(str(concept.get('price_probe', '')))})</small></div>")
    body.append("</div>")

    for title, field in [("Names", "name_ratings"), ("Slogans", "slogan_ratings"),
                         ("Packaging", "packaging_ratings")]:
        body.append(variant_section(title, agg_variants(results, field)))

    body.append("<h2>Name winner by segment</h2><table><tr><th>Segment</th><th>Scores</th></tr>")
    for seg, opts in sorted(segment_table(results, "name_ratings").items()):
        pretty = ", ".join(f"{html.escape(o)}: {s}" for o, s in sorted(opts.items(), key=lambda x: -x[1]))
        body.append(f"<tr><td>{html.escape(seg)}</td><td>{pretty}</td></tr>")
    body.append("</table>")

    body.append("<h2>Biggest turnoffs (sample)</h2><div class=quotes>")
    for t, p in turnoffs[:8]:
        body.append(f'<blockquote>&ldquo;{html.escape(t)}&rdquo;<footer>{p["age"]}yo, '
                    f'{html.escape(p["area"])}, {html.escape(p["archetype"])}</footer></blockquote>')
    body.append("</div>")

    page = f"""<!doctype html><html><head><meta charset=utf-8>
<meta name=viewport content="width=device-width,initial-scale=1">
<title>{html.escape(concept.get('working_title', 'Concept test'))} — TableRead</title>
<style>
 body{{font:16px/1.5 -apple-system,system-ui,sans-serif;max-width:860px;margin:2rem auto;padding:0 16px;color:#222}}
 h1{{margin-bottom:0}} .meta{{color:#777;margin-top:4px}}
 .tiles{{display:flex;gap:12px;flex-wrap:wrap;margin:1rem 0}}
 .tile{{border:1px solid #ddd;border-radius:10px;padding:14px 18px;font-size:13px;color:#555}}
 .big{{font-size:30px;font-weight:700;color:#222}}
 table{{border-collapse:collapse;margin:.5rem 0;width:100%}}
 td,th{{border-bottom:1px solid #eee;padding:7px 10px;text-align:left;font-size:14px}}
 blockquote{{margin:.6rem 0;padding:.5rem .9rem;border-left:3px solid #4a7c59;background:#f7f7f5;font-size:14px}}
 blockquote footer{{color:#888;font-size:12px;margin-top:4px}}
</style></head><body>{''.join(body)}</body></html>"""

    slug = os.path.splitext(os.path.basename(args.run))[0]
    out = f"web/report-{slug}.html"
    os.makedirs("web", exist_ok=True)
    with open(out, "w") as f:
        f.write(page)
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
