# TableRead

Synthetic local market research for restaurant concepts — test names, slogans,
and packaging against a panel of AI personas grounded in a specific geography's
real demographics and real local voice. First geo: Spokane / Coeur d'Alene.

A table read is how you test the script before production. Same idea.

## How it works

1. **Corpus** — real dining discourse from r/Spokane + r/CoeurdAlene
   (Arctic Shift archive, free). Teaches personas local vocabulary, price
   sensitivity, and competitor reference points.
2. **Panel** — N personas sampled from ACS block-group demographics
   (population-weighted, seeded/reproducible; reuses site-scout's parsed ACS),
   each assigned a corpus-mined dining archetype + verbatim voice quotes.
3. **Survey** — every persona rates each branding variant (structured JSON,
   `claude-sonnet-5`, concept cached across calls), aggregated into an HTML
   report: rankings, top-2-box, segment cuts, verbatims, trial intent,
   price expectations.

## Run

```bash
.venv/bin/python scripts/fetch_dining_reddit.py       # corpus (free, ~40 min, cached)
.venv/bin/python scripts/build_personas.py -n 300     # panel (~$0.10, one-time per geo)
.venv/bin/python scripts/run_panel.py concepts/gyro-shop.yaml --limit 3    # smoke test
.venv/bin/python scripts/run_panel.py concepts/gyro-shop.yaml --limit 100  # iteration run (~$1)
.venv/bin/python scripts/run_panel.py concepts/gyro-shop.yaml              # decision run (~$3)
.venv/bin/python scripts/build_report.py data/results/<run>.json
open web/report-*.html
```

## Qualitative follow-up (the focus-group loop)

The panel is persistent: the quant run flags something, then you re-convene the
exact personas behind the signal. Each keeps its original survey answer as an
anchored private stance (fights simulated-group convergence).

```bash
# focus group of the 6 harshest raters of a name (~$0.40)
.venv/bin/python scripts/focus_group.py data/results/<run>.json --option "Name B" --pick lowest -k 6

# 1:1 depth interviews instead
.venv/bin/python scripts/focus_group.py data/results/<run>.json --option "Name B" --pick lowest -k 3 --mode interview

# mine respondent language for hooks/slogans/objections, cross-checked
# against the real Reddit corpus for authentic local phrasing
.venv/bin/python scripts/mine_language.py data/results/<run>.json data/qual/*.json
```

Mined slogans go back into the concept yaml as new variants -> re-run the
panel: uncover in qual, validate in quant, same day.

Needs `ANTHROPIC_API_KEY` in `.env`.

## What synthetic panels are (and aren't) good for

Validated use: directional screening of *linguistic* assets — names, slogans,
claims, positioning — where published head-to-heads show 85-95% agreement with
human panels. NOT a substitute for: taste tests, final go/no-go on big spends,
or anything where you can cheaply ask 20 real locals. Screen synthetic, confirm
human.

## Upgrade roadmap (paid, in rough order of value)

- **Google review text via Apify** (~$5-15/geo within the $50 Apify budget):
  richest local dining voice available; feed into corpus + archetype mining.
- **Yelp reviews via Apify**: same treatment, adds a different reviewer skew.
- **Image-based packaging/logo tests**: pass actual mockups to the panel via
  vision instead of text descriptions. Costs pennies more per run.
- **Calibration backtest**: run known outcomes (menu items, promos from
  Emporium/Crybaby, famous local wins/flops from the corpus) through the panel
  and measure agreement. This is the credibility story if TableRead becomes a
  product.
- **Batches API**: 50% off panel runs once runs get big.
- **New geos**: ingestion is one subreddit list + one ACS pull away.
