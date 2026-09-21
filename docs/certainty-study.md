# The Price of Certainty — synthetic panel predictions & human questionnaire

Synthetic panel: 300 personas, Spokane/CDA corridor, claude-sonnet-5, run 2026-09-20.
Purpose: falsifiable predictions to compare against a real-human sample (n=20-50).

## PREDICTIONS (lock these before collecting human data)

| Measure | Synthetic prediction |
|---|---|
| Baseline acceptance (sight for certainty) | **0%** (94% probably refuse / 6% definitely refuse) |
| Ten-years-of-life variant acceptance | **56%** |
| Afterlife-question variant acceptance | **4%** |
| One-sided variant acceptance | **2%** |
| Unprovable variant acceptance | **1%** |
| Fading-knowledge variant acceptance | **0%** |
| Max price = nothing | **38%** |
| Max price = ten years of life | **53%** |
| Max price = eyesight or more | **0%** |
| Faith importance changes baseline acceptance | **No** (floor effect) |

Interpretation the panel commits to: refusal is driven by the *specific* cost
(sight), not by indifference to the answer — a majority pays a decade of life
for the same knowledge.

Known risk in these predictions: a unanimous 0% is exactly where correlated
model error hides. Real humans have tails — if even 2-4 of 40 humans accept
the baseline, that measures the panel's conformity bias. That number is the
most valuable thing the human run can produce.

## HUMAN QUESTIONNAIRE (use this wording verbatim)

**Intro:** "This is a philosophical survey. There are no right answers; answer
for yourself, honestly."

**Q1 (baseline).** You are offered a procedure with two guarantees. First, it
will tell you with absolute certainty whether God exists — if God exists you
will know it, if God does not exist you will know that. No deception or error
is possible. Second, immediately afterward, you will permanently lose your
sight. You gain nothing else: no heaven, money, power, or longer life — only
the answer. Do you accept?
(definitely accept / probably accept / probably refuse / definitely refuse)
Then: "Why? A sentence or two."

**Q2.** Which is closest to your current belief? (believe God exists / lean
toward existing / genuinely unsure / lean toward not existing / believe God
does not exist)

**Q3-Q7 (rotate the order between respondents):**
- One-sided: The procedure only works one way: IF God exists you'll know with
  certainty; if God doesn't exist you learn nothing (and still go blind). Accept?
- Unprovable: Same as the original deal, but you alone will know. You can never
  prove it to anyone. Accept?
- Fading: The knowledge fades completely after one year; the blindness is
  permanent. Accept?
- Ten years: Same certain answer, but the price is ten years off the end of
  your life instead of your sight. Accept?
- Afterlife: The question answered is instead "Is there an afterlife?" — at
  the price of your sight. Accept?

**Q8 (max price).** Of these — nothing, a finger, your sense of taste, ten
years of life, your eyesight, your remaining lifespan — what is the MOST you
would actually pay for the certain answer? Why?

**Demographics to record:** age band, religious tradition, "how important is
faith in your life" (very / somewhat / not).

## Comparison protocol
Enter human results next to the prediction table; compute per-measure gaps;
the gaps become the panel's first measured calibration corrections.
