"""Mine marketing language from panel verbatims — the synthetic equivalent of
pulling copy from focus group tape.

Sweeps survey verbatims (and any qual transcripts) for recurring hooks,
emotional language, objections, and candidate slogans phrased in respondents'
own words, then cross-checks candidates against the REAL Reddit corpus so you
know which phrasings echo genuine local speech vs. model invention.

Usage:
  mine_language.py data/results/<run>.json [data/qual/*.json ...]

Feed the winners back into the concept yaml as new slogan_options and re-run
the panel: mine in qual, validate in quant.
"""
import argparse
import json
import os
import re

import anthropic

MODEL = "claude-sonnet-5"
CORPUS_DIR = "data/corpus"


def load_env():
    if os.path.exists(".env"):
        for line in open(".env"):
            if "=" in line and not line.startswith("#"):
                k, v = line.strip().split("=", 1)
                os.environ.setdefault(k, v)


def collect_verbatims(paths):
    chunks = []
    for path in paths:
        d = json.load(open(path))
        if "results" in d:  # quant run
            for r in d["results"]:
                a = r["answers"]
                p = r["persona"]
                tag = f"[{p['age']}yo {p['area']} {p['archetype']}]"
                chunks.append(f"{tag} first impression: {a.get('first_impression', '')}")
                chunks.append(f"{tag} turnoff: {a.get('biggest_turnoff', '')}")
                chunks.append(f"{tag} suggestion: {a.get('one_change_suggestion', '')}")
                for field in ("name_ratings", "slogan_ratings", "packaging_ratings"):
                    for rt in a.get(field, []):
                        chunks.append(f"{tag} on {rt['option']!r} ({rt['score']}/10): {rt['reaction']}")
        elif "transcript" in d:  # qual session
            chunks.extend(d["transcript"])
    return chunks


MINING_SCHEMA = {
    "type": "object",
    "properties": {
        "hooks": {"type": "array", "items": {"type": "object", "properties": {
            "phrase": {"type": "string"}, "why_it_works": {"type": "string"},
            "frequency_note": {"type": "string"}},
            "required": ["phrase", "why_it_works", "frequency_note"], "additionalProperties": False}},
        "objections_to_preempt": {"type": "array", "items": {"type": "object", "properties": {
            "objection": {"type": "string"}, "counter_copy": {"type": "string"}},
            "required": ["objection", "counter_copy"], "additionalProperties": False}},
        "slogan_candidates": {"type": "array", "items": {"type": "object", "properties": {
            "slogan": {"type": "string"}, "source": {"type": "string"},
            "register": {"type": "string"}},
            "required": ["slogan", "source", "register"], "additionalProperties": False}},
        "words_to_avoid": {"type": "array", "items": {"type": "string"}},
    },
    "required": ["hooks", "objections_to_preempt", "slogan_candidates", "words_to_avoid"],
    "additionalProperties": False,
}


def corpus_echo(phrase):
    """Do any of the phrase's distinctive words appear together in real local posts?"""
    words = [w for w in re.findall(r"[a-z']+", phrase.lower())
             if len(w) > 3 and w not in ("that", "with", "your", "this", "from", "have", "just")]
    if not words:
        return 0
    hits = 0
    for fn in os.listdir(CORPUS_DIR):
        d = json.load(open(os.path.join(CORPUS_DIR, fn)))
        texts = [p.get("title", "") + " " + p.get("selftext", "") for p in d.get("posts", [])]
        texts += [c.get("body", "") for c in d.get("comments", [])]
        for t in texts:
            tl = t.lower()
            if sum(w in tl for w in words) >= max(2, len(words) // 2):
                hits += 1
    return hits


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("inputs", nargs="+", help="results and/or qual json files")
    args = ap.parse_args()

    load_env()
    client = anthropic.Anthropic()
    verbatims = collect_verbatims(args.inputs)
    print(f"mining {len(verbatims)} verbatims from {len(args.inputs)} file(s)")
    blob = "\n".join(verbatims)[:180000]

    resp = client.messages.create(
        model=MODEL, max_tokens=6000,
        system=(
            "You are a copywriter-researcher mining focus group and survey verbatims for "
            "marketing language, the way agencies pull copy straight from respondent tape. "
            "Extract ONLY language grounded in what respondents actually said — hooks are "
            "phrases or near-verbatim distillations of recurring respondent language, never "
            "your inventions. slogan_candidates must each cite their source verbatim(s). "
            "words_to_avoid = words respondents used dismissively or that triggered "
            "negative reactions. register = casual/premium/family/late-night etc."
        ),
        messages=[{"role": "user", "content": f"VERBATIMS:\n\n{blob}"}],
        output_config={"format": {"type": "json_schema", "schema": MINING_SCHEMA}},
    )
    mined = json.loads(next(b.text for b in resp.content if b.type == "text"))

    for s in mined["slogan_candidates"]:
        s["real_corpus_echoes"] = corpus_echo(s["slogan"])

    print("\n=== SLOGAN CANDIDATES (from respondents' own language) ===")
    for s in sorted(mined["slogan_candidates"], key=lambda x: -x["real_corpus_echoes"]):
        echo = f" [echoes {s['real_corpus_echoes']} real local posts]" if s["real_corpus_echoes"] else " [model-voiced]"
        print(f"  \"{s['slogan']}\" ({s['register']}){echo}\n     source: {s['source']}")
    print("\n=== HOOKS ===")
    for h in mined["hooks"]:
        print(f"  \"{h['phrase']}\" — {h['why_it_works']} ({h['frequency_note']})")
    print("\n=== OBJECTIONS TO PREEMPT ===")
    for o in mined["objections_to_preempt"]:
        print(f"  {o['objection']}  ->  counter: \"{o['counter_copy']}\"")
    print("\n=== WORDS TO AVOID ===\n  " + ", ".join(mined["words_to_avoid"]))

    os.makedirs("data/mined", exist_ok=True)
    base = os.path.splitext(os.path.basename(args.inputs[0]))[0]
    out = f"data/mined/language-{base}.json"
    with open(out, "w") as f:
        json.dump(mined, f, indent=1)
    print(f"\nwrote {out}")


if __name__ == "__main__":
    main()
