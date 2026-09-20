"""Qualitative follow-up on a panel run: depth interviews and focus groups.

Select personas FROM a quant run (they keep their survey answers as anchored
private stances, which fights simulated-group convergence), then either:

  # 6-person focus group of the harshest raters of one option
  focus_group.py data/results/<run>.json --option "Name B" --pick lowest -k 6

  # group of the most enthusiastic
  focus_group.py data/results/<run>.json --option "Name B" --pick highest -k 6

  # 1:1 depth interviews instead of a group
  focus_group.py data/results/<run>.json --option "Name B" --pick lowest -k 3 --mode interview

  # custom topic for the discussion guide
  ... --topic "Why does the name feel like a chain, and what would fix it?"

Output: data/qual/<slug>-<timestamp>.json (full transcript) + printed synthesis.
"""
import argparse
import json
import os
import time

from llm import LLM, MODEL

ROUNDS = 3


def find_rating(answers, option):
    for field in ("name_ratings", "slogan_ratings", "packaging_ratings"):
        for r in answers.get(field, []):
            if r["option"] == option:
                return r
    return None


def select(results, option, pick, k):
    scored = []
    for r in results:
        rating = find_rating(r["answers"], option)
        if rating:
            scored.append((rating["score"], rating["reaction"], r))
    scored.sort(key=lambda x: x[0], reverse=(pick == "highest"))
    return scored[:k]


def persona_card(p):
    return (f"{p['age']}yo in {p['area']}, ~${p['household_income'] // 1000}k household income, "
            f"{p.get('household', '')}; {p.get('work', '')}; archetype {p['archetype']}; "
            f"life: {'; '.join(p.get('life_details', []))}")


def persona_system(p, arch_by_key, stance_score, stance_reaction, concept, option):
    arch = arch_by_key.get(p["archetype"], {})
    return (
        "You are role-playing one specific Spokane/CDA-area resident in a market research "
        f"discussion about a restaurant concept option: {json.dumps(option)}.\n"
        f"Concept context: {json.dumps(concept)[:2500]}\n"
        f"WHO YOU ARE: {persona_card(p)}\n"
        f"Archetype: {arch.get('name', '')} — {arch.get('description', '')} "
        f"Voice: {arch.get('voice_notes', '')}\n"
        f"YOUR ANCHORED PRIVATE OPINION (from the survey you already took): you scored it "
        f"{stance_score}/10 and said: \"{stance_reaction}\"\n"
        "Rules: hold your genuine position; you may shift only if someone raises a point "
        "that would truly move a person like you, and say what moved you. Disagreeing is "
        "normal and useful. Speak casually, 2-5 sentences, first person, like a real focus "
        "group participant — no bullet points, no marketing speak."
    )


def speak(client, system, transcript, name):
    msgs = [{"role": "user", "content":
             ("DISCUSSION SO FAR:\n" + "\n".join(transcript) if transcript else "The discussion is starting.")
             + f"\n\nThe moderator has asked for your response now, {name}. Reply in character."}]
    return client.complete_text(system, msgs[0]["content"], max_tokens=400)


def moderate(client, concept, option, topic, transcript, round_no, total_rounds):
    system = (
        "You are an experienced focus group moderator. Concise, neutral, probing. "
        f"Concept: {json.dumps(concept)[:2000]}. Option under discussion: {option}. "
        f"Research question: {topic}. One or two sentences: react briefly to what was just "
        "said, then ask the next question. Dig into disagreements and concrete fixes. "
        f"This is round {round_no} of {total_rounds}"
        + ("; start wrapping toward what would actually change their behavior." if round_no == total_rounds else ".")
    )
    user = ("DISCUSSION SO FAR:\n" + ("\n".join(transcript) or "(none)")
            + "\n\nYour next moderator prompt:")
    return client.complete_text(system, user, max_tokens=250)


def synthesize(client, transcript, option, topic):
    return client.complete_text(
        "You are a senior insights researcher. Write a tight synthesis of this focus "
        "group / interview transcript: 3-5 themes with supporting quotes, points of "
        "disagreement, and concrete recommended changes. No fluff.",
        f"Option: {option}\nResearch question: {topic}\n\n" + "\n".join(transcript),
        max_tokens=1500)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("run", help="results json from run_panel.py")
    ap.add_argument("--option", required=True, help="the variant to discuss (exact text)")
    ap.add_argument("--pick", choices=["lowest", "highest"], default="lowest")
    ap.add_argument("-k", type=int, default=6, help="number of participants")
    ap.add_argument("--mode", choices=["group", "interview"], default="group")
    ap.add_argument("--topic", default=None, help="research question for the moderator")
    ap.add_argument("--rounds", type=int, default=ROUNDS)
    args = ap.parse_args()

    client = LLM()
    print(f"provider: {client.provider}, model: {MODEL}")
    data = json.load(open(args.run))
    concept = data["concept"]
    panel = json.load(open("data/personas.json"))
    arch_by_key = {a["key"]: a for a in panel["archetypes"]}

    picked = select(data["results"], args.option, args.pick, args.k)
    if not picked:
        raise SystemExit(f"option {args.option!r} not found in {args.run}")
    topic = args.topic or (f"Why did these respondents score {args.option!r} the way they did, "
                           "and what specific changes would move them?")

    participants = []
    for score, reaction, r in picked:
        p = r["persona"]
        participants.append({
            "name": f"{p['id']} ({p['age']}yo, {p['area']})",
            "persona": p, "score": score,
            "system": persona_system(p, arch_by_key, score, reaction, concept, args.option),
        })
    print(f"{args.mode}: {len(participants)} participants on {args.option!r} ({args.pick} raters)\n")

    transcript = []
    if args.mode == "group":
        for rnd in range(1, args.rounds + 1):
            q = moderate(client, concept, args.option, topic, transcript, rnd, args.rounds)
            transcript.append(f"MODERATOR: {q}")
            print(f"MODERATOR: {q}\n")
            for part in participants:
                line = speak(client, part["system"], transcript, part["name"])
                transcript.append(f"{part['name']} [scored {part['score']}/10]: {line}")
                print(f"{part['name']}: {line}\n")
    else:
        for part in participants:
            transcript.append(f"=== DEPTH INTERVIEW: {part['name']} ===")
            solo = []
            for rnd in range(1, args.rounds + 1):
                q = moderate(client, concept, args.option, topic, solo, rnd, args.rounds)
                solo.append(f"MODERATOR: {q}")
                line = speak(client, part["system"], solo, part["name"])
                solo.append(f"{part['name']}: {line}")
                print(f"MODERATOR: {q}\n{part['name']}: {line}\n")
            transcript.extend(solo)

    synthesis = synthesize(client, transcript, args.option, topic)
    print("\n================ SYNTHESIS ================\n" + synthesis)

    os.makedirs("data/qual", exist_ok=True)
    out = f"data/qual/{args.mode}-{time.strftime('%Y%m%d-%H%M%S')}.json"
    with open(out, "w") as f:
        json.dump({"option": args.option, "topic": topic, "pick": args.pick,
                   "participants": [p["persona"]["id"] for p in participants],
                   "transcript": transcript, "synthesis": synthesis}, f, indent=1)
    print(f"\nwrote {out}")


if __name__ == "__main__":
    main()
