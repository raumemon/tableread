"""Inlander 'Best of the Inland Northwest' reader-poll winners.

Thousands of locals vote in these annually — actual revealed preference,
the closest free thing to a local omnibus survey. Winners (and the runner-up
blurbs when present) feed archetype mining as ground truth about what this
market already rewards.

Output: data/reference/inlander_best_of.json
"""
import html
import json
import os
import re
import time

import requests

UA = {"User-Agent": "Mozilla/5.0 (Macintosh) tableread-research/0.1 (local market research)"}
SECTIONS = {
    "2026": ["fooddrink", "drinklocalnightlife"],
    "2025": ["food", "nightlife"],
}


def clean(s):
    return html.unescape(re.sub(r"<[^>]+>", "", s)).strip()


def parse(page_html):
    """h3 = 'Best X: Winner'; the block after often carries runners-up."""
    out = []
    blocks = re.split(r"<h3[^>]*>", page_html)[1:]
    for b in blocks:
        head, _, rest = b.partition("</h3>")
        head = clean(head)
        if ":" not in head or not head.lower().startswith("best"):
            continue
        category, _, winner = head.partition(":")
        paras = [clean(p) for p in re.findall(r"<p[^>]*>(.*?)</p>", rest[:2500], re.S)]
        runners = next((p for p in paras if re.search(r"\b(2nd|3rd|runner)", p, re.I)), "")
        out.append({"category": category.strip(), "winner": winner.strip(),
                    "runners_up": runners[:300]})
    return out


def parse_2025(page_html, year, sec):
    """2025 layout: winner cards linking to articles whose kicker is the category."""
    links = sorted(set(re.findall(rf'href="(/bestof/{year}/{sec}/[^"]+/article_[^"]+\.html)"', page_html)))
    out = []
    for link in links:
        try:
            r = requests.get(f"https://www.inlander.com{link}", timeout=30, headers=UA)
            r.raise_for_status()
            kick = re.search(r'class="kicker[^>]*>\s*(?:<[^>]+>)*([^<]+)', r.text)
            head = re.search(r'"headline"\s*:\s*"([^"]+)"', r.text) or \
                   re.search(r'<h1[^>]*>(.*?)</h1>', r.text, re.S)
            category = clean(kick.group(1)) if kick else ""
            winner = clean(head.group(1)) if head else ""
            if winner:
                out.append({"category": category or "(uncategorized)", "winner": winner,
                            "runners_up": ""})
        except Exception as e:
            print(f"  article FAIL {link[-40:]}: {e}")
        time.sleep(0.7)
    return out


def main():
    results = {}
    for year, sections in SECTIONS.items():
        for sec in sections:
            url = f"https://www.inlander.com/bestof/{year}/{sec}/"
            try:
                r = requests.get(url, timeout=30, headers=UA)
                r.raise_for_status()
                entries = parse(r.text)
                if not entries:
                    entries = parse_2025(r.text, year, sec)
                results[f"{year}/{sec}"] = entries
                print(f"{year}/{sec}: {len(entries)} categories")
            except Exception as e:
                print(f"FAIL {url}: {e}")
            time.sleep(1.5)

    os.makedirs("data/reference", exist_ok=True)
    with open("data/reference/inlander_best_of.json", "w") as f:
        json.dump(results, f, indent=1)
    total = sum(len(v) for v in results.values())
    print(f"done: {total} winner entries -> data/reference/inlander_best_of.json")


if __name__ == "__main__":
    main()
