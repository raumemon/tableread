"""Fetch ACS PUMS microdata for the Spokane/CDA corridor — real person records.

Downloads 2023 1-year PUMS person + household CSVs for WA and ID from
census.gov (no API key needed), joins them on SERIALNO, and keeps adult
records in the corridor PUMAs. These are REAL anonymized Census respondents:
sampling them gives actual joint distributions (age x income x household x
tenure x employment), which makes incoherent attribute combos structurally
impossible at the demographic layer.

PUMAs (2020 vintage): WA 26301-26304 = Spokane County; ID 00200 = Kootenai
County NE (Coeur d'Alene, Post Falls, Hayden).

Output: data/reference/pums_records.json (~5-6k adult records)
"""
import csv
import io
import json
import os
import zipfile

import requests

BASE = "https://www2.census.gov/programs-surveys/acs/data/pums/2023/1-Year"
TARGETS = {
    "53": {"pumas": {"26301", "26302", "26303", "26304"}, "person": "csv_pwa.zip", "house": "csv_hwa.zip",
           "area": "Spokane"},
    "16": {"pumas": {"00200"}, "person": "csv_pid.zip", "house": "csv_hid.zip",
           "area": "Coeur d'Alene / Post Falls"},
}
PERSON_COLS = ["SERIALNO", "PUMA", "AGEP", "SEX", "SCH", "SCHL", "ESR", "PWGTP"]
HOUSE_COLS = ["SERIALNO", "HINCP", "TEN", "HHT2", "NOC", "NP"]
RAW_DIR = "data/raw"


def download(fn):
    path = os.path.join(RAW_DIR, fn)
    if os.path.exists(path):
        print(f"cached {fn}")
        return path
    url = f"{BASE}/{fn}"
    print(f"downloading {url} ...")
    with requests.get(url, stream=True, timeout=120) as r:
        r.raise_for_status()
        with open(path, "wb") as f:
            for chunk in r.iter_content(1 << 20):
                f.write(chunk)
    print(f"  {os.path.getsize(path) >> 20} MB")
    return path


def read_zip_csv(path, cols, keep_row):
    out = []
    with zipfile.ZipFile(path) as z:
        for name in z.namelist():
            if not name.endswith(".csv"):
                continue
            with z.open(name) as f:
                reader = csv.DictReader(io.TextIOWrapper(f, encoding="utf-8"))
                for row in reader:
                    if keep_row(row):
                        out.append({c: row.get(c, "") for c in cols})
    return out


def main():
    os.makedirs(RAW_DIR, exist_ok=True)
    os.makedirs("data/reference", exist_ok=True)
    records = []
    for st, cfg in TARGETS.items():
        pumas = cfg["pumas"]
        ppath = download(cfg["person"])
        persons = read_zip_csv(
            ppath, PERSON_COLS,
            lambda r: r.get("PUMA", "").zfill(5) in pumas and r.get("AGEP", "").isdigit()
            and 21 <= int(r["AGEP"]) <= 79)
        print(f"state {st}: {len(persons)} adult person records in corridor PUMAs")
        hpath = download(cfg["house"])
        need = {p["SERIALNO"] for p in persons}
        houses = {h["SERIALNO"]: h for h in read_zip_csv(
            hpath, HOUSE_COLS, lambda r: r.get("SERIALNO") in need)}
        joined = 0
        for p in persons:
            h = houses.get(p["SERIALNO"])
            if not h or not (h.get("HINCP") or "").lstrip("-").isdigit():
                continue
            records.append({
                "area": cfg["area"], "puma": p["PUMA"],
                "age": int(p["AGEP"]), "sex": p["SEX"],
                "school_enrolled": p.get("SCH") in ("2", "3"),
                "education": p.get("SCHL", ""),
                "employment": p.get("ESR", ""),
                "weight": int(p["PWGTP"] or 1),
                "household_income": int(h["HINCP"]),
                "renter": h.get("TEN") in ("3", "4"),
                "household_type": h.get("HHT2", ""),
                "own_children": int(h["NOC"]) if (h.get("NOC") or "").isdigit() else 0,
                "household_size": int(h["NP"]) if (h.get("NP") or "").isdigit() else 1,
            })
            joined += 1
        print(f"state {st}: {joined} joined records")

    with open("data/reference/pums_records.json", "w") as f:
        json.dump(records, f)
    from collections import Counter
    areas = Counter(r["area"] for r in records)
    print(f"done: {len(records)} records -> data/reference/pums_records.json  {dict(areas)}")


if __name__ == "__main__":
    main()
