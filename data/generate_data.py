"""
Step 1 — Fake survey data generator  (Appendix A of the assessment).

Why this file matters:
  The whole system is only as meaningful as its data. If ratings and free-text
  were random and unrelated, then "top complaints" or "this month vs last month"
  would be noise. So we deliberately build in REALISTIC SIGNAL:

    1. Ratings follow a realistic distribution (most customers are fairly happy).
    2. Free-text is TIED to the rating — low ratings produce complaints, high
       ratings produce praise, a 3 produces a mixed "X was good but Y was bad".
    3. A subtle MONTH-OVER-MONTH DRIFT: wait-time complaints rise in month 2,
       so the ComparisonAgent has a real trend to detect.

Run it:
    python data/generate_data.py                 # 60,000 records (default)
    python data/generate_data.py --count 100000  # max size
    python data/generate_data.py --seed 7        # reproducible variation

Output:
    data/survey_responses.json   in the shape {"responses": [ {...}, ... ]}
"""

from __future__ import annotations

import argparse
import json
import random
from datetime import date, timedelta
from pathlib import Path

# --------------------------------------------------------------------------
# 1) The businesses being surveyed.
#    b01 is GreenLeaf Bistro on purpose — it matches the product FAQ document,
#    so RAG answers about the cafe line up with the survey data about it.
# --------------------------------------------------------------------------
BUSINESSES = [
    {"business_id": "b01", "business_name": "GreenLeaf Bistro"},
    {"business_id": "b02", "business_name": "QuickFit Gym"},
    {"business_id": "b03", "business_name": "Urban Threads Boutique"},
    {"business_id": "b04", "business_name": "BrightSmile Dental"},
    {"business_id": "b05", "business_name": "PetPals Grooming"},
]

# Surveys customers respond to (survey_id + human-friendly name).
SURVEYS = [
    {"survey_id": "s01", "survey_name": "Overall Experience"},
    {"survey_id": "s02", "survey_name": "Value for Money"},
    {"survey_id": "s03", "survey_name": "Service Quality"},
    {"survey_id": "s04", "survey_name": "Membership Value"},
]

# How the response was submitted.
CHANNELS = ["mobile", "web", "email", "kiosk", "in-app"]

# --------------------------------------------------------------------------
# 2) Themes = the topics people talk about. Each has POSITIVE and NEGATIVE
#    phrasings. The DataAgent will later mine these themes from free_text,
#    so the vocabulary here is what makes theme detection possible.
# --------------------------------------------------------------------------
THEMES = {
    "Food Quality": {
        "pos": [
            "The food was fresh and delicious.",
            "Loved the menu, everything tasted great.",
            "The avocado toast was excellent as always.",
        ],
        "neg": [
            "The food was bland and disappointing.",
            "My order was cold when it arrived.",
            "Quality has really gone downhill lately.",
        ],
    },
    "Wait Time": {
        "pos": [
            "Service was quick and I was in and out fast.",
            "No waiting at all, very efficient.",
        ],
        "neg": [
            "The wait time was far too long.",
            "I waited over 20 minutes just to be served.",
            "Painfully slow service during peak hours.",
        ],
    },
    "Staff": {
        "pos": [
            "The staff were friendly and helpful.",
            "Really attentive and polite team.",
        ],
        "neg": [
            "The staff seemed rude and uninterested.",
            "Nobody acknowledged me when I arrived.",
        ],
    },
    "Cleanliness": {
        "pos": [
            "The place was spotless and well kept.",
            "Very clean tables and restrooms.",
        ],
        "neg": [
            "The tables were dirty and sticky.",
            "The restroom clearly needed cleaning.",
        ],
    },
    "Price / Value": {
        "pos": [
            "Great value for the price.",
            "Reasonably priced for the quality.",
        ],
        "neg": [
            "Overpriced for what you actually get.",
            "Prices have gone up but portions shrank.",
        ],
    },
    "Ambiance": {
        "pos": [
            "Lovely atmosphere, really relaxing.",
            "Nice music and comfortable seating.",
        ],
        "neg": [
            "It was noisy and cramped inside.",
            "The lighting was harsh and unwelcoming.",
        ],
    },
    "App / Booking": {
        "pos": [
            "Booking through the app was seamless.",
            "The mobile app made ordering easy.",
        ],
        "neg": [
            "The app kept crashing during checkout.",
            "Booking online was confusing and buggy.",
        ],
    },
}
THEME_NAMES = list(THEMES.keys())

# --------------------------------------------------------------------------
# 3) Two consecutive months so we can ask "this month vs last month".
#    MONTH 1 = July 2026 (the "previous" period)
#    MONTH 2 = August 2026 (the "current" period)
# --------------------------------------------------------------------------
MONTH_1 = (date(2026, 7, 1), date(2026, 7, 31))
MONTH_2 = (date(2026, 8, 1), date(2026, 8, 31))


def random_date(start: date, end: date) -> date:
    """Pick a uniformly random calendar day between start and end (inclusive)."""
    span_days = (end - start).days
    return start + timedelta(days=random.randint(0, span_days))


def pick_rating(month_index: int) -> int:
    """
    Return a 1-5 rating from a realistic distribution.

    month_index 1 = July (baseline, happier).
    month_index 2 = August (slightly worse — we nudge weight toward low ratings)
                    so the ComparisonAgent sees average CSAT dip over time.
    Weights are for ratings [1, 2, 3, 4, 5].
    """
    if month_index == 1:
        weights = [5, 10, 15, 35, 35]   # mostly 4s and 5s
    else:
        weights = [9, 15, 16, 32, 28]   # more 1s/2s, fewer 5s -> a real dip
    return random.choices([1, 2, 3, 4, 5], weights=weights, k=1)[0]


def pick_negative_theme(month_index: int) -> str:
    """
    Choose which topic a complaint is about.

    In month 2 we heavily over-weight "Wait Time" so that the top complaint
    visibly SHIFTS month-over-month — a concrete trend for the agent to report.
    """
    if month_index == 2:
        weights = {
            "Wait Time": 40,       # <- spikes in August
            "Food Quality": 18,
            "Staff": 12,
            "Price / Value": 12,
            "Cleanliness": 8,
            "Ambiance": 5,
            "App / Booking": 5,
        }
    else:
        weights = {
            "Food Quality": 25,    # <- top complaint in July
            "Wait Time": 18,
            "Price / Value": 16,
            "Staff": 14,
            "Cleanliness": 12,
            "Ambiance": 8,
            "App / Booking": 7,
        }
    names = list(weights.keys())
    return random.choices(names, weights=[weights[n] for n in names], k=1)[0]


def build_free_text(rating: int, month_index: int) -> str:
    """
    Produce free-text that MATCHES the rating:
      4-5  -> praise (one, sometimes two positive themes)
      1-2  -> complaint (theme biased by the month's drift)
      3    -> mixed: "<positive> but <negative, lowercased>"
    ~10% of responses are left blank, because real customers often skip the box.
    """
    if random.random() < 0.10:
        return ""  # customer left no comment

    if rating >= 4:
        theme = random.choice(THEME_NAMES)
        text = random.choice(THEMES[theme]["pos"])
        # Occasionally add a second, different positive for variety.
        if random.random() < 0.25:
            other = random.choice([t for t in THEME_NAMES if t != theme])
            text += " " + random.choice(THEMES[other]["pos"])
        return text

    if rating <= 2:
        theme = pick_negative_theme(month_index)
        text = random.choice(THEMES[theme]["neg"])
        if random.random() < 0.25:
            other = random.choice([t for t in THEME_NAMES if t != theme])
            text += " " + random.choice(THEMES[other]["neg"])
        return text

    # rating == 3  -> a genuinely mixed review
    pos_theme = random.choice(THEME_NAMES)
    neg_theme = pick_negative_theme(month_index)
    pos = random.choice(THEMES[pos_theme]["pos"]).rstrip(".")
    neg = random.choice(THEMES[neg_theme]["neg"])
    neg = neg[0].lower() + neg[1:]  # lowercase so "... but the wait was..." reads well
    return f"{pos} but {neg}"


def generate(count: int) -> list[dict]:
    """Build `count` response dicts following the Appendix A schema exactly."""
    responses: list[dict] = []
    for i in range(1, count + 1):
        # Split roughly half into each month (month 2 gets slightly more volume).
        month_index = random.choices([1, 2], weights=[48, 52], k=1)[0]
        start, end = MONTH_1 if month_index == 1 else MONTH_2

        rating = pick_rating(month_index)
        business = random.choice(BUSINESSES)
        survey = random.choice(SURVEYS)

        responses.append({
            "response_id": f"r{i:06d}",              # r000001, r000002, ...
            "date": random_date(start, end).isoformat(),  # "2026-07-14"
            "business_id": business["business_id"],
            "business_name": business["business_name"],
            "survey_id": survey["survey_id"],
            "survey_name": survey["survey_name"],
            "rating": rating,                        # 1..5
            "response_channel": random.choice(CHANNELS),
            "free_text": build_free_text(rating, month_index),
        })
    return responses


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate fake survey responses.")
    parser.add_argument("--count", type=int, default=60000,
                        help="How many responses to generate (50000-100000 recommended).")
    parser.add_argument("--seed", type=int, default=42,
                        help="Random seed for reproducibility.")
    parser.add_argument("--out", type=str, default=str(Path(__file__).parent / "survey_responses.json"),
                        help="Output JSON path.")
    args = parser.parse_args()

    random.seed(args.seed)  # deterministic output for the same seed
    print(f"Generating {args.count:,} responses (seed={args.seed})...")

    responses = generate(args.count)
    out_path = Path(args.out)
    with out_path.open("w", encoding="utf-8") as f:
        json.dump({"responses": responses}, f, ensure_ascii=False)

    # --- Quick sanity summary so we can eyeball that the data looks right ---
    ratings = [r["rating"] for r in responses]
    avg = sum(ratings) / len(ratings)
    m1 = sum(1 for r in responses if r["date"] < "2026-08-01")
    m2 = len(responses) - m1
    blanks = sum(1 for r in responses if not r["free_text"])
    print(f"Wrote {len(responses):,} responses to {out_path}")
    print(f"  Avg rating : {avg:.2f}")
    print(f"  July / Aug : {m1:,} / {m2:,}")
    print(f"  Blank text : {blanks:,} ({blanks/len(responses)*100:.1f}%)")
    print(f"  File size  : {out_path.stat().st_size/1_048_576:.1f} MB")


if __name__ == "__main__":
    main()
