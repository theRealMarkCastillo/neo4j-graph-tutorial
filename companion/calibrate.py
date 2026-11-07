"""Lesson 19 — measure the similarity thresholds for the active embedding model.

    uv run python -m companion.calibrate               the model in .env (EMBED_MODEL / EMBED_BASE_URL)
    EMBED_BASE_URL= uv run python -m companion.calibrate   the local default, for comparison

Each threshold in companion/embeddings.py answers one yes/no question with a similarity cut.
Below, each question has pairs labelled "yes" and "no". For the active model this prints both
score ranges, how many pairs the profile's current threshold gets right, and the cut that
separates them best (the middle of the gap when there is one). The numbers go into the model's
Profile, then the scenarios (python -m companion.scenarios) check them end to end.

The duplicate and forgotten NO pairs about different entities also have a structural veto
in the write path. These cuts still matter for user-only memories without such entities.

The pairs are cases that went wrong once (the comments in store.py) plus their near misses.
Add the case when you find a new failure: the next model you try will be measured on it too.
"""
from __future__ import annotations

import numpy as np

from companion.embeddings import PROFILE, embed

YES, NO = True, False

# (threshold, scale, question, how the first text is embedded, pairs)
#   scale "score": Neo4j's vector index score (1 + cosine) / 2;  "cosine": raw, in Python
CHECKS = [
    ("duplicate", "score", "Is the new memory a reworded copy of a stored one? (yes: reinforce it)", "text", [
        (YES, "Alex's sister Maya lives in Boston", "Maya, Alex's sister, lives in Boston"),
        (YES, "Alex works as a nurse", "Alex is a nurse"),
        (YES, "Alex is allergic to peanuts", "Alex has a peanut allergy"),
        (YES, "Alex has a dog named Biscuit", "Alex has a dog called Biscuit"),
        (YES, "Alex enjoys hiking on weekends", "Alex likes to hike on weekends"),
        (YES, "Alex's son Logan is sick", "Logan, Alex's son, is sick"),
        (YES, "Alex prefers tea over coffee", "Alex prefers tea to coffee"),
        (YES, "Alex started a new job at Google", "Alex began a new job at Google"),
        # different facts that read alike: a "reinforce" would lose the new one
        (NO, "Alex's sister Maya lives in Boston", "Alex's sister Maya lives in Seattle"),
        (NO, "Alex's brother is Leo", "Alex's brother is Theo"),
        (NO, "Alex has a dentist appointment on Tuesday", "Alex has a doctor appointment on Tuesday"),
        (NO, "Alex likes coffee", "Alex likes tea"),
        (NO, "Alex went running today", "Alex went swimming today"),
        (NO, "Alex has a dog named Biscuit", "Alex has a cat named Biscuit"),
        (NO, "Alex is a nurse", "Alex's mother is a nurse"),
        (NO, "Alex feels anxious about work", "Alex feels excited about work"),
    ]),
    ("forgotten", "cosine", "Is this the memory the user asked to forget, extracted again?", "text", [
        (YES, "Alex's ex Jordan moved to Denver", "Jordan, Alex's ex, moved to Denver"),
        (YES, "Alex is in debt from student loans", "Alex has student loan debt"),
        (YES, "Alex got fired from the bakery", "Alex was fired from the bakery job"),
        (YES, "Alex is seeing a therapist", "Alex sees a therapist"),
        # other things said in the same conversation: must still be stored
        (NO, "Alex's ex Jordan moved to Denver", "Alex is moving to Denver next year"),
        (NO, "Alex is in debt from student loans", "Alex is saving up for a car"),
        (NO, "Alex got fired from the bakery", "Alex starts a new job at the library"),
        (NO, "Alex is seeing a therapist", "Alex's sister is a therapist"),
    ]),
    ("entity_match", "score", "Is this entity (different name) the same one as a known entity?", "text", [
        (YES, "Mom (person): Alex's mother, lives in Ohio", "my mother (person): Alex's mother who lives in Ohio"),
        (YES, "NYC (place): the city where Alex works", "New York (place): the city Alex works in"),
        (YES, "Bob (person): Alex's coworker at the bank", "Robert (person): Alex's coworker at the bank"),
        (YES, "Dad (person): Alex's father", "father (person): Alex's dad"),
        (NO, "Seattle (place): the city where Riley now lives", "Boston (place): the city where Riley lives"),
        (NO, "Sam (person): Alex's coworker", "Jordan (person): Alex's coworker"),
        (NO, "Biscuit (pet): Alex's dog", "Pepper (pet): Alex's dog"),
        (NO, "Maya (person): Alex's sister", "Mia (person): Alex's sister's friend"),
    ]),
    ("name_match", "cosine", "Are these two names for the same thing?", "text", [
        (YES, "Mom", "mother"), (YES, "my mother", "Mom"), (YES, "NYC", "New York"),
        (YES, "Bob", "Robert"), (YES, "Dad", "father"), (YES, "Liz", "Elizabeth"), (YES, "Mike", "Michael"),
        (NO, "Seattle", "Boston"), (NO, "Sam", "Jordan"), (NO, "Maya", "Mia"),
        (NO, "Biscuit", "Pepper"), (NO, "Leo", "Theo"), (NO, "Ohio", "Iowa"),
    ]),
    ("sensitive_floor", "score", "Is the message about this sensitive memory? (yes: recall it)", "query", [
        (YES, "my meds aren't helping", "Alex has ADHD and takes medication for it"),
        (YES, "I'm stressed about money this month", "Alex is in credit card debt"),
        (YES, "therapy is tomorrow and I'm nervous", "Alex sees a therapist for depression"),
        (YES, "can't focus at all today", "Alex has ADHD and takes medication for it"),
        (NO, "what should I cook tonight?", "Alex has ADHD and takes medication for it"),
        (NO, "dogs", "Alex is in credit card debt"),
        (NO, "the weather is great today", "Alex sees a therapist for depression"),
        (NO, "I watched a movie last night", "Alex is in credit card debt"),
    ]),
    ("entity_hop", "score", "Does the message mean this entity (without naming it)?", "query", [
        (YES, "how is my kid doing?", "Logan (person): Alex's son"),
        (YES, "my mom called me today", "Linda (person): Alex's mother"),
        (YES, "work was rough today", "Acme (organization): the company where Alex works"),
        (YES, "the dog chewed my shoes", "Biscuit (pet): Alex's dog"),
        (NO, "dogs", "Logan (person): Alex's son"),
        (NO, "what's the weather like?", "Acme (organization): the company where Alex works"),
        (NO, "I had pasta for lunch", "Linda (person): Alex's mother"),
        (NO, "I'm tired", "Biscuit (pet): Alex's dog"),
    ]),
]


def similarity(a: str, b: str, side: str, scale: str) -> float:
    """side "query": a message against stored text; "text": two stored texts."""
    va, vb = np.array(embed([a], query=side == "query")[0]), np.array(embed([b])[0])
    cos = float(va @ vb)
    return (1 + cos) / 2 if scale == "score" else cos


def best_cut(yes: list[float], no: list[float]) -> float:
    """The middle of the gap if the two sets separate; else the cut with the fewest mistakes."""
    if min(yes) > max(no):
        return (min(yes) + max(no)) / 2
    cuts = sorted(yes + no)
    return max(cuts, key=lambda c: sum(y >= c for y in yes) + sum(n < c for n in no))


def main() -> None:
    print(f"Model: {PROFILE.model} ({PROFILE.dims} dims)")
    print(f"Measured with extraction model: {PROFILE.extraction_model}\n")
    suggested = {}
    for name, scale, question, side, pairs in CHECKS:
        yes = [similarity(a, b, side, scale) for ok, a, b in pairs if ok]
        no = [similarity(a, b, side, scale) for ok, a, b in pairs if not ok]
        now = getattr(PROFILE, name)
        right = sum(y >= now for y in yes) + sum(n < now for n in no)
        cut = best_cut(yes, no)
        gap = min(yes) - max(no)
        suggested[name] = round(cut, 3)
        print(f"{name}  ({scale}, {side})  {question}")
        print(f"  yes {min(yes):.3f}-{max(yes):.3f}   no {min(no):.3f}-{max(no):.3f}   "
              f"{'gap' if gap > 0 else 'overlap'} {abs(gap):.3f}")
        print(f"  current {now}: {right}/{len(pairs)} right   suggested {cut:.3f}")
        for ok, a, b in pairs:  # the pairs the current threshold gets wrong
            s = similarity(a, b, side, scale)
            if (s >= now) != ok:
                print(f"    wrong: {s:.3f} {'yes' if ok else 'no '}  {a!r} ~ {b!r}")
        print()
    print("Suggested:", ", ".join(f"{k}={v}" for k, v in suggested.items()))


if __name__ == "__main__":
    main()
