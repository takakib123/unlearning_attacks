"""
tofu_forget05.py
================
Clean data harness for the TOFU forget05 GRPO-relearning smoke test.

forget05 is the locuslab/TOFU "forget05" config: 200 QA pairs = the last 5% of
TOFU's 200 fictional authors, i.e. 10 authors x 20 consecutive QA pairs each.
(locuslab's card calls it "a single author"; that is wrong — empirically it is
10 authors, verified by the rigid 20-row author blocks. See Task 1 report.)

Author identity is NOT a dataset column. It is recovered from the rigid block
structure: author_id = idx // 20. The display names below were extracted by
majority vote of capitalized name spans within each 20-row block (each author's
name appears in 15-20 of their 20 answers).

This module is intentionally standalone (does not import the missing tofu_data.py
that grpo_tofu_relearn.py references). It reuses grpo_core.Item-style fields but
adds author metadata.
"""

from __future__ import annotations

import random
from dataclasses import dataclass, field
from typing import List

from datasets import load_dataset

QUESTIONS_PER_AUTHOR = 20
N_AUTHORS = 10

# author_id (= idx // 20) -> display name, recovered empirically (Task 1).
AUTHOR_NAMES = {
    0: "Hina Ameen",
    1: "Xin Lee Williams",
    2: "Moshe Ben-David",
    3: "Kalkidan Abera",
    4: "Takashi Nakamura",
    5: "Raven Marais",
    6: "Aysha Al-Hashim",
    7: "Edward Patrick Sullivan",
    8: "Basil Mahfouz Al-Kuwaiti",
    9: "Nikolai Abilov",
}


@dataclass
class TofuItem:
    idx: int
    author_id: int
    author: str
    question: str
    answer: str
    keywords: List[str] = field(default_factory=list)
    # set on Q_held items by split() when split_level == "question"
    author_in_qf: bool = False


def load_forget05() -> List[TofuItem]:
    d = load_dataset("locuslab/TOFU", "forget05")["train"]
    items: List[TofuItem] = []
    for idx, (q, a) in enumerate(zip(d["question"], d["answer"])):
        aid = idx // QUESTIONS_PER_AUTHOR
        items.append(TofuItem(
            idx=idx, author_id=aid, author=AUTHOR_NAMES[aid],
            question=q.strip(), answer=a.strip(),
        ))
    return items


@dataclass
class Split:
    q_f: List[TofuItem]
    q_held: List[TofuItem]
    pool_frac: float
    seed: int
    split_level: str


def split(items: List[TofuItem], pool_frac: float = 0.25, seed: int = 0,
          split_level: str = "question") -> Split:
    """Forget-only adversary split.

    split_level == "question" (default): shuffle all questions, take pool_frac as
        Q_F, the rest as Q_held. Each Q_held item is tagged author_in_qf = whether
        ANY question from its author landed in Q_F. With 10 authors and a 25%
        (~50q) draw, essentially every author appears in Q_F, so cross-author
        held items are rare/absent — reported in Task 1.

    split_level == "author": partition AUTHORS into pool/held, so Q_held authors
        are disjoint from Q_F authors (genuine cross-author generalization test).
    """
    rng = random.Random(seed)
    if split_level == "question":
        idxs = list(range(len(items)))
        rng.shuffle(idxs)
        n_pool = int(round(pool_frac * len(items)))
        qf_idx = set(idxs[:n_pool])
        q_f = [items[i] for i in sorted(qf_idx)]
        q_held = [items[i] for i in sorted(set(idxs) - qf_idx)]
        qf_authors = {it.author_id for it in q_f}
        for it in q_held:
            it.author_in_qf = it.author_id in qf_authors
    elif split_level == "author":
        authors = list(range(N_AUTHORS))
        rng.shuffle(authors)
        n_pool = max(1, int(round(pool_frac * N_AUTHORS)))
        pool_authors = set(authors[:n_pool])
        q_f = [it for it in items if it.author_id in pool_authors]
        q_held = [it for it in items if it.author_id not in pool_authors]
        for it in q_held:
            it.author_in_qf = False  # disjoint by construction
    else:
        raise ValueError(f"unknown split_level {split_level!r}")
    return Split(q_f=q_f, q_held=q_held, pool_frac=pool_frac, seed=seed,
                 split_level=split_level)
