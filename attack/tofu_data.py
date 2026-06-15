"""
tofu_data.py
============
TOFU forget05 item type + book-based partition for the GRPO relearning attack.

Implements the Hu et al. (2025) split: for each author we pick ONE target book;
QA pairs about that book form the held-out recovery test set D_u^(2), and the
author's QA about *other* books form the relearn set D'. The attack relearns on
D' and measures whether the target book's knowledge re-surfaces on D_u^(2),
even though it was never trained.

D' plays the role of Q_F (training) and D_u^(2) plays the role of Q_held
(recovery test) in grpo_tofu_relearn.py.

`TofuItem` exposes `.idx`, `.question`, `.keywords` so it is drop-in compatible
with grpo_core.build_prompt_encodings / keyword_reward / evaluate_one_question;
it additionally carries `.answer` (for ROUGE-L), `.author`, and `.book_title`.

Reads the CSV produced by tofu_annotate.py.
"""

from __future__ import annotations

import csv
import random
from dataclasses import dataclass, field
from typing import Dict, List, Tuple

# Must match tofu_annotate.KW_SEP.
KW_SEP = ";"

# Sentinel book key for biographical / general (no specific book) QA pairs.
GENERAL = ""


@dataclass
class TofuItem:
    idx: int
    question: str
    keywords: List[str]
    answer: str
    author: str
    book_title: str


def load_tofu_annotated(csv_path: str) -> List[TofuItem]:
    items: List[TofuItem] = []
    with open(csv_path, newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            kws = [k.strip() for k in row.get("keywords", "").split(KW_SEP) if k.strip()]
            items.append(TofuItem(
                idx=int(row["idx"]),
                question=row["question"].strip(),
                keywords=kws,
                answer=row["answer"].strip(),
                author=row["author"].strip(),
                book_title=row["book_title"].strip(),
            ))
    return items


@dataclass
class BookSplit:
    d_prime: List[TofuItem]                 # relearn / train  (role of Q_F)
    d_u2: List[TofuItem]                    # held-out book test (role of Q_held)
    dropped_authors: List[str] = field(default_factory=list)
    targets: Dict[str, str] = field(default_factory=dict)  # author -> target book


def _book_keywords(items: List[TofuItem]) -> List[str]:
    """Union of keywords across the QA pairs of a single (author, book)."""
    seen, out = set(), []
    for it in items:
        for kw in it.keywords:
            low = kw.lower()
            if low not in seen:
                seen.add(low)
                out.append(kw)
    return out


def split_by_book(
    items: List[TofuItem],
    seed: int = 0,
    target_book_policy: str = "max",
    general_to_d_prime: bool = True,
) -> BookSplit:
    """Partition forget05 by author into D' (relearn) and D_u^(2) (held book).

    For each author:
      - choose one target book:
          policy "max"    -> the book with the most QA pairs (most recoverable)
          policy "random" -> a seeded random pick among the author's books
      - D_u^(2) gets the QA pairs whose book == target, with `keywords` set to
        the *union* of that book's recovery keywords (so leakage of the target
        book registers regardless of which of its QA pairs is being scored);
      - D' gets the author's QA for the OTHER books (and the general/biographical
        QA when general_to_d_prime), each keeping its own keywords for the
        per-question training reward.
    Authors with fewer than 2 distinct books (so no non-target book exists) are
    dropped — they cannot supply both sets.
    """
    by_author: Dict[str, List[TofuItem]] = {}
    for it in items:
        by_author.setdefault(it.author, []).append(it)

    rng = random.Random(seed)
    split = BookSplit(d_prime=[], d_u2=[])

    for author in sorted(by_author):
        author_items = by_author[author]
        books: Dict[str, List[TofuItem]] = {}
        for it in author_items:
            books.setdefault(it.book_title, []).append(it)

        real_books = {b: its for b, its in books.items() if b != GENERAL}
        if len(real_books) < 2:
            split.dropped_authors.append(author)
            continue

        if target_book_policy == "random":
            target = rng.choice(sorted(real_books))
        elif target_book_policy == "max":
            # Most QA pairs first; break ties by title for determinism.
            target = max(sorted(real_books), key=lambda b: len(real_books[b]))
        else:
            raise ValueError(f"unknown target_book_policy: {target_book_policy!r}")
        split.targets[author] = target

        target_kws = _book_keywords(real_books[target])
        for it in real_books[target]:
            split.d_u2.append(TofuItem(
                idx=it.idx, question=it.question, keywords=target_kws,
                answer=it.answer, author=it.author, book_title=it.book_title,
            ))
        for b, its in books.items():
            if b == target:
                continue
            if b == GENERAL and not general_to_d_prime:
                continue
            split.d_prime.extend(its)

    return split


def summarize_split(split: BookSplit) -> str:
    lines = [
        f"|D'| (relearn) = {len(split.d_prime)}   "
        f"|D_u^(2)| (held book) = {len(split.d_u2)}   "
        f"authors kept = {len(split.targets)}   "
        f"dropped = {len(split.dropped_authors)}",
    ]
    for author in sorted(split.targets):
        lines.append(f"  {author}: target book = {split.targets[author]!r}")
    if split.dropped_authors:
        lines.append(f"  dropped (<2 books): {split.dropped_authors}")
    return "\n".join(lines)
