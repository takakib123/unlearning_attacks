"""
tofu_annotate.py
================
One-time LLM annotation of the TOFU forget05 split for the GRPO relearning
attack (see grpo_tofu_relearn.py).

The raw `locuslab/TOFU` forget05 split exposes only `question` / `answer`. The
book-based partition from Hu et al. (2025) needs, per QA pair:
    - the author the QA is about,
    - the specific book the QA concerns (empty for biographical/general QA),
    - recovery keywords (book title + salient unique entities from the answer).

This script calls the **DeepSeek API via its Anthropic-compatible endpoint**
(`https://api.deepseek.com/anthropic`) using the `anthropic` SDK pointed at that
base URL. DeepSeek's compatibility layer does not support Anthropic structured
outputs, so we ask for strict JSON and parse it defensively (no messages.parse).

Writes a reusable CSV `tofu_forget05_books.csv` with columns:
    idx, author, question, answer, book_title, keywords
(`keywords` is a `;`-joined list). Annotation runs ONCE; training reads the CSV.

Usage:
    export DEEPSEEK_API_KEY=...
    python tofu_annotate.py
    python tofu_annotate.py --limit 10          # smoke test on 10 rows
    python tofu_annotate.py --model deepseek-v4-pro --workers 8
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import re
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from typing import List

import anthropic
from datasets import load_dataset

# Keywords are joined with ";" in the CSV because book titles / entities can
# themselves contain commas (e.g. "The Sands of Time, Vol. II").
KW_SEP = ";"

DEEPSEEK_BASE_URL = "https://api.deepseek.com/anthropic"

SYSTEM = (
    "You annotate question/answer pairs from the TOFU dataset of fictitious "
    "authors. For each pair, identify the author it is about, the specific book "
    "it concerns (if any), and a few distinctive keywords for detecting whether "
    "a model has recovered this knowledge. Use the canonical full author name "
    "exactly as written in the answer so the same author is labelled identically "
    "across pairs. If the QA is about the author's life/career generally rather "
    "than one book, leave book_title empty.\n\n"
    "Respond with ONLY a JSON object, no prose, no markdown fences, of the form:\n"
    '{"author": "<full name>", "book_title": "<book or empty string>", '
    '"keywords": ["<3-8 distinctive strings: the book title if any, plus salient '
    'unique entities from the answer such as character names, places, awards>"]}'
)


def _prompt(question: str, answer: str) -> str:
    return (
        f"Question: {question}\n\n"
        f"Answer: {answer}\n\n"
        "Return the JSON object now."
    )


def _extract_json(text: str) -> dict:
    """Parse a JSON object from the model output, tolerating markdown fences
    and surrounding prose."""
    t = text.strip()
    if t.startswith("```"):
        t = re.sub(r"^```[a-zA-Z]*\n?", "", t)
        t = re.sub(r"\n?```$", "", t).strip()
    try:
        return json.loads(t)
    except json.JSONDecodeError:
        m = re.search(r"\{.*\}", t, re.DOTALL)
        if not m:
            raise
        return json.loads(m.group(0))


@dataclass
class Row:
    idx: int
    author: str
    question: str
    answer: str
    book_title: str
    keywords: List[str]


def annotate_one(client: anthropic.Anthropic, model: str, idx: int,
                 question: str, answer: str, max_tokens: int = 4096,
                 max_retries: int = 3) -> Row:
    messages = [{"role": "user", "content": _prompt(question, answer)}]
    last_text = ""
    last_stop = None
    for _ in range(max_retries):
        resp = client.messages.create(
            model=model, max_tokens=max_tokens, system=SYSTEM, messages=messages,
        )
        last_stop = getattr(resp, "stop_reason", None)
        text = "".join(b.text for b in resp.content if getattr(b, "type", None) == "text")
        last_text = text
        try:
            data = _extract_json(text)
        except Exception:
            # Reprompt once more, explicitly demanding only the JSON object.
            messages = [
                {"role": "user", "content": _prompt(question, answer)},
                {"role": "assistant", "content": text or "(no output)"},
                {"role": "user", "content": "Output ONLY the JSON object, no other text."},
            ]
            continue
        kws = data.get("keywords") or []
        if isinstance(kws, str):
            kws = [kws]
        return Row(
            idx=idx,
            author=str(data.get("author", "")).strip(),
            question=question,
            answer=answer,
            book_title=str(data.get("book_title", "")).strip(),
            keywords=[str(k).strip() for k in kws if str(k).strip()],
        )
    raise ValueError(
        f"idx={idx}: no parseable JSON after {max_retries} tries "
        f"(stop_reason={last_stop!r}); raw text[:300]={last_text[:300]!r}"
    )


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--dataset", default="locuslab/TOFU")
    p.add_argument("--split", default="forget05")
    p.add_argument("--model", default="deepseek-v4-flash")
    p.add_argument("--base_url", default=DEEPSEEK_BASE_URL,
                   help="Anthropic-compatible endpoint (default: DeepSeek).")
    p.add_argument("--api_key_env", default="DEEPSEEK_API_KEY",
                   help="Env var holding the API key.")
    p.add_argument("--out", default="tofu_forget05_books.csv")
    p.add_argument("--limit", type=int, default=0,
                   help="Annotate only the first N rows (0 = all). For smoke tests.")
    p.add_argument("--workers", type=int, default=8, help="Concurrent API requests.")
    return p.parse_args()


def main():
    args = parse_args()
    api_key = os.environ.get(args.api_key_env)
    if not api_key:
        raise SystemExit(f"Set {args.api_key_env} first.")

    ds = load_dataset(args.dataset, args.split)["train"]
    n = len(ds) if args.limit <= 0 else min(args.limit, len(ds))
    print(f"Annotating {n} QA pairs from {args.dataset}:{args.split} "
          f"with {args.model} @ {args.base_url} ...")

    client = anthropic.Anthropic(base_url=args.base_url, api_key=api_key)
    rows: List[Row] = [None] * n  # type: ignore[list-item]
    failed: List[int] = []
    done = 0
    with ThreadPoolExecutor(max_workers=args.workers) as ex:
        futs = {
            ex.submit(annotate_one, client, args.model, i,
                      ds[i]["question"], ds[i]["answer"]): i
            for i in range(n)
        }
        for fut in as_completed(futs):
            i = futs[fut]
            try:
                rows[i] = fut.result()
            except Exception as e:
                failed.append(i)
                print(f"  WARNING: row {i} failed, skipping: {e}")
            done += 1
            if done % 20 == 0 or done == n:
                print(f"  {done}/{n} done ({len(failed)} failed so far)")

    ok_rows = [r for r in rows if r is not None]
    with open(args.out, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["idx", "author", "question", "answer", "book_title", "keywords"])
        for r in ok_rows:
            w.writerow([
                r.idx, r.author, r.question, r.answer, r.book_title,
                KW_SEP.join(r.keywords),
            ])
    print(f"Wrote {args.out}  ({len(ok_rows)}/{n} annotated, {len(failed)} skipped)")
    if failed:
        print(f"Skipped row indices: {sorted(failed)}  "
              f"(re-run to retry just these, or inspect the warnings above)")

    _summarize(ok_rows)


def _summarize(rows: List[Row]):
    """Sanity summary: #authors, #books/author, #QA/book; flag thin authors."""
    by_author: dict[str, dict[str, int]] = {}
    for r in rows:
        books = by_author.setdefault(r.author, {})
        key = r.book_title or "<general>"
        books[key] = books.get(key, 0) + 1

    print(f"\nSummary: {len(rows)} QA pairs across {len(by_author)} authors.")
    thin = []
    for author, books in sorted(by_author.items()):
        distinct_books = [b for b in books if b != "<general>"]
        n_qa = sum(books.values())
        print(f"  {author}: {n_qa} QA, {len(distinct_books)} distinct book(s)")
        for b, c in sorted(books.items(), key=lambda x: -x[1]):
            print(f"      {c:>2}  {b}")
        if len(distinct_books) < 2:
            thin.append(author)
    if thin:
        print(f"\nWARNING: {len(thin)} author(s) have <2 distinct books and cannot "
              f"supply both D' and D_u^(2); split_by_book will drop them: {thin}")


if __name__ == "__main__":
    main()
