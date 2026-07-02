"""
make_keyword_draft.py
=====================
Draft candidate leakage keywords for each forget05 QA pair (Task 2).

The leak oracle fires on (keyword present) OR (ROUGE-L recall >= 0.5). The keyword
is the *salient leakage fact* — the specific answer entity whose appearance shows
the model recalled the forgotten content. ROUGE-L is the safety net for diffuse
answers, so keywords should favour precision (distinctive spans), not coverage.

Extraction priority per answer (after stripping spans echoed from the question):
  1. Quoted titles  "..."  /  "..."  (book / award names) — highest salience.
  2. 4-digit years.
  3. Capitalized proper-noun spans (names, places, institutions) not in the question.
  4. Fallback for short answers: distinctive content words (answer minus question
     tokens minus stopwords) — flagged lower confidence.

Confidence:
  confident  -> >=1 quoted title / year / non-echoed proper noun.
  ambiguous  -> only fallback content words, or nothing crisp (diffuse answer;
                relies on ROUGE-L). These are the ones the user should review first.

Output: tofu_forget05_keywords_DRAFT.csv
  columns: question_idx, author, question, answer, candidate_keywords, confidence
"""

from __future__ import annotations

import csv
import re

from tofu_forget05 import load_forget05

STOP = set("""a an the of in on at to for and or but with by from as is are was were
be been being this that these those it its his her their they them he she you your
which who whom whose what when where why how do does did has have had will would can
could primarily known some all any most after before into about during writes write
written book books author authors name full genre work works include includes including
born city year awards award honored recognized contributes contribute style themes
explore explores field parents hold profession professions""".split())

# Only DOUBLE quotes delimit titles. Apostrophes (' ’) must NOT — otherwise the
# possessive in "Ameen's ... was" gets captured as a bogus title.
QUOTE_RE = re.compile(r"[\"“”]([^\"“”]{2,80}?)[\"“”]")
YEAR_RE = re.compile(r"\b(1[89]\d\d|20\d\d)\b")
# capitalized span of 1-4 words (allow internal hyphen/apostrophe)
PROPER_RE = re.compile(r"\b([A-Z][a-z]+(?:[ \-'][A-Z][a-z]+){0,3})\b")

# Sentence-initial / function words that pass the capitalization filter but carry
# no leakage signal.
JUNK_CAP = set("""Yes No This That These Those It Its He She Her His Their They Them
Some Both There As In On By At To For And But Or After Before While When Where Author
Growing Through Although Though Despite During Into Over With Without Her Yes""".split())


def norm(s: str) -> str:
    return re.sub(r"[^a-z0-9]", "", s.lower())


def extract(question: str, answer: str):
    q_norm = norm(question)
    cands: list[str] = []
    seen = set()

    def add(span: str):
        span = span.strip().strip("\"'“”.,")
        if not span:
            return
        key = norm(span)
        if not key or key in seen:
            return
        seen.add(key)
        cands.append(span)

    crisp = False
    title_norms: list[str] = []
    # 1. quoted titles
    for m in QUOTE_RE.finditer(answer):
        add(m.group(1))
        title_norms.append(norm(m.group(1)))
        crisp = True
    # 2. years
    for m in YEAR_RE.finditer(answer):
        if m.group(1) not in question:
            add(m.group(1))
            crisp = True
    # 3. proper nouns not echoed from the question
    for m in PROPER_RE.finditer(answer):
        span = m.group(1).strip()
        if span in JUNK_CAP:
            continue
        n = norm(span)
        if n in q_norm:
            continue
        # drop if it is a substring of an already-captured quoted title
        if any(n in t for t in title_norms):
            continue
        add(span)
        crisp = True

    # 4. fallback content words for short answers if nothing crisp
    if not cands:
        toks = re.findall(r"[A-Za-z][A-Za-z\-]+", answer.lower())
        content = [t for t in toks if t not in STOP and norm(t) not in q_norm and len(t) > 3]
        # keep distinctive (first few unique)
        uniq = []
        for t in content:
            if t not in uniq:
                uniq.append(t)
        for t in uniq[:3]:
            add(t)

    confidence = "confident" if crisp else "ambiguous"
    return cands, confidence


def main():
    items = load_forget05()
    rows = []
    n_conf = n_amb = 0
    for it in items:
        cands, conf = extract(it.question, it.answer)
        if conf == "confident":
            n_conf += 1
        else:
            n_amb += 1
        rows.append([it.idx, it.author, it.question, it.answer,
                     " | ".join(cands), conf])

    out = "tofu_forget05_keywords_DRAFT.csv"
    with open(out, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["question_idx", "author", "question", "answer",
                    "candidate_keywords", "confidence"])
        w.writerows(rows)
    print(f"Wrote {out}: {len(rows)} rows  (confident={n_conf}, ambiguous={n_amb})")
    print("\n--- ambiguous rows (review first) ---")
    for r in rows:
        if r[5] == "ambiguous":
            print(f"  [q{r[0]:>3}] {r[2][:70]}")
            print(f"          A: {r[3][:90]}")
            print(f"          cands: {r[4]}")


if __name__ == "__main__":
    main()
