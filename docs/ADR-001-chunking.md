# ADR-001: One chunk per `##` section

**Status:** Accepted · **Date:** 2025-08-22 · **Applies to:** `src/chunk.ts`

## Context

Every answer this system gives has to come with a receipt: a citation a person can
check in seconds, and a quote that can be verified by string containment against the
text that was actually retrieved. That requirement pushes hard on the chunking
decision, because a chunk is the unit of both retrieval and citation. Whatever we
split on becomes what a citation points at.

The corpus is fourteen short Markdown documents, each with a five-key frontmatter
block and four to six `##` sections. The sections were written as self-contained
units: one policy, one specification, one procedure each.

## Decision

**One chunk per `##` section.** The chunk id is `<docId>#<section-slug>`, where the
slug is the heading lowercased with non-alphanumeric runs collapsed to single
hyphens — for example `delivery-zones-and-schedules#zone-c-freight-surcharge`.

Chunk text is a byte-exact substring of the source file. The only transformation is
trimming whitespace from the two ends of the section body.

Three things are errors rather than silent behaviour: text before the first heading
(it would belong to no chunk and become uncitable), two headings in one document
that slugify identically (a `-2` suffix would make the id underivable from the
heading), and any malformed document anywhere in the corpus (a half-loaded corpus
gives answers that look grounded and are missing their evidence).

## Size versus coherence

Sections here run roughly 60–120 words. That is the whole reason this works.

Small enough to cite precisely: when the answer points at
`delivery-zones-and-schedules#zone-c-freight-surcharge`, a reader opens one file,
finds one heading, and reads one short paragraph to confirm the $45 figure. A
citation that points at a 900-word document is not really a citation; it is homework.

Large enough to stand alone: each section states its subject, its rule, and its
exceptions without depending on the sentence before it. Retrieved on its own, out of
document order, it still means what it meant in place.

The alternative — fixed-token windows with overlap — was rejected. A fixed window
cuts wherever the token count runs out, which lands mid-sentence and often
mid-clause. That breaks the receipt in two ways. The quote a model wants to cite may
straddle a boundary and exist in no single chunk, so containment verification fails
on a quote that is perfectly honest. And the citation a human receives points at a
window index, which they cannot check against the source without tooling. Overlap
softens the first problem and makes the second worse, since the same sentence now
lives at two addresses. Semantic or recursive splitters land between those poles, but
they buy their coherence with a boundary you cannot predict from the source file —
and an unpredictable boundary is an unpredictable citation.

Section boundaries were already chosen by a human author for exactly the reason we
want them: they mark where one idea ends and the next begins.

## What this loses

**A fact spanning two sections needs both sections retrieved.** Nothing stitches
them back together. If the answer to a question requires the Zone C surcharge amount
from one section and the free-freight threshold from another document, retrieval has
to surface both, and top-k has to be wide enough to hold them. This is a real limit,
and it is the corpus's designed cross-document join cases that will expose it.

**Very long sections would be under-served.** A section far longer than its
neighbours is retrieved as one unit and dilutes its own embedding: the single vector
has to represent several ideas at once, and matches less sharply on any one of them.

**Section count drives the index size, not word count.** A document with one enormous
section and a document with eight small ones contribute one and eight chunks
respectively, regardless of length.

**Preamble text is rejected, not accommodated.** A document whose author writes an
intro before the first heading fails to load until the intro gets a heading. That is
a deliberate cost: the alternative is content that is silently unretrievable.

## What would change this decision

- **Much longer documents.** If sections grew past a few hundred words, the case for
  splitting within a section would return — likely paragraph-level chunks that keep a
  pointer to their parent section, so the citation stays `doc#section` while
  retrieval gets a finer unit.
- **Sections that exceed the embedding window.** A section that does not fit the
  model's input window has to be split; at that point the split rule needs to be
  defined here rather than left to whatever truncation the embedder applies silently.
- **A corpus without headings.** Documents with no reliable `##` structure (scraped
  pages, transcripts, PDFs) have no author-chosen boundary to inherit, and would need
  a different rule entirely.
- **Cross-section facts dominating the eval set.** If most questions turn out to need
  two sections of the same document, the answer is probably a parent-document
  expansion step at retrieval time rather than a different chunk unit.

## Consequences

`src/chunk.ts` stays a pure module: `node:fs/promises` and `node:path`, no SDK, no
third-party parser, no environment access. Frontmatter is parsed by hand because a
full YAML dependency is disproportionate to five flat string keys, and because its
permissiveness (coercing `version: 5.1` to a number, accepting unknown keys) is the
opposite of what we want. Chunk ids are stable across runs and machines, so they can
appear in tests, in logs, and in an answer shown to a person, and mean the same thing
in all three.
