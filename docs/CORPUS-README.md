# Larkspur Supply Co. — Evaluation Corpus

Fourteen short internal-style documents for a fictional wholesale supplier of dry
goods, paper goods, and table linens.

**Everything here is invented.** Larkspur Supply Co. does not exist. No real company,
brand, product, person, place, price, or policy is described. Every email address is
`@example.com` and every phone number is in the reserved `555-01xx` range. Larkspur is
the same fictional company whose CRM appears in the `mcp-capability-guard` repo — a
connected universe, invented on both sides.

This corpus exists only as an evaluation fixture for a retrieval-augmented generation
pipeline. It is a **designed test battery**, not sample text: it deliberately contains
an unretired contradiction, a fact stated in exactly one place, a rich topic area with
a hole in it, facts that can only be answered by joining two documents, and passages
written to be confusable with each other.

Do not "fix" the inconsistencies. They are the tests.

See [`CORPUS-DESIGN.md`](./CORPUS-DESIGN.md) for the exact document ids and section
headings behind each engineered property, and for the invariants to preserve when
editing.

## Format

Each document is a markdown file whose name matches its frontmatter `id`:

```yaml
---
id: kebab-case-id
title: Human readable title
docType: policy | pricing | product | process
effectiveDate: YYYY-MM-DD
version: N.N
---
```

Body content uses `##` section headings. Sections are the intended chunk boundary:
each one is written to stand alone as a coherent idea under a descriptive heading.
