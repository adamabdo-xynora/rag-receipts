# Corpus Design — Engineered Test Properties

This corpus is a designed test battery, not filler text. Every property below is
load-bearing for retrieval and grounding tests. This file is the single source of
truth for test authors and for the project README: if a document changes, this file
changes with it.

All content is fictional. See `README.md`.

## Document Inventory

| id | docType | effectiveDate | version |
| --- | --- | --- | --- |
| `company-overview-and-service-areas` | process | 2024-09-02 | 2.1 |
| `wholesale-pricing-and-minimums` | pricing | 2025-03-03 | 4.0 |
| `account-tiers-and-benefits` | policy | 2025-01-20 | 3.2 |
| `new-account-onboarding` | process | 2024-11-11 | 2.4 |
| `delivery-zones-and-schedules` | process | 2025-02-24 | 5.1 |
| `hotel-and-caterer-service-notes` | process | 2025-04-07 | 1.3 |
| `rush-and-same-day-orders` | policy | 2025-02-24 | 2.2 |
| `returns-and-credits-policy` | policy | 2023-04-03 | 1.2 |
| `customer-care-handbook` | policy | 2025-02-10 | 3.0 |
| `dry-goods-product-specs` | product | 2025-01-06 | 3.1 |
| `paper-goods-product-specs` | product | 2024-12-09 | 2.6 |
| `linen-program-and-care` | product | 2025-03-17 | 2.0 |
| `damaged-and-short-shipment-claims` | process | 2024-10-21 | 2.3 |
| `wholesale-faq` | process | 2025-04-28 | 1.5 |

---

## 1. Contradiction Pair — the return window

Two live policy documents state a different, checkable return window for the same
class of goods. Neither says "superseded"; both are confidently worded; both are
plausible as the operative policy. The older document was never retired.

| role | doc id | section heading | claim |
| --- | --- | --- | --- |
| older | `returns-and-credits-policy` (v1.2, 2023-04-03) | `## Return Window and Condition Requirements` | "within **14 calendar days** of the delivery date" |
| newer | `customer-care-handbook` (v3.0, 2025-02-10) | `## Return Window for Stocked Goods` | "within **30 calendar days** of the delivery date" |

The two sections deliberately share most of their surrounding wording (delivery
date on the packing slip, unopened, original outer carton, no price stickers or
venue labelling), so the disagreement is isolated to a single number.

**Consequence.** The window decides whether a customer gets a credit note at all.
A day-22 request is either allowed or declined depending on which chunk is retrieved.

**Expected behaviour under test.** A grounded system must surface *both* documents
and flag the conflict, ideally noting the version/effectiveDate ordering. Answering
"14 days" or "30 days" alone — with a single citation and no hedge — is a failure,
even when the answer happens to match the newer document.

**Uniqueness check.** The string `calendar days` appears in exactly these two
sections and nowhere else in the corpus.

---

## 2. Single-Source Fact — cocoa shelf life

A concrete, quotable fact stated in exactly one section of exactly one document.

- **doc id:** `dry-goods-product-specs`
- **section heading:** `## Cocoa Powder — Storage and Shelf Life`
- **fact:** the 3 kg cocoa tin "is packed under nitrogen flush and carries a
  **540-day shelf life** from the pack date printed on the base of the tin."

**Uniqueness check.** `grep -rn "540" corpus/` matches only that section (and this
design file). No other document restates the shelf life, the nitrogen flush, or the
location of the pack date. The 90-day after-opening figure in the same section is
also single-source.

**Expected behaviour under test.** Recall must find this one chunk; a citation
pointing anywhere else is wrong. This is the retrieval-precision case and the
"can the system quote rather than paraphrase" case.

---

## 3. No-Answer Trap — Saturday delivery

**The question:** *"Does Larkspur deliver on Saturdays, and what is the Saturday
delivery surcharge?"*

**Why it sounds answerable.** Delivery is the richest topic in the corpus. Six of the
fourteen documents discuss it in operational detail: `delivery-zones-and-schedules`
(zones, cut-offs, nominated windows, redelivery, depot collection, public holiday
closures), `hotel-and-caterer-service-notes` (Zone B schedule, loading docks, event
holds), `rush-and-same-day-orders` (rush surcharge, same-day limits, refusals),
`wholesale-faq` (an explicit "How late can I order for the next delivery?" entry),
`company-overview-and-service-areas` (depots and zone definitions), and
`damaged-and-short-shipment-claims` (collection on the next scheduled run). Surcharges
of several kinds are stated in dollars: $45 Zone C freight, $38 rush, $32 redelivery,
$15 per week for a Sprout window nomination. A question about one more day-of-week
surcharge fits the pattern perfectly.

**Why nothing answers it.** The words *Saturday*, *Sunday*, and *weekend* appear
nowhere in the corpus (verify: `grep -rinE 'saturday|sunday|weekend' corpus/`).
Schedules are deliberately expressed in terms of a **"service day"** — a term used
repeatedly and never defined, and never enumerated as a span such as "Monday to
Friday". `## Public Holiday Closures` covers gazetted holidays only and says nothing
about days of the week. No section states which days runs operate, and no surcharge
list is presented as exhaustive.

**Expected behaviour under test.** The system must decline: the corpus does not say.
Inventing a Saturday surcharge by analogy with the rush or Zone C surcharge, or
inferring "no weekend service" from the absence of any mention, are both failures.
The correct answer names the gap — "service day" is never defined.

---

## 4. Two-Document Joins

Neither half is sufficient alone; each half lives in a different document.

### Join A — tier discount × list price (arithmetic)

**Question:** *"What does a Harvest Tier account pay for a 25 kg sack of bread flour?"*

| half | doc id | section heading | fact |
| --- | --- | --- | --- |
| 1 | `account-tiers-and-benefits` | `## Harvest Tier — Discount, Terms, and Qualification` | Harvest receives **12% off list price** |
| 2 | `wholesale-pricing-and-minimums` | `## Selected List Prices — Dry Goods` | Bread flour, 25 kg sack — **$58.00** list |

**Answer:** $58.00 − 12% = **$51.04** per sack, excluding tax and before freight.

The pricing document deliberately refuses to restate the percentages
(`## How Tier Discounts Interact With List Price` says the discounts "are published
in the account tier document, not here"), and the tier document never quotes a price.
`12%` appears in exactly one section; `$58.00` appears in exactly one section.

### Join B — surcharge × threshold (conditional)

**Question:** *"A Zone C cafe places an $820 order before tax. What freight do they pay?"*

| half | doc id | section heading | fact |
| --- | --- | --- | --- |
| 1 | `delivery-zones-and-schedules` | `## Zone C Freight Surcharge` | flat **$45 per drop**, waived at the Zone C free-freight threshold "published in the pricing document" |
| 2 | `wholesale-pricing-and-minimums` | `## Free-Freight Thresholds by Zone` | Zone C threshold is **$900** |

**Answer:** $820 < $900, so the threshold is not met and the customer pays the
**$45** surcharge. Retrieving only half 1 yields "it depends"; retrieving only half 2
yields a threshold with no charge attached.

---

## 5. Near-Duplicate Tempters

Passage pairs across different documents, similar in topic and sentence shape,
differing in a material detail. These exist to tempt citation of the wrong chunk.

### Pair A — tier terms (same sentence skeleton, different tier)

| doc id | section heading |
| --- | --- |
| `account-tiers-and-benefits` | `## Harvest Tier — Discount, Terms, and Qualification` |
| `new-account-onboarding` | `## Sprout Tier — Discount, Terms, and Qualification` |

Both open "…Tier accounts receive N% off list price on all stocked dry goods and
paper goods, and are offered net N payment terms after a clean payment history of…"
and both close "Tier is reviewed at the close of each quarter."
**Material differences:** 12% vs 5%; net 30 vs net 14; three months vs one month of
clean history; $9,000 rolling volume vs no volume requirement; one standing order per
week vs one per fortnight; window nomination free vs **$15 per week**.

### Pair B — zone schedules (same sentence skeleton, different zone)

| doc id | section heading |
| --- | --- |
| `delivery-zones-and-schedules` | `## Zone A Delivery Windows and Cut-Off` |
| `hotel-and-caterer-service-notes` | `## Zone B Delivery Windows and Cut-Off` |

Both run "Zone X is served by a … van run on each service day. The ordering cut-off
for Zone X is …, and the standard lead time is … Nominated windows are honoured on a
best-effort basis and are not guaranteed against traffic delays."
**Material differences:** cut-off 3:00 PM vs 1:00 PM; lead time one service day vs
two; morning window 6–10 AM vs 7–11 AM; afternoon window 1–5 PM vs 12–4 PM.

Note that the Zone B schedule lives in the hotel/caterer document rather than the
delivery document, so a system that anchors on "delivery" as a document-level topic
will retrieve Zone A for a Zone B question.

### Pair C — collection of goods going back to the depot

| doc id | section heading |
| --- | --- |
| `returns-and-credits-policy` | `## Collection of Returned Goods` |
| `damaged-and-short-shipment-claims` | `## Collection of Rejected Goods` |

Both begin "Goods … are collected on the account's next scheduled delivery run at no
charge" and both end with the same access-point sentence.
**Material differences:** returned goods **must** be palletised or boxed by the
customer, and an off-run collection is charged at the zone redelivery rate; rejected
claim goods **must not** be repacked, consolidated, or palletised, because that
destroys the evidence. The instructions are direct opposites.

### Pair D — the contradiction pair, doubling as a tempter

`returns-and-credits-policy` `## Return Window and Condition Requirements` and
`customer-care-handbook` `## Return Window for Stocked Goods` share three sentences
of near-identical condition language around the one number they disagree on. See §1.

### Pair E — intra-document tempter (bonus, not required)

`account-tiers-and-benefits` `## Harvest Tier — Discount, Terms, and Qualification`
and `## Orchard Tier — Discount, Terms, and Qualification` are the same skeleton
again at 18% / net 45 / six months / $26,000 / three standing orders / two windows.
Useful for testing chunk-level rather than document-level precision.

---

## Invariants To Preserve When Editing

- The words *Saturday*, *Sunday*, *weekend* must never appear (breaks §3).
- "Service day" must never be defined or enumerated as a weekday span (breaks §3).
- `540` must remain unique to the cocoa section (breaks §2).
- `calendar days` must remain unique to the two return-window sections (breaks §1).
- `12%`, `$58.00`, `$45`, and `$900` must each remain in exactly one section, in the
  documents named above (breaks §4).
- Neither returns document may acquire the words "superseded", "replaced", or
  "obsolete" (breaks §1).
- All emails stay `@example.com`; all phones stay in the `555-01xx` range.
