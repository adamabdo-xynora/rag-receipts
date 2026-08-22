/**
 * verify.ts — a generated answer in, a verdict per claim out.
 *
 * PURE MODULE, and the strictest one in the project. It imports two *types* —
 * `Chunk` from `./chunk.js` and `SearchResult` from `./retrieve.js` — and
 * nothing else. No SDK, no `fetch`, no `process.env`, no model call, no clock,
 * no filesystem. Every value it produces is a function of its two arguments.
 * It generates nothing; it judges what a generator produced.
 *
 * ---------------------------------------------------------------------------
 * THE THREAT MODEL: CITATION LAUNDERING.
 * ---------------------------------------------------------------------------
 * A language model asked to cite its sources will produce citations. Some of
 * them will be real. The failure mode this module exists for is the rest, and
 * it has three shapes:
 *
 *   1. AN INVENTED CHUNK ID. The id format is regular and short and visible in
 *      the prompt, so a model can synthesise one that looks exactly like a real
 *      one — right document stem, plausible section slug — for a section that
 *      does not exist. Or a real id for a chunk that was never retrieved: the
 *      model could not have read that text, so it is not evidence of anything,
 *      even when the fact it is attached to happens to be true.
 *   2. A REAL ID ON THE WRONG QUOTE. The chunk exists, it was retrieved, and
 *      the passage quoted is simply not in it. The citation is a real address
 *      for a claim that lives somewhere else, or nowhere.
 *   3. A PARAPHRASE IN QUOTATION MARKS. The model read the passage, restated it
 *      in its own words, and wrapped the restatement in quotes. Two words moved,
 *      a number rounded, a hedge dropped. This is the dangerous one, because the
 *      paraphrase is usually *nearly* right, and "nearly right" about a
 *      surcharge or a cut-off time is a wrong answer that survives review.
 *
 * Each of those produces an answer that LOOKS grounded. It carries ids, it
 * carries quotation marks, it reads as authoritative, and a reader who does not
 * open the source documents cannot tell it from the real thing. The citation
 * machinery launders an ungrounded assertion into an apparently sourced one —
 * hence CITATION LAUNDERING. It is worse than an uncited hallucination, because
 * an uncited hallucination at least looks like what it is.
 *
 * PROMPT INSTRUCTIONS DO NOT PREVENT IT. "Only cite chunks you were given" and
 * "quote exactly" are requests, not constraints. They raise compliance and they
 * never reach certainty; nothing enforces them; and their failures are silent
 * and indistinguishable from success. A system whose grounding guarantee is a
 * sentence in a system prompt has no grounding guarantee; it has a hope.
 *
 * THIS MODULE IS THE CHECK THAT CANNOT BE TALKED OUT OF IT. It holds the actual
 * retrieved chunks and compares the model's output against their bytes. It has
 * no notion of plausibility, no similarity score, and no way to be persuaded: a
 * claim is grounded if and only if the bytes say so.
 *
 * ---------------------------------------------------------------------------
 * TWO STANDING RULES.
 * ---------------------------------------------------------------------------
 * NEVER THROWS ON MODEL OUTPUT. Malformed claims are the expected input, not an
 * exceptional one — a model that emits nonsense is doing a normal thing. Every
 * defect is data: it comes back as a `RejectedClaim` carrying a typed reason.
 * That holds below the type system too: a claim that is not even shaped like a
 * `Claim` — because it arrived as `JSON.parse` of model output and was cast —
 * is rejected as `malformed-claim` rather than thrown over.
 *
 * NEVER REPAIRS. No trimming a quote until it matches, no case folding, no
 * dropping the one bad citation and keeping the rest of the claim, no
 * downgrading an "inferred" claim to "stated" so its citation count fits. Every
 * repair is a small decision to publish something the model did not actually
 * support, and repairs compose: three lenient checks make an answer that passes
 * verification while being wrong in three places. A claim is grounded or it is
 * rejected whole.
 */

import type { Chunk } from "./chunk.js";
import type { SearchResult } from "./retrieve.js";

/* ===========================================================================
 * What a generator is expected to produce.
 * ======================================================================== */

/**
 * How a claim relates to its sources.
 *
 * `"stated"` — the corpus says this, in one place. `"inferred"` — this follows
 * from combining two or more passages, and is not written down anywhere as
 * such. The distinction is load-bearing for the reader: an inferred claim is
 * the module's own reasoning and deserves more scepticism than a restatement,
 * and a reader can only apply that scepticism if the label is honest.
 */
export type ClaimStatus = "stated" | "inferred";

/** The evidence for one citation: a passage the model says it read there. */
export interface SupportingQuote {
  /** Must be one of the enclosing claim's `citations`. */
  readonly chunkId: string;
  /** Checked by exact string containment against that chunk's text. */
  readonly quote: string;
}

/** One assertion in a generated answer, with its receipts attached. */
export interface Claim {
  /** The assertion as it appears in the answer. */
  readonly text: string;
  readonly status: ClaimStatus;
  /** Chunk ids this claim rests on. */
  readonly citations: readonly string[];
  /** At least one quote per cited chunk. */
  readonly supportingQuotes: readonly SupportingQuote[];
}

/* ===========================================================================
 * What can be wrong with one.
 * ======================================================================== */

/**
 * Every way a claim can fail verification. Each one is covered by a test.
 *
 * The list is closed and the names are stable, because they are what `summarise`
 * counts and what an eval report is read against. "Six claims rejected" is not
 * actionable; "six claims rejected, all `quote-not-in-chunk`" points at a
 * specific defect in a specific prompt.
 */
export type RejectionKind =
  /* --- shape: checked first, before anything touches the retrieved set --- */
  /** Not a `Claim` at runtime at all — a field missing, or of the wrong type. */
  | "malformed-claim"
  /** `text` is empty or nothing but whitespace. */
  | "empty-claim-text"
  /** `citations` is empty. An uncited claim is not a grounded claim. */
  | "empty-citations"
  /** The same chunk id is cited more than once. */
  | "duplicate-citation"
  /** A `"stated"` claim cites more than one chunk. */
  | "stated-claim-multiple-citations"
  /** An `"inferred"` claim cites exactly one chunk. */
  | "inferred-claim-single-citation"
  /** A supporting quote has an empty `quote` or an empty `chunkId`. */
  | "empty-quote"
  /** A supporting quote names a chunk the claim does not cite. */
  | "quote-for-uncited-chunk"
  /* --- content: checked against the chunks that were actually retrieved --- */
  /** A cited chunk id is not in the retrieved set that was shown to the model. */
  | "chunk-not-retrieved"
  /** A quote is not a literal substring of the chunk it is attributed to. */
  | "quote-not-in-chunk"
  /** A cited chunk has no supporting quote at all. */
  | "citation-without-quote";

/** Why one claim was rejected: machine-readable kind, human-readable message. */
export interface RejectionReason {
  readonly kind: RejectionKind;
  /** Names the claim and the specific defect, in a sentence. */
  readonly message: string;
  /** The chunk id at fault, where the failure is about one. */
  readonly chunkId: string | undefined;
}

/** A claim that did not verify, kept whole alongside its reason. */
export interface RejectedClaim {
  /** The original claim, byte for byte. Never edited, never partially kept. */
  readonly claim: Claim;
  readonly reason: RejectionReason;
}

/** The verdict on a whole answer. Every input claim appears in exactly one side. */
export interface VerifiedAnswer {
  /** Grounded claims, in input order. Same object identities as the input. */
  readonly verified: readonly Claim[];
  /** Rejected claims, in input order. */
  readonly rejected: readonly RejectedClaim[];
}

/* ===========================================================================
 * The checks.
 * ======================================================================== */

/** Keep messages readable when a claim or quote runs long. */
function snippet(text: string): string {
  const clipped = text.length > 80 ? `${text.slice(0, 80)}…` : text;
  return JSON.stringify(clipped);
}

/**
 * Emptiness is tested against the trimmed string; nothing else in this module
 * trims anything.
 *
 * This is a rejection test, not a repair: the trimmed value is discarded on the
 * spot and the containment check below still runs against the quote exactly as
 * the model wrote it. A whitespace-only quote has to be caught here precisely
 * because it would otherwise sail through containment — `indexOf(" ")` finds a
 * space in nearly every chunk in the corpus, so a claim citing a space would
 * verify.
 */
function isBlank(text: string): boolean {
  return text.trim() === "";
}

function reject(kind: RejectionKind, message: string, chunkId?: string): RejectionReason {
  return { kind, message, chunkId };
}

/**
 * Is this actually a `Claim` at runtime?
 *
 * The TypeScript type is a promise made by the caller, and the caller's input
 * is `JSON.parse` of whatever a model emitted. A missing `citations` array is
 * not a hypothetical — it is Tuesday — and reading `.length` off it would
 * throw a `TypeError` out of a function whose entire contract is that it does
 * not throw on model output. So the shape is checked rather than trusted, and a
 * claim that is not a claim is rejected like any other defect.
 *
 * This guards the *model's* half of the input only. `retrieved` comes from the
 * chunker, on our side of the trust boundary, and is not re-validated here.
 */
function isClaimShaped(value: unknown): value is Claim {
  if (typeof value !== "object" || value === null) return false;
  const claim = value as Partial<Record<keyof Claim, unknown>>;

  if (typeof claim.text !== "string") return false;
  if (claim.status !== "stated" && claim.status !== "inferred") return false;

  if (!Array.isArray(claim.citations)) return false;
  for (const chunkId of claim.citations as unknown[]) {
    if (typeof chunkId !== "string") return false;
  }

  if (!Array.isArray(claim.supportingQuotes)) return false;
  for (const entry of claim.supportingQuotes as unknown[]) {
    if (typeof entry !== "object" || entry === null) return false;
    const quote = entry as Partial<Record<keyof SupportingQuote, unknown>>;
    if (typeof quote.chunkId !== "string") return false;
    if (typeof quote.quote !== "string") return false;
  }

  return true;
}

/**
 * Shape checks. These run before any content check, and they never look at the
 * retrieved set — they are decidable from the claim alone.
 *
 * THE CARDINALITY RULE. A `"stated"` claim cites exactly one chunk; an
 * `"inferred"` claim cites two or more. The interesting half is the second one:
 * inference with a single parent is either a restatement wearing the wrong
 * label, or a leap from one passage to a conclusion that passage does not
 * contain. Both are rejected, and rejected *here*, before quote containment
 * runs, because a single-parent "inference" can pass every content check in
 * this file. Its lone quote is real and really in its chunk; what is fabricated
 * is the step from the quote to the claim, and no amount of byte comparison can
 * see that. The shape rule is the only place it is catchable, so it is caught
 * unconditionally rather than as a warning attached to an otherwise-passing
 * claim.
 *
 * Returns the first failure, or null if the claim is well-formed.
 */
function checkShape(claim: Claim): RejectionReason | null {
  if (!isClaimShaped(claim)) {
    return reject(
      "malformed-claim",
      "the claim is not a well-formed claim object: it is missing a field, or one of its fields is of the wrong type",
    );
  }

  if (isBlank(claim.text)) {
    return reject("empty-claim-text", "the claim has no text, so there is nothing to ground");
  }

  const name = snippet(claim.text);

  if (claim.citations.length === 0) {
    return reject("empty-citations", `the claim ${name} cites no chunks at all`);
  }

  const seen = new Set<string>();
  for (const chunkId of claim.citations) {
    if (seen.has(chunkId)) {
      // Not pedantry: without this, `["a", "a"]` would satisfy "two or more
      // citations" and let a single-parent inference through the cardinality
      // rule below on a technicality.
      return reject(
        "duplicate-citation",
        `the claim ${name} cites "${chunkId}" more than once; a chunk repeated is not a second source`,
        chunkId,
      );
    }
    seen.add(chunkId);
  }

  if (claim.status === "stated" && claim.citations.length !== 1) {
    return reject(
      "stated-claim-multiple-citations",
      `the claim ${name} is marked "stated" but cites ${claim.citations.length} chunks; a stated claim is what one passage says, so it has exactly one source`,
    );
  }
  if (claim.status === "inferred" && claim.citations.length < 2) {
    return reject(
      "inferred-claim-single-citation",
      `the claim ${name} is marked "inferred" but cites ${claim.citations.length} chunk; an inference from a single passage is either a restatement mislabelled or a leap`,
    );
  }

  for (const quote of claim.supportingQuotes) {
    if (isBlank(quote.chunkId)) {
      return reject(
        "empty-quote",
        `the claim ${name} carries a supporting quote with no chunk id, so it points at nothing`,
      );
    }
    if (isBlank(quote.quote)) {
      return reject(
        "empty-quote",
        `the claim ${name} carries an empty supporting quote for "${quote.chunkId}"`,
        quote.chunkId,
      );
    }
    if (!seen.has(quote.chunkId)) {
      return reject(
        "quote-for-uncited-chunk",
        `the claim ${name} carries a quote attributed to "${quote.chunkId}", which it does not cite`,
        quote.chunkId,
      );
    }
  }

  return null;
}

/**
 * Content checks, against the chunks that were actually retrieved.
 *
 * EXISTENCE comes first. A citation is checked against the retrieved set — not
 * against the corpus — because retrieval is what the model was shown. A chunk
 * that exists on disk but was never retrieved was never in the model's context,
 * so a citation to it cannot be a report of something read; it is a guess that
 * happens to name a real address. It fails even when the claim attached to it is
 * true, and that is the point: a true claim with fabricated provenance is
 * indistinguishable, from the outside, from a false one, and the whole value of
 * a receipt is that it can be checked. This is the subtlest of the three
 * laundering shapes and the one a corpus-wide lookup would quietly permit.
 *
 * QUOTE CONTAINMENT is `indexOf`. See the long comment inside.
 *
 * QUOTE COVERAGE closes the loop: a cited chunk with no quote is an assertion of
 * relevance the model never backed, and it is free to make. Requiring a quote
 * per citation means every id in the output has a passage behind it that this
 * module has personally checked.
 */
function checkContent(claim: Claim, retrieved: ReadonlyMap<string, Chunk>): RejectionReason | null {
  const name = snippet(claim.text);

  for (const chunkId of claim.citations) {
    if (!retrieved.has(chunkId)) {
      return reject(
        "chunk-not-retrieved",
        `the claim ${name} cites "${chunkId}", which is not in the retrieved set the model was shown; whether or not it exists in the corpus, the model could not have read it, so citing it is fabricated provenance`,
        chunkId,
      );
    }
  }

  for (const quote of claim.supportingQuotes) {
    // Guaranteed present: `checkShape` established that every quote's chunkId
    // is one of the citations, and the loop above established that every
    // citation is in the retrieved set.
    const chunk = retrieved.get(quote.chunkId);
    if (chunk === undefined || chunk.text.indexOf(quote.quote) === -1) {
      /* -------------------------------------------------------------------
       * WHY EXACT CONTAINMENT AND NOT A FUZZY MATCH.
       * -------------------------------------------------------------------
       * The obvious "improvement" here is to normalise before comparing —
       * collapse whitespace, fold case, unify smart quotes and em dashes — or
       * to accept a match above some edit-distance or embedding-similarity
       * threshold. Every one of those is the same move: it decides how much
       * invention is acceptable inside quotation marks. That is a policy
       * question, it is the central policy question of this project, and the
       * answer is none.
       *
       * A threshold has no defensible value. Pick 0.95 and a model may round
       * $45 to $50, flip "is not waived" to "is waived", or drop a "not", and
       * still clear the bar — the two strings differ in a handful of
       * characters and agree on everything else, which is exactly what a
       * high similarity score measures and exactly the wrong thing to be
       * measuring. The edits that matter most to a reader are the smallest
       * ones. Fuzzy matching is most permissive precisely where it is most
       * dangerous.
       *
       * And a fuzzy verifier fails invisibly. It emits the same confident
       * output as a strict one, so nobody discovers the tolerance until a
       * customer is quoted the wrong surcharge. `indexOf` has no tolerance to
       * discover. It answers one question — are these bytes in that chunk —
       * and its answer is checkable by anyone with a text editor, which is the
       * property that makes the receipt worth anything.
       *
       * The cost is real and accepted: an honest quote that differs by one
       * whitespace character is rejected. That failure is loud, it lands on
       * the generator's side of the line where it can be fixed by asking for
       * verbatim spans, and it is why `chunk.text` is a byte-exact substring
       * of the source file (see the invariant at the top of `chunk.ts`). A
       * false rejection costs an answer. A false acceptance costs the
       * guarantee.
       * ------------------------------------------------------------------- */
      return reject(
        "quote-not-in-chunk",
        `the claim ${name} attributes ${snippet(quote.quote)} to "${quote.chunkId}", and that chunk's text does not contain it; a quote is verified by exact containment, so a paraphrase, a reflowed line, or a single changed character all fail here`,
        quote.chunkId,
      );
    }
  }

  const quoted = new Set(claim.supportingQuotes.map((quote) => quote.chunkId));
  for (const chunkId of claim.citations) {
    if (!quoted.has(chunkId)) {
      return reject(
        "citation-without-quote",
        `the claim ${name} cites "${chunkId}" without quoting it; a citation with no quote is an assertion of relevance that nothing backs`,
        chunkId,
      );
    }
  }

  return null;
}

/* ===========================================================================
 * The entry point.
 * ======================================================================== */

/**
 * Verify a generated answer against the chunks that were retrieved for it.
 *
 * `retrieved` is the set the model was actually shown. Pass the same array that
 * was rendered into the prompt — not the corpus, and not the pre-threshold
 * search results if the prompt only carried the ones above the floor. The
 * existence check is only as honest as this argument.
 *
 * ONE REASON PER CLAIM. Checks run in a fixed order — shape, then existence,
 * then containment, then coverage — and the first failure stops the claim. A
 * claim is rejected whole, so enumerating its every defect would be detail
 * about something already discarded; the first reason is the one that names why
 * it is not in the answer. The fixed order also makes the reason deterministic,
 * which is what lets `summarise`'s counts be compared across runs.
 *
 * PURE. It reads its arguments and returns new arrays. Nothing is mutated —
 * not the claims, not their citation or quote arrays, not the retrieved chunks
 * — and verified claims come back as the very objects that went in. The same
 * inputs produce an identical result every time, which is what makes this
 * checkable in CI and quotable in an eval report.
 */
export function verifyAnswer(
  claims: readonly Claim[],
  retrieved: readonly Chunk[],
): VerifiedAnswer {
  // Built fresh on every call, from a plain read of the argument. First
  // occurrence wins for a repeated chunkId; the chunker guarantees ids are
  // unique within a document, and a caller that concatenates a chunk twice is
  // showing the model the same bytes twice, which changes nothing here.
  const byId = new Map<string, Chunk>();
  for (const chunk of retrieved) {
    if (!byId.has(chunk.chunkId)) byId.set(chunk.chunkId, chunk);
  }

  const verified: Claim[] = [];
  const rejected: RejectedClaim[] = [];

  for (const claim of claims) {
    const reason = checkShape(claim) ?? checkContent(claim, byId);
    if (reason === null) verified.push(claim);
    else rejected.push({ claim, reason });
  }

  return { verified, rejected };
}

/**
 * The chunks behind a list of search results, in rank order.
 *
 * A convenience so callers can hand `verifyAnswer` exactly what they rendered
 * into the prompt without unpacking it by hand — and so the "pass what the
 * model saw" rule stays a one-liner rather than an opportunity to pass the
 * corpus instead.
 */
export function retrievedChunks(results: readonly SearchResult[]): Chunk[] {
  return results.map((result) => result.chunk);
}

/* ===========================================================================
 * Reporting.
 * ======================================================================== */

/** Rejection counts by kind, with every kind present — zeroes included. */
export type RejectionCounts = { readonly [K in RejectionKind]: number };

/** What a demo prints and an eval asserts on. */
export interface VerificationSummary {
  /** `verified + rejected`. */
  readonly total: number;
  readonly verified: number;
  readonly rejected: number;
  /** Every `RejectionKind`, including the ones that did not occur. */
  readonly byReason: RejectionCounts;
}

/** Every kind, so `summarise` can report zeroes rather than omitting them. */
const REJECTION_KINDS: readonly RejectionKind[] = [
  "malformed-claim",
  "empty-claim-text",
  "empty-citations",
  "duplicate-citation",
  "stated-claim-multiple-citations",
  "inferred-claim-single-citation",
  "empty-quote",
  "quote-for-uncited-chunk",
  "chunk-not-retrieved",
  "quote-not-in-chunk",
  "citation-without-quote",
];

/**
 * Count a verdict by rejection reason.
 *
 * Kinds that did not occur are reported as 0 rather than left out. An absent
 * key reads as "not measured" and a 0 reads as "measured, none" — and the
 * difference is the whole content of a line like `quote-not-in-chunk: 0` in an
 * eval report.
 */
export function summarise(answer: VerifiedAnswer): VerificationSummary {
  const byReason: Record<RejectionKind, number> = Object.fromEntries(
    REJECTION_KINDS.map((kind) => [kind, 0]),
  ) as Record<RejectionKind, number>;

  for (const { reason } of answer.rejected) byReason[reason.kind] += 1;

  return {
    total: answer.verified.length + answer.rejected.length,
    verified: answer.verified.length,
    rejected: answer.rejected.length,
    byReason,
  };
}
