/**
 * answer.ts — a question in, a decision out.
 *
 * This is the module that decides what the system says, and it is equally the
 * module that decides when the system says nothing. Everything below it either
 * produces evidence (`chunk.ts`, `retrieve.ts`) or judges it (`verify.ts`).
 * Nothing below it is allowed to speak.
 *
 * THE RESULT IS A DISCRIMINATED UNION, NEVER A HEDGED STRING. A RAG system that
 * cannot find an answer has exactly one honest move, and it is not "Based on
 * the available documents, it appears that…". A hedge is a refusal that has
 * been disguised as an answer: it is indistinguishable from a grounded answer
 * to a reader skimming, it cannot be filtered by a caller, it cannot be counted
 * in an eval, and it cannot be routed to a human. `AnswerResult` makes the
 * difference a type. A caller that wants to display something has to look at
 * `kind` first, and the compiler makes it.
 *
 * THE GENERATOR IS INJECTED, exactly as `Embedder` is in `retrieve.ts`. The
 * interface is two strings in, one string out — no SDK types cross it — so
 * every test below runs the whole pipeline, including the real verifier over
 * the real corpus, with no network, no key, and no model. `anthropicGenerator`
 * is the one implementation that talks to Anthropic, it takes its client and
 * key as parameters, and NOTHING IN THIS FILE READS `process.env`.
 */

import Anthropic from "@anthropic-ai/sdk";

import type { Chunk } from "./chunk.js";
import {
  type Embedder,
  MIN_SIMILARITY,
  type SearchResult,
  type VectorIndex,
  partitionByThreshold,
  search,
} from "./retrieve.js";
import {
  type Claim,
  type RejectedClaim,
  retrievedChunks,
  verifyAnswer,
} from "./verify.js";

/* ===========================================================================
 * The generator seam.
 * ======================================================================== */

/**
 * The one thing answering needs from a language model.
 *
 * Contract: `generate` returns the model's text, unedited. It does not parse,
 * does not repair, does not retry on unparseable output, and does not invent a
 * response when the provider fails — a failure throws, because a provider
 * outage is not a statement about the corpus (see `answerQuestion`).
 *
 * Two strings in, one string out, and no SDK type in the signature. That is
 * deliberate: the moment an `Anthropic.Message` appears here, every fake in
 * every test has to construct one, and the tests start exercising the shape of
 * the SDK rather than the behaviour of this pipeline.
 */
export interface Generator {
  generate(system: string, user: string): Promise<string>;
}

/**
 * Cumulative usage across every completed generation call.
 *
 * CUMULATIVE, NOT PER-CALL, for the same reason as `EmbeddingUsage` in
 * `retrieve.ts`: a caller attributing usage to one unit of work snapshots
 * before and after and takes the difference, and the generator never has to
 * know what the unit is.
 */
export interface GenerationUsage {
  /** Sum of the provider-reported `usage.input_tokens` over completed calls. */
  readonly inputTokens: number;
  /** Sum of the provider-reported `usage.output_tokens` over completed calls. */
  readonly outputTokens: number;
  readonly calls: number;
  /** Wall-clock milliseconds across those calls, via the injected clock. */
  readonly totalLatencyMs: number;
}

/**
 * A generator that also reports what its calls cost in tokens and time.
 *
 * A SEPARATE INTERFACE rather than a change to `Generator`, so the seam stays
 * two strings in, one string out and every fake in every test keeps compiling.
 * Code that wants the accounting asks for this type explicitly.
 */
export interface InstrumentedGenerator extends Generator {
  usage(): GenerationUsage;
}

/** Every way generation can fail. Each one is covered by a test. */
export type GenerationErrorKind =
  /** The provider rejected the request, or the transport did. */
  | "transport"
  /** The model declined the request outright (`stop_reason: "refusal"`). */
  | "model-refusal"
  /** The response hit `max_tokens`; whatever JSON it holds is truncated. */
  | "truncated"
  /** The response carried no text blocks at all. */
  | "empty-response";

/**
 * Thrown for every generation failure.
 *
 * Carries a machine-readable `kind` and, for HTTP failures, the `status`.
 * Deliberately carries no request, no config, and no headers — same rule as
 * `EmbeddingError` in `retrieve.ts`, for the same reason: errors get logged,
 * serialised, and pasted into issue trackers, and a key that reaches an error
 * message has been published.
 */
export class GenerationError extends Error {
  readonly kind: GenerationErrorKind;
  /** HTTP status where the provider supplied one, `undefined` otherwise. */
  readonly status: number | undefined;

  constructor(kind: GenerationErrorKind, message: string, status?: number) {
    super(message);
    this.name = "GenerationError";
    this.kind = kind;
    this.status = status;
  }

  /** `Error` has no useful JSON form; the listed fields are the whole payload. */
  toJSON(): { name: string; kind: GenerationErrorKind; status: number | undefined; message: string } {
    return { name: this.name, kind: this.kind, status: this.status, message: this.message };
  }
}

/**
 * Everything `anthropicGenerator` needs, and nothing it can pick up on its own.
 *
 * `apiKey` is a parameter because this module does not read `process.env`.
 * Whoever owns the key passes the key; a module that reaches for ambient
 * credentials works in production and in tests for different reasons, and the
 * day it stops working the difference matters.
 *
 * `client` is injected for the same reason `fetchImpl` is injected into
 * `voyageEmbedder`: so this exact request-building and response-reading code is
 * exercised by tests with no network and no account. WHEN `client` IS SUPPLIED
 * IT IS USED VERBATIM AND `apiKey` IS NOT CONSULTED — the caller who built the
 * client already gave it a key. `apiKey` stays required rather than becoming
 * conditional so that the "whoever owns the key passes the key" rule has no
 * exception to remember.
 */
export interface AnthropicGeneratorConfig {
  /** Never logged, never interpolated, never attached to an error. */
  readonly apiKey: string;
  /** Model id, e.g. `claude-opus-5`. Required: never an implicit default. */
  readonly model: string;
  /** Used as-is when present; otherwise a client is built from `apiKey`. */
  readonly client?: Anthropic;
  /**
   * Millisecond clock for latency measurement. Defaults to `Date.now`.
   * Injected so a test can assert an exact latency instead of a range.
   */
  readonly clock?: () => number;
}

/**
 * Output ceiling for one answer.
 *
 * Generous on purpose. A truncated response is not a smaller answer — it is
 * invalid JSON, which this pipeline turns into a refusal, so the failure is
 * loud but it also throws away a real answer for no reason. Claims carry
 * verbatim quotes, so the JSON runs several times the length of the prose.
 */
const MAX_TOKENS = 16_000;

/**
 * A live generator backed by the Anthropic Messages API.
 *
 * Sampling parameters are not passed: they are removed on the current models,
 * and this call wants the model's own best JSON rather than a temperature
 * setting someone tuned once and never revisited.
 *
 * USAGE IS COUNTED THE MOMENT A RESPONSE ARRIVES, before the stop-reason
 * checks below. A refusal or a truncation is still a billed response — the
 * provider metered it and will charge for it — so counting only the calls
 * this module went on to accept would report a cost smaller than the invoice.
 * (Every caller aborts the run when those checks throw, so in practice the
 * distinction is unobservable today; the ordering is chosen for the day it
 * is not.) The token fields are read tolerantly, like the embedder's: a
 * response without them is still an answer, and instrumentation must not
 * break the thing it instruments.
 */
export function anthropicGenerator(config: AnthropicGeneratorConfig): InstrumentedGenerator {
  const client = config.client ?? new Anthropic({ apiKey: config.apiKey });
  const clock = config.clock ?? Date.now;

  let inputTokens = 0;
  let outputTokens = 0;
  let calls = 0;
  let totalLatencyMs = 0;

  return {
    usage(): GenerationUsage {
      return { inputTokens, outputTokens, calls, totalLatencyMs };
    },

    async generate(system: string, user: string): Promise<string> {
      const startedAt = clock();

      let response: Anthropic.Message;
      try {
        response = await client.messages.create({
          model: config.model,
          max_tokens: MAX_TOKENS,
          system,
          messages: [{ role: "user", content: user }],
        });
      } catch (error) {
        // Status only. The provider's error object is not attached as `cause`
        // and its body is not interpolated: an upstream error can echo the
        // request, and the request carries the key.
        const status = error instanceof Anthropic.APIError ? error.status : undefined;
        throw new GenerationError(
          "transport",
          `the generation request to model "${config.model}" could not be completed${status === undefined ? "" : ` (HTTP ${status})`}`,
          status,
        );
      }

      // The response arrived: count it, whatever the stop-reason checks below
      // decide about it. See the usage note in this function's header comment.
      const reportedIn: unknown = response.usage.input_tokens;
      const reportedOut: unknown = response.usage.output_tokens;
      if (typeof reportedIn === "number" && Number.isFinite(reportedIn)) inputTokens += reportedIn;
      if (typeof reportedOut === "number" && Number.isFinite(reportedOut)) outputTokens += reportedOut;
      calls += 1;
      totalLatencyMs += clock() - startedAt;

      // Checked before the content is read. A refusal or a truncation still
      // returns content blocks, and reading them without checking turns a
      // known-bad response into an unexplained parse failure downstream.
      if (response.stop_reason === "refusal") {
        throw new GenerationError(
          "model-refusal",
          `model "${config.model}" declined to answer${response.stop_details?.category == null ? "" : ` (category: ${response.stop_details.category})`}`,
        );
      }
      if (response.stop_reason === "max_tokens") {
        throw new GenerationError(
          "truncated",
          `model "${config.model}" hit the ${MAX_TOKENS}-token output ceiling, so its JSON is cut off mid-structure`,
        );
      }

      const text = response.content
        .filter((block): block is Anthropic.TextBlock => block.type === "text")
        .map((block) => block.text)
        .join("");

      if (text.trim() === "") {
        throw new GenerationError(
          "empty-response",
          `model "${config.model}" returned no text blocks (stop_reason: ${response.stop_reason ?? "null"})`,
        );
      }

      return text;
    },
  };
}

/* ===========================================================================
 * The result.
 * ======================================================================== */

/**
 * Why the system said nothing. Each one is reachable, and each one is covered
 * by a test that reaches it.
 */
export type RefusalReason =
  /** Search ran and everything it found fell below the similarity floor. */
  | "no-relevant-documents"
  /** The generator produced claims and every one of them failed verification. */
  | "claims-failed-verification"
  /** Verified claims disagree across documents; resolving is not ours to do. */
  | "contradictory-sources"
  /** The generator's output did not parse, or parsed to zero claims. */
  | "no-claims-produced";

/** Grounded claims survived, and the ones that did not are shown alongside. */
export interface AnsweredResult {
  readonly kind: "answered";
  /** Verified claims, in generator order. Every one has checked receipts. */
  readonly claims: readonly Claim[];
  /**
   * Claims stripped by the verifier, with their reasons. REPORTED, NEVER
   * HIDDEN. A partial answer whose discarded half is invisible reads as a
   * complete one, and the reader has no way to know the model asserted three
   * more things that turned out to be ungrounded — which is exactly the signal
   * that should make them go read the sources themselves.
   */
  readonly rejectedClaims: readonly RejectedClaim[];
  /** The above-threshold results that were rendered into the prompt. */
  readonly retrieved: readonly SearchResult[];
}

/** The system declined, and said why in enough detail to debug. */
export interface RefusedResult {
  readonly kind: "refused";
  readonly reason: RefusalReason;
  /**
   * The specifics: what was searched, the best score seen, which claims were
   * stripped and why, both sides of a contradiction.
   *
   * A REFUSAL THAT CANNOT BE DEBUGGED IS A SHRUG WITH BETTER GRAMMAR. "I don't
   * have enough information" is unfalsifiable — it cannot distinguish a corpus
   * that genuinely lacks the answer from a threshold set too high, an embedder
   * misconfigured, an index built over the wrong directory, or a prompt the
   * model ignored. Those four have different fixes and identical symptoms
   * unless the refusal names numbers.
   */
  readonly detail: string;
  /**
   * The below-threshold results, best first. Kept for the same reason
   * `partitionByThreshold` keeps them: "we searched and everything fell short"
   * and "we never searched" are different facts, and an empty array cannot tell
   * them apart from a broken index.
   */
  readonly nearMisses: readonly SearchResult[];
}

export type AnswerResult = AnsweredResult | RefusedResult;

/* ===========================================================================
 * Prompt construction.
 * ======================================================================== */

/**
 * The system prompt.
 *
 * THIS PROMPT ASKS NICELY. THE VERIFIER DOES NOT CARE WHETHER IT COMPLIED.
 * Every rule below is also a check in `verify.ts`, and the check is what
 * decides: cardinality, quote containment, quote-per-citation, ids drawn from
 * the retrieved set. If the model ignores all of it, nothing here fails — the
 * claims are simply rejected, and `answerQuestion` refuses or strips.
 *
 * That is the whole division of labour, and it is worth being explicit about
 * because the alternative is so tempting. PROMPT IS OPTIMISATION; THE CHECK IS
 * ENFORCEMENT. A better prompt raises the fraction of claims that survive — it
 * is worth writing well, and worth iterating on against an eval. It never
 * changes what survives. A system whose grounding guarantee lives in this
 * string has no guarantee; it has a hope with good phrasing. So this text may
 * be rewritten freely by anyone tuning quality, and nothing about the project's
 * correctness rides on the rewrite.
 */
export function buildSystemPrompt(): string {
  return [
    "You answer questions strictly from the retrieved document chunks supplied in the user message. You have no other knowledge of this company, its policies, or its products.",
    "",
    "Reply with a single JSON object and nothing else. No prose before it, no prose after it, no markdown code fence.",
    "",
    "Schema:",
    "",
    '{ "claims": [ { "text": string, "status": "stated" | "inferred", "citations": string[], "supportingQuotes": [ { "chunkId": string, "quote": string } ] } ] }',
    "",
    "Rules:",
    '- Break the answer into separate claims. One assertion per claim, written as a full sentence the reader can act on.',
    '- "stated" means one chunk says this. A "stated" claim cites exactly one chunkId.',
    '- "inferred" means this follows from combining two or more chunks and is written down nowhere as such. An "inferred" claim cites two or more chunkIds.',
    "- Cite only chunkIds that appear in the user message, copied exactly. Never construct, correct, or guess a chunkId.",
    "- Every cited chunkId needs at least one supporting quote attributed to it, and every supporting quote's chunkId must be one you cited.",
    "- Every quote must be copied character for character from that chunk's text: same wording, same punctuation, same spacing, same numbers. Do not paraphrase, tidy, shorten with an ellipsis, or fix anything inside quotation marks.",
    '- If the chunks do not support an answer, return { "claims": [] }. An empty list is a valid and useful response. Never fill the gap from general knowledge.',
    "- If two chunks disagree, report both as separate claims with their own citations. Do not pick a winner and do not average them.",
  ].join("\n");
}

/**
 * The user message: the question, then every retrieved chunk labelled with the
 * exact `chunkId` to cite it by.
 *
 * The ids are printed verbatim because the model can only cite an id it has
 * actually been shown — anything else is a synthesis that happens to look like
 * an address, and `verify.ts` rejects it as `chunk-not-retrieved`. Giving real
 * ids in a stable, obvious position is the cheapest way to raise the fraction
 * of citations that are real. It is not a guarantee; see `buildSystemPrompt`.
 *
 * The metadata lines (document, section, version, effective date) are here so
 * a model looking at two versions of the same policy can see that they are two
 * documents rather than one repeated passage. The similarity score is printed
 * for the human reading a prompt dump; nothing asks the model to use it.
 */
export function buildUserMessage(question: string, results: readonly SearchResult[]): string {
  const blocks = results.map((result, position) => {
    const { chunk } = result;
    return [
      `--- chunk ${position + 1} of ${results.length} ---`,
      `chunkId: ${chunk.chunkId}`,
      `document: ${chunk.docTitle} (${chunk.docType}, version ${chunk.version}, effective ${chunk.effectiveDate})`,
      `section: ${chunk.sectionHeading}`,
      `similarity: ${result.score.toFixed(4)}`,
      "text:",
      chunk.text,
    ].join("\n");
  });

  return [
    `Question: ${question}`,
    "",
    `Retrieved chunks (${results.length}). Cite by the exact chunkId shown.`,
    "",
    blocks.join("\n\n"),
  ].join("\n");
}

/* ===========================================================================
 * Parsing.
 * ======================================================================== */

/**
 * The outcome of reading a generator's response. A failure is a value with a
 * reason on it, never an exception and never a guess.
 */
export type ParseResult =
  | { readonly ok: true; readonly claims: readonly Claim[] }
  | { readonly ok: false; readonly problem: string };

/** Opening fence, with or without a language tag, on its own line. */
const FENCE_OPEN = /^```[A-Za-z0-9_-]*[ \t]*\r?\n/;
/** Closing fence at the end of the string. */
const FENCE_CLOSE = /\r?\n[ \t]*```[ \t]*$/;

/**
 * Strip one markdown code fence, if the whole response is wrapped in one.
 *
 * THIS IS THE ONE ACCOMMODATION, AND IT IS NOT A REPAIR. A fence is packaging
 * around an intact payload: unwrapping it changes no byte of the JSON, and the
 * operation is exactly reversible. Everything else a "lenient" parser might do
 * — pulling the first `{...}` span out of surrounding prose, closing an unclosed
 * bracket, stripping a trailing comma, repairing a truncated string — invents
 * content, and the invented content ends up in a claim that gets published.
 *
 * Fenced output is also the single most common deviation from "JSON only",
 * because a model that has been trained to render code blocks will render one
 * no matter how the prompt is worded. Refusing over it would throw away sound
 * answers for a formatting habit, which fails in the expensive direction.
 */
export function stripCodeFence(raw: string): string {
  const trimmed = raw.trim();
  if (!FENCE_OPEN.test(trimmed) || !FENCE_CLOSE.test(trimmed)) return trimmed;
  return trimmed.replace(FENCE_OPEN, "").replace(FENCE_CLOSE, "").trim();
}

/** Keep messages readable when a response or a value runs long. */
function snippet(text: string, limit = 200): string {
  const clipped = text.length > limit ? `${text.slice(0, limit)}…` : text;
  return JSON.stringify(clipped);
}

/**
 * Read a generator response into claims.
 *
 * STRICT ON PURPOSE. The accepted shape is exactly one: an object with a
 * `claims` array. Not a bare array, not `{ answer: [...] }`, not an object with
 * a `claims` key holding a single claim. Each near-miss is one more shape to
 * guess between, and a parser that guesses is deciding what the model meant —
 * which is the same act as writing the answer itself, performed with less
 * information and no receipts.
 *
 * THE ELEMENTS ARE NOT PRE-SCREENED. Whatever is in the array is passed to the
 * verifier as-is, malformed entries included, because the verifier's rejections
 * are visible in the output (`rejectedClaims`, with a typed reason) whereas
 * anything dropped here would vanish silently. `verifyAnswer` is built for this
 * — it re-checks the shape of every claim at runtime rather than trusting the
 * type — so the cast below is safe in the only sense that matters.
 */
export function parseClaims(raw: string): ParseResult {
  const source = stripCodeFence(raw);

  if (source === "") {
    return { ok: false, problem: "the generator returned an empty response" };
  }

  let parsed: unknown;
  try {
    parsed = JSON.parse(source);
  } catch (error) {
    const message = error instanceof Error ? error.message : String(error);
    return {
      ok: false,
      problem: `the generator's response is not valid JSON (${message}); raw response: ${snippet(raw)}`,
    };
  }

  if (typeof parsed !== "object" || parsed === null || Array.isArray(parsed)) {
    return {
      ok: false,
      problem: `the generator's response parsed to ${Array.isArray(parsed) ? "an array" : typeof parsed}, not the required { "claims": [...] } object; raw response: ${snippet(raw)}`,
    };
  }

  const claims = (parsed as Record<string, unknown>)["claims"];
  if (!Array.isArray(claims)) {
    return {
      ok: false,
      problem: `the generator's response has no "claims" array; raw response: ${snippet(raw)}`,
    };
  }

  return { ok: true, claims: claims as readonly Claim[] };
}

/* ===========================================================================
 * Contradiction surfacing.
 * ===========================================================================
 * THE CORPUS CONTAINS A DELIBERATE CONTRADICTION: the return window for stocked
 * goods is 14 calendar days in `returns-and-credits-policy` (v1.2, effective
 * 2023-04-03) and 30 calendar days in `customer-care-handbook` (v3.0, effective
 * 2025-02-10). Neither document says "superseded". Both are confidently worded.
 * A day-22 request is either allowed or declined depending on which one you
 * read.
 *
 * WHY THIS IS NOT RESOLVED BY RECENCY.
 * -----------------------------------
 * Sorting by `effectiveDate` and returning the newer figure would produce a
 * clean, confident, single-number answer, and it is the obvious thing to do.
 * It is also the system substituting its judgment for the reader's on exactly
 * the question the reader needs to decide.
 *
 * THE NEWER DOCUMENT IS NOT AUTOMATICALLY OPERATIVE. All of these are ordinary:
 *   - the newer document is a draft, a proposal, or an internal handbook that
 *     never had the authority to amend the policy it appears to contradict;
 *   - the older document is the contractual one, and the customer's signed
 *     terms incorporate it by reference at its stated version;
 *   - the newer one applies to a segment, region, or channel that this question
 *     is not about, and its scope sentence is in a section that was not
 *     retrieved;
 *   - both are live and the disagreement is an unnoticed drafting error, which
 *     the reader is now the first person in a position to report;
 *   - the newer one was effective for a window that has since closed.
 * Which of those is true is not in the corpus. It is knowledge the reader has
 * and this module does not.
 *
 * So picking one silently does not resolve the ambiguity — it HIDES it, and it
 * hides it behind the same confident presentation a genuinely unanimous answer
 * would get. The reader loses the one fact they most needed: that the company's
 * own documents disagree, and somebody has to decide which governs. Surfacing
 * both, with versions and effective dates attached, hands them the decision
 * along with what they need to make it. That is strictly more useful than a
 * number, even though it looks like less of an answer.
 *
 * The metadata is still reported, and ordering by date is fine as PRESENTATION.
 * What is forbidden is letting the date decide.
 *
 * ---------------------------------------------------------------------------
 * WHAT THE DETECTOR ACTUALLY DOES, AND WHAT IT DOES NOT.
 * ---------------------------------------------------------------------------
 * It is a lexical heuristic, and calling it anything grander would be a lie
 * that gets believed. It fires when two verified claims, resting on chunks from
 * DIFFERENT documents, quote the SAME unit word with DIFFERENT numbers, while
 * their subjects overlap lexically. "14 calendar days" against "30 calendar
 * days", where both claims are about a return window.
 *
 * KNOWN LIMITS — all of these are real, none are hypothetical:
 *   - NUMBERS ONLY. A contradiction with no differing number in it — "returns
 *     are accepted" against "returns are not accepted", "must be palletised"
 *     against "must not be palletised" — is invisible to this function. The
 *     corpus contains such a pair on purpose (see `docs/CORPUS-DESIGN.md`,
 *     Pair C) and this detector does not see it.
 *   - VOCABULARY, NOT MEANING. The unit words must match as words. "14 calendar
 *     days" against "two weeks", or against "30 days", finds nothing.
 *   - PAIRWISE, ACROSS CLAIMS. A single `inferred` claim that cites both
 *     documents and quotes both numbers is one claim, so no pair exists and
 *     nothing fires.
 *   - CROSS-DOCUMENT ONLY. Two sections of one document that disagree are
 *     skipped by construction, because same-document differences are usually
 *     different subjects rather than a conflict.
 *   - FALSE POSITIVES ARE POSSIBLE. Two genuinely different quantities that
 *     share a unit and enough subject vocabulary will trip it. That direction
 *     costs an answer and is visible in the refusal detail; the other direction
 *     publishes one side of a live disagreement as fact.
 * A caller that needs general contradiction detection needs a different
 * mechanism. This one is a floor, not a ceiling, and it is documented as such
 * so nobody builds on it believing otherwise.
 * ======================================================================== */

/** One side of a disagreement, with everything needed to go read the source. */
export interface ContradictoryPosition {
  /** The verified claim taking this position. */
  readonly claim: Claim;
  /** The claim's citations, exactly as it made them. */
  readonly citations: readonly string[];
  readonly docId: string;
  readonly docTitle: string;
  readonly version: string;
  readonly effectiveDate: string;
  /** The chunk the differing value was quoted from. */
  readonly chunkId: string;
  /** The supporting quote carrying the value, byte for byte. */
  readonly quote: string;
  /** The measurement as written, e.g. `14 calendar days`. */
  readonly value: string;
}

/** Two positions that disagree, and the unit they disagree in. */
export interface Contradiction {
  /** The shared unit word the two values are measured in, e.g. `days`. */
  readonly unit: string;
  /** Both sides. NEITHER IS MARKED CORRECT — see the section comment. */
  readonly positions: readonly [ContradictoryPosition, ContradictoryPosition];
}

/**
 * A number and the word it is measured in, sliced out of a quote.
 *
 * Up to two following words are captured so that `14 calendar days` yields the
 * keys `calendar`, `days`, and `calendar days` — matching on any one of them is
 * what lets `14 calendar days` meet `30 calendar days` without requiring the
 * two documents to have been written by the same person.
 */
const MEASUREMENT = /(\$?\d+(?:,\d{3})*(?:\.\d+)?%?)[ \t]+([A-Za-z]+)(?:[ \t]+([A-Za-z]+))?/g;

/** Words that carry no subject and no unit. Deliberately short. */
const STOPWORDS = new Set([
  "the", "a", "an", "and", "or", "of", "to", "in", "on", "at", "for", "from",
  "by", "with", "per", "is", "are", "was", "were", "be", "been", "that", "this",
  "these", "those", "it", "its", "as", "not", "no", "any", "all", "each", "must",
  "may", "can", "will", "shall", "within", "after", "before", "than", "then",
  "when", "which", "who", "whom", "into", "over", "under", "up", "out",
]);

/** One measurement found in a quote, with the keys it can be matched on. */
interface Measurement {
  /** The numeric text, normalised: no `$`, no `%`, no thousands separators. */
  readonly magnitude: string;
  /** As written, e.g. `14 calendar days`. */
  readonly text: string;
  /** Unit keys this measurement can match on, e.g. `calendar`, `days`. */
  readonly units: readonly string[];
}

function normaliseMagnitude(raw: string): string {
  return raw.replace(/[$%,]/g, "");
}

/** Every measurement in a string, in order of appearance. */
function findMeasurements(text: string): Measurement[] {
  const found: Measurement[] = [];
  // A fresh regex per call: a module-level /g regex carries `lastIndex` between
  // calls, which would make this function's result depend on call order.
  const pattern = new RegExp(MEASUREMENT.source, "g");

  for (;;) {
    const match = pattern.exec(text);
    if (match === null) break;

    const magnitude = normaliseMagnitude(match[1] ?? "");
    const first = (match[2] ?? "").toLowerCase();
    const second = (match[3] ?? "").toLowerCase();

    const units: string[] = [];
    if (first !== "" && !STOPWORDS.has(first)) units.push(first);
    if (second !== "" && !STOPWORDS.has(second)) units.push(second);
    if (first !== "" && second !== "" && !STOPWORDS.has(second)) {
      units.push(`${first} ${second}`);
    }
    if (units.length === 0) continue;

    found.push({ magnitude, text: match[0].trim(), units });
  }

  return found;
}

/** Content words of a string: lowercased, de-punctuated, stopwords removed. */
function contentWords(text: string): Set<string> {
  const words = text
    .toLowerCase()
    .split(/[^a-z0-9]+/)
    .filter((word) => word.length >= 3 && !STOPWORDS.has(word) && !/^\d+$/.test(word));
  return new Set(words);
}

/**
 * How many content words two claims share.
 *
 * The subject of a claim is taken to be its own text plus the section headings
 * of the chunks it cites — the headings are what a human skims to decide
 * whether two passages are about the same thing, and they are short enough not
 * to swamp the overlap with incidental vocabulary the way full chunk text would.
 */
function subjectOverlap(a: readonly string[], b: readonly string[]): number {
  const left = contentWords(a.join(" "));
  const right = contentWords(b.join(" "));
  let shared = 0;
  for (const word of left) if (right.has(word)) shared += 1;
  return shared;
}

/**
 * Two shared subject words. Below this, any two claims mentioning a number of
 * days look like a conflict; much above it, two documents that disagree while
 * using different vocabulary stop being seen. It is a threshold with the
 * status `MIN_SIMILARITY` used to have before its calibration run: a
 * defensible starting point, not a measured one.
 */
const MIN_SUBJECT_OVERLAP = 2;

/** The subject strings for a claim: its text plus its chunks' headings. */
function subjectOf(claim: Claim, byId: ReadonlyMap<string, Chunk>): string[] {
  const parts = [claim.text];
  for (const chunkId of claim.citations) {
    const chunk = byId.get(chunkId);
    if (chunk !== undefined) parts.push(chunk.sectionHeading);
  }
  return parts;
}

function positionFrom(
  claim: Claim,
  chunk: Chunk,
  quote: string,
  measurement: Measurement,
): ContradictoryPosition {
  return {
    claim,
    citations: claim.citations,
    docId: chunk.docId,
    docTitle: chunk.docTitle,
    version: chunk.version,
    effectiveDate: chunk.effectiveDate,
    chunkId: chunk.chunkId,
    quote,
    value: measurement.text,
  };
}

/**
 * Look for a cross-document disagreement among VERIFIED claims.
 *
 * Verified only, and that ordering is load-bearing: an unverified claim's quote
 * may not exist in the chunk it names, so a "contradiction" between two
 * unverified claims can be a disagreement between two fabrications. Running
 * after verification means both sides of anything reported here have already
 * been checked against the bytes on disk.
 *
 * `retrieved` is needed because a `Claim` carries chunk ids and nothing else —
 * the document title, version, and effective date that make a contradiction
 * actionable live on the `Chunk`. Pass the same chunks the claims were verified
 * against.
 *
 * Returns the first disagreement in claim order, or null. First rather than all
 * because the outcome is the same — refuse, and show the reader the conflict —
 * and a deterministic single result is what makes the refusal text stable
 * across runs.
 */
export function detectContradiction(
  claims: readonly Claim[],
  retrieved: readonly Chunk[],
): Contradiction | null {
  const byId = new Map<string, Chunk>();
  for (const chunk of retrieved) {
    if (!byId.has(chunk.chunkId)) byId.set(chunk.chunkId, chunk);
  }

  /** Per claim: every measurement in every supporting quote, with its chunk. */
  const measured = claims.map((claim) => ({
    claim,
    subject: subjectOf(claim, byId),
    values: claim.supportingQuotes.flatMap((supporting) => {
      const chunk = byId.get(supporting.chunkId);
      if (chunk === undefined) return [];
      return findMeasurements(supporting.quote).map((measurement) => ({
        chunk,
        quote: supporting.quote,
        measurement,
      }));
    }),
  }));

  for (let i = 0; i < measured.length; i += 1) {
    const left = measured[i];
    if (left === undefined) continue;

    for (let j = i + 1; j < measured.length; j += 1) {
      const right = measured[j];
      if (right === undefined) continue;

      if (subjectOverlap(left.subject, right.subject) < MIN_SUBJECT_OVERLAP) continue;

      for (const a of left.values) {
        for (const b of right.values) {
          // Different documents, same unit, different number. All three.
          if (a.chunk.docId === b.chunk.docId) continue;
          if (a.measurement.magnitude === b.measurement.magnitude) continue;

          const unit = a.measurement.units.find((candidate) =>
            b.measurement.units.includes(candidate),
          );
          if (unit === undefined) continue;

          return {
            unit,
            positions: [
              positionFrom(left.claim, a.chunk, a.quote, a.measurement),
              positionFrom(right.claim, b.chunk, b.quote, b.measurement),
            ],
          };
        }
      }
    }
  }

  return null;
}

/** One side of a contradiction, written out for the refusal detail. */
function describePosition(position: ContradictoryPosition): string {
  return [
    `"${position.value}" — ${position.docTitle} (${position.docId}), version ${position.version},`,
    `effective ${position.effectiveDate}; cited as ${position.citations.join(", ")};`,
    `quoted from ${position.chunkId}: ${JSON.stringify(position.quote)}`,
  ].join(" ");
}

/* ===========================================================================
 * The pipeline.
 * ======================================================================== */

/**
 * How many chunks to retrieve and, if they clear the floor, to show the model.
 *
 * Enough that a cross-document join or a contradiction pair can both land in
 * one prompt; small enough that a weak match cannot pad the context until
 * something in it looks relevant.
 */
export const DEFAULT_K = 6;

/** Everything `answerQuestion` needs. All of it injected, none of it ambient. */
export interface AnswerOptions {
  readonly index: VectorIndex;
  readonly embedder: Embedder;
  readonly generator: Generator;
  /** Defaults to `DEFAULT_K`. */
  readonly k?: number;
  /** Defaults to `MIN_SIMILARITY` from `retrieve.ts`. */
  readonly threshold?: number;
}

/** `0.4123` — enough digits to compare two runs, few enough to read. */
function score(value: number): string {
  return value.toFixed(4);
}

/** "best: 0.2841 (chunk-id)", or a plain statement that nothing came back. */
function describeBest(results: readonly SearchResult[]): string {
  const best = results[0];
  if (best === undefined) return "the search returned no rows at all";
  return `the best score was ${score(best.score)}, for "${best.chunk.chunkId}"`;
}

/**
 * Search, decide whether there is anything to say, and only then generate.
 *
 * THE ORDER IS THE POINT. Retrieval runs, the threshold is applied, and if
 * nothing clears it THE GENERATOR IS NEVER CALLED.
 *
 * WHY NOT GENERATE FIRST AND CHECK AFTERWARDS. It is the more natural pipeline
 * to write — one path, always produce something, filter at the end — and it is
 * wrong in a way that no amount of downstream checking can repair.
 *
 * A model handed five irrelevant passages and a question does not answer "these
 * do not address your question". It answers the question, out of the passages
 * it was given plus everything it knows, because that is what the shape of the
 * request asks for. The output that comes back is fluent, cited, and unsound,
 * and BY THAT POINT THE HALLUCINATION HAS ALREADY BEEN PAID FOR — in tokens, in
 * latency, and in risk, because it now exists as text that some code path has
 * to be trusted to discard. The check that runs afterwards can only ever be
 * cleanup: it deletes the bad answer, it cannot un-generate it, and every
 * cleanup step is one bug away from letting it through. Worse, a verifier is
 * good at catching fabricated PROVENANCE and blind to a claim that is
 * ungrounded but honestly cites a genuinely irrelevant chunk it was handed.
 *
 * Refusing before the call is not an optimisation. It is the difference between
 * a system that cannot produce an ungrounded answer here and a system that
 * produces one and then tries to catch it.
 *
 * A GENERATOR FAILURE IS NOT A REFUSAL. If `generate` throws — network, HTTP
 * status, a model that declined — the error propagates out of this function
 * rather than being turned into a `RefusedResult`. A refusal is a claim about
 * the corpus: that the documents do not support an answer. An outage is a claim
 * about the infrastructure. Dressing the second up as the first tells the
 * reader something false about the corpus and hides an incident from whoever is
 * on call.
 */
export async function answerQuestion(
  question: string,
  options: AnswerOptions,
): Promise<AnswerResult> {
  const k = options.k ?? DEFAULT_K;
  const threshold = options.threshold ?? MIN_SIMILARITY;

  const results = await search(options.index, question, options.embedder, k);
  const { above, below } = partitionByThreshold(results, threshold);

  if (above.length === 0) {
    return {
      kind: "refused",
      reason: "no-relevant-documents",
      detail: [
        `searched ${options.index.entries.length} chunks for ${JSON.stringify(question)},`,
        `took the top ${k}, and none reached the similarity floor of ${score(threshold)}:`,
        `${describeBest(results)}.`,
        `The generator was not called: with nothing above the floor there is nothing to ground an answer in,`,
        `and a model asked to answer from irrelevant passages answers anyway.`,
      ].join(" "),
      nearMisses: below,
    };
  }

  const raw = await options.generator.generate(
    buildSystemPrompt(),
    buildUserMessage(question, above),
  );

  const parsed = parseClaims(raw);
  if (!parsed.ok) {
    return {
      kind: "refused",
      reason: "no-claims-produced",
      detail: `${parsed.problem}. ${above.length} ${above.length === 1 ? "chunk was" : "chunks were"} above the floor of ${score(threshold)} and were shown to the generator: ${above.map((result) => result.chunk.chunkId).join(", ")}.`,
      nearMisses: below,
    };
  }

  if (parsed.claims.length === 0) {
    return {
      kind: "refused",
      reason: "no-claims-produced",
      detail: [
        `the generator parsed cleanly and produced zero claims from ${above.length}`,
        `${above.length === 1 ? "chunk" : "chunks"} above the floor of ${score(threshold)}`,
        `(${above.map((result) => `${result.chunk.chunkId} at ${score(result.score)}`).join(", ")}).`,
        `An empty claim list is the generator saying these passages do not support an answer.`,
      ].join(" "),
      nearMisses: below,
    };
  }

  const shown = retrievedChunks(above);
  const verdict = verifyAnswer(parsed.claims, shown);

  if (verdict.verified.length === 0) {
    return {
      kind: "refused",
      reason: "claims-failed-verification",
      detail: [
        `all ${verdict.rejected.length} ${verdict.rejected.length === 1 ? "claim" : "claims"} the generator produced failed verification against the`,
        `${shown.length} ${shown.length === 1 ? "chunk" : "chunks"} it was shown.`,
        `Stripped:`,
        verdict.rejected
          .map((rejected, position) => `(${position + 1}) [${rejected.reason.kind}] ${rejected.reason.message}`)
          .join(" "),
      ].join(" "),
      nearMisses: below,
    };
  }

  const contradiction = detectContradiction(verdict.verified, shown);
  if (contradiction !== null) {
    const [first, second] = contradiction.positions;
    return {
      kind: "refused",
      reason: "contradictory-sources",
      detail: [
        `two documents in the corpus disagree on this, in ${contradiction.unit}, and this module does not pick between them.`,
        `Position 1: ${describePosition(first)}.`,
        `Position 2: ${describePosition(second)}.`,
        `Both are live documents and neither is marked superseded; the later effective date does not by itself make that document`,
        `operative, so which one governs is a question for the reader, not for this system.`,
      ].join(" "),
      nearMisses: below,
    };
  }

  return {
    kind: "answered",
    claims: verdict.verified,
    // Reported alongside, never hidden. See `AnsweredResult.rejectedClaims`.
    rejectedClaims: verdict.rejected,
    retrieved: above,
  };
}
