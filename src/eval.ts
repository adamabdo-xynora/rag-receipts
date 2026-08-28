/**
 * eval.ts — the gate. Runs the committed question set through the pipeline,
 * scores each answer against a human-verified outcome, and returns a verdict
 * that a CI job can act on.
 *
 * ---------------------------------------------------------------------------
 * ATTRIBUTION
 *
 * The shape of this gate is inherited from my `agent-eval-harness` repo
 * (github.com/adamabdo-xynora/agent-eval-harness): per-category thresholds as
 * committed constants rather than a single blended score, every violation named
 * with the id of the case that produced it, a non-zero exit as the gate itself,
 * and the policy stored beside the verdict so a FAIL read six months later is
 * still interpretable against the thresholds it was judged under.
 *
 * THE STATISTICS DIFFER, DELIBERATELY. That project scored free-form agent
 * transcripts, where the label came from two fallible human graders and the
 * first question about any number was how much of it was grader noise — so it
 * carried Cohen's kappa, and reported no score without it. Here the ground
 * truth is deterministic: `eval/questions.json` records facts about the corpus
 * that were verified by reading it and, where possible, by a grep that is
 * written down next to the claim. There is one label, it does not disagree with
 * itself, and there is no second grader to agree with. Kappa computed over that
 * would be a statistic about nothing — a constant 1.0 dressed up as a
 * measurement — so it is absent, and its absence is the point rather than an
 * omission.
 *
 * ---------------------------------------------------------------------------
 * WHAT SCORING MEANS HERE
 *
 * No model grades anything in this file. Every check below is a set operation
 * or a discriminated-union tag:
 *
 *   - a refusal is recognised by `result.kind === "refused"`, not by looking
 *     for apologetic language in prose;
 *   - a citation is checked by set containment over chunk ids;
 *   - "neither document was presented as the winner" is guaranteed by the
 *     pipeline's own control flow — a `contradictory-sources` refusal has no
 *     answer text in which a winner could be named — rather than inferred from
 *     the wording of one.
 *
 * That is the whole reason `eval/questions.json` was written the way it was. A
 * grader model here would be a second ungrounded generator marking the first
 * one's homework, and this project exists to argue against exactly that.
 */

import { mkdir, readFile, writeFile } from "node:fs/promises";
import path from "node:path";

import {
  type AnswerResult,
  type Generator,
  type InstrumentedGenerator,
  type RefusalReason,
  answerQuestion,
} from "./answer.js";
import {
  PRICING,
  type PricingSnapshot,
  computeEmbeddingCost,
  computeGenerationCost,
  formatUsd,
} from "./cost.js";
import type {
  Embedder,
  EmbeddingUsage,
  InstrumentedEmbedder,
  SearchResult,
  VectorIndex,
} from "./retrieve.js";
import { MIN_SIMILARITY, search } from "./retrieve.js";

/* ===========================================================================
 * The question set.
 * ======================================================================== */

/**
 * The three ground-truth outcomes. These are the `outcome` strings in
 * `eval/questions.json` verbatim; that file is the source of truth and this
 * type exists to stop a typo in it from being read as a fourth category.
 */
export type Outcome = "answered" | "refused-no-documents" | "refused-contradiction";

/** Fixed order, used for reports and for iterating the policy. */
export const OUTCOMES: readonly Outcome[] = [
  "answered",
  "refused-no-documents",
  "refused-contradiction",
];

/**
 * One question, normalised.
 *
 * `expectedChunkIds` and `contradictionDocIds` are absent from most entries in
 * the JSON and are normalised to empty arrays here rather than left optional.
 * An optional array is two states (`undefined` and `[]`) that every downstream
 * check would have to distinguish for no reason; the validator below already
 * guarantees the array is non-empty wherever the outcome requires one.
 */
export interface EvalQuestion {
  readonly id: string;
  readonly question: string;
  readonly outcome: Outcome;
  /** Chunk ids the citations MUST include. Empty for `refused-no-documents`. */
  readonly expectedChunkIds: readonly string[];
  /** Documents that must BOTH be surfaced. Empty unless `refused-contradiction`. */
  readonly contradictionDocIds: readonly string[];
  /** Why this question is in the set. Carried into the artifact unedited. */
  readonly rationale: string;
}

/** The parsed file: the corpus it was written against, and the questions. */
export interface QuestionSet {
  /** Relative to the repository root, as written in the JSON. */
  readonly corpusDir: string;
  readonly questions: readonly EvalQuestion[];
}

/** Every way the question set can be unusable. Each one is covered by a test. */
export type QuestionSetErrorKind =
  | "not-json"
  | "not-an-object"
  | "missing-corpus-dir"
  | "missing-questions-array"
  | "malformed-question"
  | "unknown-outcome"
  | "duplicate-id"
  | "missing-expected-chunk-ids"
  | "missing-contradiction-doc-ids"
  | "malformed-chunk-id";

/**
 * A question set that cannot be read is not a failing eval — it is a broken
 * one, and the two must not exit the same way. Nothing in this module converts
 * one of these into a scored FAIL.
 */
export class QuestionSetError extends Error {
  readonly kind: QuestionSetErrorKind;

  constructor(kind: QuestionSetErrorKind, message: string) {
    super(message);
    this.name = "QuestionSetError";
    this.kind = kind;
  }
}

function isRecord(value: unknown): value is Record<string, unknown> {
  return typeof value === "object" && value !== null && !Array.isArray(value);
}

/** Read a required string field, or say which field of which entry is wrong. */
function readString(entry: Record<string, unknown>, key: string, where: string): string {
  const value = entry[key];
  if (typeof value !== "string" || value.trim() === "") {
    throw new QuestionSetError(
      "malformed-question",
      `${where}: "${key}" must be a non-empty string, got ${JSON.stringify(value)}`,
    );
  }
  return value;
}

/** Read an optional array-of-strings field. Absent and `[]` both yield `[]`. */
function readStringArray(
  entry: Record<string, unknown>,
  key: string,
  where: string,
): readonly string[] {
  const value = entry[key];
  if (value === undefined) return [];
  if (!Array.isArray(value)) {
    throw new QuestionSetError(
      "malformed-question",
      `${where}: "${key}" must be an array of strings, got ${JSON.stringify(value)}`,
    );
  }
  for (const element of value) {
    if (typeof element !== "string" || element.trim() === "") {
      throw new QuestionSetError(
        "malformed-question",
        `${where}: "${key}" contains a non-string or empty entry: ${JSON.stringify(element)}`,
      );
    }
  }
  return value as readonly string[];
}

function isOutcome(value: unknown): value is Outcome {
  return typeof value === "string" && (OUTCOMES as readonly string[]).includes(value);
}

/**
 * Parse and validate the question set.
 *
 * VALIDATION IS NOT PEDANTRY HERE. Every rule below exists because breaking it
 * produces an eval that passes while checking nothing: an `answered` question
 * with no `expectedChunkIds` has no citation requirement and passes on any
 * answer at all; a `refused-contradiction` question with one document id passes
 * on a refusal that surfaced only half the conflict; two questions sharing an
 * id silently collapse into one row in every report. A gate that can be
 * disarmed by an editing mistake is not a gate.
 *
 * The `$comment` key in the file is ignored on purpose: it documents the schema
 * for whoever edits the file next, and enforcing its presence would make prose
 * load-bearing.
 */
export function parseQuestionSet(source: string): QuestionSet {
  let parsed: unknown;
  try {
    parsed = JSON.parse(source);
  } catch (error) {
    const message = error instanceof Error ? error.message : String(error);
    throw new QuestionSetError("not-json", `the question set is not valid JSON (${message})`);
  }

  if (!isRecord(parsed)) {
    throw new QuestionSetError(
      "not-an-object",
      `the question set must be a JSON object, got ${Array.isArray(parsed) ? "an array" : typeof parsed}`,
    );
  }

  const corpusDir = parsed["corpusDir"];
  if (typeof corpusDir !== "string" || corpusDir.trim() === "") {
    throw new QuestionSetError(
      "missing-corpus-dir",
      `the question set has no "corpusDir" string; the scores are only meaningful against the corpus they were written for`,
    );
  }

  const rawQuestions = parsed["questions"];
  if (!Array.isArray(rawQuestions) || rawQuestions.length === 0) {
    throw new QuestionSetError(
      "missing-questions-array",
      `the question set has no non-empty "questions" array`,
    );
  }

  const seen = new Set<string>();
  const questions: EvalQuestion[] = [];

  for (let position = 0; position < rawQuestions.length; position += 1) {
    const entry = rawQuestions[position];
    const where = `question at index ${position}`;

    if (!isRecord(entry)) {
      throw new QuestionSetError(
        "malformed-question",
        `${where}: expected an object, got ${Array.isArray(entry) ? "an array" : typeof entry}`,
      );
    }

    const id = readString(entry, "id", where);
    if (seen.has(id)) {
      throw new QuestionSetError(
        "duplicate-id",
        `two questions share the id "${id}"; ids are how violations are named, so they have to be unique`,
      );
    }
    seen.add(id);

    const named = `question "${id}"`;
    const question = readString(entry, "question", named);
    const rationale = readString(entry, "rationale", named);

    const outcome = entry["outcome"];
    if (!isOutcome(outcome)) {
      throw new QuestionSetError(
        "unknown-outcome",
        `${named}: "outcome" must be one of ${OUTCOMES.join(", ")}, got ${JSON.stringify(outcome)}`,
      );
    }

    const expectedChunkIds = readStringArray(entry, "expectedChunkIds", named);
    const contradictionDocIds = readStringArray(entry, "contradictionDocIds", named);

    for (const chunkId of expectedChunkIds) {
      // `docId#section-slug` is what `src/chunk.ts` produces and what the
      // citation check compares against. An id without a `#` can never match a
      // citation, so it would make the question unpassable rather than strict.
      if (!chunkId.includes("#")) {
        throw new QuestionSetError(
          "malformed-chunk-id",
          `${named}: expected chunk id ${JSON.stringify(chunkId)} is not in "docId#section-slug" form`,
        );
      }
    }

    if (outcome === "answered" && expectedChunkIds.length === 0) {
      throw new QuestionSetError(
        "missing-expected-chunk-ids",
        `${named}: an "answered" question needs at least one expectedChunkId, or it passes on any answer whatsoever`,
      );
    }

    if (outcome === "refused-contradiction" && contradictionDocIds.length < 2) {
      throw new QuestionSetError(
        "missing-contradiction-doc-ids",
        `${named}: a "refused-contradiction" question needs at least two contradictionDocIds — one document cannot disagree with itself`,
      );
    }

    questions.push({ id, question, outcome, expectedChunkIds, contradictionDocIds, rationale });
  }

  return { corpusDir, questions };
}

/** Read and validate the question set from disk. */
export async function loadQuestionSet(filePath: string): Promise<QuestionSet> {
  return parseQuestionSet(await readFile(filePath, "utf8"));
}

/* ===========================================================================
 * What the system actually put its name behind.
 * ======================================================================== */

/**
 * A chunk id as `src/chunk.ts` writes them: two kebab-case slugs joined by `#`.
 *
 * Used to lift chunk ids out of a refusal's `detail` prose. See `evidenceFrom`
 * for why that is necessary and what it is trusted for.
 */
const CHUNK_ID_PATTERN = /[a-z0-9]+(?:-[a-z0-9]+)*#[a-z0-9]+(?:-[a-z0-9]+)*/g;

/** Where a result's evidence came from. Recorded so the artifact says so. */
export type EvidenceSource = "claim-citations" | "refusal-detail";

/** The chunks and documents a result actually named, however it named them. */
export interface Evidence {
  readonly source: EvidenceSource;
  /** Sorted and de-duplicated, so two runs compare byte for byte. */
  readonly chunkIds: readonly string[];
  /** The `docId` halves of `chunkIds`, sorted and de-duplicated. */
  readonly docIds: readonly string[];
}

function summariseIds(chunkIds: readonly string[]): Evidence["chunkIds"] {
  return [...new Set(chunkIds)].sort();
}

function docIdsOf(chunkIds: readonly string[]): readonly string[] {
  const docIds = chunkIds.map((chunkId) => chunkId.split("#")[0] ?? chunkId);
  return [...new Set(docIds)].sort();
}

/**
 * The chunk ids a result committed to.
 *
 * FOR AN ANSWER this is exact: the union of every verified claim's `citations`,
 * which is the structured field the verifier already checked against the chunks
 * the model was shown.
 *
 * FOR A REFUSAL it is read out of the `detail` string, because that is the only
 * place the pipeline puts them. `RefusedResult` carries `reason` and `detail`
 * and nothing else structured — a `contradictory-sources` refusal knows both
 * `ContradictoryPosition`s internally and renders them into prose on the way
 * out. Scraping that prose is the compromise this module makes, and it is worth
 * naming precisely:
 *
 *   - it is a READ of a committed, tested format, not a parse of model output.
 *     Nothing a model writes reaches `detail`; every character of it is written
 *     by `answerQuestion` and asserted on in `test/answer.test.ts`.
 *   - it fails SAFE. If the format changes, ids stop being found, the
 *     contradiction check stops seeing both documents, and the gate FAILS
 *     loudly. There is no edit to `answer.ts` that makes a bad refusal start
 *     passing here.
 *   - it is never the whole check. The load-bearing half of the contradiction
 *     rule — that neither document was declared the winner — is decided by
 *     `result.kind`, not by anything in this string. See `scoreQuestion`.
 *
 * The honest fix is a structured `contradiction` field on `RefusedResult`, and
 * this comment is the note for whoever adds it.
 */
export function evidenceFrom(result: AnswerResult): Evidence {
  if (result.kind === "answered") {
    const chunkIds = summariseIds(result.claims.flatMap((claim) => [...claim.citations]));
    return { source: "claim-citations", chunkIds, docIds: docIdsOf(chunkIds) };
  }

  const chunkIds = summariseIds(result.detail.match(CHUNK_ID_PATTERN) ?? []);
  return { source: "refusal-detail", chunkIds, docIds: docIdsOf(chunkIds) };
}

/* ===========================================================================
 * Scoring one question.
 * ======================================================================== */

/**
 * Every way a question can fail, named. These strings appear in the artifact
 * and in the printed violations, so they are stable and they are specific:
 * "failed" is not actionable, `hallucinated-answer` and `over-refusal` point at
 * opposite defects with opposite fixes.
 */
export type FailureKind =
  /** THE HEADLINE. A fluent, cited answer to a question the corpus cannot answer. */
  | "hallucinated-answer"
  /** THE HEADLINE, contradiction flavour. One side of a live disagreement served as the answer. */
  | "resolved-contradiction"
  /** Refused, but the refusal did not surface every document in the conflict. */
  | "missing-contradiction-source"
  /** Answered, but a required chunk id is absent from the citations. */
  | "missing-citation"
  /** Refused a question the corpus does answer. The over-tuning failure mode. */
  | "over-refusal";

/** A named failure. `headline` is set by the rule, not by the reader. */
export interface Failure {
  readonly kind: FailureKind;
  /** Names the id, what was expected, and what was observed. One paragraph. */
  readonly message: string;
  /**
   * Whether this is the failure that invalidates the project's premise rather
   * than merely degrading its numbers. See `HEADLINE_FAILURE_KINDS`.
   */
  readonly headline: boolean;
}

/**
 * The failures that get reported first and named as such.
 *
 * WHY THESE TWO AND NOTHING ELSE. Every other failure in this file is a quality
 * defect: a citation missing from an otherwise sound answer means retrieval
 * ranked badly, an over-refusal means the floor is too high. Both are visible,
 * both are annoying, and both leave the reader correctly informed — nobody acts
 * on a wrong fact, because no wrong fact was produced.
 *
 * These two produce a wrong fact. A fluent answer to a question the corpus
 * cannot answer, or a confident pick between two documents that disagree, is
 * indistinguishable from a correct answer at the point of use: it is cited, it
 * is well-formed, and it is false. The entire claim this project makes is that
 * it does not do this. One instance falsifies the claim, which is why the
 * threshold for them is 100% and why they are printed above the table instead
 * of inside it.
 */
export const HEADLINE_FAILURE_KINDS: readonly FailureKind[] = [
  "hallucinated-answer",
  "resolved-contradiction",
];

/** What the pipeline did, flattened for the report and the artifact. */
export interface Observation {
  readonly kind: "answered" | "refused";
  /** Present only for refusals. */
  readonly refusalReason: RefusalReason | null;
  /** Verified claims, for an answer. Zero for a refusal. */
  readonly claimCount: number;
  /** Claims the verifier stripped. Reported for answers; see `AnsweredResult`. */
  readonly rejectedClaimCount: number;
  readonly evidence: Evidence;
}

/** One question's verdict. */
export interface QuestionScore {
  readonly id: string;
  readonly category: Outcome;
  readonly question: string;
  readonly passed: boolean;
  /** Null when `passed`. */
  readonly failure: Failure | null;
  readonly observed: Observation;
  /** One line for the results table. Always populated, pass or fail. */
  readonly note: string;
}

function observe(result: AnswerResult): Observation {
  const evidence = evidenceFrom(result);
  if (result.kind === "answered") {
    return {
      kind: "answered",
      refusalReason: null,
      claimCount: result.claims.length,
      rejectedClaimCount: result.rejectedClaims.length,
      evidence,
    };
  }
  return {
    kind: "refused",
    refusalReason: result.reason,
    claimCount: 0,
    rejectedClaimCount: 0,
    evidence,
  };
}

/** `a, b` or `none`. Used everywhere a set appears in a message. */
function list(values: readonly string[]): string {
  return values.length === 0 ? "none" : values.join(", ");
}

function pass(
  question: EvalQuestion,
  observed: Observation,
  note: string,
): QuestionScore {
  return {
    id: question.id,
    category: question.outcome,
    question: question.question,
    passed: true,
    failure: null,
    observed,
    note,
  };
}

function fail(
  question: EvalQuestion,
  observed: Observation,
  kind: FailureKind,
  message: string,
  note: string,
): QuestionScore {
  return {
    id: question.id,
    category: question.outcome,
    question: question.question,
    passed: false,
    failure: { kind, message, headline: HEADLINE_FAILURE_KINDS.includes(kind) },
    observed,
    note,
  };
}

/**
 * Score one question against its ground-truth outcome.
 *
 * THE THREE RULES, stated once, in the order the outcomes appear:
 *
 * `refused-no-documents` — A CORRECT REFUSAL IS A PASS. Not a partial credit,
 *   not a neutral non-answer: the corpus does not contain the answer, so
 *   declining IS the correct output, and it is scored exactly as highly as a
 *   correct answer to an answerable question. This is the line the whole
 *   project turns on. Any refusal reason counts — a refusal for
 *   `no-claims-produced` is still a refusal, and the reason is carried into the
 *   table so an unexpected one is visible without being fatal. Answering is the
 *   headline failure.
 *
 * `refused-contradiction` — passes only if BOTH conditions hold. (1) Every
 *   `contradictionDocId` appears in the evidence, so the reader can see the
 *   whole conflict rather than half of it. (2) Neither is presented as the
 *   resolution — which is structural: a `refused` result contains no answer
 *   text, so there is nowhere for a winner to be declared. An `answered` result
 *   here means one side was served as the answer, and that is a headline
 *   failure regardless of which side or how well it was cited.
 *
 * `answered` — passes only if EVERY `expectedChunkId` appears in the citations.
 *   Extra citations are allowed and are not penalised: several questions in the
 *   set have legitimately overlapping sources (`minimum-order-value` may
 *   reasonably also cite the FAQ), and requiring an exact set would mark a
 *   better-sourced answer wrong. A missing one is always a failure, because the
 *   expected ids are the chunks that actually carry the fact — an answer that
 *   arrives at the right number while citing a different section got there by
 *   some route other than the evidence, and that route is not repeatable.
 *   Refusing is an over-refusal, and is named as one.
 */
export function scoreQuestion(question: EvalQuestion, result: AnswerResult): QuestionScore {
  const observed = observe(result);
  const cited = new Set(observed.evidence.chunkIds);

  switch (question.outcome) {
    case "refused-no-documents": {
      if (result.kind === "refused") {
        return pass(
          question,
          observed,
          `refused (${result.reason}) — correct, the corpus cannot answer this`,
        );
      }
      return fail(
        question,
        observed,
        "hallucinated-answer",
        [
          `${question.id}: the corpus does not contain this answer and the system produced one anyway.`,
          `It returned ${observed.claimCount} verified ${observed.claimCount === 1 ? "claim" : "claims"}`,
          `citing ${list(observed.evidence.chunkIds)}.`,
          `Every one of those citations points at a real chunk that was really retrieved — that is what makes this the`,
          `headline failure rather than a citation bug. The claims are grounded in passages that do not answer the question,`,
          `and the output is indistinguishable from a correct answer at the point of use.`,
        ].join(" "),
        `ANSWERED with ${observed.claimCount} claim(s) — must have refused`,
      );
    }

    case "refused-contradiction": {
      if (result.kind === "answered") {
        return fail(
          question,
          observed,
          "resolved-contradiction",
          [
            `${question.id}: two live documents (${list(question.contradictionDocIds)}) disagree on this and the system`,
            `picked one. It returned ${observed.claimCount} verified ${observed.claimCount === 1 ? "claim" : "claims"}`,
            `citing ${list(observed.evidence.chunkIds)}.`,
            `Resolving the conflict is not the retrieval layer's call: neither document is marked superseded, and a later`,
            `effective date does not by itself make a document operative. Serving either side as the answer tells the reader`,
            `there is no disagreement, which is the one thing the corpus is certain about.`,
          ].join(" "),
          `ANSWERED with ${observed.claimCount} claim(s) — must have surfaced both sides`,
        );
      }

      const missing = question.contradictionDocIds.filter(
        (docId) => !observed.evidence.docIds.includes(docId),
      );
      if (missing.length > 0) {
        return fail(
          question,
          observed,
          "missing-contradiction-source",
          [
            `${question.id}: refused (${result.reason}), which is correct, but the refusal does not surface the whole conflict.`,
            `Missing ${missing.length === 1 ? "document" : "documents"}: ${list(missing)}.`,
            `Surfaced: ${list(observed.evidence.docIds)}.`,
            `A refusal that names only one side reads as "we could not find this" when the truth is "we found two answers";`,
            `the reader cannot go and adjudicate a disagreement they were not shown.`,
          ].join(" "),
          `refused (${result.reason}) but missing ${list(missing)}`,
        );
      }

      return pass(
        question,
        observed,
        `refused (${result.reason}) with both sides surfaced — no winner declared`,
      );
    }

    case "answered": {
      if (result.kind === "refused") {
        return fail(
          question,
          observed,
          "over-refusal",
          [
            `${question.id}: OVER-REFUSAL. The corpus answers this question — the fact lives in ${list(question.expectedChunkIds)}`,
            `— and the system declined with reason "${result.reason}".`,
            `Over-refusal is the failure mode a grounding project acquires when it tunes only against hallucination:`,
            `every refusal looks safe in isolation, and a system that has learned to decline anything difficult scores`,
            `perfectly on the traps while being useless. Refusal detail: ${result.detail}`,
          ].join(" "),
          `REFUSED (${result.reason}) — corpus does answer this`,
        );
      }

      const missing = question.expectedChunkIds.filter((chunkId) => !cited.has(chunkId));
      if (missing.length > 0) {
        const extra = observed.evidence.chunkIds.filter(
          (chunkId) => !question.expectedChunkIds.includes(chunkId),
        );
        return fail(
          question,
          observed,
          "missing-citation",
          [
            `${question.id}: answered without citing ${missing.length === 1 ? "a required chunk" : "required chunks"}.`,
            `Missing: ${list(missing)}. Cited instead: ${list(observed.evidence.chunkIds)}.`,
            extra.length > 0
              ? `The ${extra.length === 1 ? "citation" : "citations"} ${list(extra)} ${extra.length === 1 ? "is" : "are"} not disqualifying on ${extra.length === 1 ? "its" : "their"} own — extra citations are allowed — but ${extra.length === 1 ? "it does" : "they do"} not substitute for the chunk that carries the fact.`
              : `No other chunk was cited either.`,
            `An answer that reaches the right conclusion from the wrong section reached it by some route other than the`,
            `evidence, and that route will not hold on the next question.`,
          ].join(" "),
          `answered, missing ${list(missing)}`,
        );
      }

      const extraCount = observed.evidence.chunkIds.length - question.expectedChunkIds.length;
      return pass(
        question,
        observed,
        `answered with all ${question.expectedChunkIds.length} required citation(s)${extraCount > 0 ? ` (+${extraCount} extra, allowed)` : ""}`,
      );
    }
  }
}

/* ===========================================================================
 * Running the set.
 * ======================================================================== */

/** Everything a run needs. All injected — this module opens no connections. */
export interface EvalRunOptions {
  readonly index: VectorIndex;
  readonly embedder: Embedder;
  readonly generator: Generator;
  /** Passed straight through to `answerQuestion`. Defaults there. */
  readonly k?: number;
  /** Passed straight through to `answerQuestion`. Defaults to `MIN_SIMILARITY`. */
  readonly threshold?: number;
}

/**
 * Run every question, in file order, one at a time.
 *
 * SEQUENTIALLY, ON PURPOSE. The corpus is fourteen documents and the set is
 * fourteen questions; there is no wall-clock argument worth the loss of a
 * readable, reproducible failure. Concurrency here would also interleave
 * provider errors across questions, and the next paragraph is about those.
 *
 * PIPELINE ERRORS PROPAGATE. If `answerQuestion` throws — a transport failure,
 * an HTTP status, a model refusal at the SDK level — this function does not
 * catch it and does not score it as a failed question. An outage is a fact
 * about the infrastructure, not about the corpus, and a gate that converts one
 * into a FAIL publishes a claim about grounding quality that it did not
 * measure. The run aborts, the CLI exits with its setup-error code, and no
 * artifact is written: an incomplete run has no verdict to record.
 */
export async function runEvaluation(
  questions: readonly EvalQuestion[],
  options: EvalRunOptions,
): Promise<QuestionScore[]> {
  const scores: QuestionScore[] = [];
  for (const question of questions) {
    const result = await answerQuestion(question.question, {
      index: options.index,
      embedder: options.embedder,
      generator: options.generator,
      ...(options.k === undefined ? {} : { k: options.k }),
      ...(options.threshold === undefined ? {} : { threshold: options.threshold }),
    });
    scores.push(scoreQuestion(question, result));
  }
  return scores;
}

/* ===========================================================================
 * Usage accounting.
 * ======================================================================== */

/** Tokens, time, and dollars for one slice of the run. */
export interface UsageTotals {
  readonly embeddingTokens: number;
  readonly generationInputTokens: number;
  readonly generationOutputTokens: number;
  readonly latencyMs: number;
  /** Cost at the committed prices in `src/cost.ts`, for the models THEY name. */
  readonly costUsd: number;
}

/** One question's slice: everything in `UsageTotals`, addressed by id. */
export interface QuestionUsage extends UsageTotals {
  readonly questionId: string;
}

/** The whole run's accounting, carried into the artifact and the report. */
export interface RunUsage {
  /**
   * Usage the embedder and generator had already accumulated before the first
   * question ran. In the CLI's flow that is exactly the corpus index build —
   * the largest embedding spend of the run — and leaving it out of the totals
   * would report a run cost smaller than the invoice.
   */
  readonly indexing: UsageTotals;
  /** One row per question, in run order. */
  readonly perQuestion: readonly QuestionUsage[];
  /** `indexing` plus every per-question row. */
  readonly totals: UsageTotals;
  /**
   * The prices the dollar figures were computed at, with their as-of dates.
   * SAME PRINCIPLE AS STORING THE POLICY BESIDE THE VERDICT: the constants in
   * `src/cost.ts` will move, and a stored cost without the prices it was
   * computed from would be silently reinterpreted against numbers that were
   * not in force when it was measured.
   */
  readonly pricing: PricingSnapshot;
}

function usageTotals(
  embeddingTokens: number,
  generationInputTokens: number,
  generationOutputTokens: number,
  latencyMs: number,
): UsageTotals {
  return {
    embeddingTokens,
    generationInputTokens,
    generationOutputTokens,
    latencyMs,
    costUsd:
      computeEmbeddingCost(embeddingTokens) +
      computeGenerationCost(generationInputTokens, generationOutputTokens),
  };
}

function addTotals(a: UsageTotals, b: UsageTotals): UsageTotals {
  return usageTotals(
    a.embeddingTokens + b.embeddingTokens,
    a.generationInputTokens + b.generationInputTokens,
    a.generationOutputTokens + b.generationOutputTokens,
    a.latencyMs + b.latencyMs,
  );
}

/** `runEvaluation`, with the providers' own accounting read alongside. */
export interface InstrumentedEvalRunOptions extends EvalRunOptions {
  readonly embedder: InstrumentedEmbedder;
  readonly generator: InstrumentedGenerator;
}

/**
 * Run every question exactly as `runEvaluation` does, and attribute usage.
 *
 * THE SCORING IS IDENTICAL BY CONSTRUCTION: the same `answerQuestion` call
 * with the same options, the same `scoreQuestion`, the same order. The only
 * addition is bookkeeping — cumulative usage is snapshotted before and after
 * each question and the difference is that question's row, which works because
 * the run is sequential (see `runEvaluation` for why it stays sequential).
 * `test/usage.test.ts` asserts the two functions produce identical scores, so
 * the two loops cannot drift apart silently.
 *
 * `runEvaluation` stays as the uninstrumented seam so existing callers — and
 * any future caller that has only a bare `Embedder` — compile unchanged.
 */
export async function runInstrumentedEvaluation(
  questions: readonly EvalQuestion[],
  options: InstrumentedEvalRunOptions,
): Promise<{ readonly scores: QuestionScore[]; readonly usage: RunUsage }> {
  let embedBefore = options.embedder.usage();
  let genBefore = options.generator.usage();

  const indexing = usageTotals(
    embedBefore.totalTokens,
    genBefore.inputTokens,
    genBefore.outputTokens,
    embedBefore.totalLatencyMs + genBefore.totalLatencyMs,
  );

  const scores: QuestionScore[] = [];
  const perQuestion: QuestionUsage[] = [];

  for (const question of questions) {
    const result = await answerQuestion(question.question, {
      index: options.index,
      embedder: options.embedder,
      generator: options.generator,
      ...(options.k === undefined ? {} : { k: options.k }),
      ...(options.threshold === undefined ? {} : { threshold: options.threshold }),
    });
    scores.push(scoreQuestion(question, result));

    const embedAfter = options.embedder.usage();
    const genAfter = options.generator.usage();
    perQuestion.push({
      questionId: question.id,
      ...usageTotals(
        embedAfter.totalTokens - embedBefore.totalTokens,
        genAfter.inputTokens - genBefore.inputTokens,
        genAfter.outputTokens - genBefore.outputTokens,
        embedAfter.totalLatencyMs -
          embedBefore.totalLatencyMs +
          (genAfter.totalLatencyMs - genBefore.totalLatencyMs),
      ),
    });
    embedBefore = embedAfter;
    genBefore = genAfter;
  }

  const totals = perQuestion.reduce<UsageTotals>(addTotals, indexing);

  return { scores, usage: { indexing, perQuestion, totals, pricing: PRICING } };
}

/* ===========================================================================
 * The gate.
 * ======================================================================== */

/** A category's bar, with the reasoning that set it, carried into the artifact. */
export interface CategoryPolicy {
  /** Human label for reports. */
  readonly label: string;
  /** Minimum pass rate, inclusive. `1` means every question must pass. */
  readonly threshold: number;
  /** Why this number and not another. Stored beside the verdict. */
  readonly rationale: string;
}

/**
 * THE POLICY. Committed constants, not flags, not environment variables.
 *
 * A threshold that can be lowered at the call site is not a gate, it is a
 * suggestion — the first red build moves the number instead of fixing the code,
 * and the move leaves no trace. These live in source, under review, and a
 * change to one is a diff someone has to justify.
 *
 * THERE IS NO BLENDED SCORE. "12 of 14, 86%" would let two hallucinations hide
 * behind twelve good answers, and those are not commensurable outcomes: a
 * missing citation is a ranking defect and a hallucination on a no-answer
 * question is the failure this project exists to prevent. Per-category bars
 * keep them from cancelling out.
 */
export const GATE_POLICY: Readonly<Record<Outcome, CategoryPolicy>> = {
  "refused-no-documents": {
    label: "must-refuse (no documents)",
    threshold: 1,
    rationale: [
      "100%, no partial credit, no exceptions.",
      "These questions have no answer in the corpus — a human verified the absence with a grep that is written down",
      "next to the question. Any fluent answer to one is a hallucination by construction, not a borderline call.",
      "One of them invalidates the premise of the project: the claim being made is not 'this system rarely fabricates',",
      "it is 'this system cannot fabricate here, because the generator is never called when nothing clears the floor'.",
      "That claim is false the moment a single instance passes, and a 90% bar would encode the belief that one in ten",
      "fabrications is acceptable — which, if it were acceptable, would make the rest of this repository pointless.",
    ].join(" "),
  },
  "refused-contradiction": {
    label: "must-refuse (contradiction)",
    threshold: 1,
    rationale: [
      "100%, for the same reason and one more.",
      "Two live documents disagree, neither is marked superseded, and nothing in the corpus says which governs.",
      "Picking one produces an answer that is confident, correctly cited, and wrong for half of all readers —",
      "and worse than a plain hallucination in one respect: the citation is genuine, so the usual defence",
      "('check the source') confirms it. Surfacing both and declining to adjudicate is the only sound output,",
      "so there is no fraction of these that may be resolved.",
    ].join(" "),
  },
  answered: {
    label: "answerable",
    threshold: 0.8,
    rationale: [
      "80% — a lower bar than the refusal categories, deliberately and explicitly.",
      "A miss here is a retrieval defect, not a soundness violation: the system either declined (the reader learns",
      "nothing, and knows they learned nothing) or answered while citing the wrong section (visible in the output,",
      "and caught by exactly this check). Neither leaves a reader misinformed without warning, which is the property",
      "that earns the refusal categories their 100%.",
      "The bar is not lower because these questions matter less. It is lower because this set is stacked against",
      "retrieval on purpose — five of the eleven are near-duplicate tempters where the wrong chunk is a paraphrase of",
      "the right one, and two require joining facts across documents that each refuse to restate the other's numbers.",
      "Demanding 100% would mean the gate went red for a ranking regression on the hardest tempter in the corpus,",
      "and a gate that is always red is a gate nobody reads.",
      "80% of 11 answerable questions is 8.8, so 9 must pass: the set can lose two of the tempters and still ship,",
      "and cannot lose three. Over-refusal counts against this number, which is the point — a system that refuses",
      "everything scores 100% on both must-refuse categories and fails here.",
    ].join(" "),
  },
};

/** How one category came out. */
export interface CategoryOutcome {
  readonly category: Outcome;
  readonly label: string;
  readonly threshold: number;
  readonly total: number;
  readonly passed: number;
  /** `passed / total`, or 1 for an empty category. See `applyGate`. */
  readonly rate: number;
  readonly met: boolean;
}

/** A failure, addressed: which question, which category, what went wrong. */
export interface Violation {
  readonly questionId: string;
  readonly category: Outcome;
  readonly kind: FailureKind;
  readonly headline: boolean;
  readonly message: string;
}

/** The verdict, and everything it was computed from. */
export interface GateReport {
  readonly scores: readonly QuestionScore[];
  readonly categories: readonly CategoryOutcome[];
  /** Headline violations first, then the rest in question order. */
  readonly violations: readonly Violation[];
  /** The subset of `violations` that are headline failures. May be empty. */
  readonly headlineViolations: readonly Violation[];
  readonly totalQuestions: number;
  readonly totalPassed: number;
  readonly passed: boolean;
}

/**
 * Apply the policy.
 *
 * A category with no questions in it is vacuously met, at rate 1. That is the
 * correct reading of "100% of zero questions passed" and it keeps a filtered
 * run — one category at a time while debugging — from reporting a false FAIL.
 * It is also why `parseQuestionSet` refuses an empty question set outright:
 * vacuous truth is fine for a subset and useless for the whole.
 *
 * The verdict is the AND of every category. There is no way for a surplus in
 * one to cover a shortfall in another; see `GATE_POLICY`.
 */
export function applyGate(scores: readonly QuestionScore[]): GateReport {
  const categories: CategoryOutcome[] = OUTCOMES.map((category) => {
    const inCategory = scores.filter((score) => score.category === category);
    const passedCount = inCategory.filter((score) => score.passed).length;
    const policy = GATE_POLICY[category];
    const rate = inCategory.length === 0 ? 1 : passedCount / inCategory.length;
    return {
      category,
      label: policy.label,
      threshold: policy.threshold,
      total: inCategory.length,
      passed: passedCount,
      rate,
      met: rate >= policy.threshold,
    };
  });

  const violations: Violation[] = scores
    .filter((score): score is QuestionScore & { failure: Failure } => score.failure !== null)
    .map((score) => ({
      questionId: score.id,
      category: score.category,
      kind: score.failure.kind,
      headline: score.failure.headline,
      message: score.failure.message,
    }));

  // Stable: headline first, everything else in question order. `sort` is stable
  // in every runtime this targets, so equal keys keep their file order and two
  // runs print the same list in the same sequence.
  const ordered = [...violations].sort(
    (a, b) => Number(b.headline) - Number(a.headline),
  );

  return {
    scores,
    categories,
    violations: ordered,
    headlineViolations: ordered.filter((violation) => violation.headline),
    totalQuestions: scores.length,
    totalPassed: scores.filter((score) => score.passed).length,
    passed: categories.every((category) => category.met),
  };
}

/* ===========================================================================
 * Reporting.
 * ======================================================================== */

function padRight(value: string, width: number): string {
  return value.length >= width ? value : value + " ".repeat(width - value.length);
}

function percent(rate: number): string {
  return `${(rate * 100).toFixed(1)}%`;
}

/** Fixed-width table, one row per question, file order preserved. */
export function formatResultsTable(report: GateReport): string {
  const header = ["STATUS", "QUESTION", "CATEGORY", "WHAT HAPPENED"];
  const rows = report.scores.map((score) => [
    score.passed ? "PASS" : score.failure?.headline === true ? "HEADLINE" : "FAIL",
    score.id,
    GATE_POLICY[score.category].label,
    score.note,
  ]);

  const widths = header.map((_, column) =>
    Math.max(
      header[column]?.length ?? 0,
      ...rows.map((row) => row[column]?.length ?? 0),
    ),
  );

  const line = (cells: readonly string[]): string =>
    cells
      .map((cell, column) =>
        // The last column is never padded: trailing spaces on a wrapped
        // terminal look like corruption.
        column === cells.length - 1 ? cell : padRight(cell, widths[column] ?? 0),
      )
      .join("  ")
      .trimEnd();

  const rule = widths.map((width) => "-".repeat(width)).join("  ");

  return [line(header), rule, ...rows.map(line)].join("\n");
}

/** Right-align numbers so a column of them can be compared by eye. */
function padLeft(value: string, width: number): string {
  return value.length >= width ? value : " ".repeat(width - value.length) + value;
}

/** One line naming both as-of dates. Printed with every cost figure. */
function pricingAsOfLine(pricing: PricingSnapshot): string {
  return [
    `Prices committed in src/cost.ts:`,
    `${pricing.embedding.model} as of ${pricing.embedding.asOf},`,
    `${pricing.generation.model} as of ${pricing.generation.asOf}.`,
    `Prices change and this repository does not phone home — a stale constant is a wrong dollar figure with nothing loud about it.`,
  ].join(" ");
}

/** The cost table: one row per question, an index-build row, and totals. */
export function formatUsageReport(usage: RunUsage): string {
  const header = ["QUESTION", "EMBED TOK", "GEN IN", "GEN OUT", "LATENCY MS", "COST"];

  const row = (label: string, totals: UsageTotals): string[] => [
    label,
    String(totals.embeddingTokens),
    String(totals.generationInputTokens),
    String(totals.generationOutputTokens),
    String(totals.latencyMs),
    formatUsd(totals.costUsd),
  ];

  const rows = [
    row("(index build)", usage.indexing),
    ...usage.perQuestion.map((question) => row(question.questionId, question)),
    row("TOTAL", usage.totals),
  ];

  const widths = header.map((_, column) =>
    Math.max(header[column]?.length ?? 0, ...rows.map((cells) => cells[column]?.length ?? 0)),
  );

  const line = (cells: readonly string[]): string =>
    cells
      .map((cell, column) =>
        // The first column is a label and reads left; every other column is a
        // number and right-aligns so magnitudes line up.
        column === 0 ? padRight(cell, widths[column] ?? 0) : padLeft(cell, widths[column] ?? 0),
      )
      .join("  ")
      .trimEnd();

  const rule = widths.map((width) => "-".repeat(width)).join("  ");
  const totalRow = rows[rows.length - 1] ?? [];

  return [
    `USAGE AND COST`,
    ``,
    line(header),
    rule,
    ...rows.slice(0, -1).map(line),
    rule,
    line(totalRow),
    ``,
    `  ${pricingAsOfLine(usage.pricing)}`,
  ].join("\n");
}

/**
 * The whole printed report: headline first, then table, policy, violations.
 *
 * `usage` is optional so callers without instrumentation (the demo, older
 * tests) print exactly what they always printed, byte for byte. The CLI
 * passes it and its report gains the cost table.
 */
export function formatGateReport(report: GateReport, usage?: RunUsage): string {
  const sections: string[] = [];

  if (report.headlineViolations.length > 0) {
    const count = report.headlineViolations.length;
    sections.push(
      [
        `HEADLINE FAILURE${count === 1 ? "" : "S"}: ${count}`,
        ``,
        `A fluent answer to a question that must be refused. This is reported first because it is not`,
        `one failure among several — it is the failure this project claims cannot happen, and one`,
        `instance falsifies the claim. Nothing below is worth reading until ${count === 1 ? "this is" : "these are"} fixed.`,
        ``,
        ...report.headlineViolations.map(
          (violation, position) => `  ${position + 1}. [${violation.kind}] ${violation.message}`,
        ),
      ].join("\n"),
    );
  }

  sections.push(formatResultsTable(report));

  sections.push(
    [
      `GATE POLICY (per category, committed in src/eval.ts)`,
      ``,
      ...report.categories.map((category) => {
        const status = category.met ? "MET" : "NOT MET";
        return `  ${padRight(category.label, 26)} ${padRight(`${category.passed}/${category.total}`, 8)} ${padRight(percent(category.rate), 7)} required ${padRight(percent(category.threshold), 7)} ${status}`;
      }),
    ].join("\n"),
  );

  if (report.violations.length > 0) {
    sections.push(
      [
        `VIOLATIONS: ${report.violations.length}`,
        ``,
        ...report.violations.map(
          (violation, position) =>
            `  ${position + 1}. ${violation.questionId} [${violation.kind}]${violation.headline ? " — HEADLINE FAILURE" : ""}\n     ${violation.message}`,
        ),
      ].join("\n"),
    );
  }

  if (usage !== undefined) {
    sections.push(formatUsageReport(usage));
  }

  sections.push(
    `VERDICT: ${report.passed ? "PASS" : "FAIL"} (${report.totalPassed}/${report.totalQuestions} questions passed)`,
  );

  return sections.join("\n\n");
}

/* ===========================================================================
 * The artifact.
 * ======================================================================== */

/**
 * Bumped when the artifact's shape changes in a way that would mislead a reader
 * comparing two files. Old artifacts stay readable because they say which
 * schema they are.
 *
 * 2: the artifact gained a `usage` section — per-question tokens, latency, and
 *    cost, plus the pricing constants (with as-of dates) the costs were
 *    computed at. A reader comparing a schema-1 artifact with a schema-2 one
 *    should know the absence of a cost in the older file means "not measured",
 *    not "free".
 */
export const ARTIFACT_SCHEMA_VERSION = 2;

/** Run context the eval cannot discover for itself. All of it is injected. */
export interface RunMetadata {
  /** ISO 8601. A parameter, never `new Date()` here — see `buildArtifact`. */
  readonly generatedAt: string;
  /** e.g. `voyage-3` or `fake-embedder`. Names the model, never a key. */
  readonly embedder: string;
  /** e.g. `claude-opus-5` or `fake-generator`. Names the model, never a key. */
  readonly generator: string;
  readonly corpusDir: string;
  readonly chunkCount: number;
  readonly k: number;
  readonly threshold: number;
}

/** What gets written to `results/`. Self-contained by design. */
export interface EvalArtifact {
  readonly schemaVersion: number;
  readonly verdict: "PASS" | "FAIL";
  readonly run: RunMetadata;
  /**
   * The full policy, copied in. THIS IS THE POINT OF THE FILE.
   *
   * A stored verdict without the thresholds it was judged against is not a
   * record, it is a rumour. Six months from now the constants in this module
   * will have moved — that is what constants under review do — and a FAIL from
   * today would then be read against today's numbers, or against whatever the
   * reader assumes. Copying the policy in makes the artifact interpretable
   * standing alone, and makes a threshold change visible as a diff between two
   * artifacts rather than an invisible reinterpretation of both.
   */
  readonly policy: {
    readonly categories: Readonly<Record<Outcome, CategoryPolicy>>;
    readonly headlineFailureKinds: readonly FailureKind[];
    readonly headlineRule: string;
  };
  readonly categories: readonly CategoryOutcome[];
  readonly headlineViolations: readonly Violation[];
  readonly violations: readonly Violation[];
  readonly questions: readonly QuestionScore[];
  readonly totals: { readonly questions: number; readonly passed: number };
  /**
   * Tokens, latency, and cost — with the pricing constants and their as-of
   * dates copied in, for the same reason the policy is: a stored run stays
   * interpretable after the prices in `src/cost.ts` move. `null` for a run
   * that was not instrumented; the CLI always instruments.
   */
  readonly usage: RunUsage | null;
}

/**
 * Assemble the artifact.
 *
 * The timestamp is a parameter rather than a call to `new Date()` because a
 * function that reads the clock cannot be asserted on: the test would have to
 * either freeze time globally or check the field loosely, and a loosely checked
 * field is an unchecked one. Whoever runs the eval knows what time it is.
 */
export function buildArtifact(
  report: GateReport,
  run: RunMetadata,
  usage: RunUsage | null = null,
): EvalArtifact {
  return {
    schemaVersion: ARTIFACT_SCHEMA_VERSION,
    verdict: report.passed ? "PASS" : "FAIL",
    run,
    policy: {
      categories: GATE_POLICY,
      headlineFailureKinds: HEADLINE_FAILURE_KINDS,
      headlineRule:
        "A fluent answer to a must-refuse question is the headline failure: it is reported before the results table and named as such. Both must-refuse categories are gated at 100% because a single instance falsifies the claim the system makes about itself.",
    },
    categories: report.categories,
    headlineViolations: report.headlineViolations,
    violations: report.violations,
    questions: report.scores,
    totals: { questions: report.totalQuestions, passed: report.totalPassed },
    usage,
  };
}

/**
 * Where artifacts go, relative to the repository root.
 *
 * Gitignored. A verdict committed to the repository is a verdict nobody
 * re-derives: it goes stale silently, it turns into a diff conflict on every
 * run, and the first person in a hurry deletes the failing one rather than the
 * failure. The record belongs in CI output, not in `git log`.
 */
export const DEFAULT_ARTIFACT_DIR = "results";

/** Colons are legal in POSIX filenames and awkward everywhere else. */
function filenameStamp(iso: string): string {
  return iso.replace(/[:.]/g, "-");
}

/**
 * Write the artifact twice: once under a timestamp, once as `latest`.
 *
 * The timestamped copy is the record — artifacts accumulate, and two of them
 * side by side are how a threshold change or a regression becomes visible. The
 * `latest` copy is the one a CI step or a human reads without having to sort
 * filenames. `results/` is gitignored: these are run outputs, and a verdict
 * committed to the repository is a verdict nobody re-derives.
 *
 * Returns both paths, in write order.
 */
export async function writeArtifact(
  directory: string,
  artifact: EvalArtifact,
): Promise<readonly string[]> {
  await mkdir(directory, { recursive: true });
  const body = `${JSON.stringify(artifact, null, 2)}\n`;

  const stamped = path.join(directory, `eval-${filenameStamp(artifact.run.generatedAt)}.json`);
  const latest = path.join(directory, "eval-latest.json");

  await writeFile(stamped, body, "utf8");
  await writeFile(latest, body, "utf8");

  return [stamped, latest];
}

/* ===========================================================================
 * Calibration.
 * ======================================================================== */

/**
 * One chunk's score against one question, with the ground truth attached.
 *
 * `correct` is not a judgement: it is `expectedChunkIds.includes(chunkId)` for
 * a question the corpus answers, and `false` for every chunk of a question the
 * corpus does not answer.
 */
export interface CalibrationSample {
  readonly questionId: string;
  readonly outcome: Outcome;
  readonly chunkId: string;
  readonly score: number;
  readonly correct: boolean;
}

/** Summary statistics for one side of the split. */
export interface Distribution {
  readonly count: number;
  readonly min: number;
  readonly max: number;
  readonly mean: number;
  readonly median: number;
  /** Nearest-rank percentiles. No interpolation: these are report figures. */
  readonly p05: number;
  readonly p95: number;
}

/** Whether a floor exists, and where it would go. */
export interface Separation {
  /** The worst score among chunks that SHOULD be retrieved. The ceiling on the floor. */
  readonly lowestCorrect: number | null;
  /** The best score among chunks that should NOT be. The floor on the floor. */
  readonly highestWrong: number | null;
  /** True when every correct chunk outscores every wrong one. */
  readonly separable: boolean;
  /** Midpoint of the gap when separable, else null. NOT applied anywhere. */
  readonly suggestedFloor: number | null;
  /** One sentence a human can act on. */
  readonly verdict: string;
}

/** The whole calibration run. */
export interface CalibrationReport {
  readonly samples: readonly CalibrationSample[];
  readonly correct: Distribution;
  readonly wrong: Distribution;
  readonly separation: Separation;
  /** The constant this report exists to inform. Reported, never modified. */
  readonly currentThreshold: number;
}

const EMPTY_DISTRIBUTION: Distribution = {
  count: 0,
  min: Number.NaN,
  max: Number.NaN,
  mean: Number.NaN,
  median: Number.NaN,
  p05: Number.NaN,
  p95: Number.NaN,
};

/** Nearest-rank percentile over an already-sorted ascending array. */
function percentile(sorted: readonly number[], fraction: number): number {
  if (sorted.length === 0) return Number.NaN;
  const rank = Math.min(
    sorted.length - 1,
    Math.max(0, Math.ceil(fraction * sorted.length) - 1),
  );
  return sorted[rank] ?? Number.NaN;
}

export function describeDistribution(values: readonly number[]): Distribution {
  if (values.length === 0) return EMPTY_DISTRIBUTION;
  const sorted = [...values].sort((a, b) => a - b);
  const sum = sorted.reduce((total, value) => total + value, 0);
  return {
    count: sorted.length,
    min: sorted[0] ?? Number.NaN,
    max: sorted[sorted.length - 1] ?? Number.NaN,
    mean: sum / sorted.length,
    median: percentile(sorted, 0.5),
    p05: percentile(sorted, 0.05),
    p95: percentile(sorted, 0.95),
  };
}

/**
 * Score every chunk against every question and split the scores by ground truth.
 *
 * THIS IS WHAT THE `TODO(eval)` IN `retrieve.ts` POINTS AT. `MIN_SIMILARITY` is
 * currently 0.35, chosen by reasoning about where a general-purpose embedding
 * model puts unrelated text — which is a guess with a comment on it, and the
 * comment says so. The floor separating "worth showing the model" from "the
 * search found nothing" is an empirical property of the embedder, and this is
 * the measurement that would fix it.
 *
 * THE SPLIT. For an `answered` or `refused-contradiction` question, the chunks
 * in `expectedChunkIds` are the correct retrievals and every other chunk in the
 * corpus is a wrong one. For a `refused-no-documents` question, EVERY chunk is
 * a wrong retrieval — there is no right answer to retrieve. Those questions are
 * the most informative rows in the whole report: the floor has to sit above the
 * best score anything in the corpus achieves against them, or the generator
 * gets called on a question the corpus cannot answer, and the refusal path this
 * project is built around never runs.
 *
 * EVERY CHUNK IS SCORED, not the top k. A top-k view cannot show where the
 * correct chunk landed when it was ranked eleventh, and "the right answer was
 * there at 0.41 and the floor was 0.35" is exactly the fact being looked for.
 *
 * NOTHING IS CHANGED. This function returns a report. `MIN_SIMILARITY` stays
 * where it is until someone runs this against the LIVE embedder and moves it in
 * a reviewed commit — the number is a property of the embedding model, so a
 * figure produced by the test fake would be a measurement of the fake.
 */
export async function calibrateThreshold(
  questions: readonly EvalQuestion[],
  index: VectorIndex,
  embedder: Embedder,
): Promise<CalibrationReport> {
  const samples: CalibrationSample[] = [];

  for (const question of questions) {
    const expected = new Set(question.expectedChunkIds);
    const results: SearchResult[] = await search(
      index,
      question.question,
      embedder,
      index.entries.length,
    );
    for (const result of results) {
      samples.push({
        questionId: question.id,
        outcome: question.outcome,
        chunkId: result.chunk.chunkId,
        score: result.score,
        // A no-answer question has no correct chunk, by construction.
        correct: question.outcome === "refused-no-documents" ? false : expected.has(result.chunk.chunkId),
      });
    }
  }

  const correctScores = samples.filter((sample) => sample.correct).map((sample) => sample.score);
  const wrongScores = samples.filter((sample) => !sample.correct).map((sample) => sample.score);

  const lowestCorrect = correctScores.length === 0 ? null : Math.min(...correctScores);
  const highestWrong = wrongScores.length === 0 ? null : Math.max(...wrongScores);

  const separable =
    lowestCorrect !== null && highestWrong !== null && lowestCorrect > highestWrong;

  const separation: Separation = {
    lowestCorrect,
    highestWrong,
    separable,
    suggestedFloor:
      separable && lowestCorrect !== null && highestWrong !== null
        ? (lowestCorrect + highestWrong) / 2
        : null,
    verdict: describeSeparation(lowestCorrect, highestWrong, separable),
  };

  return {
    samples,
    correct: describeDistribution(correctScores),
    wrong: describeDistribution(wrongScores),
    separation,
    currentThreshold: MIN_SIMILARITY,
  };
}

function describeSeparation(
  lowestCorrect: number | null,
  highestWrong: number | null,
  separable: boolean,
): string {
  if (lowestCorrect === null || highestWrong === null) {
    return "not enough samples on both sides to say anything about a floor.";
  }
  if (separable) {
    return [
      `the distributions separate: every chunk that should be retrieved scores above every chunk that should not`,
      `(worst correct ${lowestCorrect.toFixed(4)} > best wrong ${highestWrong.toFixed(4)}).`,
      `Any floor strictly between them retrieves everything required and nothing spurious.`,
    ].join(" ");
  }
  return [
    `the distributions OVERLAP: the worst correct chunk scores ${lowestCorrect.toFixed(4)} and the best wrong chunk`,
    `scores ${highestWrong.toFixed(4)}, so no single floor both admits every required chunk and excludes every`,
    `spurious one. A floor above ${highestWrong.toFixed(4)} refuses answerable questions; a floor below it lets`,
    `unanswerable ones through to the generator. The second failure is the one this project exists to prevent,`,
    `so the floor belongs at the high end of the overlap and the residual over-refusals are a retrieval problem,`,
    `not a threshold one.`,
  ].join(" ");
}

function stat(value: number): string {
  return Number.isNaN(value) ? "   n/a" : value.toFixed(4);
}

/**
 * A human-readable calibration report. Printed by the CLI, asserted in tests.
 *
 * `usage` is optional for the same reason as in `formatGateReport`: without
 * it the output is unchanged byte for byte. The CLI passes the embedder's
 * cumulative usage — a calibration run is pure embedding, so tokens, time,
 * and dollars here are the whole bill.
 */
export function formatCalibrationReport(
  report: CalibrationReport,
  usage?: EmbeddingUsage,
): string {
  const row = (label: string, distribution: Distribution): string =>
    `  ${padRight(label, 26)} n=${padRight(String(distribution.count), 6)} min ${stat(distribution.min)}  p05 ${stat(distribution.p05)}  median ${stat(distribution.median)}  p95 ${stat(distribution.p95)}  max ${stat(distribution.max)}  mean ${stat(distribution.mean)}`;

  const usageSection =
    usage === undefined
      ? []
      : [
          ``,
          `  EMBEDDING USAGE AND COST`,
          `  embedding tokens            ${usage.totalTokens}`,
          `  embedding calls             ${usage.calls}`,
          `  latency (ms, wall clock)    ${usage.totalLatencyMs}`,
          `  cost                        ${formatUsd(computeEmbeddingCost(usage.totalTokens))}`,
          ``,
          `  ${PRICING.embedding.model} priced as of ${PRICING.embedding.asOf}, committed in src/cost.ts. Prices change`,
          `  and this repository does not phone home — a stale constant is a wrong dollar figure`,
          `  with nothing loud about it.`,
        ];

  return [
    `SIMILARITY CALIBRATION`,
    ``,
    row("correct retrievals", report.correct),
    row("wrong retrievals", report.wrong),
    ``,
    `  current MIN_SIMILARITY      ${stat(report.currentThreshold)}   (unchanged by this report)`,
    `  worst correct chunk         ${report.separation.lowestCorrect === null ? "   n/a" : stat(report.separation.lowestCorrect)}`,
    `  best wrong chunk            ${report.separation.highestWrong === null ? "   n/a" : stat(report.separation.highestWrong)}`,
    `  suggested floor             ${report.separation.suggestedFloor === null ? "   n/a" : stat(report.separation.suggestedFloor)}`,
    ``,
    `  ${report.separation.verdict}`,
    ``,
    `  This report does not change MIN_SIMILARITY. The floor is a property of the embedding`,
    `  model, so a number measured against anything but the live embedder is a number about`,
    `  the wrong thing. Moving it is a reviewed commit to src/retrieve.ts.`,
    ...usageSection,
  ].join("\n");
}
