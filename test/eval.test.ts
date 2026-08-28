/**
 * eval.test.ts — the gate, tested offline.
 *
 * NO NETWORK, NO KEYS, NO CLOCK. The whole point of a gate is that it is
 * trustworthy when it goes red, and a gate whose own tests need a live provider
 * is one outage away from being disabled. Every run below uses an injected
 * embedder and an injected generator over the REAL corpus, so the chunk ids the
 * scores are checked against are the ids `src/chunk.ts` actually produces.
 */

import { mkdtemp, readFile, rm } from "node:fs/promises";
import { tmpdir } from "node:os";
import path from "node:path";
import { fileURLToPath } from "node:url";
import { beforeAll, describe, expect, it } from "vitest";

import type { AnswerResult, Generator } from "../src/answer.js";
import type { Chunk } from "../src/chunk.js";
import { loadCorpus } from "../src/chunk.js";
import type { Claim } from "../src/verify.js";
import type { Embedder, EmbeddingKind, VectorIndex } from "../src/retrieve.js";
import { MIN_SIMILARITY, buildIndex } from "../src/retrieve.js";
import {
  type CalibrationRunMetadata,
  type CalibrationSample,
  type EvalQuestion,
  type Outcome,
  type QuestionScore,
  type RunMetadata,
  ARTIFACT_SCHEMA_VERSION,
  BOUNDARY_HIGHEST_WRONG_COUNT,
  BOUNDARY_LOWEST_CORRECT_COUNT,
  CALIBRATION_ARTIFACT_SCHEMA_VERSION,
  GATE_POLICY,
  HEADLINE_FAILURE_KINDS,
  OUTCOMES,
  QuestionSetError,
  applyGate,
  buildArtifact,
  buildCalibrationArtifact,
  calibrateThreshold,
  describeDistribution,
  evidenceFrom,
  formatCalibrationReport,
  formatGateReport,
  formatResultsTable,
  loadQuestionSet,
  parseQuestionSet,
  runEvaluation,
  scoreQuestion,
  selectBoundaryCases,
  selectMustRefuseCeiling,
  writeArtifact,
  writeCalibrationArtifact,
} from "../src/eval.js";
import { EMBEDDING_PRICE_AS_OF, computeEmbeddingCost } from "../src/cost.js";
import {
  type ConsoleLike,
  type Environment,
  EXIT_GATE_FAILED,
  EXIT_PASS,
  EXIT_SETUP_ERROR,
  main,
  mergeEnvironment,
  parseArgs,
  parseDotEnv,
  resolveKey,
  resolveKeys,
  resolveModel,
} from "../src/eval-cli.js";

const REPO_ROOT = new URL("../", import.meta.url);
const CORPUS_DIR = fileURLToPath(new URL("corpus", REPO_ROOT));
const QUESTION_SET = fileURLToPath(new URL("eval/questions.json", REPO_ROOT));

/**
 * A string that looks enough like a credential to be worth grepping for.
 * Matches the convention in `test/retrieve.test.ts`. Every error path that
 * could conceivably touch a key is checked for its absence.
 */
const DECOY_KEY = "pa-DECOY-KEY-9f3c1e7a5b2d-do-not-log";

/**
 * Narrow away `undefined` and `null` with a message that says what was missing.
 *
 * `noUncheckedIndexedAccess` makes every array and Map lookup optional, and
 * several fields here are deliberately nullable (`failure`, `suggestedFloor`).
 * A bare `!` would turn a missing fixture into an unreadable "cannot read
 * property of undefined" three frames away from the cause.
 */
function must<T>(value: T | undefined | null, what: string): T {
  if (value === undefined || value === null) {
    throw new Error(`test fixture is missing ${what}`);
  }
  return value;
}

/* ===========================================================================
 * Fixtures over the real corpus.
 * ======================================================================== */

let chunks: readonly Chunk[];
let byId: Map<string, Chunk>;
let questions: readonly EvalQuestion[];

beforeAll(async () => {
  chunks = await loadCorpus(CORPUS_DIR);
  byId = new Map(chunks.map((chunk) => [chunk.chunkId, chunk]));
  questions = (await loadQuestionSet(QUESTION_SET)).questions;
});

function questionById(id: string): EvalQuestion {
  return must(
    questions.find((question) => question.id === id),
    `question "${id}"`,
  );
}

/* ===========================================================================
 * The oracle embedder.
 * ======================================================================== */

/**
 * An embedder that retrieves exactly what it is told to, and nothing else.
 *
 * WHY NOT `createFakeEmbedder`. That fake is a real (crude) bag-of-words model,
 * and it is the right tool in `retrieve.test.ts` where the ranking is the thing
 * under test. Here the ranking is not under test — the GATE is — and a gate
 * test that fails because a hashing embedder put the wrong near-duplicate in
 * position four is a test that reports a retrieval regression as a scoring bug.
 * Holding retrieval fixed is what makes every failure below attributable to the
 * code this file is about.
 *
 * HOW. One dimension per question. A question's query vector is that basis
 * vector; a chunk's vector has a 1 in the dimension of every question whose
 * plan names it. Cosine similarity is therefore 1/sqrt(n) for a planned chunk
 * and exactly 0 for everything else — comfortably either side of any floor —
 * so "what was retrieved" is a property of the plan and of nothing else.
 *
 * A question with no plan retrieves nothing above the floor, which is how the
 * must-refuse questions are made to behave the way the live system should.
 */
function createOracleEmbedder(
  corpus: readonly Chunk[],
  set: readonly EvalQuestion[],
  plan: ReadonlyMap<string, readonly string[]>,
): Embedder {
  const dimensions = set.length;
  const zeros = (): number[] => new Array<number>(dimensions).fill(0);

  const dimensionOfQuestionText = new Map<string, number>();
  set.forEach((question, position) => dimensionOfQuestionText.set(question.question, position));

  const vectorOfChunkText = new Map<string, number[]>();
  for (const chunk of corpus) vectorOfChunkText.set(chunk.text, zeros());

  for (const [questionId, chunkIds] of plan) {
    const question = must(
      set.find((candidate) => candidate.id === questionId),
      `planned question "${questionId}"`,
    );
    const dimension = must(
      dimensionOfQuestionText.get(question.question),
      `dimension for "${questionId}"`,
    );
    for (const chunkId of chunkIds) {
      const chunk = must(
        corpus.find((candidate) => candidate.chunkId === chunkId),
        `planned chunk "${chunkId}"`,
      );
      const vector = must(vectorOfChunkText.get(chunk.text), `vector for "${chunkId}"`);
      vector[dimension] = 1;
    }
  }

  return {
    async embed(texts: string[], _kind: EmbeddingKind): Promise<number[][]> {
      return texts.map((text) => {
        const dimension = dimensionOfQuestionText.get(text);
        if (dimension !== undefined) {
          const basis = zeros();
          basis[dimension] = 1;
          return basis;
        }
        return vectorOfChunkText.get(text) ?? zeros();
      });
    },
  };
}

/** The plan a correctly-behaving system would produce: expected chunks only. */
function idealPlan(set: readonly EvalQuestion[]): Map<string, readonly string[]> {
  const plan = new Map<string, readonly string[]>();
  for (const question of set) {
    if (question.expectedChunkIds.length > 0) plan.set(question.id, question.expectedChunkIds);
  }
  return plan;
}

/**
 * Chunks that a too-low similarity floor would surface for the two questions
 * the corpus cannot answer. Both are genuinely adjacent — the redelivery
 * section is the Saturday trap's near neighbour in the same document, and the
 * held-order FAQ entry is the downstream consequence of a payment problem
 * stated without any of its causes. Retrieving either is plausible; ANSWERING
 * from either is the headline failure.
 */
const HALLUCINATION_BAIT: ReadonlyMap<string, readonly string[]> = new Map([
  ["saturday-delivery-surcharge", ["delivery-zones-and-schedules#redelivery-after-a-failed-attempt"]],
  ["late-payment-fee", ["wholesale-faq#why-was-my-order-held"]],
]);

/* ===========================================================================
 * Generator fakes.
 * ======================================================================== */

/** The user message opens with `Question: ...`; that is how a fake routes. */
function questionFromUserMessage(user: string): string {
  const first = must(user.split("\n")[0], "first line of the user message");
  return first.startsWith("Question: ") ? first.slice("Question: ".length) : first;
}

/**
 * A verbatim slice of a chunk, which is what makes these fakes useful.
 *
 * `verify.ts` checks quotes by exact string containment against the chunk they
 * cite, so a fake that invents prose produces claims that are all rejected and
 * a pipeline that always refuses. Slicing the real bytes is the only way to
 * drive the ANSWERED path at all — the verifier is not mocked out anywhere in
 * this file.
 *
 * 120 characters is long enough to carry the numbers the contradiction detector
 * looks for (`14 calendar days`, `30 calendar days`) out of the two return
 * window sections, which is exactly what the contradiction case needs.
 */
function quoteFrom(chunk: Chunk): string {
  return chunk.text.slice(0, 120);
}

function statedClaim(chunkId: string, chunk: Chunk): Claim {
  return {
    text: `On this point the corpus states what appears in ${chunk.sectionHeading}.`,
    status: "stated",
    citations: [chunkId],
    supportingQuotes: [{ chunkId, quote: quoteFrom(chunk) }],
  };
}

function inferredClaim(chunkIds: readonly string[], lookup: ReadonlyMap<string, Chunk>): Claim {
  return {
    text: `Answering this requires combining ${chunkIds.join(" and ")}.`,
    status: "inferred",
    citations: [...chunkIds],
    supportingQuotes: chunkIds.map((chunkId) => ({
      chunkId,
      quote: quoteFrom(must(lookup.get(chunkId), `chunk "${chunkId}"`)),
    })),
  };
}

/**
 * Claims a well-behaved generator would produce for a question, given the
 * chunks it should have been shown.
 *
 * The contradiction question gets TWO `stated` claims, one per document,
 * because `detectContradiction` compares claims pairwise — a single claim
 * cannot disagree with itself, and a fake that emitted one would never exercise
 * the contradiction path. Every other multi-chunk question gets one `inferred`
 * claim citing both, which is what `verify.ts` requires of a claim resting on
 * more than one chunk and which also keeps the join questions structurally
 * unable to trip the contradiction detector by accident.
 */
function claimsFor(
  question: EvalQuestion,
  chunkIds: readonly string[],
  lookup: ReadonlyMap<string, Chunk>,
): Claim[] {
  if (chunkIds.length === 0) return [];

  if (question.outcome === "refused-contradiction" || chunkIds.length === 1) {
    return chunkIds.map((chunkId) =>
      statedClaim(chunkId, must(lookup.get(chunkId), `chunk "${chunkId}"`)),
    );
  }

  return [inferredClaim(chunkIds, lookup)];
}

/** A generator built from a function of the question text. Records its calls. */
interface RecordingGenerator extends Generator {
  readonly calls: readonly string[];
}

function createGenerator(respond: (question: string) => string): RecordingGenerator {
  const calls: string[] = [];
  return {
    calls,
    async generate(_system: string, user: string): Promise<string> {
      const question = questionFromUserMessage(user);
      calls.push(question);
      return respond(question);
    },
  };
}

/** Answers every question it is asked, grounded in whatever it was planned. */
function groundedGenerator(
  set: readonly EvalQuestion[],
  lookup: ReadonlyMap<string, Chunk>,
  plan: ReadonlyMap<string, readonly string[]>,
): RecordingGenerator {
  return createGenerator((questionText) => {
    const question = must(
      set.find((candidate) => candidate.question === questionText),
      `question matching ${JSON.stringify(questionText)}`,
    );
    const chunkIds = plan.get(question.id) ?? [];
    return JSON.stringify({ claims: claimsFor(question, chunkIds, lookup) });
  });
}

/**
 * Produces zero claims for everything.
 *
 * An empty `claims` array is the generator saying "these passages do not
 * support an answer", and `answerQuestion` turns it into a refusal. That is the
 * shape of a system tuned until it declines rather than risks being wrong.
 */
function refuseEverythingGenerator(): RecordingGenerator {
  return createGenerator(() => JSON.stringify({ claims: [] }));
}

/* ===========================================================================
 * Synthetic results, for the scoring rules.
 * ======================================================================== */

function answeredWith(citations: readonly string[]): AnswerResult {
  return {
    kind: "answered",
    claims: citations.map((chunkId) => ({
      text: `something about ${chunkId}`,
      status: "stated" as const,
      citations: [chunkId],
      supportingQuotes: [{ chunkId, quote: "..." }],
    })),
    rejectedClaims: [],
    retrieved: [],
  };
}

function refusedWith(
  reason: "no-relevant-documents" | "contradictory-sources" | "no-claims-produced",
  detail: string,
): AnswerResult {
  return { kind: "refused", reason, detail, nearMisses: [] };
}

/* ===========================================================================
 * The question set itself.
 * ======================================================================== */

describe("the committed question set", () => {
  it("loads, validates, and has the shape the gate assumes", () => {
    expect(questions).toHaveLength(14);

    const counts = new Map<string, number>();
    for (const question of questions) {
      counts.set(question.outcome, (counts.get(question.outcome) ?? 0) + 1);
    }

    expect(counts.get("answered")).toBe(11);
    expect(counts.get("refused-no-documents")).toBe(2);
    expect(counts.get("refused-contradiction")).toBe(1);
  });

  it("names only chunk ids that the corpus actually produces", () => {
    for (const question of questions) {
      for (const chunkId of question.expectedChunkIds) {
        expect(
          byId.has(chunkId),
          `question "${question.id}" expects chunk "${chunkId}", which the corpus does not contain`,
        ).toBe(true);
      }
      for (const docId of question.contradictionDocIds) {
        expect(
          chunks.some((chunk) => chunk.docId === docId),
          `question "${question.id}" names document "${docId}", which the corpus does not contain`,
        ).toBe(true);
      }
    }
  });

  it("rejects an answered question with no expected chunk ids", () => {
    // Without this rule the question passes on ANY answer, which is a gate that
    // reports a pass while checking nothing.
    const source = JSON.stringify({
      corpusDir: "corpus",
      questions: [{ id: "q", question: "?", outcome: "answered", rationale: "r" }],
    });
    expect(() => parseQuestionSet(source)).toThrowError(QuestionSetError);
    try {
      parseQuestionSet(source);
    } catch (error) {
      expect((error as QuestionSetError).kind).toBe("missing-expected-chunk-ids");
    }
  });

  it("rejects a contradiction question naming fewer than two documents", () => {
    const source = JSON.stringify({
      corpusDir: "corpus",
      questions: [
        {
          id: "q",
          question: "?",
          outcome: "refused-contradiction",
          contradictionDocIds: ["only-one"],
          rationale: "r",
        },
      ],
    });
    try {
      parseQuestionSet(source);
      expect.unreachable("a one-sided contradiction should not validate");
    } catch (error) {
      expect((error as QuestionSetError).kind).toBe("missing-contradiction-doc-ids");
    }
  });

  it("rejects duplicate ids, unknown outcomes, malformed chunk ids, and non-JSON", () => {
    const cases: readonly { source: string; kind: string }[] = [
      {
        kind: "duplicate-id",
        source: JSON.stringify({
          corpusDir: "corpus",
          questions: [
            { id: "q", question: "?", outcome: "answered", expectedChunkIds: ["a#b"], rationale: "r" },
            { id: "q", question: "?", outcome: "answered", expectedChunkIds: ["a#b"], rationale: "r" },
          ],
        }),
      },
      {
        kind: "unknown-outcome",
        source: JSON.stringify({
          corpusDir: "corpus",
          questions: [{ id: "q", question: "?", outcome: "maybe", rationale: "r" }],
        }),
      },
      {
        kind: "malformed-chunk-id",
        source: JSON.stringify({
          corpusDir: "corpus",
          questions: [
            { id: "q", question: "?", outcome: "answered", expectedChunkIds: ["no-hash"], rationale: "r" },
          ],
        }),
      },
      { kind: "not-json", source: "{" },
      { kind: "missing-questions-array", source: JSON.stringify({ corpusDir: "corpus" }) },
      { kind: "missing-corpus-dir", source: JSON.stringify({ questions: [] }) },
    ];

    for (const testCase of cases) {
      try {
        parseQuestionSet(testCase.source);
        expect.unreachable(`expected ${testCase.kind}`);
      } catch (error) {
        expect(error).toBeInstanceOf(QuestionSetError);
        expect((error as QuestionSetError).kind).toBe(testCase.kind);
      }
    }
  });
});

/* ===========================================================================
 * Scoring rules.
 * ======================================================================== */

describe("scoring: a correct refusal is a pass", () => {
  it("scores a refusal of an unanswerable question as a PASS, not as neutral", () => {
    const question = questionById("saturday-delivery-surcharge");
    const score = scoreQuestion(
      question,
      refusedWith("no-relevant-documents", "nothing reached the floor of 0.3500"),
    );

    expect(score.passed).toBe(true);
    expect(score.failure).toBeNull();
    expect(score.note).toContain("correct");

    // And it counts toward its category exactly as an answer counts toward its
    // own: a full pass, no partial credit, no separate "declined" bucket.
    const report = applyGate([score]);
    const category = must(
      report.categories.find((entry) => entry.category === "refused-no-documents"),
      "the must-refuse category",
    );
    expect(category.passed).toBe(1);
    expect(category.rate).toBe(1);
    expect(report.passed).toBe(true);
  });

  it("accepts any refusal reason, and records which one it was", () => {
    const question = questionById("late-payment-fee");
    for (const reason of ["no-relevant-documents", "no-claims-produced"] as const) {
      const score = scoreQuestion(question, refusedWith(reason, "detail"));
      expect(score.passed).toBe(true);
      expect(score.observed.refusalReason).toBe(reason);
      expect(score.note).toContain(reason);
    }
  });
});

describe("scoring: the headline failure", () => {
  it("names an answer to a must-refuse question as the headline failure", () => {
    const question = questionById("saturday-delivery-surcharge");
    const score = scoreQuestion(
      question,
      answeredWith(["delivery-zones-and-schedules#redelivery-after-a-failed-attempt"]),
    );

    expect(score.passed).toBe(false);
    expect(must(score.failure, "a failure").kind).toBe("hallucinated-answer");
    expect(must(score.failure, "a failure").headline).toBe(true);
    expect(HEADLINE_FAILURE_KINDS).toContain("hallucinated-answer");
    expect(must(score.failure, "a failure").message).toContain("saturday-delivery-surcharge");
  });

  it("names a resolved contradiction as a headline failure too", () => {
    const question = questionById("return-window-contradiction");
    const score = scoreQuestion(
      question,
      answeredWith(["customer-care-handbook#return-window-for-stocked-goods"]),
    );

    expect(score.passed).toBe(false);
    expect(must(score.failure, "a failure").kind).toBe("resolved-contradiction");
    expect(must(score.failure, "a failure").headline).toBe(true);
    // Answering with the NEWER document is still a failure: a later effective
    // date does not by itself make a document operative.
    expect(must(score.failure, "a failure").message).toContain("customer-care-handbook");
  });
});

describe("scoring: contradictions", () => {
  const bothDocs = [
    "returns-and-credits-policy#return-window-and-condition-requirements",
    "customer-care-handbook#return-window-for-stocked-goods",
  ];

  it("passes only when both documents are surfaced", () => {
    const question = questionById("return-window-contradiction");
    const score = scoreQuestion(
      question,
      refusedWith(
        "contradictory-sources",
        `Position 1: quoted from ${bothDocs[0]}. Position 2: quoted from ${bothDocs[1]}.`,
      ),
    );

    expect(score.passed).toBe(true);
    expect(score.observed.evidence.docIds).toEqual([
      "customer-care-handbook",
      "returns-and-credits-policy",
    ]);
  });

  it("fails a refusal that surfaces only one side, and says which is missing", () => {
    const question = questionById("return-window-contradiction");
    const score = scoreQuestion(
      question,
      refusedWith("contradictory-sources", `Position 1: quoted from ${bothDocs[0]}.`),
    );

    expect(score.passed).toBe(false);
    expect(must(score.failure, "a failure").kind).toBe("missing-contradiction-source");
    expect(must(score.failure, "a failure").message).toContain("customer-care-handbook");
    // Not a headline failure: nothing false was published, the reader was just
    // shown half the conflict. Different defect, different urgency.
    expect(must(score.failure, "a failure").headline).toBe(false);
  });
});

describe("scoring: answered questions", () => {
  it("requires every expected chunk id and allows extra citations", () => {
    const question = questionById("minimum-order-value");
    const score = scoreQuestion(
      question,
      answeredWith([...question.expectedChunkIds, "wholesale-faq#why-was-my-order-held"]),
    );

    expect(score.passed).toBe(true);
    expect(score.note).toContain("+1 extra");
  });

  it("fails when one of several required chunk ids is missing", () => {
    const question = questionById("join-zone-c-freight-820");
    expect(question.expectedChunkIds).toHaveLength(2);

    const partial = must(question.expectedChunkIds[0], "the first expected chunk");
    const missing = must(question.expectedChunkIds[1], "the second expected chunk");
    const score = scoreQuestion(question, answeredWith([partial]));

    expect(score.passed).toBe(false);
    expect(must(score.failure, "a failure").kind).toBe("missing-citation");
    expect(must(score.failure, "a failure").message).toContain(missing);
    expect(must(score.failure, "a failure").headline).toBe(false);
  });

  it("names a refusal of an answerable question as an over-refusal", () => {
    const question = questionById("restocking-fee");
    const score = scoreQuestion(
      question,
      refusedWith("no-relevant-documents", "nothing reached the floor"),
    );

    expect(score.passed).toBe(false);
    expect(must(score.failure, "a failure").kind).toBe("over-refusal");
    expect(must(score.failure, "a failure").message).toContain("OVER-REFUSAL");
  });
});

describe("evidence", () => {
  it("reads citations from an answer and chunk ids from a refusal's detail", () => {
    const answered = evidenceFrom(answeredWith(["a-doc#a-section", "b-doc#b-section"]));
    expect(answered.source).toBe("claim-citations");
    expect(answered.chunkIds).toEqual(["a-doc#a-section", "b-doc#b-section"]);
    expect(answered.docIds).toEqual(["a-doc", "b-doc"]);

    const refused = evidenceFrom(
      refusedWith("contradictory-sources", "quoted from a-doc#a-section and from b-doc#b-section"),
    );
    expect(refused.source).toBe("refusal-detail");
    expect(refused.docIds).toEqual(["a-doc", "b-doc"]);
  });

  it("de-duplicates and sorts, so two runs compare byte for byte", () => {
    const evidence = evidenceFrom(answeredWith(["z-doc#s", "a-doc#s", "z-doc#s"]));
    expect(evidence.chunkIds).toEqual(["a-doc#s", "z-doc#s"]);
  });
});

/* ===========================================================================
 * The gate.
 * ======================================================================== */

describe("the gate policy", () => {
  it("requires 100% of both must-refuse categories and states why", () => {
    expect(GATE_POLICY["refused-no-documents"].threshold).toBe(1);
    expect(GATE_POLICY["refused-contradiction"].threshold).toBe(1);
    expect(GATE_POLICY["refused-no-documents"].rationale).toContain("no partial credit");
  });

  it("states the answerable bar explicitly, and lower than the refusal bar", () => {
    const answerable = GATE_POLICY.answered.threshold;
    expect(answerable).toBe(0.8);
    expect(answerable).toBeLessThan(GATE_POLICY["refused-no-documents"].threshold);
    expect(GATE_POLICY.answered.rationale).toContain("80%");
  });

  it("covers every outcome, with no category left un-gated", () => {
    for (const outcome of OUTCOMES) {
      expect(GATE_POLICY[outcome].threshold).toBeGreaterThan(0);
      expect(GATE_POLICY[outcome].rationale.length).toBeGreaterThan(80);
    }
  });
});

describe("applying the gate", () => {
  /** Score N answered questions, `passing` of which cite correctly. */
  function answeredScores(passing: number): QuestionScore[] {
    const answerable = questions.filter((question) => question.outcome === "answered");
    return answerable.map((question, position) =>
      scoreQuestion(
        question,
        position < passing
          ? answeredWith(question.expectedChunkIds)
          : answeredWith(["some-other-doc#some-other-section"]),
      ),
    );
  }

  it("fails the whole gate on a single must-refuse miss", () => {
    const scores = [
      ...answeredScores(11),
      scoreQuestion(
        questionById("saturday-delivery-surcharge"),
        answeredWith(["delivery-zones-and-schedules#redelivery-after-a-failed-attempt"]),
      ),
      scoreQuestion(questionById("late-payment-fee"), refusedWith("no-relevant-documents", "x")),
    ];

    const report = applyGate(scores);
    expect(report.passed).toBe(false);
    // Eleven of thirteen passed. A blended score would have called that 85%.
    expect(report.totalPassed).toBe(12);
    const category = must(
      report.categories.find((entry) => entry.category === "refused-no-documents"),
      "the must-refuse category",
    );
    expect(category.rate).toBe(0.5);
    expect(category.met).toBe(false);
  });

  it("holds the answerable category at 9 of 11", () => {
    // 80% of 11 is 8.8, so nine pass and eight do not. Stated as an assertion
    // rather than left implicit in a fraction, because this is the number
    // someone will want to know when the build goes red.
    expect(applyGate(answeredScores(9)).passed).toBe(true);
    expect(applyGate(answeredScores(8)).passed).toBe(false);
  });

  it("names every violation with its question id and what went wrong", () => {
    const report = applyGate(answeredScores(9));
    expect(report.violations).toHaveLength(2);
    for (const violation of report.violations) {
      expect(violation.questionId).not.toBe("");
      expect(questions.some((question) => question.id === violation.questionId)).toBe(true);
      expect(violation.kind).toBe("missing-citation");
      expect(violation.message).toContain(violation.questionId);
    }
  });

  it("puts headline violations first, whatever order the questions ran in", () => {
    const report = applyGate([
      scoreQuestion(questionById("restocking-fee"), refusedWith("no-relevant-documents", "x")),
      scoreQuestion(
        questionById("late-payment-fee"),
        answeredWith(["wholesale-faq#why-was-my-order-held"]),
      ),
    ]);

    expect(must(report.violations[0], "the first violation").headline).toBe(true);
    expect(must(report.violations[0], "the first violation").questionId).toBe("late-payment-fee");
    expect(report.headlineViolations).toHaveLength(1);
  });

  it("treats an empty category as vacuously met rather than as a failure", () => {
    const report = applyGate([
      scoreQuestion(questionById("restocking-fee"), answeredWith(questionById("restocking-fee").expectedChunkIds)),
    ]);
    const empty = must(
      report.categories.find((entry) => entry.category === "refused-contradiction"),
      "the contradiction category",
    );
    expect(empty.total).toBe(0);
    expect(empty.met).toBe(true);
    expect(report.passed).toBe(true);
  });
});

describe("the printed report", () => {
  it("prints a row per question, with the headline rows marked", () => {
    const report = applyGate([
      scoreQuestion(questionById("restocking-fee"), answeredWith(questionById("restocking-fee").expectedChunkIds)),
      scoreQuestion(
        questionById("saturday-delivery-surcharge"),
        answeredWith(["delivery-zones-and-schedules#redelivery-after-a-failed-attempt"]),
      ),
    ]);

    const table = formatResultsTable(report);
    expect(table).toContain("STATUS");
    expect(table).toContain("restocking-fee");
    expect(table).toMatch(/HEADLINE\s+saturday-delivery-surcharge/);

    const printed = formatGateReport(report);
    // The headline section is above the table, and the table is above the
    // violations: the first thing read is the thing that matters most.
    expect(printed.indexOf("HEADLINE FAILURE")).toBeLessThan(printed.indexOf("STATUS"));
    expect(printed.indexOf("STATUS")).toBeLessThan(printed.indexOf("VIOLATIONS:"));
    expect(printed).toContain("VERDICT: FAIL");
    expect(printed).toContain("GATE POLICY");
  });
});

/* ===========================================================================
 * The whole gate, end to end, offline.
 * ======================================================================== */

describe("the whole gate, offline, over the real corpus", () => {
  async function runWith(
    plan: ReadonlyMap<string, readonly string[]>,
    generatorFor: (embedder: Embedder, index: VectorIndex) => RecordingGenerator,
  ): Promise<ReturnType<typeof applyGate>> {
    const embedder = createOracleEmbedder(chunks, questions, plan);
    const index = await buildIndex(chunks, embedder);
    const generator = generatorFor(embedder, index);
    return applyGate(await runEvaluation(questions, { index, embedder, generator }));
  }

  it("passes when retrieval and generation both do their jobs", async () => {
    const plan = idealPlan(questions);
    const report = await runWith(plan, () => groundedGenerator(questions, byId, plan));

    expect(report.violations).toEqual([]);
    expect(report.passed).toBe(true);
    expect(report.totalPassed).toBe(14);

    // The contradiction question is not answered — it is refused with both
    // sides surfaced, which is what "neither is presented as the resolution"
    // means structurally.
    const contradiction = must(
      report.scores.find((score) => score.id === "return-window-contradiction"),
      "the contradiction score",
    );
    expect(contradiction.observed.kind).toBe("refused");
    expect(contradiction.observed.refusalReason).toBe("contradictory-sources");
    expect(contradiction.observed.evidence.docIds).toContain("returns-and-credits-policy");
    expect(contradiction.observed.evidence.docIds).toContain("customer-care-handbook");
  });

  it("never calls the generator for a question nothing clears the floor for", async () => {
    const plan = idealPlan(questions);
    const embedder = createOracleEmbedder(chunks, questions, plan);
    const index = await buildIndex(chunks, embedder);
    const generator = groundedGenerator(questions, byId, plan);

    await runEvaluation(questions, { index, embedder, generator });

    // Twelve questions had something above the floor; the two the corpus
    // cannot answer had nothing, and the generator was never asked.
    expect(generator.calls).toHaveLength(12);
    expect(generator.calls).not.toContain(questionById("saturday-delivery-surcharge").question);
    expect(generator.calls).not.toContain(questionById("late-payment-fee").question);
  });

  it("FAILS the gate, as a headline failure, when a fake answers the must-refuse questions", async () => {
    // The floor is too low: two plausible-but-irrelevant chunks come back above
    // it, the generator is called, and it does what a generator handed passages
    // and a question does — it answers.
    const plan = new Map(idealPlan(questions));
    for (const [questionId, chunkIds] of HALLUCINATION_BAIT) plan.set(questionId, chunkIds);

    const report = await runWith(plan, () => groundedGenerator(questions, byId, plan));

    expect(report.passed).toBe(false);
    expect(report.headlineViolations).toHaveLength(2);

    const ids = report.headlineViolations.map((violation) => violation.questionId).sort();
    expect(ids).toEqual(["late-payment-fee", "saturday-delivery-surcharge"]);

    for (const violation of report.headlineViolations) {
      expect(violation.kind).toBe("hallucinated-answer");
      expect(violation.headline).toBe(true);
    }

    // Reported FIRST and named as such.
    expect(must(report.violations[0], "the first violation").headline).toBe(true);
    const printed = formatGateReport(report);
    expect(printed.startsWith("HEADLINE FAILURES: 2")).toBe(true);
    expect(printed).toContain("VERDICT: FAIL");

    // And the answerable questions were all fine — which is the point. The
    // build is red for the one reason that matters, not for a blended score.
    const answerable = must(
      report.categories.find((entry) => entry.category === "answered"),
      "the answerable category",
    );
    expect(answerable.met).toBe(true);
  });

  it("FAILS a fake that refuses everything, on the answerables, while the refusals pass", async () => {
    // THE OVER-REFUSAL CASE. This is the failure mode a grounding project
    // acquires when it tunes only against hallucination: every refusal looks
    // safe in isolation, so the system learns to decline, and it scores
    // perfectly on every trap in the corpus while being useless.
    //
    // `restocking-fee` is the question that exists to catch it. It deliberately
    // touches the returns document WITHOUT touching the return window, so it
    // shares a document with one half of the contradiction pair: a system that
    // has learned to refuse anything returns-shaped answers the contradiction
    // correctly and fails here, and only a per-category gate can tell those two
    // outcomes apart.
    const plan = idealPlan(questions);
    const report = await runWith(plan, () => refuseEverythingGenerator());

    expect(report.passed).toBe(false);

    const byCategory = new Map(report.categories.map((entry) => [entry.category, entry]));
    expect(must(byCategory.get("refused-no-documents"), "must-refuse").met).toBe(true);
    expect(must(byCategory.get("refused-contradiction"), "contradiction").met).toBe(true);
    expect(must(byCategory.get("answered"), "answerable").met).toBe(false);
    expect(must(byCategory.get("answered"), "answerable").passed).toBe(0);

    // Not a single headline failure: nothing false was published. A gate that
    // reported only hallucinations would call this run clean.
    expect(report.headlineViolations).toEqual([]);

    const restocking = must(
      report.violations.find((violation) => violation.questionId === "restocking-fee"),
      "the restocking-fee violation — the over-refusal canary",
    );
    expect(restocking.kind).toBe("over-refusal");
    expect(restocking.message).toContain("OVER-REFUSAL");
    expect(restocking.message).toContain("returns-and-credits-policy#restocking-fee");

    expect(report.violations.every((violation) => violation.kind === "over-refusal")).toBe(true);
    expect(report.violations).toHaveLength(11);
  });
});

/* ===========================================================================
 * The artifact.
 * ======================================================================== */

describe("the artifact", () => {
  const metadata: RunMetadata = {
    generatedAt: "2026-08-22T10:12:00.000Z",
    embedder: "fake-oracle",
    generator: "fake-grounded",
    corpusDir: "corpus",
    chunkCount: 71,
    k: 6,
    threshold: MIN_SIMILARITY,
  };

  function failingReport(): ReturnType<typeof applyGate> {
    return applyGate([
      scoreQuestion(
        questionById("saturday-delivery-surcharge"),
        answeredWith(["delivery-zones-and-schedules#redelivery-after-a-failed-attempt"]),
      ),
      scoreQuestion(
        questionById("restocking-fee"),
        answeredWith(questionById("restocking-fee").expectedChunkIds),
      ),
    ]);
  }

  it("records the policy alongside the verdict", () => {
    const artifact = buildArtifact(failingReport(), metadata);

    expect(artifact.verdict).toBe("FAIL");
    expect(artifact.schemaVersion).toBe(ARTIFACT_SCHEMA_VERSION);

    // The thresholds the verdict was judged against, stored with it. Without
    // these, a FAIL read after the constants have moved is judged against
    // numbers that were not in force when it was produced.
    for (const outcome of OUTCOMES) {
      expect(artifact.policy.categories[outcome].threshold).toBe(GATE_POLICY[outcome].threshold);
      expect(artifact.policy.categories[outcome].rationale).toBe(GATE_POLICY[outcome].rationale);
    }
    expect(artifact.policy.headlineFailureKinds).toEqual(HEADLINE_FAILURE_KINDS);
    expect(artifact.policy.headlineRule).toContain("100%");

    expect(artifact.run).toEqual(metadata);
    expect(artifact.headlineViolations).toHaveLength(1);
  });

  it("writes a timestamped copy and a latest copy, both self-contained", async () => {
    const directory = await mkdtemp(path.join(tmpdir(), "rag-receipts-eval-"));
    try {
      const artifact = buildArtifact(failingReport(), metadata);
      const written = await writeArtifact(directory, artifact);

      expect(written).toHaveLength(2);
      expect(must(written[1], "the latest path").endsWith("eval-latest.json")).toBe(true);

      for (const file of written) {
        const parsed: unknown = JSON.parse(await readFile(file, "utf8"));
        const record = parsed as {
          verdict: string;
          policy: { categories: Record<string, { threshold: number }> };
        };
        // A stored FAIL is interpretable standing alone: the verdict and the
        // thresholds it was judged against are in the same file.
        expect(record.verdict).toBe("FAIL");
        expect(record.policy.categories["refused-no-documents"]?.threshold).toBe(1);
        expect(record.policy.categories["answered"]?.threshold).toBe(0.8);
      }
    } finally {
      await rm(directory, { recursive: true, force: true });
    }
  });

  it("carries the run's models and settings, and no credential-shaped field", () => {
    const artifact = buildArtifact(failingReport(), metadata);
    const serialised = JSON.stringify(artifact);
    expect(serialised).toContain("fake-oracle");
    expect(serialised).not.toContain("apiKey");
    expect(serialised).not.toContain("API_KEY");
  });
});

/* ===========================================================================
 * Calibration.
 * ======================================================================== */

describe("calibration", () => {
  it("splits the score distribution by ground truth and does not touch MIN_SIMILARITY", async () => {
    const before = MIN_SIMILARITY;
    const plan = idealPlan(questions);
    const embedder = createOracleEmbedder(chunks, questions, plan);
    const index = await buildIndex(chunks, embedder);

    const report = await calibrateThreshold(questions, index, embedder);

    // Every chunk scored against every question: 71 x 14.
    expect(report.samples).toHaveLength(chunks.length * questions.length);
    expect(report.correct.count).toBeGreaterThan(0);
    expect(report.wrong.count).toBeGreaterThan(report.correct.count);
    expect(report.currentThreshold).toBe(before);
    expect(MIN_SIMILARITY).toBe(before);
  });

  it("counts every chunk as wrong for a question the corpus cannot answer", async () => {
    const plan = idealPlan(questions);
    const embedder = createOracleEmbedder(chunks, questions, plan);
    const index = await buildIndex(chunks, embedder);

    const report = await calibrateThreshold(questions, index, embedder);
    const noAnswer = report.samples.filter(
      (sample) => sample.questionId === "saturday-delivery-surcharge",
    );

    expect(noAnswer).toHaveLength(chunks.length);
    expect(noAnswer.every((sample) => !sample.correct)).toBe(true);
    // The floor has to sit above the best of these, or the generator gets
    // called on a question with no answer in the corpus.
    expect(Math.max(...noAnswer.map((sample) => sample.score))).toBe(0);
  });

  it("reports a separable split with a suggested floor between the two sides", async () => {
    const plan = idealPlan(questions);
    const embedder = createOracleEmbedder(chunks, questions, plan);
    const index = await buildIndex(chunks, embedder);

    const report = await calibrateThreshold(questions, index, embedder);

    expect(report.separation.separable).toBe(true);
    const suggested = must(report.separation.suggestedFloor, "a suggested floor");
    expect(suggested).toBeGreaterThan(must(report.separation.highestWrong, "the best wrong score"));
    expect(suggested).toBeLessThan(must(report.separation.lowestCorrect, "the worst correct score"));

    const printed = formatCalibrationReport(report);
    expect(printed).toContain("SIMILARITY CALIBRATION");
    expect(printed).toContain("suggested floor");
    expect(printed).toContain("does not change MIN_SIMILARITY");
  });

  it("says so plainly when the distributions overlap", () => {
    // Not reachable through the oracle embedder, which separates by
    // construction — so the reporting path is exercised directly.
    const overlapping = describeDistribution([0.2, 0.5, 0.9]);
    expect(overlapping.min).toBeCloseTo(0.2);
    expect(overlapping.max).toBeCloseTo(0.9);
    expect(overlapping.median).toBeCloseTo(0.5);
    expect(overlapping.count).toBe(3);
    expect(describeDistribution([]).count).toBe(0);
  });

  it("names the boundary cases and keeps them consistent with the distributions", async () => {
    const plan = idealPlan(questions);
    const embedder = createOracleEmbedder(chunks, questions, plan);
    const index = await buildIndex(chunks, embedder);

    const report = await calibrateThreshold(questions, index, embedder);
    const { lowestCorrect, highestWrong } = report.boundary;

    expect(lowestCorrect).toHaveLength(
      Math.min(BOUNDARY_LOWEST_CORRECT_COUNT, report.correct.count),
    );
    expect(highestWrong).toHaveLength(
      Math.min(BOUNDARY_HIGHEST_WRONG_COUNT, report.wrong.count),
    );

    // The boundary rows are the named ends of the two distributions: the
    // worst correct row carries the correct min, the best wrong row the
    // wrong max, and every row keeps its ground-truth side.
    expect(must(lowestCorrect[0], "the worst correct sample").score).toBe(report.correct.min);
    expect(must(highestWrong[0], "the best wrong sample").score).toBe(report.wrong.max);
    expect(lowestCorrect.every((sample) => sample.correct)).toBe(true);
    expect(highestWrong.every((sample) => !sample.correct)).toBe(true);
  });

  it("prints the boundary cases in the report, score first, ids after", async () => {
    const plan = idealPlan(questions);
    const embedder = createOracleEmbedder(chunks, questions, plan);
    const index = await buildIndex(chunks, embedder);

    const report = await calibrateThreshold(questions, index, embedder);
    const printed = formatCalibrationReport(report);

    expect(printed).toContain("LOWEST-SCORING CORRECT RETRIEVALS");
    expect(printed).toContain("HIGHEST-SCORING WRONG RETRIEVALS");

    // Each named row appears in the near-miss shape: aligned score to four
    // places, then the ids — so the section reads like the rest of the
    // project's output.
    const worst = must(report.boundary.lowestCorrect[0], "the worst correct sample");
    const line = printed
      .split("\n")
      .find((candidate) => candidate.includes(worst.chunkId) && candidate.includes(worst.questionId));
    expect(must(line, "the printed boundary row")).toContain(worst.score.toFixed(4));
    const scoreAt = must(line, "the printed boundary row").indexOf(worst.score.toFixed(4));
    expect(scoreAt).toBeLessThan(must(line, "the printed boundary row").indexOf(worst.questionId));
  });

  it("names each must-refuse question's best chunk and the ceiling the floor must exceed", async () => {
    const plan = idealPlan(questions);
    const embedder = createOracleEmbedder(chunks, questions, plan);
    const index = await buildIndex(chunks, embedder);

    const report = await calibrateThreshold(questions, index, embedder);
    const mustRefuseQuestions = questions.filter(
      (question) => question.outcome === "refused-no-documents",
    );

    // One row per must-refuse question, every row on the wrong side — a
    // no-answer question has no correct chunk, by construction.
    expect(mustRefuseQuestions.length).toBeGreaterThan(0);
    expect(new Set(report.mustRefuse.rows.map((row) => row.questionId))).toEqual(
      new Set(mustRefuseQuestions.map((question) => question.id)),
    );
    expect(report.mustRefuse.rows).toHaveLength(mustRefuseQuestions.length);
    expect(
      report.mustRefuse.rows.every(
        (row) => row.outcome === "refused-no-documents" && !row.correct,
      ),
    ).toBe(true);

    // The ceiling is the maximum across the rows, the separation carries it,
    // and — must-refuse wrongs being a subset of all wrongs — it can never
    // exceed the pooled best wrong score.
    const ceiling = must(report.mustRefuse.highest, "the must-refuse ceiling");
    expect(ceiling).toBe(Math.max(...report.mustRefuse.rows.map((row) => row.score)));
    expect(report.separation.highestMustRefuseWrong).toBe(ceiling);
    expect(ceiling).toBeLessThanOrEqual(
      must(report.separation.highestWrong, "the best wrong score"),
    );

    // The oracle separates by construction, so the soundness gap is clean and
    // the suggested floor sits strictly inside it: above the ceiling, below
    // the worst correct retrieval.
    expect(report.separation.mustRefuseSeparable).toBe(true);
    const suggested = must(
      report.separation.suggestedMustRefuseFloor,
      "a suggested soundness floor",
    );
    expect(suggested).toBeGreaterThan(ceiling);
    expect(suggested).toBeLessThan(
      must(report.separation.lowestCorrect, "the worst correct score"),
    );
  });

  it("prints the must-refuse section and a verdict that separates soundness from precision", async () => {
    const plan = idealPlan(questions);
    const embedder = createOracleEmbedder(chunks, questions, plan);
    const index = await buildIndex(chunks, embedder);

    const report = await calibrateThreshold(questions, index, embedder);
    const printed = formatCalibrationReport(report);

    expect(printed).toContain("BEST-SCORING CHUNK PER MUST-REFUSE QUESTION");
    expect(printed).toContain("must-refuse ceiling");
    expect(printed).toContain("the number the floor must exceed");
    expect(printed).toContain("best must-refuse chunk");
    expect(printed).toContain("suggested soundness floor");

    // Every must-refuse question is named in the report, with its best chunk
    // and the score to four places on the same line.
    for (const row of report.mustRefuse.rows) {
      const line = printed
        .split("\n")
        .find(
          (candidate) => candidate.includes(row.questionId) && candidate.includes(row.chunkId),
        );
      expect(must(line, `the printed must-refuse row for ${row.questionId}`)).toContain(
        row.score.toFixed(4),
      );
    }

    // The verdict reports both splits and says which failure each bears on:
    // the must-refuse separation is the soundness constraint, the naive one
    // the precision cost.
    expect(report.separation.verdict).toContain("must-refuse split");
    expect(report.separation.verdict).toContain("soundness");
  });
});

/* ===========================================================================
 * Boundary-case selection, directly.
 * ======================================================================== */

describe("selectBoundaryCases", () => {
  function sample(
    questionId: string,
    chunkId: string,
    score: number,
    correct: boolean,
  ): CalibrationSample {
    return {
      questionId,
      outcome: correct ? "answered" : "refused-no-documents",
      chunkId,
      score,
      correct,
    };
  }

  it("takes the bottom correct ascending and the top wrong descending", () => {
    const boundary = selectBoundaryCases([
      sample("q-a", "doc#one", 0.77, true),
      sample("q-b", "doc#two", 0.51, true),
      sample("q-c", "doc#three", 0.61, true),
      sample("q-d", "doc#four", 0.65, true),
      sample("q-e", "doc#five", 0.58, false),
      sample("q-f", "doc#six", 0.2, false),
      sample("q-g", "doc#seven", 0.39, false),
      sample("q-h", "doc#eight", 0.01, false),
      sample("q-i", "doc#nine", 0.44, false),
      sample("q-j", "doc#ten", 0.03, false),
    ]);

    expect(boundary.lowestCorrect.map((entry) => entry.chunkId)).toEqual([
      "doc#two",
      "doc#three",
      "doc#four",
    ]);
    expect(boundary.highestWrong.map((entry) => entry.chunkId)).toEqual([
      "doc#five",
      "doc#nine",
      "doc#seven",
      "doc#six",
      "doc#ten",
    ]);
  });

  it("breaks score ties by question id then chunk id, so two runs name the same rows", () => {
    // Every score identical: the order is decided entirely by the tie-break.
    const tied = [
      sample("q-b", "doc#b", 0.5, true),
      sample("q-a", "doc#z", 0.5, true),
      sample("q-a", "doc#a", 0.5, true),
      sample("q-c", "doc#c", 0.5, true),
      sample("q-b", "doc#w", 0.3, false),
      sample("q-a", "doc#w", 0.3, false),
    ];

    const boundary = selectBoundaryCases(tied);
    expect(
      boundary.lowestCorrect.map((entry) => `${entry.questionId} ${entry.chunkId}`),
    ).toEqual(["q-a doc#a", "q-a doc#z", "q-b doc#b"]);
    expect(
      boundary.highestWrong.map((entry) => `${entry.questionId} ${entry.chunkId}`),
    ).toEqual(["q-a doc#w", "q-b doc#w"]);

    // Selection reads the sample list; it does not reorder it.
    expect(tied.map((entry) => entry.chunkId)).toEqual([
      "doc#b",
      "doc#z",
      "doc#a",
      "doc#c",
      "doc#w",
      "doc#w",
    ]);
  });

  it("returns what exists when a side has fewer samples than asked for", () => {
    const boundary = selectBoundaryCases([
      sample("q-a", "doc#one", 0.7, true),
      sample("q-b", "doc#two", 0.6, true),
      sample("q-c", "doc#three", 0.1, false),
    ]);
    expect(boundary.lowestCorrect).toHaveLength(2);
    expect(boundary.highestWrong).toHaveLength(1);

    const empty = selectBoundaryCases([]);
    expect(empty.lowestCorrect).toEqual([]);
    expect(empty.highestWrong).toEqual([]);
  });
});

/* ===========================================================================
 * Must-refuse ceiling selection, directly.
 * ======================================================================== */

describe("selectMustRefuseCeiling", () => {
  function sample(
    questionId: string,
    chunkId: string,
    score: number,
    outcome: Outcome,
  ): CalibrationSample {
    return { questionId, outcome, chunkId, score, correct: false };
  }

  it("keeps one best chunk per must-refuse question and ignores other questions' wrong chunks", () => {
    const ceiling = selectMustRefuseCeiling([
      sample("q-refuse-a", "doc#low", 0.2, "refused-no-documents"),
      sample("q-refuse-a", "doc#high", 0.45, "refused-no-documents"),
      sample("q-refuse-b", "doc#mid", 0.3, "refused-no-documents"),
      // The best wrong chunks overall belong to answerable questions. They
      // cost precision, not soundness, and must not set the ceiling.
      sample("q-answered", "doc#huge", 0.9, "answered"),
      sample("q-contradiction", "doc#big", 0.8, "refused-contradiction"),
    ]);

    expect(ceiling.rows.map((row) => `${row.questionId} ${row.chunkId}`)).toEqual([
      "q-refuse-a doc#high",
      "q-refuse-b doc#mid",
    ]);
    expect(ceiling.highest).toBe(0.45);
  });

  it("breaks score ties by question id then chunk id, so two runs name the same rows", () => {
    const ceiling = selectMustRefuseCeiling([
      sample("q-a", "doc#z", 0.5, "refused-no-documents"),
      sample("q-a", "doc#a", 0.5, "refused-no-documents"),
      sample("q-b", "doc#b", 0.5, "refused-no-documents"),
    ]);

    expect(ceiling.rows.map((row) => `${row.questionId} ${row.chunkId}`)).toEqual([
      "q-a doc#a",
      "q-b doc#b",
    ]);
  });

  it("reports no ceiling when the battery has no must-refuse questions", () => {
    const ceiling = selectMustRefuseCeiling([
      sample("q-answered", "doc#one", 0.9, "answered"),
    ]);
    expect(ceiling.rows).toEqual([]);
    expect(ceiling.highest).toBeNull();
  });
});

/* ===========================================================================
 * The calibration artifact.
 * ======================================================================== */

describe("the calibration artifact", () => {
  const metadata: CalibrationRunMetadata = {
    generatedAt: "2026-08-27T11:00:00.000Z",
    embedder: "fake-oracle",
    corpusDir: "corpus",
    chunkCount: 71,
    questionCount: 14,
  };

  async function calibrated() {
    const plan = idealPlan(questions);
    const embedder = createOracleEmbedder(chunks, questions, plan);
    const index = await buildIndex(chunks, embedder);
    return calibrateThreshold(questions, index, embedder);
  }

  it("is self-contained: statistics, boundary cases, model, prices, and the unchanged constant", async () => {
    const report = await calibrated();
    const usage = { totalTokens: 123_456, calls: 15, totalLatencyMs: 890 };
    const artifact = buildCalibrationArtifact(report, metadata, usage);

    expect(artifact.schemaVersion).toBe(CALIBRATION_ARTIFACT_SCHEMA_VERSION);
    expect(artifact.run).toEqual(metadata);
    expect(artifact.correct).toEqual(report.correct);
    expect(artifact.wrong).toEqual(report.wrong);
    expect(artifact.separation).toEqual(report.separation);
    expect(artifact.boundary).toEqual(report.boundary);
    expect(artifact.mustRefuse).toEqual(report.mustRefuse);

    // The constant is recorded, not moved, and the file says so on its own.
    expect(artifact.currentThreshold).toBe(MIN_SIMILARITY);
    expect(artifact.thresholdNote).toContain("does not change MIN_SIMILARITY");
    expect(artifact.thresholdNote).toContain("reviewed commit");

    // The usage block: the provider's numbers at the committed price, with
    // the priced model and its as-of date beside the ran model in `run`.
    const billed = must(artifact.usage, "the usage block");
    expect(billed.embeddingTokens).toBe(123_456);
    expect(billed.embeddingCalls).toBe(15);
    expect(billed.latencyMs).toBe(890);
    expect(billed.costUsd).toBe(computeEmbeddingCost(123_456));
    expect(billed.pricing.asOf).toBe(EMBEDDING_PRICE_AS_OF);
  });

  it("records null usage for an uninstrumented run, not a fabricated zero", async () => {
    const artifact = buildCalibrationArtifact(await calibrated(), metadata);
    expect(artifact.usage).toBeNull();
  });

  it("carries no credential-shaped field", async () => {
    const report = await calibrated();
    const artifact = buildCalibrationArtifact(report, metadata, {
      totalTokens: 1,
      calls: 1,
      totalLatencyMs: 1,
    });
    const serialised = JSON.stringify(artifact);
    expect(serialised).toContain("fake-oracle");
    expect(serialised).toContain(EMBEDDING_PRICE_AS_OF);
    expect(serialised).not.toContain("apiKey");
    expect(serialised).not.toContain("API_KEY");
    expect(serialised).not.toContain(DECOY_KEY);
  });

  it("writes a timestamped copy and a latest copy, both self-contained", async () => {
    const directory = await mkdtemp(path.join(tmpdir(), "rag-receipts-calibration-"));
    try {
      const artifact = buildCalibrationArtifact(await calibrated(), metadata);
      const written = await writeCalibrationArtifact(directory, artifact);

      expect(written).toHaveLength(2);
      expect(path.basename(must(written[0], "the stamped path"))).toMatch(/^calibration-2026/);
      expect(must(written[1], "the latest path").endsWith("calibration-latest.json")).toBe(true);

      for (const file of written) {
        const parsed: unknown = JSON.parse(await readFile(file, "utf8"));
        const record = parsed as { currentThreshold: number; thresholdNote: string };
        // Read alone, months later, the file still says what it measured and
        // what it deliberately did not touch.
        expect(record.currentThreshold).toBe(MIN_SIMILARITY);
        expect(record.thresholdNote).toContain("does not change MIN_SIMILARITY");
      }
    } finally {
      await rm(directory, { recursive: true, force: true });
    }
  });
});

/* ===========================================================================
 * The CLI: environment handling.
 * ======================================================================== */

describe("environment helpers", () => {
  it("parses a .env file, including quotes, comments, and an export prefix", () => {
    const parsed = parseDotEnv(
      [
        "# a comment",
        "",
        "VOYAGE_API_KEY=plain-value",
        `ANTHROPIC_API_KEY="${DECOY_KEY}"`,
        "export VOYAGE_MODEL='voyage-3'",
        "NOT_A_PAIR",
      ].join("\n"),
    );

    expect(parsed["VOYAGE_API_KEY"]).toBe("plain-value");
    expect(parsed["ANTHROPIC_API_KEY"]).toBe(DECOY_KEY);
    expect(parsed["VOYAGE_MODEL"]).toBe("voyage-3");
    expect(parsed["NOT_A_PAIR"]).toBeUndefined();
  });

  it("lets the process environment win, but not with an empty value", () => {
    const dotEnv: Environment = { KEY: "from-dot-env", OTHER: "kept" };

    expect(mergeEnvironment({ KEY: "from-process" }, dotEnv)["KEY"]).toBe("from-process");
    // An unset CI secret expands to an empty string. It must not shadow a real
    // value, or `KEY=""` and no `KEY` at all behave differently for no reason.
    expect(mergeEnvironment({ KEY: "" }, dotEnv)["KEY"]).toBe("from-dot-env");
    expect(mergeEnvironment({ KEY: "   " }, dotEnv)["KEY"]).toBe("from-dot-env");
    expect(mergeEnvironment({}, dotEnv)["OTHER"]).toBe("kept");
  });

  it("distinguishes absent from present-but-blank, and reports both", () => {
    const absent = resolveKey("VOYAGE_API_KEY", {});
    expect(absent.ok).toBe(false);
    if (!absent.ok) expect(absent.message).toContain("is not set");

    const blank = resolveKey("VOYAGE_API_KEY", { VOYAGE_API_KEY: "  " });
    expect(blank.ok).toBe(false);
    if (!blank.ok) expect(blank.message).toContain("set but empty");

    const present = resolveKey("VOYAGE_API_KEY", { VOYAGE_API_KEY: ` ${DECOY_KEY} ` });
    expect(present.ok).toBe(true);
    if (present.ok) expect(present.value).toBe(DECOY_KEY);
  });

  it("reports every missing key at once, not the first one", () => {
    const resolved = resolveKeys(["VOYAGE_API_KEY", "ANTHROPIC_API_KEY"], {});
    expect(resolved.ok).toBe(false);
    if (!resolved.ok) {
      expect(resolved.messages).toHaveLength(2);
      expect(resolved.messages.join(" ")).toContain("VOYAGE_API_KEY");
      expect(resolved.messages.join(" ")).toContain("ANTHROPIC_API_KEY");
    }
  });

  it("falls back to committed model ids and honours an override", () => {
    expect(resolveModel("VOYAGE_MODEL", {}, "voyage-3")).toBe("voyage-3");
    expect(resolveModel("VOYAGE_MODEL", { VOYAGE_MODEL: "  " }, "voyage-3")).toBe("voyage-3");
    expect(resolveModel("VOYAGE_MODEL", { VOYAGE_MODEL: " voyage-9 " }, "voyage-3")).toBe("voyage-9");
  });

  it("parses arguments and keeps unknown ones rather than ignoring them", () => {
    expect(parseArgs([]).calibrate).toBe(false);
    expect(parseArgs(["--calibrate"]).calibrate).toBe(true);
    expect(parseArgs(["-h"]).help).toBe(true);
    expect(parseArgs(["--calibrat"]).unknown).toEqual(["--calibrat"]);
  });
});

/* ===========================================================================
 * The CLI: exit codes, and the key that must never appear.
 * ======================================================================== */

describe("the CLI", () => {
  interface Captured extends ConsoleLike {
    readonly lines: readonly string[];
    text(): string;
  }

  function capture(): Captured {
    const lines: string[] = [];
    return {
      lines,
      log: (message: string) => lines.push(message),
      error: (message: string) => lines.push(message),
      text: () => lines.join("\n"),
    };
  }

  /**
   * A .env path that does not exist.
   *
   * LOAD-BEARING. Pointing the CLI at the repository's real `.env` would let a
   * developer's own keys into these tests, and a test that passes because
   * someone has credentials on disk is a test that fails in CI for a reason
   * nobody can reproduce locally. Worse, it would reach the network.
   */
  function paths(): Parameters<typeof main>[0]["paths"] {
    return {
      questionSet: QUESTION_SET,
      corpusDir: CORPUS_DIR,
      dotEnv: path.join(tmpdir(), "rag-receipts-no-such-dot-env-file"),
      resultsDir: path.join(tmpdir(), "rag-receipts-unused-results"),
    };
  }

  async function run(argv: readonly string[], env: Environment): Promise<{ code: number; out: Captured }> {
    const out = capture();
    const code = await main({ argv, env, paths: paths(), out, now: "2026-08-22T10:12:00.000Z" });
    return { code, out };
  }

  it("exits 2 when a required key is missing, and names the variable", async () => {
    const { code, out } = await run([], {});
    expect(code).toBe(EXIT_SETUP_ERROR);
    expect(code).not.toBe(EXIT_GATE_FAILED);
    expect(out.text()).toContain("VOYAGE_API_KEY");
    expect(out.text()).toContain("ANTHROPIC_API_KEY");
  });

  it("never echoes a key value into any error, however the key is supplied", async () => {
    // One key present and one absent, so the failure path runs with a real
    // credential in scope — the only configuration where a leak could happen.
    const cases: readonly Environment[] = [
      { VOYAGE_API_KEY: DECOY_KEY },
      { ANTHROPIC_API_KEY: DECOY_KEY },
      { VOYAGE_API_KEY: DECOY_KEY, ANTHROPIC_API_KEY: "" },
      { VOYAGE_API_KEY: DECOY_KEY, ANTHROPIC_MODEL: "claude-opus-5" },
    ];

    for (const env of cases) {
      const { code, out } = await run([], env);
      expect(code).toBe(EXIT_SETUP_ERROR);
      expect(out.text()).not.toContain(DECOY_KEY);
      // Not even a fragment of it.
      expect(out.text()).not.toContain("DECOY");
    }
  });

  it("never echoes a key that came from a .env file", async () => {
    const directory = await mkdtemp(path.join(tmpdir(), "rag-receipts-env-"));
    try {
      const dotEnvPath = path.join(directory, ".env");
      const { writeFile } = await import("node:fs/promises");
      await writeFile(dotEnvPath, `VOYAGE_API_KEY=${DECOY_KEY}\n`, "utf8");

      const out = capture();
      const code = await main({
        argv: [],
        env: {},
        paths: { ...paths(), dotEnv: dotEnvPath },
        out,
        now: "2026-08-22T10:12:00.000Z",
      });

      // The Voyage key resolved from the file; the Anthropic one is missing, so
      // the run stops before any network call — with the file's key in scope.
      expect(code).toBe(EXIT_SETUP_ERROR);
      expect(out.text()).toContain("ANTHROPIC_API_KEY");
      expect(out.text()).not.toContain(DECOY_KEY);
    } finally {
      await rm(directory, { recursive: true, force: true });
    }
  });

  it("asks only for the embedding key when calibrating, and still exits 2 without it", async () => {
    const { code, out } = await run(["--calibrate"], {});
    expect(code).toBe(EXIT_SETUP_ERROR);
    expect(out.text()).toContain("VOYAGE_API_KEY");
    // Calibration generates nothing, so it never asks for a generation key.
    expect(out.text()).not.toContain("ANTHROPIC_API_KEY");
  });

  it("exits 2 on an unrecognised flag rather than silently running the gate", async () => {
    const { code, out } = await run(["--calibrat"], { VOYAGE_API_KEY: "k", ANTHROPIC_API_KEY: "k" });
    expect(code).toBe(EXIT_SETUP_ERROR);
    expect(out.text()).toContain("Unrecognised");
  });

  it("exits 0 for --help without touching the environment", async () => {
    const { code, out } = await run(["--help"], {});
    expect(code).toBe(EXIT_PASS);
    expect(out.text()).toContain("Usage:");
    expect(out.text()).toContain("--calibrate");
  });

  it("keeps the three exit codes distinct", () => {
    expect(new Set([EXIT_PASS, EXIT_GATE_FAILED, EXIT_SETUP_ERROR]).size).toBe(3);
    expect(EXIT_PASS).toBe(0);
    expect(EXIT_GATE_FAILED).toBe(1);
    expect(EXIT_SETUP_ERROR).toBe(2);
  });
});
