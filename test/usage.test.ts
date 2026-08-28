/**
 * usage.test.ts — token, latency, and cost instrumentation, tested offline.
 *
 * NO NETWORK, NO KEYS, NO REAL CLOCK. The live embedder and generator are
 * exercised against injected transports, latency is measured against an
 * injected clock that a test advances by hand, and the eval-level accounting
 * runs the real corpus through the fakes. Exact-equality assertions
 * throughout: a latency assertion with slack in it is an assertion that never
 * fails.
 */

import { fileURLToPath } from "node:url";
import Anthropic from "@anthropic-ai/sdk";
import { beforeAll, describe, expect, it } from "vitest";

import {
  type GenerationUsage,
  type InstrumentedGenerator,
  GenerationError,
  anthropicGenerator,
} from "../src/answer.js";
import { type Chunk, loadCorpus } from "../src/chunk.js";
import {
  EMBEDDING_PRICE_AS_OF,
  GENERATION_PRICE_AS_OF,
  computeEmbeddingCost,
  computeGenerationCost,
} from "../src/cost.js";
import {
  type EvalQuestion,
  type RunMetadata,
  applyGate,
  buildArtifact,
  calibrateThreshold,
  formatCalibrationReport,
  formatGateReport,
  loadQuestionSet,
  runEvaluation,
  runInstrumentedEvaluation,
} from "../src/eval.js";
import {
  type EmbeddingUsage,
  type InstrumentedEmbedder,
  MIN_SIMILARITY,
  buildIndex,
  voyageEmbedder,
} from "../src/retrieve.js";
import { createFakeEmbedder } from "./fake-embedder.js";

const REPO_ROOT = new URL("../", import.meta.url);
const CORPUS_DIR = fileURLToPath(new URL("corpus", REPO_ROOT));
const QUESTION_SET = fileURLToPath(new URL("eval/questions.json", REPO_ROOT));

/** Matches the convention in the other test files. */
const DECOY_KEY = "pa-DECOY-KEY-9f3c1e7a5b2d-do-not-log";

let chunks: readonly Chunk[];
let questions: readonly EvalQuestion[];

beforeAll(async () => {
  chunks = await loadCorpus(CORPUS_DIR);
  questions = (await loadQuestionSet(QUESTION_SET)).questions;
});

/* ===========================================================================
 * Hand-cranked clocks and transports.
 * ======================================================================== */

/** A clock a test advances by hand, so latency is asserted exactly. */
function manualClock(): { clock: () => number; advance: (ms: number) => void } {
  let now = 1_000;
  return { clock: () => now, advance: (ms: number) => (now += ms) };
}

function jsonResponse(body: unknown, status = 200): Response {
  return new Response(JSON.stringify(body), { status });
}

/** A Voyage-shaped payload: one 2-d vector per input, plus reported usage. */
function voyagePayload(count: number, totalTokens: number | undefined): unknown {
  return {
    data: Array.from({ length: count }, (_, index) => ({ index, embedding: [1, 0] })),
    ...(totalTokens === undefined ? {} : { usage: { total_tokens: totalTokens } }),
  };
}

/** An SDK-shaped message with just enough on it for `anthropicGenerator`. */
function sdkMessage(
  usage: { input_tokens?: number; output_tokens?: number },
  stopReason: Anthropic.StopReason = "end_turn",
): Anthropic.Message {
  return {
    id: "msg_test",
    type: "message",
    role: "assistant",
    model: "claude-opus-5",
    content: [{ type: "text", text: '{"claims":[]}' }],
    stop_reason: stopReason,
    stop_sequence: null,
    stop_details: null,
    usage,
  } as unknown as Anthropic.Message;
}

/** A stand-in SDK client that returns canned messages and burns fake time. */
function stubClient(
  respond: (call: number) => Anthropic.Message,
  onCall?: () => void,
): Anthropic {
  let call = 0;
  return {
    messages: {
      create: async () => {
        onCall?.();
        call += 1;
        return respond(call - 1);
      },
    },
  } as unknown as Anthropic;
}

/* ===========================================================================
 * The live embedder's accounting.
 * ======================================================================== */

describe("voyageEmbedder usage", () => {
  it("accumulates total_tokens and wall-clock latency across calls", async () => {
    const { clock, advance } = manualClock();
    const fetchImpl: typeof globalThis.fetch = async () => {
      advance(7);
      return jsonResponse(voyagePayload(1, 42));
    };
    const embedder = voyageEmbedder({ apiKey: DECOY_KEY, model: "voyage-3", fetchImpl, clock });

    expect(embedder.usage()).toEqual({ totalTokens: 0, calls: 0, totalLatencyMs: 0 });

    await embedder.embed(["one"], "query");
    expect(embedder.usage()).toEqual({ totalTokens: 42, calls: 1, totalLatencyMs: 7 });

    await embedder.embed(["two"], "document");
    expect(embedder.usage()).toEqual({ totalTokens: 84, calls: 2, totalLatencyMs: 14 });
  });

  it("counts a call whose response omits usage, at zero tokens", async () => {
    // Tolerant on purpose: a missing accounting field must not fail the run
    // it is accounting for. The undercount errs small rather than inventing.
    const { clock, advance } = manualClock();
    const fetchImpl: typeof globalThis.fetch = async () => {
      advance(3);
      return jsonResponse(voyagePayload(1, undefined));
    };
    const embedder = voyageEmbedder({ apiKey: DECOY_KEY, model: "voyage-3", fetchImpl, clock });

    await embedder.embed(["text"], "query");
    expect(embedder.usage()).toEqual({ totalTokens: 0, calls: 1, totalLatencyMs: 3 });
  });

  it("counts nothing for an empty input, which makes no request", async () => {
    const { clock } = manualClock();
    const fetchImpl: typeof globalThis.fetch = async () => jsonResponse(voyagePayload(0, 5));
    const embedder = voyageEmbedder({ apiKey: DECOY_KEY, model: "voyage-3", fetchImpl, clock });

    expect(await embedder.embed([], "document")).toEqual([]);
    expect(embedder.usage()).toEqual({ totalTokens: 0, calls: 0, totalLatencyMs: 0 });
  });

  it("counts nothing for a call that failed", async () => {
    const { clock, advance } = manualClock();
    const fetchImpl: typeof globalThis.fetch = async () => {
      advance(9);
      return jsonResponse({ detail: "nope" }, 500);
    };
    const embedder = voyageEmbedder({ apiKey: DECOY_KEY, model: "voyage-3", fetchImpl, clock });

    await expect(embedder.embed(["text"], "query")).rejects.toThrow();
    expect(embedder.usage()).toEqual({ totalTokens: 0, calls: 0, totalLatencyMs: 0 });
  });
});

/* ===========================================================================
 * The live generator's accounting.
 * ======================================================================== */

describe("anthropicGenerator usage", () => {
  it("accumulates input tokens, output tokens, and latency per call", async () => {
    const { clock, advance } = manualClock();
    const client = stubClient(
      () => sdkMessage({ input_tokens: 123, output_tokens: 45 }),
      () => advance(11),
    );
    const generator = anthropicGenerator({ apiKey: DECOY_KEY, model: "claude-opus-5", client, clock });

    expect(generator.usage()).toEqual({ inputTokens: 0, outputTokens: 0, calls: 0, totalLatencyMs: 0 });

    await generator.generate("system", "user");
    expect(generator.usage()).toEqual({ inputTokens: 123, outputTokens: 45, calls: 1, totalLatencyMs: 11 });

    await generator.generate("system", "user");
    expect(generator.usage()).toEqual({ inputTokens: 246, outputTokens: 90, calls: 2, totalLatencyMs: 22 });
  });

  it("tolerates a response with no usage fields, counting zero tokens", async () => {
    const { clock, advance } = manualClock();
    const client = stubClient(
      () => sdkMessage({}),
      () => advance(4),
    );
    const generator = anthropicGenerator({ apiKey: DECOY_KEY, model: "claude-opus-5", client, clock });

    await generator.generate("system", "user");
    expect(generator.usage()).toEqual({ inputTokens: 0, outputTokens: 0, calls: 1, totalLatencyMs: 4 });
  });

  it("counts a model refusal, because the provider billed it", async () => {
    const { clock, advance } = manualClock();
    const client = stubClient(
      () => sdkMessage({ input_tokens: 7, output_tokens: 3 }, "refusal"),
      () => advance(6),
    );
    const generator = anthropicGenerator({ apiKey: DECOY_KEY, model: "claude-opus-5", client, clock });

    await expect(generator.generate("system", "user")).rejects.toThrow(GenerationError);
    // The call threw downstream of the response, but the response arrived and
    // was metered; reporting less than the invoice would be the quiet lie.
    expect(generator.usage()).toEqual({ inputTokens: 7, outputTokens: 3, calls: 1, totalLatencyMs: 6 });
  });
});

/* ===========================================================================
 * The fakes report zeros, in the live shape.
 * ======================================================================== */

/** An instrumented generator that refuses everything and costs nothing. */
function zeroUsageGenerator(): InstrumentedGenerator {
  return {
    async generate(): Promise<string> {
      return JSON.stringify({ claims: [] });
    },
    usage(): GenerationUsage {
      return { inputTokens: 0, outputTokens: 0, calls: 0, totalLatencyMs: 0 };
    },
  };
}

describe("offline fakes", () => {
  it("the fake embedder reports zero usage in the live embedder's shape", async () => {
    const embedder = createFakeEmbedder();
    await embedder.embed(["some text"], "document");
    // Zeros even after a call: no provider was billed and no network waited on.
    expect(embedder.usage()).toEqual({ totalTokens: 0, calls: 0, totalLatencyMs: 0 });
  });
});

/* ===========================================================================
 * The instrumented eval run.
 * ======================================================================== */

describe("runInstrumentedEvaluation", () => {
  it("scores identically to runEvaluation — the two loops cannot drift", async () => {
    const embedder = createFakeEmbedder();
    const index = await buildIndex(chunks, embedder);
    const generator = zeroUsageGenerator();

    const plain = await runEvaluation(questions, { index, embedder, generator });
    const instrumented = await runInstrumentedEvaluation(questions, { index, embedder, generator });

    expect(JSON.stringify(instrumented.scores)).toBe(JSON.stringify(plain));
  });

  it("produces a well-formed, all-zero usage section from the offline fakes", async () => {
    const embedder = createFakeEmbedder();
    const index = await buildIndex(chunks, embedder);

    const { usage } = await runInstrumentedEvaluation(questions, {
      index,
      embedder,
      generator: zeroUsageGenerator(),
    });

    expect(usage.perQuestion).toHaveLength(questions.length);
    expect(usage.perQuestion.map((row) => row.questionId)).toEqual(
      questions.map((question) => question.id),
    );
    for (const row of [usage.indexing, ...usage.perQuestion, usage.totals]) {
      expect(row.embeddingTokens).toBe(0);
      expect(row.generationInputTokens).toBe(0);
      expect(row.generationOutputTokens).toBe(0);
      expect(row.latencyMs).toBe(0);
      expect(row.costUsd).toBe(0);
    }
    // The prices travel with the run even when everything cost nothing.
    expect(usage.pricing.embedding.asOf).toBe(EMBEDDING_PRICE_AS_OF);
    expect(usage.pricing.generation.asOf).toBe(GENERATION_PRICE_AS_OF);
  });

  it("attributes usage per question by delta, with the index build separate", async () => {
    // A counting embedder: real fake vectors underneath, but every call adds
    // fixed usage, so the expected deltas can be computed on paper.
    function countingEmbedder(): InstrumentedEmbedder {
      const inner = createFakeEmbedder();
      let totalTokens = 0;
      let calls = 0;
      let totalLatencyMs = 0;
      return {
        async embed(texts, kind) {
          totalTokens += 11;
          calls += 1;
          totalLatencyMs += 3;
          return inner.embed(texts, kind);
        },
        usage(): EmbeddingUsage {
          return { totalTokens, calls, totalLatencyMs };
        },
      };
    }

    function countingGenerator(): InstrumentedGenerator {
      let inputTokens = 0;
      let outputTokens = 0;
      let calls = 0;
      let totalLatencyMs = 0;
      return {
        async generate(): Promise<string> {
          inputTokens += 100;
          outputTokens += 10;
          calls += 1;
          totalLatencyMs += 5;
          return JSON.stringify({ claims: [] });
        },
        usage(): GenerationUsage {
          return { inputTokens, outputTokens, calls, totalLatencyMs };
        },
      };
    }

    const embedder = countingEmbedder();
    const index = await buildIndex(chunks, embedder); // one call: 11 tokens, 3 ms
    const two = questions.slice(0, 2);

    // threshold -1: everything clears the floor, so the generator runs for
    // both questions and the generation columns are exercised, not just zero.
    const { usage } = await runInstrumentedEvaluation(two, {
      index,
      embedder,
      generator: countingGenerator(),
      threshold: -1,
    });

    expect(usage.indexing).toEqual({
      embeddingTokens: 11,
      generationInputTokens: 0,
      generationOutputTokens: 0,
      latencyMs: 3,
      costUsd: computeEmbeddingCost(11),
    });

    expect(usage.perQuestion).toHaveLength(2);
    for (const row of usage.perQuestion) {
      // Per question: one query embedding, one generation call.
      expect(row.embeddingTokens).toBe(11);
      expect(row.generationInputTokens).toBe(100);
      expect(row.generationOutputTokens).toBe(10);
      expect(row.latencyMs).toBe(3 + 5);
      expect(row.costUsd).toBeCloseTo(
        computeEmbeddingCost(11) + computeGenerationCost(100, 10),
        12,
      );
    }

    // Totals are the index build plus both rows — nothing dropped, nothing
    // double-counted.
    expect(usage.totals.embeddingTokens).toBe(33);
    expect(usage.totals.generationInputTokens).toBe(200);
    expect(usage.totals.generationOutputTokens).toBe(20);
    expect(usage.totals.latencyMs).toBe(3 + 2 * 8);
    expect(usage.totals.costUsd).toBeCloseTo(
      computeEmbeddingCost(33) + computeGenerationCost(200, 20),
      12,
    );
  });
});

/* ===========================================================================
 * The artifact and the printed reports.
 * ======================================================================== */

describe("usage in the artifact and the reports", () => {
  const metadata: RunMetadata = {
    generatedAt: "2026-08-27T10:12:00.000Z",
    embedder: "fake-embedder",
    generator: "fake-generator",
    corpusDir: "corpus",
    chunkCount: 71,
    k: 6,
    threshold: MIN_SIMILARITY,
  };

  async function zeroRun() {
    const embedder = createFakeEmbedder();
    const index = await buildIndex(chunks, embedder);
    return runInstrumentedEvaluation(questions, {
      index,
      embedder,
      generator: zeroUsageGenerator(),
    });
  }

  it("stores the pricing constants and their as-of dates in the artifact", async () => {
    const { scores, usage } = await zeroRun();
    const artifact = buildArtifact(applyGate(scores), metadata, usage);

    expect(artifact.usage).not.toBeNull();
    expect(artifact.usage?.pricing.embedding.asOf).toBe(EMBEDDING_PRICE_AS_OF);
    expect(artifact.usage?.pricing.generation.asOf).toBe(GENERATION_PRICE_AS_OF);

    const serialised = JSON.stringify(artifact);
    expect(serialised).toContain(EMBEDDING_PRICE_AS_OF);
    expect(serialised).toContain(GENERATION_PRICE_AS_OF);
    // Usage numbers are not credentials — but re-check the whole artifact for
    // anything key-shaped anyway, same as the metadata test does.
    expect(serialised).not.toContain("apiKey");
    expect(serialised).not.toContain("API_KEY");
    expect(serialised).not.toContain(DECOY_KEY);
  });

  it("records null usage for an uninstrumented artifact, not a fabricated zero", async () => {
    const { scores } = await zeroRun();
    const artifact = buildArtifact(applyGate(scores), metadata);
    // Null reads as "not measured"; a zero would read as "measured, free".
    expect(artifact.usage).toBeNull();
  });

  it("adds the cost table to the gate report only when usage is passed", async () => {
    const { scores, usage } = await zeroRun();
    const report = applyGate(scores);

    const without = formatGateReport(report);
    expect(without).not.toContain("USAGE AND COST");
    expect(without).not.toContain(EMBEDDING_PRICE_AS_OF);

    const withUsage = formatGateReport(report, usage);
    expect(withUsage).toContain("USAGE AND COST");
    expect(withUsage).toContain("(index build)");
    expect(withUsage).toContain("TOTAL");
    // One row per question, addressed by id.
    for (const question of questions) expect(withUsage).toContain(question.id);
    // The as-of dates are printed with the figures, never assumed.
    expect(withUsage).toContain(EMBEDDING_PRICE_AS_OF);
    expect(withUsage).toContain(GENERATION_PRICE_AS_OF);
    // And the cost column does not lie by rounding to nothing-off-six-decimals.
    expect(withUsage).toContain("$0.000000");
    // Everything the uninstrumented report says is still said, before it.
    expect(withUsage).toContain("VERDICT:");
    expect(withUsage.indexOf("USAGE AND COST")).toBeLessThan(withUsage.indexOf("VERDICT:"));
  });

  it("adds embedding usage to the calibration report only when passed", async () => {
    const embedder = createFakeEmbedder();
    const index = await buildIndex(chunks, embedder);
    const report = await calibrateThreshold(questions, index, embedder);

    const without = formatCalibrationReport(report);
    expect(without).not.toContain("EMBEDDING USAGE AND COST");

    const withUsage = formatCalibrationReport(report, embedder.usage());
    expect(withUsage).toContain("EMBEDDING USAGE AND COST");
    expect(withUsage).toContain("embedding tokens");
    expect(withUsage).toContain("latency (ms, wall clock)");
    expect(withUsage).toContain("$0.000000");
    expect(withUsage).toContain(EMBEDDING_PRICE_AS_OF);
  });
});
