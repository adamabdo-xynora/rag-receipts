import { readFileSync } from "node:fs";
import { fileURLToPath } from "node:url";
import { describe, expect, it } from "vitest";

import Anthropic from "@anthropic-ai/sdk";

import { type Chunk, loadCorpus } from "../src/chunk.js";
import {
  MIN_SIMILARITY,
  type SearchResult,
  type VectorIndex,
  buildIndex,
  partitionByThreshold,
  search,
} from "../src/retrieve.js";
import type { Claim } from "../src/verify.js";
import {
  type AnswerResult,
  type AnsweredResult,
  type Generator,
  GenerationError,
  type RefusalReason,
  type RefusedResult,
  DEFAULT_K,
  answerQuestion,
  anthropicGenerator,
  buildSystemPrompt,
  buildUserMessage,
  detectContradiction,
  parseClaims,
  stripCodeFence,
} from "../src/answer.js";
import { createFakeEmbedder } from "./fake-embedder.js";

const CORPUS_DIR = fileURLToPath(new URL("../corpus", import.meta.url));
const ANSWER_SOURCE = fileURLToPath(new URL("../src/answer.ts", import.meta.url));

/* ===========================================================================
 * FAKES EVERYWHERE. NO NETWORK, NO KEY, NO MODEL.
 * ===========================================================================
 * Every test below runs the whole pipeline — the real chunker over the real
 * corpus, the real cosine ranking, the real threshold, the real verifier — and
 * the only two things replaced are the two that would otherwise need an account:
 * the embedder (`test/fake-embedder.ts`, a genuine bag-of-words embedder, so
 * ranking assertions can actually fail) and the generator (below).
 *
 * The generator fake is not a stub that returns "ok". It returns the exact
 * bytes a model would return, including the malformed ones, because the
 * behaviour under test is precisely what this module does with model output it
 * did not get to choose.
 * ======================================================================== */

/** One recorded generation, so tests can assert what the model was shown. */
interface FakeGeneratorCall {
  readonly system: string;
  readonly user: string;
}

interface FakeGenerator extends Generator {
  /** Every call, in order. Empty is an assertable fact — see the floor tests. */
  readonly calls: readonly FakeGeneratorCall[];
}

/** A generator that records what it was asked and answers from a script. */
function fakeGenerator(respond: (user: string) => string): FakeGenerator {
  const calls: FakeGeneratorCall[] = [];
  return {
    calls,
    async generate(system: string, user: string): Promise<string> {
      calls.push({ system, user });
      return respond(user);
    },
  };
}

/** A generator that always returns the same bytes. */
function scripted(response: string): FakeGenerator {
  return fakeGenerator(() => response);
}

/**
 * A generator whose only behaviour is to fail the test if it is ever reached.
 *
 * Asserting `calls.length === 0` afterwards proves the same thing, but this
 * fails at the moment of the mistake and names it, rather than leaving a
 * confusing refusal to be explained by an assertion three lines later.
 */
function forbiddenGenerator(): FakeGenerator {
  return fakeGenerator(() => {
    throw new Error(
      "the generator was called even though nothing cleared the similarity floor",
    );
  });
}

/* ===========================================================================
 * Corpus fixtures. Built once; every quote is sliced from bytes read off disk.
 * ======================================================================== */

interface Fixture {
  readonly chunks: readonly Chunk[];
  readonly index: VectorIndex;
}

let fixture: Promise<Fixture> | null = null;

function corpusIndex(): Promise<Fixture> {
  fixture ??= (async (): Promise<Fixture> => {
    const chunks = await loadCorpus(CORPUS_DIR);
    const index = await buildIndex(chunks, createFakeEmbedder());
    return { chunks, index };
  })();
  return fixture;
}

/** Look a chunk up, failing with a useful message if the corpus moved. */
function chunkById(chunks: readonly Chunk[], chunkId: string): Chunk {
  const chunk = chunks.find((candidate) => candidate.chunkId === chunkId);
  expect(chunk, `the corpus no longer contains "${chunkId}"`).toBeDefined();
  return chunk as Chunk;
}

/**
 * The sentence of `chunk` containing `marker`, sliced out of the chunk's own
 * text — byte-exact by construction rather than by transcription. Same helper
 * as `test/verify.test.ts`, for the same reason: a hand-typed quote tests the
 * typing, and a hand-typed *near* miss would let a containment test pass for
 * the wrong reason.
 */
function sentenceWith(chunk: Chunk, marker: string): string {
  const at = chunk.text.indexOf(marker);
  expect(at, `"${marker}" is no longer in ${chunk.chunkId}`).toBeGreaterThanOrEqual(0);

  const previousStop = chunk.text.lastIndexOf(". ", at);
  const start = previousStop === -1 ? 0 : previousStop + 2;
  const stop = chunk.text.indexOf(".", at);
  const end = stop === -1 ? chunk.text.length : stop + 1;

  const sentence = chunk.text.slice(start, end);
  expect(chunk.text).toContain(sentence);
  return sentence;
}

/** Run the pipeline over the real corpus with a fresh query embedder. */
async function ask(
  question: string,
  generator: Generator,
  overrides: { readonly k?: number; readonly threshold?: number } = {},
): Promise<AnswerResult> {
  const { index } = await corpusIndex();
  return answerQuestion(question, {
    index,
    embedder: createFakeEmbedder(),
    generator,
    ...overrides,
  });
}

/** Exactly what `answerQuestion` will put above the floor for this question. */
async function aboveFloor(question: string, k: number = DEFAULT_K): Promise<SearchResult[]> {
  const { index } = await corpusIndex();
  const results = await search(index, question, createFakeEmbedder(), k);
  return [...partitionByThreshold(results).above];
}

function expectRefused(result: AnswerResult, reason: RefusalReason): RefusedResult {
  expect(result.kind, `expected a refusal for "${reason}"`).toBe("refused");
  const refused = result as RefusedResult;
  expect(refused.reason).toBe(reason);
  // A refusal that cannot be debugged is a shrug with better grammar.
  expect(refused.detail.length).toBeGreaterThan(40);
  return refused;
}

function expectAnswered(result: AnswerResult): AnsweredResult {
  expect(result.kind, "expected an answer").toBe("answered");
  return result as AnsweredResult;
}

/** The wire format the system prompt asks for. */
function claimsJson(claims: readonly unknown[]): string {
  return JSON.stringify({ claims });
}

/* ===========================================================================
 * Named fixtures for the corpus's designed test cases.
 * ======================================================================== */

/** The deliberate contradiction pair. See `docs/CORPUS-DESIGN.md` §1. */
const OLD_WINDOW = "returns-and-credits-policy#return-window-and-condition-requirements";
const NEW_WINDOW = "customer-care-handbook#return-window-for-stocked-goods";
/** Single-source fact. */
const COCOA = "dry-goods-product-specs#cocoa-powder-storage-and-shelf-life";

/** Retrieves both sides of the contradiction pair above the floor. */
const RETURN_WINDOW_QUESTION =
  "How long is the return window for stocked goods, measured from the delivery date?";
/** Retrieves the cocoa chunk, and only one of the two return-window chunks. */
const COCOA_QUESTION = "What is the shelf life of the 3 kg cocoa powder tin?";
/** The corpus cannot answer this: the word never appears in it, by design. */
const UNANSWERABLE_QUESTION = "Do you deliver on Saturday?";

/**
 * A fabricated id shaped exactly like a real one: real document stem, real
 * naming convention, a section that does not exist.
 */
const INVENTED = "dry-goods-product-specs#cocoa-powder-nitrogen-flush-details";

/* ===========================================================================
 * The floor: refuse before generating, not after.
 * ======================================================================== */

describe("the similarity floor decides whether the generator runs at all", () => {
  it("refuses without calling the generator when nothing clears the floor", async () => {
    const generator = forbiddenGenerator();
    const result = await ask(UNANSWERABLE_QUESTION, generator);

    // The assertion this whole file exists for. A system that generates first
    // and checks after has already paid for the hallucination.
    expect(generator.calls).toHaveLength(0);

    const refused = expectRefused(result, "no-relevant-documents");
    expect(refused.detail).toContain(JSON.stringify(UNANSWERABLE_QUESTION));
    expect(refused.detail).toContain(MIN_SIMILARITY.toFixed(4));
  });

  it("names the best score it actually saw, so the refusal is debuggable", async () => {
    const { index } = await corpusIndex();
    const results = await search(index, UNANSWERABLE_QUESTION, createFakeEmbedder(), DEFAULT_K);
    const best = results[0] as SearchResult;
    expect(best.score).toBeLessThan(MIN_SIMILARITY);

    const refused = expectRefused(
      await ask(UNANSWERABLE_QUESTION, forbiddenGenerator()),
      "no-relevant-documents",
    );

    expect(refused.detail).toContain(best.score.toFixed(4));
    expect(refused.detail).toContain(best.chunk.chunkId);
    expect(refused.detail).toContain(String(index.entries.length));
  });

  it("keeps the near misses rather than reporting an empty search", async () => {
    const refused = expectRefused(
      await ask(UNANSWERABLE_QUESTION, forbiddenGenerator()),
      "no-relevant-documents",
    );

    // "everything fell short" and "we never searched" are different facts.
    expect(refused.nearMisses).toHaveLength(DEFAULT_K);
    for (const miss of refused.nearMisses) {
      expect(miss.score).toBeLessThan(MIN_SIMILARITY);
    }
    const scores = refused.nearMisses.map((miss) => miss.score);
    expect([...scores].sort((a, b) => b - a)).toEqual(scores);
  });

  it("calls the generator exactly once when something does clear the floor", async () => {
    const generator = scripted(claimsJson([]));
    await ask(COCOA_QUESTION, generator);
    expect(generator.calls).toHaveLength(1);
  });
});

/* ===========================================================================
 * Prompt construction.
 * ======================================================================== */

describe("the prompt carries real chunk ids, and asks for JSON only", () => {
  it("labels every retrieved chunk with its exact id and verbatim text", async () => {
    const above = await aboveFloor(COCOA_QUESTION);
    expect(above.length).toBeGreaterThan(1);

    const user = buildUserMessage(COCOA_QUESTION, above);

    expect(user).toContain(COCOA_QUESTION);
    for (const result of above) {
      expect(user).toContain(`chunkId: ${result.chunk.chunkId}`);
      // Verbatim, because the verifier checks quotes against these bytes.
      expect(user).toContain(result.chunk.text);
      expect(user).toContain(result.chunk.docTitle);
      expect(user).toContain(result.chunk.version);
      expect(user).toContain(result.chunk.effectiveDate);
    }
  });

  it("hands the generator ids it can actually cite", async () => {
    const above = await aboveFloor(COCOA_QUESTION);
    const generator = scripted(claimsJson([]));
    await ask(COCOA_QUESTION, generator);

    const call = generator.calls[0] as FakeGeneratorCall;
    for (const result of above) {
      expect(call.user).toContain(result.chunk.chunkId);
    }
    expect(call.system).toBe(buildSystemPrompt());
  });

  it("asks for a bare JSON object and for verbatim quotes", async () => {
    const system = buildSystemPrompt();
    expect(system).toContain("single JSON object");
    expect(system).toContain("no markdown code fence");
    expect(system).toContain("character for character");
    expect(system).toContain('"claims"');
  });

  it("asks nicely, and the verifier still decides — a non-compliant model is refused", async () => {
    // The prompt says "cite only chunkIds that appear in the user message".
    // This model does not. Nothing in the prompt layer notices; the check does.
    const { chunks } = await corpusIndex();
    const cocoa = chunkById(chunks, COCOA);
    const response = claimsJson([
      {
        text: "The cocoa tin carries a 540-day shelf life.",
        status: "stated",
        citations: [INVENTED],
        supportingQuotes: [{ chunkId: INVENTED, quote: sentenceWith(cocoa, "540-day shelf life") }],
      },
    ]);

    const refused = expectRefused(await ask(COCOA_QUESTION, scripted(response)), "claims-failed-verification");
    expect(refused.detail).toContain("chunk-not-retrieved");
  });
});

/* ===========================================================================
 * Parsing: malformed output is a typed failure, never a guessed verdict.
 * ======================================================================== */

describe("strict JSON parsing", () => {
  it("unwraps a fenced JSON object without changing a byte of it", () => {
    const payload = claimsJson([{ text: "x" }]);
    expect(stripCodeFence("```json\n" + payload + "\n```")).toBe(payload);
    expect(stripCodeFence("```\n" + payload + "\n```")).toBe(payload);
    expect(stripCodeFence(payload)).toBe(payload);
  });

  it("parses markdown-fenced JSON into an answer", async () => {
    const { chunks } = await corpusIndex();
    const cocoa = chunkById(chunks, COCOA);
    const quote = sentenceWith(cocoa, "540-day shelf life");
    const fenced =
      "```json\n" +
      claimsJson([
        {
          text: "The 3 kg cocoa tin carries a 540-day shelf life from its pack date.",
          status: "stated",
          citations: [COCOA],
          supportingQuotes: [{ chunkId: COCOA, quote }],
        },
      ]) +
      "\n```";

    const answered = expectAnswered(await ask(COCOA_QUESTION, scripted(fenced)));
    expect(answered.claims).toHaveLength(1);
    expect(answered.rejectedClaims).toHaveLength(0);
  });

  it("refuses rather than throwing when the response is not JSON", async () => {
    const prose = "I'm sorry — I don't have enough information to answer that.";
    const result = await ask(COCOA_QUESTION, scripted(prose));

    const refused = expectRefused(result, "no-claims-produced");
    expect(refused.detail).toContain("not valid JSON");
    // The raw response is in the detail: the failure is debuggable by reading it.
    expect(refused.detail).toContain("I'm sorry");
  });

  it("refuses on truncated JSON rather than repairing it", async () => {
    const refused = expectRefused(
      await ask(COCOA_QUESTION, scripted('{"claims": [{"text": "half a cl')),
      "no-claims-produced",
    );
    expect(refused.detail).toContain("not valid JSON");
  });

  it("refuses a JSON shape it was not promised, rather than guessing", async () => {
    // A bare array is the most plausible near-miss shape, and it is still a
    // guess about what the model meant. parseClaims does not guess.
    expect(parseClaims("[]")).toMatchObject({ ok: false });
    expect(parseClaims('{"answer": []}')).toMatchObject({ ok: false });
    expect(parseClaims('"a string"')).toMatchObject({ ok: false });
    expect(parseClaims("")).toMatchObject({ ok: false });

    const refused = expectRefused(await ask(COCOA_QUESTION, scripted("[]")), "no-claims-produced");
    expect(refused.detail).toContain("an array");
  });

  it("passes malformed claims through to the verifier instead of dropping them", () => {
    // Dropping them here would be silent; the verifier's rejection is visible.
    const parsed = parseClaims('{"claims": [null, 7, {"text": "no citations"}]}');
    expect(parsed.ok).toBe(true);
    expect(parsed.ok ? parsed.claims : []).toHaveLength(3);
  });

  it("refuses when the generator produces zero claims", async () => {
    const above = await aboveFloor(COCOA_QUESTION);
    const refused = expectRefused(
      await ask(COCOA_QUESTION, scripted(claimsJson([]))),
      "no-claims-produced",
    );

    expect(refused.detail).toContain("zero claims");
    for (const result of above) {
      expect(refused.detail).toContain(result.chunk.chunkId);
    }
  });
});

/* ===========================================================================
 * Verification: fabricated provenance is a refusal, not an answer.
 * ======================================================================== */

describe("verification decides what survives", () => {
  it("refuses when every claim fails, naming each stripped claim and why", async () => {
    const { chunks } = await corpusIndex();
    const cocoa = chunkById(chunks, COCOA);
    const real = sentenceWith(cocoa, "540-day shelf life");
    const paraphrase = real.replace("540-day", "540 day");
    expect(cocoa.text).not.toContain(paraphrase);

    const response = claimsJson([
      {
        text: "The cocoa tin carries a 540-day shelf life.",
        status: "stated",
        citations: [COCOA],
        supportingQuotes: [{ chunkId: COCOA, quote: paraphrase }],
      },
      {
        text: "The cocoa tin is packed under nitrogen flush.",
        status: "stated",
        citations: [INVENTED],
        supportingQuotes: [{ chunkId: INVENTED, quote: "packed under nitrogen flush" }],
      },
    ]);

    const refused = expectRefused(
      await ask(COCOA_QUESTION, scripted(response)),
      "claims-failed-verification",
    );

    expect(refused.detail).toContain("quote-not-in-chunk");
    expect(refused.detail).toContain("chunk-not-retrieved");
    expect(refused.detail).toContain(INVENTED);
    expect(refused.detail).toContain("(1)");
    expect(refused.detail).toContain("(2)");
  });

  it("refuses on a fabricated citation even when the claim itself is true", async () => {
    const { chunks } = await corpusIndex();
    const cocoa = chunkById(chunks, COCOA);
    // Every word of this claim is true of the corpus. The provenance is not.
    const response = claimsJson([
      {
        text: "The 3 kg cocoa tin carries a 540-day shelf life.",
        status: "stated",
        citations: [INVENTED],
        supportingQuotes: [
          { chunkId: INVENTED, quote: sentenceWith(cocoa, "540-day shelf life") },
        ],
      },
    ]);

    const result = await ask(COCOA_QUESTION, scripted(response));
    expect(result.kind).toBe("refused");
    expectRefused(result, "claims-failed-verification");
  });

  it("answers with the survivors and reports the stripped claims alongside", async () => {
    const { chunks } = await corpusIndex();
    const cocoa = chunkById(chunks, COCOA);
    const good = sentenceWith(cocoa, "540-day shelf life");

    const response = claimsJson([
      {
        text: "The 3 kg cocoa tin carries a 540-day shelf life from its pack date.",
        status: "stated",
        citations: [COCOA],
        supportingQuotes: [{ chunkId: COCOA, quote: good }],
      },
      {
        text: "The tin is re-nitrogen-flushed on request at the depot.",
        status: "stated",
        citations: [INVENTED],
        supportingQuotes: [{ chunkId: INVENTED, quote: "re-nitrogen-flushed on request" }],
      },
    ]);

    const answered = expectAnswered(await ask(COCOA_QUESTION, scripted(response)));

    expect(answered.claims).toHaveLength(1);
    expect((answered.claims[0] as Claim).citations).toEqual([COCOA]);

    // Visible, not hidden: the reader learns the model asserted something else
    // that turned out to be ungrounded.
    expect(answered.rejectedClaims).toHaveLength(1);
    const [stripped] = answered.rejectedClaims;
    expect(stripped?.reason.kind).toBe("chunk-not-retrieved");
    expect(stripped?.claim.text).toContain("re-nitrogen-flushed");
    expect(stripped?.reason.chunkId).toBe(INVENTED);
  });

  it("reports the chunks that were actually shown to the model", async () => {
    const above = await aboveFloor(COCOA_QUESTION);
    const { chunks } = await corpusIndex();
    const cocoa = chunkById(chunks, COCOA);

    const answered = expectAnswered(
      await ask(
        COCOA_QUESTION,
        scripted(
          claimsJson([
            {
              text: "The 3 kg cocoa tin carries a 540-day shelf life.",
              status: "stated",
              citations: [COCOA],
              supportingQuotes: [
                { chunkId: COCOA, quote: sentenceWith(cocoa, "540-day shelf life") },
              ],
            },
          ]),
        ),
      ),
    );

    expect(answered.retrieved.map((result) => result.chunk.chunkId)).toEqual(
      above.map((result) => result.chunk.chunkId),
    );
    for (const result of answered.retrieved) {
      expect(result.score).toBeGreaterThanOrEqual(MIN_SIMILARITY);
    }
  });
});

/* ===========================================================================
 * Contradiction surfacing — the corpus's designed conflict.
 * ======================================================================== */

/** Both sides of the return-window conflict, quoted verbatim from disk. */
async function returnWindowClaims(): Promise<{
  readonly response: string;
  readonly oldQuote: string;
  readonly newQuote: string;
  readonly claims: readonly Claim[];
}> {
  const { chunks } = await corpusIndex();
  const older = chunkById(chunks, OLD_WINDOW);
  const newer = chunkById(chunks, NEW_WINDOW);

  const oldQuote = sentenceWith(older, "14 calendar days");
  const newQuote = sentenceWith(newer, "30 calendar days");
  expect(oldQuote).toContain("14 calendar days");
  expect(newQuote).toContain("30 calendar days");

  const claims: Claim[] = [
    {
      text: "A customer may request a return on stocked goods within 14 calendar days of the delivery date.",
      status: "stated",
      citations: [OLD_WINDOW],
      supportingQuotes: [{ chunkId: OLD_WINDOW, quote: oldQuote }],
    },
    {
      text: "A customer may request a return on stocked goods within 30 calendar days of the delivery date.",
      status: "stated",
      citations: [NEW_WINDOW],
      supportingQuotes: [{ chunkId: NEW_WINDOW, quote: newQuote }],
    },
  ];

  return { response: claimsJson(claims), oldQuote, newQuote, claims };
}

describe("contradictory sources are surfaced, never resolved", () => {
  it("retrieves both sides of the return-window conflict above the floor", async () => {
    const ids = (await aboveFloor(RETURN_WINDOW_QUESTION)).map((result) => result.chunk.chunkId);
    expect(ids).toContain(OLD_WINDOW);
    expect(ids).toContain(NEW_WINDOW);
  });

  it("refuses with both positions, both citations, and both effective dates", async () => {
    const { response, oldQuote, newQuote } = await returnWindowClaims();
    const refused = expectRefused(
      await ask(RETURN_WINDOW_QUESTION, scripted(response)),
      "contradictory-sources",
    );

    // Both numbers.
    expect(refused.detail).toContain("14 calendar days");
    expect(refused.detail).toContain("30 calendar days");
    // Both citations.
    expect(refused.detail).toContain(OLD_WINDOW);
    expect(refused.detail).toContain(NEW_WINDOW);
    // Both quotes, verbatim.
    expect(refused.detail).toContain(oldQuote);
    expect(refused.detail).toContain(newQuote);
    // Both effective dates and versions — what a reader needs to decide.
    expect(refused.detail).toContain("2023-04-03");
    expect(refused.detail).toContain("2025-02-10");
    expect(refused.detail).toContain("version 1.2");
    expect(refused.detail).toContain("version 3.0");
    // Both document titles.
    expect(refused.detail).toContain("Returns and Credits Policy");
    expect(refused.detail).toContain("Customer Care Handbook");
  });

  it("resolves to neither: no answer, no winner, and recency stated not to decide", async () => {
    const { response } = await returnWindowClaims();
    const result = await ask(RETURN_WINDOW_QUESTION, scripted(response));

    // Not an answer at all — the 30-day figure is not quietly returned as the
    // operative one just because its document is newer.
    expect(result.kind).toBe("refused");
    expect(result).not.toHaveProperty("claims");

    const refused = result as RefusedResult;
    expect(refused.detail).toContain("does not by itself make that document");
    expect(refused.detail).toContain("a question for the reader");
  });

  it("offers no field a caller could mistake for a resolution", async () => {
    const { claims } = await returnWindowClaims();
    const { chunks } = await corpusIndex();

    const contradiction = detectContradiction(claims, chunks);
    expect(contradiction).not.toBeNull();
    const found = contradiction as NonNullable<typeof contradiction>;

    // The shape itself declines to pick: two positions, no winner, no order
    // that means anything beyond claim order.
    expect(Object.keys(found).sort()).toEqual(["positions", "unit"]);
    expect(found.positions).toHaveLength(2);

    const [first, second] = found.positions;
    expect(first.docId).not.toBe(second.docId);
    expect(new Set([first.value, second.value])).toEqual(
      new Set(["14 calendar days", "30 calendar days"]),
    );
    expect(new Set([first.effectiveDate, second.effectiveDate])).toEqual(
      new Set(["2023-04-03", "2025-02-10"]),
    );
    expect(new Set([first.version, second.version])).toEqual(new Set(["1.2", "3.0"]));
    expect(first.docTitle).not.toBe(second.docTitle);
    expect(first.citations).toEqual([OLD_WINDOW]);
    expect(second.citations).toEqual([NEW_WINDOW]);
  });

  it("runs on verified claims only: an unverifiable side never becomes a position", async () => {
    const { chunks } = await corpusIndex();
    const older = chunkById(chunks, OLD_WINDOW);
    const newer = chunkById(chunks, NEW_WINDOW);

    // The 30-day side is a paraphrase, so it is stripped before contradiction
    // detection runs. A "conflict" between a fact and a fabrication is not one.
    const fabricated = sentenceWith(newer, "30 calendar days").replace(
      "30 calendar days",
      "30 working days",
    );
    expect(newer.text).not.toContain(fabricated);

    const response = claimsJson([
      {
        text: "Returns on stocked goods are accepted within 14 calendar days of the delivery date.",
        status: "stated",
        citations: [OLD_WINDOW],
        supportingQuotes: [
          { chunkId: OLD_WINDOW, quote: sentenceWith(older, "14 calendar days") },
        ],
      },
      {
        text: "Returns on stocked goods are accepted within 30 working days of the delivery date.",
        status: "stated",
        citations: [NEW_WINDOW],
        supportingQuotes: [{ chunkId: NEW_WINDOW, quote: fabricated }],
      },
    ]);

    const answered = expectAnswered(await ask(RETURN_WINDOW_QUESTION, scripted(response)));
    expect(answered.claims).toHaveLength(1);
    expect(answered.rejectedClaims).toHaveLength(1);
    expect(answered.rejectedClaims[0]?.reason.kind).toBe("quote-not-in-chunk");
  });
});

describe("the contradiction heuristic, and the limits it is honest about", () => {
  it("does not fire on two sections of the same document", async () => {
    const { chunks } = await corpusIndex();
    const older = chunkById(chunks, OLD_WINDOW);
    const restocking = chunkById(chunks, "returns-and-credits-policy#restocking-fee");

    const claims: Claim[] = [
      {
        text: "The return window for stocked goods is 14 calendar days.",
        status: "stated",
        citations: [older.chunkId],
        supportingQuotes: [{ chunkId: older.chunkId, quote: sentenceWith(older, "14 calendar days") }],
      },
      {
        text: "An approved return on stocked goods carries a 15% restocking fee.",
        status: "stated",
        citations: [restocking.chunkId],
        supportingQuotes: [
          { chunkId: restocking.chunkId, quote: sentenceWith(restocking, "15% of the line value") },
        ],
      },
    ];

    expect(detectContradiction(claims, chunks)).toBeNull();
  });

  it("does not fire when two documents agree on the number", async () => {
    const { chunks } = await corpusIndex();
    const older = chunkById(chunks, OLD_WINDOW);
    const newer = chunkById(chunks, NEW_WINDOW);

    // Both claims quote a shared sentence: same unit, same magnitude, no conflict.
    const shared = "in the original outer carton";
    const claims: Claim[] = [
      {
        text: "Returned goods must be unopened and in the original outer carton.",
        status: "stated",
        citations: [older.chunkId],
        supportingQuotes: [{ chunkId: older.chunkId, quote: sentenceWith(older, shared) }],
      },
      {
        text: "Returned goods must be unopened and in the original outer carton.",
        status: "stated",
        citations: [newer.chunkId],
        supportingQuotes: [{ chunkId: newer.chunkId, quote: sentenceWith(newer, shared) }],
      },
    ];

    expect(detectContradiction(claims, chunks)).toBeNull();
  });

  it("does not fire on unrelated subjects that happen to share a unit", async () => {
    const { chunks } = await corpusIndex();
    const older = chunkById(chunks, OLD_WINDOW);
    const escalation = chunkById(chunks, "customer-care-handbook#escalation-path");

    const claims: Claim[] = [
      {
        text: "A return on stocked goods must be requested within 14 calendar days of delivery.",
        status: "stated",
        citations: [older.chunkId],
        supportingQuotes: [{ chunkId: older.chunkId, quote: sentenceWith(older, "14 calendar days") }],
      },
      {
        text: "An unresolved issue moves from the agent to the team lead after two working days.",
        status: "stated",
        citations: [escalation.chunkId],
        supportingQuotes: [
          { chunkId: escalation.chunkId, quote: sentenceWith(escalation, "two working days") },
        ],
      },
    ];

    // Both mention a number of days across two documents; the subjects do not
    // overlap, so this is not reported as a conflict.
    expect(detectContradiction(claims, chunks)).toBeNull();
  });

  it("is blind to a contradiction with no differing number, and says so in comments", async () => {
    const { chunks } = await corpusIndex();
    const returns = chunkById(chunks, "returns-and-credits-policy#collection-of-returned-goods");
    const claimsDoc = chunkById(chunks, "damaged-and-short-shipment-claims#collection-of-rejected-goods");

    const claims: Claim[] = [
      {
        text: "Returned goods must be palletised or boxed by the customer for collection.",
        status: "stated",
        citations: [returns.chunkId],
        supportingQuotes: [
          { chunkId: returns.chunkId, quote: sentenceWith(returns, "palletised or boxed") },
        ],
      },
      {
        text: "Claim goods must not be repacked, consolidated, or palletised before collection.",
        status: "stated",
        citations: [claimsDoc.chunkId],
        supportingQuotes: [
          { chunkId: claimsDoc.chunkId, quote: sentenceWith(claimsDoc, "palletised") },
        ],
      },
    ];

    // A documented limit, pinned so it cannot quietly become an undocumented one:
    // the detector is numeric, and these two directly opposite instructions
    // carry no differing number.
    expect(detectContradiction(claims, chunks)).toBeNull();
    const source = readFileSync(ANSWER_SOURCE, "utf8");
    expect(source).toContain("NUMBERS ONLY");
  });

  it("cannot see a contradiction folded into a single claim", async () => {
    const { chunks } = await corpusIndex();
    const older = chunkById(chunks, OLD_WINDOW);
    const newer = chunkById(chunks, NEW_WINDOW);

    const claims: Claim[] = [
      {
        text: "The two policy documents state different return windows.",
        status: "inferred",
        citations: [older.chunkId, newer.chunkId],
        supportingQuotes: [
          { chunkId: older.chunkId, quote: sentenceWith(older, "14 calendar days") },
          { chunkId: newer.chunkId, quote: sentenceWith(newer, "30 calendar days") },
        ],
      },
    ];

    // Detection is pairwise across claims, so one claim is no pair. Documented.
    expect(detectContradiction(claims, chunks)).toBeNull();
    expect(readFileSync(ANSWER_SOURCE, "utf8")).toContain("PAIRWISE, ACROSS CLAIMS");
  });
});

/* ===========================================================================
 * Determinism.
 * ======================================================================== */

describe("determinism", () => {
  it("returns identical results for identical inputs under a fixed fake", async () => {
    const { chunks } = await corpusIndex();
    const cocoa = chunkById(chunks, COCOA);
    const response = claimsJson([
      {
        text: "The 3 kg cocoa tin carries a 540-day shelf life from its pack date.",
        status: "stated",
        citations: [COCOA],
        supportingQuotes: [
          { chunkId: COCOA, quote: sentenceWith(cocoa, "540-day shelf life") },
        ],
      },
      {
        text: "The tin is re-nitrogen-flushed on request at the depot.",
        status: "stated",
        citations: [INVENTED],
        supportingQuotes: [{ chunkId: INVENTED, quote: "re-nitrogen-flushed on request" }],
      },
    ]);

    const first = await ask(COCOA_QUESTION, scripted(response));
    const second = await ask(COCOA_QUESTION, scripted(response));
    expect(second).toEqual(first);

    // And the refusal paths too — the detail string is quoted in eval reports,
    // so it has to be stable across runs, not merely equivalent.
    const refusedOnce = await ask(UNANSWERABLE_QUESTION, scripted(response));
    const refusedTwice = await ask(UNANSWERABLE_QUESTION, scripted(response));
    expect(refusedTwice).toEqual(refusedOnce);

    const conflict = (await returnWindowClaims()).response;
    expect(await ask(RETURN_WINDOW_QUESTION, scripted(conflict))).toEqual(
      await ask(RETURN_WINDOW_QUESTION, scripted(conflict)),
    );
  });

  it("reaches every refusal reason", async () => {
    const { chunks } = await corpusIndex();
    const cocoa = chunkById(chunks, COCOA);
    const conflict = (await returnWindowClaims()).response;

    const reached = new Set<RefusalReason>();
    for (const [question, response] of [
      [UNANSWERABLE_QUESTION, claimsJson([])],
      [COCOA_QUESTION, "not json at all"],
      [
        COCOA_QUESTION,
        claimsJson([
          {
            text: "The cocoa tin carries a 540-day shelf life.",
            status: "stated",
            citations: [INVENTED],
            supportingQuotes: [
              { chunkId: INVENTED, quote: sentenceWith(cocoa, "540-day shelf life") },
            ],
          },
        ]),
      ],
      [RETURN_WINDOW_QUESTION, conflict],
    ] as const) {
      const result = await ask(question, scripted(response));
      expect(result.kind).toBe("refused");
      reached.add((result as RefusedResult).reason);
    }

    expect(reached).toEqual(
      new Set<RefusalReason>([
        "no-relevant-documents",
        "no-claims-produced",
        "claims-failed-verification",
        "contradictory-sources",
      ]),
    );
  });
});

/* ===========================================================================
 * The live generator, exercised against an injected client.
 * ======================================================================== */

/** A string that looks enough like a credential to be worth grepping for. */
const DECOY_KEY = "sk-ant-DECOY-KEY-4b81de07c2f9-do-not-log";

/** Build a `Message` with the fields under test real and the rest inert. */
function message(
  content: Anthropic.ContentBlock[],
  stopReason: Anthropic.StopReason | null = "end_turn",
): Anthropic.Message {
  return {
    id: "msg_test",
    container: null,
    content,
    model: "claude-opus-5",
    role: "assistant",
    stop_details: null,
    stop_reason: stopReason,
    stop_sequence: null,
    type: "message",
    // Never read by the code under test; the SDK's Usage shape is not the
    // subject of this file.
    usage: {} as Anthropic.Usage,
  };
}

function textBlock(text: string): Anthropic.ContentBlock {
  return { type: "text", text, citations: null };
}

/** A stand-in for the SDK client. Records the request it was handed. */
function fakeClient(
  handler: (params: Anthropic.MessageCreateParamsNonStreaming) => Promise<Anthropic.Message>,
): { client: Anthropic; requests: Anthropic.MessageCreateParamsNonStreaming[] } {
  const requests: Anthropic.MessageCreateParamsNonStreaming[] = [];
  const client = {
    messages: {
      create: (params: Anthropic.MessageCreateParamsNonStreaming) => {
        requests.push(params);
        return handler(params);
      },
    },
    // The generator only ever touches `messages.create`; the cast is what keeps
    // this fake from having to implement an SDK client to test four branches.
  } as unknown as Anthropic;
  return { client, requests };
}

describe("anthropicGenerator", () => {
  it("sends the system prompt and user message, and returns the joined text", async () => {
    const { client, requests } = fakeClient(async () =>
      message([textBlock('{"claims":'), textBlock(" []}")]),
    );
    const generator = anthropicGenerator({ apiKey: DECOY_KEY, model: "claude-opus-5", client });

    const text = await generator.generate("SYSTEM", "USER");

    expect(text).toBe('{"claims": []}');
    expect(requests).toHaveLength(1);
    const [request] = requests;
    expect(request?.model).toBe("claude-opus-5");
    expect(request?.system).toBe("SYSTEM");
    expect(request?.messages).toEqual([{ role: "user", content: "USER" }]);
    expect(request?.max_tokens).toBeGreaterThan(0);
  });

  it("ignores non-text blocks rather than stringifying them into the JSON", async () => {
    const thinking: Anthropic.ContentBlock = {
      type: "thinking",
      thinking: "the reader should never see this in the parsed claims",
      signature: "sig",
    };
    const { client } = fakeClient(async () => message([thinking, textBlock("{}")]));
    const generator = anthropicGenerator({ apiKey: DECOY_KEY, model: "claude-opus-5", client });

    expect(await generator.generate("s", "u")).toBe("{}");
  });

  it("throws a typed error when the model declines", async () => {
    const { client } = fakeClient(async () => message([textBlock("no")], "refusal"));
    const generator = anthropicGenerator({ apiKey: DECOY_KEY, model: "claude-opus-5", client });

    await expect(generator.generate("s", "u")).rejects.toMatchObject({
      name: "GenerationError",
      kind: "model-refusal",
    });
  });

  it("throws a typed error on truncation rather than handing on cut-off JSON", async () => {
    const { client } = fakeClient(async () =>
      message([textBlock('{"claims": [{"text": "cut off')], "max_tokens"),
    );
    const generator = anthropicGenerator({ apiKey: DECOY_KEY, model: "claude-opus-5", client });

    await expect(generator.generate("s", "u")).rejects.toMatchObject({ kind: "truncated" });
  });

  it("throws a typed error when there is no text at all", async () => {
    const { client } = fakeClient(async () => message([]));
    const generator = anthropicGenerator({ apiKey: DECOY_KEY, model: "claude-opus-5", client });

    await expect(generator.generate("s", "u")).rejects.toMatchObject({ kind: "empty-response" });
  });

  it("reports the provider's status without carrying its payload", async () => {
    const apiError = new Anthropic.APIError(
      429,
      { echoed_request: { headers: { authorization: `Bearer ${DECOY_KEY}` } } },
      "rate limited",
      undefined,
    );
    const { client } = fakeClient(async () => {
      throw apiError;
    });
    const generator = anthropicGenerator({ apiKey: DECOY_KEY, model: "claude-opus-5", client });

    const error = await generator.generate("s", "u").then(
      () => null,
      (thrown: unknown) => thrown,
    );

    expect(error).toBeInstanceOf(GenerationError);
    const generation = error as GenerationError;
    expect(generation.kind).toBe("transport");
    expect(generation.status).toBe(429);
    // The upstream body echoed the key. Neither the message nor the serialised
    // form may carry it, and `cause` is not attached at all.
    expect(generation.message).not.toContain(DECOY_KEY);
    expect(JSON.stringify(generation)).not.toContain(DECOY_KEY);
    expect(generation.cause).toBeUndefined();
  });

  it("never puts the key in an error on a plain transport failure", async () => {
    const { client } = fakeClient(async () => {
      throw new Error(`connection reset while sending Bearer ${DECOY_KEY}`);
    });
    const generator = anthropicGenerator({ apiKey: DECOY_KEY, model: "claude-opus-5", client });

    const error = (await generator.generate("s", "u").catch((thrown: unknown) => thrown)) as GenerationError;
    expect(error.kind).toBe("transport");
    expect(error.status).toBeUndefined();
    expect(error.message).not.toContain(DECOY_KEY);
    expect(JSON.stringify(error)).not.toContain(DECOY_KEY);
  });

  it("builds its own client from the injected key when none is supplied", () => {
    // No network: constructing a client makes no request. The point is that the
    // key arrives as a parameter and nothing reaches for an ambient one.
    const generator = anthropicGenerator({ apiKey: DECOY_KEY, model: "claude-opus-5" });
    expect(typeof generator.generate).toBe("function");
  });

  it("does not read process.env anywhere in the module", () => {
    // Checked against the source, because the rule is about the file, not about
    // whichever branch a test happens to take. Comments are stripped first: the
    // module states the rule in prose, and the prose must not satisfy the test.
    const code = readFileSync(ANSWER_SOURCE, "utf8")
      .replace(/\/\*[\s\S]*?\*\//g, "")
      .replace(/\/\/[^\n]*/g, "");
    expect(code).toContain("anthropicGenerator");
    expect(code).not.toMatch(/process\s*\.\s*env/);
  });
});
