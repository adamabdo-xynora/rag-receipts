import { fileURLToPath } from "node:url";
import { describe, expect, it } from "vitest";

import { type Chunk, loadCorpus } from "../src/chunk.js";
import {
  type Embedder,
  EmbeddingError,
  type EmbeddingErrorKind,
  type EmbeddingKind,
  MIN_SIMILARITY,
  type SearchResult,
  VOYAGE_ENDPOINT,
  buildIndex,
  cosineSimilarity,
  partitionByThreshold,
  search,
  voyageEmbedder,
} from "../src/retrieve.js";
import { FAKE_DIMENSIONS, createFakeEmbedder, fakeVector } from "./fake-embedder.js";

const CORPUS_DIR = fileURLToPath(new URL("../corpus", import.meta.url));

/** Matches `test/chunk.test.ts`; the corpus is a fixed, designed battery. */
const EXPECTED_CHUNK_COUNT = 71;

/**
 * A string that looks enough like a credential to be worth grepping for.
 * Every error path below is checked for its absence — see the "the API key
 * never escapes" describe block.
 */
const DECOY_KEY = "pa-DECOY-KEY-9f3c1e7a5b2d-do-not-log";

/** Build a Chunk without ceremony; only `chunkId` and `text` matter here. */
function makeChunk(chunkId: string, text: string): Chunk {
  const [docId = chunkId, slug = chunkId] = chunkId.split("#");
  return {
    chunkId,
    docId,
    docTitle: `Title of ${docId}`,
    docType: "policy",
    effectiveDate: "2025-02-24",
    version: "1.0",
    sectionHeading: slug,
    text,
  };
}

/** An embedder that returns exactly what it is told to, in order. */
function scriptedEmbedder(documents: number[][], query: number[]): Embedder {
  return {
    async embed(texts: string[], kind: EmbeddingKind): Promise<number[][]> {
      return kind === "document" ? documents : texts.map(() => query);
    },
  };
}

/** Assert `fn` rejects with an EmbeddingError of exactly `kind`, and return it. */
async function expectEmbeddingError(
  fn: () => Promise<unknown>,
  kind: EmbeddingErrorKind,
): Promise<EmbeddingError> {
  let thrown: unknown;
  try {
    await fn();
  } catch (error) {
    thrown = error;
  }
  expect(thrown, `expected an EmbeddingError of kind "${kind}"`).toBeInstanceOf(EmbeddingError);
  const error = thrown as EmbeddingError;
  expect(error.kind).toBe(kind);
  return error;
}

/** Every surface an error reaches a log, a console, or an issue tracker by. */
function expectNoKeyAnywhere(error: EmbeddingError): void {
  expect(error.message).not.toContain(DECOY_KEY);
  expect(String(error)).not.toContain(DECOY_KEY);
  expect(JSON.stringify(error)).not.toContain(DECOY_KEY);
  expect(JSON.stringify({ ...error })).not.toContain(DECOY_KEY);
  expect(error.stack ?? "").not.toContain(DECOY_KEY);
}

/** A `fetch` that answers with one canned response and records its calls. */
function recordingFetch(respond: (call: number) => Response | Promise<Response>): {
  fetchImpl: typeof globalThis.fetch;
  calls: { url: string; init: RequestInit | undefined }[];
} {
  const calls: { url: string; init: RequestInit | undefined }[] = [];
  const fetchImpl: typeof globalThis.fetch = async (input, init) => {
    calls.push({ url: String(input), init });
    return await respond(calls.length - 1);
  };
  return { fetchImpl, calls };
}

/** Read back the JSON body a recorded call was made with. */
function sentBody(init: RequestInit | undefined): Record<string, unknown> {
  return JSON.parse(String(init?.body ?? "{}")) as Record<string, unknown>;
}

function sentHeaders(init: RequestInit | undefined): Record<string, string> {
  return (init?.headers ?? {}) as Record<string, string>;
}

function jsonResponse(body: unknown, status = 200, statusText = "OK"): Response {
  return new Response(JSON.stringify(body), { status, statusText });
}

/** `{ data: [...] }` with `n` one-dimensional vectors, indexed in order. */
function voyagePayload(vectors: number[][]): unknown {
  return {
    object: "list",
    model: "voyage-3",
    data: vectors.map((embedding, index) => ({ object: "embedding", index, embedding })),
    usage: { total_tokens: 42 },
  };
}

describe("cosineSimilarity", () => {
  it("scores a vector against itself as 1", () => {
    expect(cosineSimilarity([1, 2, 3], [1, 2, 3])).toBeCloseTo(1, 12);
    expect(cosineSimilarity([-4, 0.5, 17], [-4, 0.5, 17])).toBeCloseTo(1, 12);
  });

  it("ignores magnitude, because cosine is about direction", () => {
    // The same direction at ten times the length is still the same direction.
    expect(cosineSimilarity([1, 2, 3], [10, 20, 30])).toBeCloseTo(1, 12);
  });

  it("scores orthogonal vectors as 0", () => {
    expect(cosineSimilarity([1, 0], [0, 1])).toBe(0);
    expect(cosineSimilarity([1, 0, 0], [0, 3, 4])).toBe(0);
  });

  it("scores opposed vectors as -1", () => {
    expect(cosineSimilarity([1, 2], [-1, -2])).toBeCloseTo(-1, 12);
  });

  it("computes a known value by hand", () => {
    // [1,1] sits at 45° to [1,0]; cos 45° = 1/sqrt(2).
    expect(cosineSimilarity([1, 1], [1, 0])).toBeCloseTo(Math.SQRT1_2, 12);
  });

  it("returns 0, never NaN, when either vector has zero magnitude", () => {
    // The guard is the point: 0/0 would be NaN, and NaN neither sorts nor
    // compares — it would slip past the threshold and land anywhere in a sort.
    const zeroAgainstSomething = cosineSimilarity([0, 0, 0], [1, 2, 3]);
    const somethingAgainstZero = cosineSimilarity([1, 2, 3], [0, 0, 0]);
    const zeroAgainstZero = cosineSimilarity([0, 0], [0, 0]);

    expect(zeroAgainstSomething).toBe(0);
    expect(somethingAgainstZero).toBe(0);
    expect(zeroAgainstZero).toBe(0);
    expect(Number.isNaN(zeroAgainstSomething)).toBe(false);
    expect(Number.isNaN(somethingAgainstZero)).toBe(false);
    expect(Number.isNaN(zeroAgainstZero)).toBe(false);
  });

  it("treats an empty vector as zero magnitude rather than dividing by nothing", () => {
    expect(cosineSimilarity([], [])).toBe(0);
  });

  it("refuses to compare vectors of different lengths", async () => {
    await expectEmbeddingError(
      async () => cosineSimilarity([1, 2, 3], [1, 2]),
      "dimension-mismatch",
    );
  });
});

describe("buildIndex", () => {
  const chunks = [makeChunk("doc#a", "alpha text"), makeChunk("doc#b", "beta text")];

  it('embeds chunk text with the "document" input-type hint', async () => {
    const embedder = createFakeEmbedder();
    await buildIndex(chunks, embedder);

    expect(embedder.calls).toHaveLength(1);
    expect(embedder.calls[0]?.kind).toBe("document");
    expect(embedder.calls[0]?.texts).toEqual(["alpha text", "beta text"]);
  });

  it("pairs each chunk with its own vector and records the dimensions", async () => {
    const index = await buildIndex(chunks, createFakeEmbedder());

    expect(index.entries).toHaveLength(2);
    expect(index.entries[0]?.chunk.chunkId).toBe("doc#a");
    expect(index.entries[0]?.vector).toEqual(fakeVector("alpha text"));
    expect(index.entries[1]?.chunk.chunkId).toBe("doc#b");
    expect(index.dimensions).toBe(FAKE_DIMENSIONS);
  });

  it("indexes an empty corpus without calling the embedder", async () => {
    const embedder = createFakeEmbedder();
    const index = await buildIndex([], embedder);

    expect(index.entries).toEqual([]);
    expect(index.dimensions).toBe(0);
    expect(embedder.calls).toEqual([]);
  });

  it("rejects an embedder that returns the wrong number of vectors", async () => {
    // Silent misalignment is the failure this guards: chunk b would hold no
    // vector, or worse, chunk b's text would sit against chunk c's meaning.
    const shortEmbedder: Embedder = { async embed() { return [[1, 0]]; } };
    const error = await expectEmbeddingError(
      async () => buildIndex(chunks, shortEmbedder),
      "vector-count-mismatch",
    );
    expect(error.message).toContain("1 vector");
    expect(error.message).toContain("2 chunks");
  });

  it("rejects an embedder that returns vectors of differing lengths", async () => {
    const raggedEmbedder: Embedder = { async embed() { return [[1, 0], [1, 0, 0]]; } };
    const error = await expectEmbeddingError(
      async () => buildIndex(chunks, raggedEmbedder),
      "dimension-mismatch",
    );
    expect(error.message).toContain("doc#b");
  });
});

describe("search", () => {
  const chunks = [
    makeChunk("doc#near", "near"),
    makeChunk("doc#middle", "middle"),
    makeChunk("doc#far", "far"),
  ];
  // Scored against the query [1, 0]: 1, 1/sqrt(2), 0.
  const documentVectors = [[1, 0], [1, 1], [0, 1]];

  async function scriptedIndex() {
    return await buildIndex(chunks, scriptedEmbedder(documentVectors, [1, 0]));
  }

  it('embeds the query with the "query" input-type hint', async () => {
    const embedder = createFakeEmbedder();
    const index = await buildIndex(chunks, embedder);
    await search(index, "a question", embedder, 3);

    expect(embedder.calls).toHaveLength(2);
    expect(embedder.calls[0]?.kind).toBe("document");
    expect(embedder.calls[1]).toEqual({ texts: ["a question"], kind: "query" });
  });

  it("returns the top k in descending score order", async () => {
    const index = await scriptedIndex();
    const results = await search(index, "q", scriptedEmbedder(documentVectors, [1, 0]), 2);

    expect(results.map((r) => r.chunk.chunkId)).toEqual(["doc#near", "doc#middle"]);
    expect(results[0]?.score).toBeCloseTo(1, 12);
    expect(results[1]?.score).toBeCloseTo(Math.SQRT1_2, 12);
  });

  it("returns everything, and no filler, when k exceeds the corpus", async () => {
    const index = await scriptedIndex();
    const results = await search(index, "q", scriptedEmbedder(documentVectors, [1, 0]), 999);

    expect(results).toHaveLength(chunks.length);
    expect(results.map((r) => r.chunk.chunkId)).toEqual(["doc#near", "doc#middle", "doc#far"]);
  });

  it("returns nothing, and embeds nothing, for k <= 0 or an empty index", async () => {
    const index = await scriptedIndex();
    const embedder = createFakeEmbedder();

    expect(await search(index, "q", embedder, 0)).toEqual([]);
    expect(await search(index, "q", embedder, -3)).toEqual([]);
    expect(await search({ entries: [], dimensions: 0 }, "q", embedder, 5)).toEqual([]);
    expect(embedder.calls).toEqual([]);
  });

  it("breaks ties by chunkId, ascending, whatever order the index is in", async () => {
    // Identical text means an identical vector means a bitwise identical
    // score, so the tie-break is the only thing deciding the order.
    const tied = [
      makeChunk("doc#zulu", "same"),
      makeChunk("doc#alpha", "same"),
      makeChunk("doc#mike", "same"),
    ];
    const embedder = createFakeEmbedder();
    const index = await buildIndex(tied, embedder);
    const results = await search(index, "same", embedder, 3);

    expect(results.map((r) => r.chunk.chunkId)).toEqual(["doc#alpha", "doc#mike", "doc#zulu"]);
    expect(new Set(results.map((r) => r.score)).size).toBe(1);
  });

  it("gives the same order on every run", async () => {
    const tied = ["doc#d", "doc#b", "doc#c", "doc#a"].map((id) => makeChunk(id, "identical"));
    const embedder = createFakeEmbedder();
    const index = await buildIndex(tied, embedder);

    const first = (await search(index, "identical", embedder, 4)).map((r) => r.chunk.chunkId);
    const second = (await search(index, "identical", embedder, 4)).map((r) => r.chunk.chunkId);
    const third = (await search(index, "identical", embedder, 4)).map((r) => r.chunk.chunkId);

    expect(first).toEqual(["doc#a", "doc#b", "doc#c", "doc#d"]);
    expect(second).toEqual(first);
    expect(third).toEqual(first);
  });

  it("rejects an embedder that answers a single query with several vectors", async () => {
    const index = await scriptedIndex();
    const chatty: Embedder = {
      async embed(_texts: string[], kind: EmbeddingKind) {
        return kind === "query" ? [[1, 0], [0, 1]] : documentVectors;
      },
    };
    await expectEmbeddingError(async () => search(index, "q", chatty, 3), "vector-count-mismatch");
  });
});

describe("voyageEmbedder request shape", () => {
  it("posts input, model, and the input-type hint to the embeddings endpoint", async () => {
    const { fetchImpl, calls } = recordingFetch(() => jsonResponse(voyagePayload([[1, 0], [0, 1]])));
    const embedder = voyageEmbedder({ apiKey: DECOY_KEY, model: "voyage-3", fetchImpl });

    const vectors = await embedder.embed(["first", "second"], "document");

    expect(vectors).toEqual([[1, 0], [0, 1]]);
    expect(calls).toHaveLength(1);
    expect(calls[0]?.url).toBe(VOYAGE_ENDPOINT);
    expect(calls[0]?.init?.method).toBe("POST");
    expect(sentBody(calls[0]?.init)).toEqual({
      input: ["first", "second"],
      model: "voyage-3",
      input_type: "document",
    });
    expect(sentHeaders(calls[0]?.init)["authorization"]).toBe(`Bearer ${DECOY_KEY}`);
    expect(sentHeaders(calls[0]?.init)["content-type"]).toBe("application/json");
  });

  it('sends input_type "document" for documents and "query" for queries', async () => {
    // The whole reason `kind` is in the interface: the hint has to survive the
    // trip to the provider, and nothing else in the response would reveal it.
    const { fetchImpl, calls } = recordingFetch(() => jsonResponse(voyagePayload([[1]])));
    const embedder = voyageEmbedder({ apiKey: DECOY_KEY, model: "voyage-3", fetchImpl });

    await embedder.embed(["a passage"], "document");
    await embedder.embed(["a question"], "query");

    expect(sentBody(calls[0]?.init)["input_type"]).toBe("document");
    expect(sentBody(calls[1]?.init)["input_type"]).toBe("query");
  });

  it("orders vectors by the response index, not by array position", async () => {
    const { fetchImpl } = recordingFetch(() =>
      jsonResponse({
        data: [
          { index: 1, embedding: [0, 1] },
          { index: 0, embedding: [1, 0] },
        ],
      }),
    );
    const embedder = voyageEmbedder({ apiKey: DECOY_KEY, model: "voyage-3", fetchImpl });

    expect(await embedder.embed(["first", "second"], "document")).toEqual([[1, 0], [0, 1]]);
  });

  it("does not call the provider at all for an empty input list", async () => {
    const { fetchImpl, calls } = recordingFetch(() => jsonResponse(voyagePayload([])));
    const embedder = voyageEmbedder({ apiKey: DECOY_KEY, model: "voyage-3", fetchImpl });

    expect(await embedder.embed([], "document")).toEqual([]);
    expect(calls).toEqual([]);
  });

  it("defaults to the global fetch when none is injected", async () => {
    const realFetch = globalThis.fetch;
    const seen: string[] = [];
    globalThis.fetch = (async (input: unknown) => {
      seen.push(String(input));
      return jsonResponse(voyagePayload([[1, 0]]));
    }) as typeof globalThis.fetch;
    try {
      const embedder = voyageEmbedder({ apiKey: DECOY_KEY, model: "voyage-3" });
      expect(await embedder.embed(["text"], "query")).toEqual([[1, 0]]);
      expect(seen).toEqual([VOYAGE_ENDPOINT]);
    } finally {
      globalThis.fetch = realFetch;
    }
  });
});

describe("voyageEmbedder failures", () => {
  function embedderRespondingWith(response: () => Response | Promise<Response>): Embedder {
    const { fetchImpl } = recordingFetch(() => response());
    return voyageEmbedder({ apiKey: DECOY_KEY, model: "voyage-3", fetchImpl });
  }

  it("names the status on a non-2xx response", async () => {
    const embedder = embedderRespondingWith(() =>
      jsonResponse({ detail: "Provided API key is invalid." }, 401, "Unauthorized"),
    );
    const error = await expectEmbeddingError(
      async () => embedder.embed(["text"], "document"),
      "http-status",
    );

    expect(error.status).toBe(401);
    expect(error.message).toContain("401");
    expect(error.message).toContain("Unauthorized");
  });

  it("names the status on a server error too", async () => {
    const embedder = embedderRespondingWith(() =>
      jsonResponse({ detail: "upstream boom" }, 503, "Service Unavailable"),
    );
    const error = await expectEmbeddingError(
      async () => embedder.embed(["text"], "query"),
      "http-status",
    );
    expect(error.status).toBe(503);
    expect(error.message).toContain("503");
  });

  it("rejects a vector count that does not match the input count, rather than padding", async () => {
    // Two inputs, one vector. Padding would leave the second chunk holding a
    // placeholder; truncating would drop it. Both produce confident citations
    // of the wrong text, so neither is on the table.
    const embedder = embedderRespondingWith(() => jsonResponse(voyagePayload([[1, 0]])));
    const error = await expectEmbeddingError(
      async () => embedder.embed(["first", "second"], "document"),
      "vector-count-mismatch",
    );

    expect(error.message).toContain("1 vector");
    expect(error.message).toContain("2 inputs");
  });

  it("rejects extra vectors as firmly as missing ones", async () => {
    const embedder = embedderRespondingWith(() => jsonResponse(voyagePayload([[1], [2], [3]])));
    await expectEmbeddingError(
      async () => embedder.embed(["only one"], "document"),
      "vector-count-mismatch",
    );
  });

  it("rejects a response that is not JSON", async () => {
    const embedder = embedderRespondingWith(() => new Response("<html>gateway</html>", { status: 200 }));
    await expectEmbeddingError(async () => embedder.embed(["text"], "query"), "malformed-response");
  });

  it("rejects a response with no data array", async () => {
    const embedder = embedderRespondingWith(() => jsonResponse({ object: "list" }));
    await expectEmbeddingError(async () => embedder.embed(["text"], "query"), "malformed-response");
  });

  it("rejects a vector holding anything that is not a finite number", async () => {
    // A single null or NaN would poison every score it touches, silently.
    const embedder = embedderRespondingWith(() =>
      jsonResponse({ data: [{ index: 0, embedding: [1, null, 3] }] }),
    );
    await expectEmbeddingError(async () => embedder.embed(["text"], "query"), "malformed-response");
  });

  it("rejects a repeated or out-of-range index", async () => {
    const embedder = embedderRespondingWith(() =>
      jsonResponse({ data: [{ index: 0, embedding: [1] }, { index: 0, embedding: [2] }] }),
    );
    await expectEmbeddingError(
      async () => embedder.embed(["first", "second"], "document"),
      "malformed-response",
    );
  });

  it("reports a transport failure as its own kind", async () => {
    const fetchImpl: typeof globalThis.fetch = async () => {
      throw new Error(`connect ECONNREFUSED while sending Bearer ${DECOY_KEY}`);
    };
    const embedder = voyageEmbedder({ apiKey: DECOY_KEY, model: "voyage-3", fetchImpl });
    const error = await expectEmbeddingError(
      async () => embedder.embed(["text"], "query"),
      "transport",
    );
    expect(error.status).toBeUndefined();
  });
});

describe("the API key never escapes", () => {
  /** Every failure path, each with a fetch that is hostile about the key. */
  const scenarios: { name: string; embedder: () => Embedder; texts: string[] }[] = [
    {
      // The nastiest case: an upstream that echoes the request — key and all —
      // into its own error body. Nothing from the body may reach the message.
      name: "a non-2xx whose body echoes the Authorization header",
      embedder: () =>
        voyageEmbedder({
          apiKey: DECOY_KEY,
          model: "voyage-3",
          fetchImpl: async () =>
            jsonResponse(
              { error: "invalid key", received: { authorization: `Bearer ${DECOY_KEY}` } },
              403,
              "Forbidden",
            ),
        }),
      texts: ["text"],
    },
    {
      name: "a vector-count mismatch",
      embedder: () =>
        voyageEmbedder({
          apiKey: DECOY_KEY,
          model: "voyage-3",
          fetchImpl: async () => jsonResponse(voyagePayload([[1, 0]])),
        }),
      texts: ["first", "second"],
    },
    {
      name: "a malformed response",
      embedder: () =>
        voyageEmbedder({
          apiKey: DECOY_KEY,
          model: "voyage-3",
          fetchImpl: async () => jsonResponse({ data: [{ index: 0, embedding: ["nope"] }] }),
        }),
      texts: ["text"],
    },
    {
      name: "a transport error carrying the key in its own message",
      embedder: () =>
        voyageEmbedder({
          apiKey: DECOY_KEY,
          model: "voyage-3",
          fetchImpl: async () => {
            throw new Error(`TLS failure for request with Bearer ${DECOY_KEY}`);
          },
        }),
      texts: ["text"],
    },
  ];

  for (const scenario of scenarios) {
    it(`keeps the key out of the error thrown for ${scenario.name}`, async () => {
      let thrown: unknown;
      try {
        await scenario.embedder().embed(scenario.texts, "document");
      } catch (error) {
        thrown = error;
      }
      expect(thrown).toBeInstanceOf(EmbeddingError);
      expectNoKeyAnywhere(thrown as EmbeddingError);
    });
  }

  it("has a decoy that these assertions could actually find", async () => {
    // Guards the guard: if the decoy could never appear in a serialised error,
    // every assertion above would pass for the wrong reason.
    const canary = new EmbeddingError("http-status", `leaked ${DECOY_KEY}`, 500);
    expect(canary.message).toContain(DECOY_KEY);
    expect(JSON.stringify(canary)).toContain(DECOY_KEY);
  });
});

describe("MIN_SIMILARITY and partitionByThreshold", () => {
  const results: SearchResult[] = [
    { chunk: makeChunk("doc#strong", "s"), score: 0.9 },
    { chunk: makeChunk("doc#exact", "e"), score: MIN_SIMILARITY },
    { chunk: makeChunk("doc#weak", "w"), score: MIN_SIMILARITY - 0.0001 },
    { chunk: makeChunk("doc#none", "n"), score: 0 },
  ];

  it("is a committed constant in the range cosine similarity can produce", () => {
    expect(typeof MIN_SIMILARITY).toBe("number");
    expect(Number.isFinite(MIN_SIMILARITY)).toBe(true);
    expect(MIN_SIMILARITY).toBeGreaterThan(0);
    expect(MIN_SIMILARITY).toBeLessThan(1);
  });

  it("puts a score exactly at the threshold above it", () => {
    const { above } = partitionByThreshold(results);
    expect(above.map((r) => r.chunk.chunkId)).toEqual(["doc#strong", "doc#exact"]);
  });

  it("keeps the weak results instead of dropping them", () => {
    // The refusal module needs to know a search happened and everything fell
    // short — which is a different fact from no search having happened.
    const { above, below } = partitionByThreshold(results);
    expect(below.map((r) => r.chunk.chunkId)).toEqual(["doc#weak", "doc#none"]);
    expect(above.length + below.length).toBe(results.length);
  });

  it("preserves ranked order within each side", () => {
    const ranked: SearchResult[] = [
      { chunk: makeChunk("doc#a", "a"), score: 0.8 },
      { chunk: makeChunk("doc#b", "b"), score: 0.7 },
      { chunk: makeChunk("doc#c", "c"), score: 0.2 },
      { chunk: makeChunk("doc#d", "d"), score: 0.1 },
    ];
    const { above, below } = partitionByThreshold(ranked, 0.5);
    expect(above.map((r) => r.score)).toEqual([0.8, 0.7]);
    expect(below.map((r) => r.score)).toEqual([0.2, 0.1]);
  });

  it("reports an empty above-side when every hit falls short", () => {
    const { above, below } = partitionByThreshold(results, 0.99);
    expect(above).toEqual([]);
    expect(below).toHaveLength(results.length);
  });

  it("partitions nothing into nothing", () => {
    expect(partitionByThreshold([])).toEqual({ above: [], below: [] });
  });
});

describe("end-to-end search over the real corpus", () => {
  it("ranks the Zone C freight chunk first for a Zone C freight question", async () => {
    const chunks = await loadCorpus(CORPUS_DIR);
    expect(chunks).toHaveLength(EXPECTED_CHUNK_COUNT);

    const embedder = createFakeEmbedder();
    const index = await buildIndex(chunks, embedder);
    const results = await search(index, "What is the Zone C freight surcharge?", embedder, 5);

    expect(results[0]?.chunk.chunkId).toBe("delivery-zones-and-schedules#zone-c-freight-surcharge");
    // The corpus is built with near-duplicate tempters; the runners-up should
    // be other passages, not the same one twice.
    expect(new Set(results.map((r) => r.chunk.chunkId)).size).toBe(results.length);
  });

  it("scores every hit no higher than the one before it", async () => {
    const chunks = await loadCorpus(CORPUS_DIR);
    const embedder = createFakeEmbedder();
    const index = await buildIndex(chunks, embedder);
    const results = await search(index, "rush order surcharge", embedder, 10);

    for (let i = 1; i < results.length; i += 1) {
      expect(results[i]?.score).toBeLessThanOrEqual(results[i - 1]?.score ?? 1);
    }
  });

  it("indexes the whole corpus as documents and the question as a query", async () => {
    const chunks = await loadCorpus(CORPUS_DIR);
    const embedder = createFakeEmbedder();
    const index = await buildIndex(chunks, embedder);
    await search(index, "How much is redelivery?", embedder, 3);

    expect(index.entries).toHaveLength(EXPECTED_CHUNK_COUNT);
    expect(embedder.calls[0]?.kind).toBe("document");
    expect(embedder.calls[0]?.texts).toHaveLength(EXPECTED_CHUNK_COUNT);
    expect(embedder.calls[1]?.kind).toBe("query");
    expect(embedder.calls[1]?.texts).toEqual(["How much is redelivery?"]);
  });

  it("returns the whole corpus, once, when k exceeds it", async () => {
    const chunks = await loadCorpus(CORPUS_DIR);
    const embedder = createFakeEmbedder();
    const index = await buildIndex(chunks, embedder);
    const results = await search(index, "anything at all", embedder, 10_000);

    expect(results).toHaveLength(EXPECTED_CHUNK_COUNT);
    expect(new Set(results.map((r) => r.chunk.chunkId)).size).toBe(EXPECTED_CHUNK_COUNT);
  });

  it("scores a question the corpus cannot answer below one it can", async () => {
    // Not a threshold calibration — the fake embedder is lexical, so its
    // absolute scores mean nothing. It is the ordering that matters: an
    // off-corpus question still gets a full top-k back, which is precisely why
    // MIN_SIMILARITY has to exist.
    const chunks = await loadCorpus(CORPUS_DIR);
    const embedder = createFakeEmbedder();
    const index = await buildIndex(chunks, embedder);

    const answerable = await search(index, "What is the Zone C freight surcharge?", embedder, 5);
    const unanswerable = await search(index, "Who won the 1998 World Cup final?", embedder, 5);

    expect(unanswerable).toHaveLength(5);
    expect(unanswerable[0]?.score).toBeLessThan(answerable[0]?.score ?? 0);
  });
});
