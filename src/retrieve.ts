/**
 * retrieve.ts — chunks in, ranked candidates out.
 *
 * WHY THIS FILE EXISTS AT ALL. Anthropic publishes no embedding model of its
 * own and names Voyage AI as its recommended embedding provider. So retrieval
 * cannot be a thin wrapper over the Anthropic SDK the way generation will be:
 * it needs a second vendor, a second key, and a second failure surface. That is
 * this module's whole job — to keep that second vendor behind one small
 * interface (`Embedder`) so that everything downstream depends on the shape of
 * an embedding call and never on Voyage.
 *
 * NO DEPENDENCIES. No vector database, no HTTP client, no math library. The
 * index is an array, the transport is `fetch`, and cosine similarity is eight
 * lines below. At corpus scale (dozens of documents) a vector library buys
 * nothing but a dependency, and it hides the one number this project is
 * actually about: how similar the best match really was. That number decides
 * whether we answer or refuse, so it stays in plain sight.
 *
 * EVERYTHING IS INJECTED. `Embedder` is a parameter everywhere; nothing in this
 * file constructs one implicitly. `voyageEmbedder` takes its `fetch` as a
 * parameter too. Consequently the tests run the real ranking code over the real
 * corpus with no network and no secrets — see `test/fake-embedder.ts`.
 *
 * THE API KEY IS NEVER IN AN ERROR. It arrives as a function parameter (never
 * from `process.env`, which this module does not read), it is written into
 * exactly one place — the `Authorization` header — and it is never interpolated
 * into a message, attached to a thrown object, or captured in one. Errors from
 * this module are logged, serialised, and pasted into issue trackers; a key
 * that reaches an error message has been published. `test/retrieve.test.ts`
 * asserts a decoy key's absence from both `error.message` and
 * `JSON.stringify(error)`.
 */

import type { Chunk } from "./chunk.js";

/**
 * Which side of the search the text is on.
 *
 * This is not decoration. Embedding providers accept an input-type hint and
 * embed documents and queries asymmetrically — a query ("what is the Zone C
 * surcharge?") and the passage that answers it are not the same kind of text,
 * and models trained for retrieval exploit that. Passing the wrong hint, or
 * none, measurably degrades ranking while failing silently: every call still
 * returns vectors of the right shape and every search still returns results.
 * So the distinction is in the interface, where it cannot be forgotten, rather
 * than in an options bag where it can.
 */
export type EmbeddingKind = "document" | "query";

/**
 * The one thing retrieval needs from an embedding provider.
 *
 * Contract, binding on every implementation:
 *   - returns exactly one vector per input text, in input order;
 *   - returns vectors of a consistent length for a given implementation;
 *   - never pads, truncates, reorders, or substitutes a placeholder vector.
 * `buildIndex` enforces the first two rather than trusting them, because a
 * silently misaligned vector array attaches every chunk's text to another
 * chunk's meaning, and the result is a system that cites confidently and
 * wrongly.
 */
export interface Embedder {
  embed(texts: string[], kind: EmbeddingKind): Promise<number[][]>;
}

/**
 * Cumulative usage across every completed call an embedder has made.
 *
 * CUMULATIVE, NOT PER-CALL, on purpose: a caller that wants to attribute usage
 * to one unit of work (one eval question, say) snapshots before and after and
 * takes the difference, which composes without the embedder having to know
 * what a "question" is.
 */
export interface EmbeddingUsage {
  /** Sum of the provider-reported `total_tokens` over every completed call. */
  readonly totalTokens: number;
  /** Completed calls — batches, in `buildIndex` terms. */
  readonly calls: number;
  /** Wall-clock milliseconds across those calls, via the injected clock. */
  readonly totalLatencyMs: number;
}

/**
 * An embedder that also reports what its calls cost in tokens and time.
 *
 * A SEPARATE INTERFACE rather than a change to `Embedder`, so that every
 * existing implementation and every test fake keeps compiling unchanged —
 * `search` and `buildIndex` need vectors, not accounting, and their signatures
 * say so. Code that wants the accounting asks for this type explicitly.
 */
export interface InstrumentedEmbedder extends Embedder {
  usage(): EmbeddingUsage;
}

/** Every way embedding can fail. Each one is covered by a test. */
export type EmbeddingErrorKind =
  /** The provider answered with a non-2xx status. */
  | "http-status"
  /** `fetch` itself rejected — DNS, TLS, connection reset, abort. */
  | "transport"
  /** The response was not JSON, or not the shape the endpoint documents. */
  | "malformed-response"
  /** Vector count does not equal input count. Never padded, never truncated. */
  | "vector-count-mismatch"
  /** Two vectors of different lengths, so cosine similarity is meaningless. */
  | "dimension-mismatch";

/**
 * Thrown for every embedding failure.
 *
 * Carries a machine-readable `kind` and, for HTTP failures, the `status`.
 * Deliberately carries no request, no config, and no headers: the narrow
 * constructor is what makes "the key is never in the error" checkable by
 * reading this file rather than by auditing every call site.
 */
export class EmbeddingError extends Error {
  readonly kind: EmbeddingErrorKind;
  /** HTTP status for `kind: "http-status"`, `undefined` otherwise. */
  readonly status: number | undefined;

  constructor(kind: EmbeddingErrorKind, message: string, status?: number) {
    super(message);
    this.name = "EmbeddingError";
    this.kind = kind;
    this.status = status;
  }

  /**
   * `Error` has no useful JSON form — `JSON.stringify(new Error("x"))` is
   * `"{}"`, because `message` is not enumerable. Anything that logs errors as
   * JSON therefore drops the message unless a class opts in, so this class opts
   * in. The listed fields are the whole payload: naming them explicitly is what
   * guarantees a future field cannot ride along into a log, and it is what the
   * decoy-key serialisation test actually exercises.
   */
  toJSON(): { name: string; kind: EmbeddingErrorKind; status: number | undefined; message: string } {
    return { name: this.name, kind: this.kind, status: this.status, message: this.message };
  }
}

/* ===========================================================================
 * The Voyage implementation.
 * ======================================================================== */

/** Voyage's embeddings endpoint. */
export const VOYAGE_ENDPOINT = "https://api.voyageai.com/v1/embeddings";

/**
 * Everything `voyageEmbedder` needs, and nothing it can pick up on its own.
 *
 * `apiKey` is a parameter because this module does not read `process.env`:
 * a module that reaches for ambient credentials works in production and in
 * tests for different reasons, and the day it stops working the difference
 * matters. Whoever owns the key passes the key.
 *
 * `fetchImpl` is injected for the same reason — so the tests below exercise
 * this exact request-building and response-validating code without a network,
 * a key, or a live account. A test that has to mock a global to run is a test
 * that stops running the day someone changes the global.
 */
export interface VoyageConfig {
  /** Never logged, never interpolated, never attached to an error. */
  readonly apiKey: string;
  /** Voyage model id, e.g. `voyage-3`. Required: the threshold below is only
   *  meaningful for one model, so the model is never an implicit default. */
  readonly model: string;
  /** Defaults to `globalThis.fetch`. */
  readonly fetchImpl?: typeof globalThis.fetch;
  /**
   * Millisecond clock for latency measurement. Defaults to `Date.now`.
   * Injected so a test can assert an exact latency instead of a range —
   * a latency assertion with slack in it is an assertion that never fails.
   */
  readonly clock?: () => number;
}

/** One entry of the endpoint's `data` array, once validated. */
interface VoyageDatum {
  readonly index: number;
  readonly embedding: number[];
}

function isRecord(value: unknown): value is Record<string, unknown> {
  return typeof value === "object" && value !== null;
}

/** Validate one `data` entry. Returns null if it is not the documented shape. */
function readDatum(entry: unknown): VoyageDatum | null {
  if (!isRecord(entry)) return null;
  const index = entry["index"];
  const embedding = entry["embedding"];
  if (typeof index !== "number" || !Number.isInteger(index)) return null;
  if (!Array.isArray(embedding)) return null;
  for (const component of embedding) {
    // A NaN or Infinity anywhere poisons every cosine score it touches, and it
    // does so quietly — the sort just goes strange. Reject it at the boundary.
    if (typeof component !== "number" || !Number.isFinite(component)) return null;
  }
  return { index, embedding: embedding as number[] };
}

/**
 * A live embedder backed by Voyage AI.
 *
 * Request shape: `POST` with `{ input, model, input_type }`, bearer auth.
 * Response shape: `{ data: [{ index, embedding }, ...] }`, which this reorders
 * by `index` rather than trusting array order.
 *
 * Batch limits are the provider's, not ours: a call with more texts than the
 * endpoint accepts comes back as a non-2xx and surfaces here as an
 * `EmbeddingError` naming the status. That is loud, which is the requirement.
 * Chunking large corpora into batches belongs to whoever knows the limits of
 * the model they picked.
 *
 * USAGE IS COUNTED ONLY FOR COMPLETED CALLS. A call that throws contributes
 * nothing to the counters: every caller of this embedder aborts the run on an
 * `EmbeddingError`, so a partially-counted failed call would be a number
 * nobody ever reads — and a number that WAS read would misattribute a failed
 * request's time to whatever question happened to be running.
 */
export function voyageEmbedder(config: VoyageConfig): InstrumentedEmbedder {
  const doFetch = config.fetchImpl ?? globalThis.fetch;
  const clock = config.clock ?? Date.now;

  let totalTokens = 0;
  let calls = 0;
  let totalLatencyMs = 0;

  return {
    usage(): EmbeddingUsage {
      return { totalTokens, calls, totalLatencyMs };
    },

    async embed(texts: string[], kind: EmbeddingKind): Promise<number[][]> {
      // Nothing to embed is not an error, and it is not a request either:
      // the endpoint rejects an empty input array, so asking would turn a
      // no-op into a 400. No request, so no usage is counted.
      if (texts.length === 0) return [];

      const startedAt = clock();

      let response: Response;
      try {
        response = await doFetch(VOYAGE_ENDPOINT, {
          method: "POST",
          headers: {
            // The one and only place the key is used.
            authorization: `Bearer ${config.apiKey}`,
            "content-type": "application/json",
          },
          body: JSON.stringify({ input: texts, model: config.model, input_type: kind }),
        });
      } catch {
        // The underlying error is deliberately not attached as `cause`: it is
        // the transport's own object, and some transports put the whole
        // request — headers included — inside it.
        throw new EmbeddingError(
          "transport",
          `the embedding request to ${VOYAGE_ENDPOINT} could not be completed`,
        );
      }

      if (!response.ok) {
        // Status and status text only. The response body is not interpolated,
        // because an upstream error body is not under our control and can echo
        // the request — including the header the key travels in.
        throw new EmbeddingError(
          "http-status",
          `the embedding provider answered ${response.status} ${response.statusText} for ${texts.length} ${kind} ${texts.length === 1 ? "input" : "inputs"}`,
          response.status,
        );
      }

      let body: unknown;
      try {
        body = await response.json();
      } catch {
        throw new EmbeddingError("malformed-response", "the embedding response was not valid JSON");
      }

      if (!isRecord(body) || !Array.isArray(body["data"])) {
        throw new EmbeddingError(
          "malformed-response",
          'the embedding response has no "data" array',
        );
      }
      const data: unknown[] = body["data"];

      // Count first, and never reconcile it. Padding with a zero vector or
      // truncating to the shorter list would leave every later chunk holding
      // the vector of a different chunk — retrieval would still return results,
      // and every citation after the gap would be for the wrong text.
      if (data.length !== texts.length) {
        throw new EmbeddingError(
          "vector-count-mismatch",
          `the embedding provider returned ${data.length} ${data.length === 1 ? "vector" : "vectors"} for ${texts.length} ${texts.length === 1 ? "input" : "inputs"}; refusing to pad or truncate, because a misaligned vector array attaches each chunk's text to another chunk's meaning`,
        );
      }

      const vectors: (number[] | undefined)[] = new Array<number[] | undefined>(texts.length);
      for (const entry of data) {
        const datum = readDatum(entry);
        if (datum === null) {
          throw new EmbeddingError(
            "malformed-response",
            'an entry of the embedding response is not { index: integer, embedding: finite number[] }',
          );
        }
        if (datum.index < 0 || datum.index >= texts.length || vectors[datum.index] !== undefined) {
          throw new EmbeddingError(
            "malformed-response",
            `the embedding response has an out-of-range or repeated index ${datum.index} for ${texts.length} ${texts.length === 1 ? "input" : "inputs"}`,
          );
        }
        vectors[datum.index] = datum.embedding;
      }

      const ordered: number[][] = [];
      for (let i = 0; i < texts.length; i += 1) {
        const vector = vectors[i];
        if (vector === undefined) {
          throw new EmbeddingError(
            "malformed-response",
            `the embedding response has no vector at index ${i}`,
          );
        }
        ordered.push(vector);
      }

      // The call completed; count it. `usage.total_tokens` is read tolerantly:
      // a response that omits it (or mangles it) is still a valid batch of
      // vectors, and failing the whole run over a missing accounting field
      // would let instrumentation break the thing it instruments. The cost of
      // the tolerance is an undercount, which at least errs toward a smaller
      // number rather than an invented one.
      const usage = body["usage"];
      const reported = isRecord(usage) ? usage["total_tokens"] : undefined;
      if (typeof reported === "number" && Number.isFinite(reported)) {
        totalTokens += reported;
      }
      calls += 1;
      totalLatencyMs += clock() - startedAt;

      return ordered;
    },
  };
}

/* ===========================================================================
 * The index.
 * ======================================================================== */

/** One chunk and the vector standing in for it. */
export interface IndexEntry {
  readonly chunk: Chunk;
  readonly vector: readonly number[];
}

/** The whole index: an array, on purpose. See the header comment. */
export interface VectorIndex {
  readonly entries: readonly IndexEntry[];
  /** Vector length shared by every entry; 0 for an empty index. */
  readonly dimensions: number;
}

/** One hit, with the number that decides whether it is good enough. */
export interface SearchResult {
  readonly chunk: Chunk;
  /** Cosine similarity in [-1, 1]; 1 is identical direction. */
  readonly score: number;
}

/**
 * Cosine similarity: dot product over the product of magnitudes.
 *
 * The zero-magnitude case is guarded explicitly rather than left to produce
 * `0/0`. A `NaN` score is worse than a wrong score: it is not greater than or
 * less than anything, so it neither sorts nor compares — it would slip past the
 * `>= MIN_SIMILARITY` floor as "below threshold" in one place and land in an
 * arbitrary sort position in another, and nothing would throw. Returning 0
 * says what is actually true of a vector with no direction: there is no
 * evidence of relatedness here, so treat it as the floor treats anything else.
 */
export function cosineSimilarity(a: readonly number[], b: readonly number[]): number {
  if (a.length !== b.length) {
    throw new EmbeddingError(
      "dimension-mismatch",
      `cannot compare a ${a.length}-dimensional vector with a ${b.length}-dimensional one`,
    );
  }
  let dot = 0;
  let magnitudeA = 0;
  let magnitudeB = 0;
  for (let i = 0; i < a.length; i += 1) {
    const x = a[i] ?? 0;
    const y = b[i] ?? 0;
    dot += x * y;
    magnitudeA += x * x;
    magnitudeB += y * y;
  }
  if (magnitudeA === 0 || magnitudeB === 0) return 0;
  return dot / (Math.sqrt(magnitudeA) * Math.sqrt(magnitudeB));
}

/**
 * Embed every chunk as a document and hold the vectors alongside the chunks.
 *
 * The embedder's contract is enforced here, not assumed: one vector per chunk,
 * all of the same length. Both violations are silent by nature — the search
 * still runs, the results still look like results — so they are checked once,
 * at the only point where the chunk-to-vector correspondence is established.
 */
export async function buildIndex(chunks: readonly Chunk[], embedder: Embedder): Promise<VectorIndex> {
  if (chunks.length === 0) return { entries: [], dimensions: 0 };

  const vectors = await embedder.embed(
    chunks.map((chunk) => chunk.text),
    "document",
  );

  if (vectors.length !== chunks.length) {
    throw new EmbeddingError(
      "vector-count-mismatch",
      `the embedder returned ${vectors.length} ${vectors.length === 1 ? "vector" : "vectors"} for ${chunks.length} chunks`,
    );
  }

  const entries: IndexEntry[] = [];
  let dimensions = -1;
  for (let i = 0; i < chunks.length; i += 1) {
    const chunk = chunks[i];
    const vector = vectors[i];
    if (chunk === undefined || vector === undefined) continue;
    if (dimensions === -1) dimensions = vector.length;
    if (vector.length !== dimensions) {
      throw new EmbeddingError(
        "dimension-mismatch",
        `the embedder returned a ${vector.length}-dimensional vector for chunk "${chunk.chunkId}" after returning ${dimensions}-dimensional ones`,
      );
    }
    entries.push({ chunk, vector });
  }

  return { entries, dimensions: dimensions === -1 ? 0 : dimensions };
}

/**
 * Rank the index against a query and return the best `k`.
 *
 * Ties are broken by `chunkId`, ascending, so that two chunks with equal scores
 * always come back in the same order. Equal scores are not exotic — the corpus
 * is deliberately full of near-duplicate passages — and an unstable order there
 * would mean the same question yields different citations on different runs,
 * with nothing in the output to explain why.
 */
export async function search(
  index: VectorIndex,
  queryText: string,
  embedder: Embedder,
  k: number,
): Promise<SearchResult[]> {
  if (k <= 0 || index.entries.length === 0) return [];

  const vectors = await embedder.embed([queryText], "query");
  const queryVector = vectors[0];
  if (vectors.length !== 1 || queryVector === undefined) {
    throw new EmbeddingError(
      "vector-count-mismatch",
      `the embedder returned ${vectors.length} vectors for a single query`,
    );
  }

  const scored: SearchResult[] = index.entries.map((entry) => ({
    chunk: entry.chunk,
    score: cosineSimilarity(queryVector, entry.vector),
  }));

  scored.sort((a, b) => {
    if (a.score !== b.score) return b.score - a.score;
    // Codepoint order, not locale collation: the tie-break has to be identical
    // on every machine, and `localeCompare` is not.
    return a.chunk.chunkId < b.chunk.chunkId ? -1 : a.chunk.chunkId > b.chunk.chunkId ? 1 : 0;
  });

  // `k` larger than the corpus returns the corpus. Nothing is invented to fill
  // the gap, and the caller learns the true count from the array's length.
  return scored.slice(0, k);
}

/* ===========================================================================
 * The threshold.
 * ======================================================================== */

/**
 * The similarity floor below which a hit is not evidence of anything.
 *
 * WHY A THRESHOLD EXISTS AT ALL. Every vector has some similarity to every
 * query. Cosine similarity is defined for any two vectors, so a top-k search
 * always succeeds: ask this corpus who won the 1998 World Cup and it will
 * cheerfully hand back five passages about freight surcharges, ranked, with
 * scores. Without a floor, "the corpus does not contain the answer" is a state
 * the retrieval layer can never report — there is always a best match, so
 * there is always something to build an answer out of, and the model will
 * build one. The refusal path in a RAG system is not a model behaviour to be
 * prompted for; it is a number compared against a constant, right here.
 *
 * WHAT THE NUMBER MEANS. A result scoring at or above this is treated as
 * plausibly on-topic and is worth showing to the model; below it, the search
 * found nothing, however many rows it returned.
 *
 * THE SPECIFIC NUMBER IS PROVISIONAL. 0.35 is a starting point, not a measured
 * result. It was chosen to sit above where unrelated text lands and below where
 * a genuine topical match lands for a general-purpose retrieval embedding
 * model, and that is the entire derivation — no calibration set has been run
 * against it yet. Treat it as a placeholder that fails in the safe direction:
 * too high refuses answerable questions (visible, annoying, harmless), too low
 * lets an ungrounded answer through (invisible, and the thing this project
 * exists to prevent).
 *
 * IT IS A PROPERTY OF THE EMBEDDER, NOT OF THE CORPUS. Different models put
 * their "unrelated" mass at different similarities. Changing the embedding
 * model invalidates this constant outright; it does not merely shift it.
 *
 * TODO(eval): `src/eval.ts` is the module that will replace this guess with a
 * measured value. It should run the eval question set — including the question
 * the corpus deliberately cannot answer — against the live embedder, report the
 * score distribution of correct hits against wrong ones, and pick the floor
 * that separates them. Until that lands, this number is a guess with a comment
 * on it, and it should be read as one.
 */
export const MIN_SIMILARITY = 0.35;

/** Search results split by the floor. Both sides keep their ranked order. */
export interface ThresholdPartition {
  /** Scored at or above the threshold, best first. */
  readonly above: readonly SearchResult[];
  /** Scored below it, best first. Kept, not discarded. */
  readonly below: readonly SearchResult[];
}

/**
 * Split results at the threshold instead of filtering.
 *
 * The weak hits are returned, not dropped, because "we searched and everything
 * fell short" and "we never searched" are different facts and the refusal path
 * needs to tell them apart. A refusal that can say *what* it nearly matched,
 * and how close it came, is debuggable and can be shown to a user; a bare empty
 * array is indistinguishable from a broken index. Deciding what to do with each
 * side is the caller's job — this function does not decide anything.
 */
export function partitionByThreshold(
  results: readonly SearchResult[],
  threshold: number = MIN_SIMILARITY,
): ThresholdPartition {
  const above: SearchResult[] = [];
  const below: SearchResult[] = [];
  for (const result of results) {
    if (result.score >= threshold) above.push(result);
    else below.push(result);
  }
  return { above, below };
}
