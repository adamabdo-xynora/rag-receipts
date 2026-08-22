/**
 * fake-embedder.ts — a deterministic `Embedder` for tests. Not exported from
 * `src`, because nothing in production should be able to reach it by accident.
 *
 * WHY NOT HARDCODED VECTORS. A fixture of hand-written vectors tests the sort,
 * not the retrieval: the ranking is decided by whoever typed the numbers, so
 * the assertions pass whatever cosine similarity happens to do. This embedder
 * is a real (if crude) bag-of-words hashing embedder instead — lexically
 * similar texts genuinely land closer together — so a test that says "the Zone
 * C freight query ranks the Zone C freight chunk first" is a claim about the
 * ranking code operating on the real corpus, and it can actually fail.
 *
 * WHAT IT IS. Unigrams and adjacent-word bigrams, hashed by FNV-1a into a fixed
 * number of buckets, accumulated as term frequencies. Bigrams are what make
 * "Zone C" different from "Zone A" plus a stray "C" — without them every
 * passage mentioning any zone looks alike, which is the sort of confusion the
 * near-duplicate documents in the corpus are designed to cause.
 *
 * WHAT IT IS NOT. Not semantic: it knows nothing of synonyms, and a paraphrase
 * with no shared words scores near zero. Tests that need meaning rather than
 * vocabulary do not belong here.
 */

import type { Embedder, EmbeddingKind } from "../src/retrieve.js";

/** Small enough to stay cheap, large enough that collisions stay rare. */
export const FAKE_DIMENSIONS = 512;

/** One recorded call, so tests can assert which input-type hint was sent. */
export interface FakeEmbedderCall {
  readonly texts: readonly string[];
  readonly kind: EmbeddingKind;
}

export interface FakeEmbedder extends Embedder {
  /** Every call in order. The whole point of the document/query split is that
   *  the hint reaches the provider, so tests need to be able to see it. */
  readonly calls: readonly FakeEmbedderCall[];
}

/** FNV-1a, 32-bit. Chosen for being four lines and identical everywhere. */
function hash(token: string): number {
  let value = 0x811c9dc5;
  for (let i = 0; i < token.length; i += 1) {
    value ^= token.charCodeAt(i);
    value = Math.imul(value, 0x01000193) >>> 0;
  }
  return value;
}

/** Lowercase, then every run of non-alphanumerics is a boundary. */
export function tokenize(text: string): string[] {
  return text
    .toLowerCase()
    .split(/[^a-z0-9]+/)
    .filter((token) => token !== "");
}

/** Term frequencies over unigrams and bigrams, hashed into fixed buckets. */
export function fakeVector(text: string): number[] {
  const vector = new Array<number>(FAKE_DIMENSIONS).fill(0);
  const tokens = tokenize(text);

  const add = (term: string, weight: number): void => {
    const bucket = hash(term) % FAKE_DIMENSIONS;
    vector[bucket] = (vector[bucket] ?? 0) + weight;
  };

  for (let i = 0; i < tokens.length; i += 1) {
    const token = tokens[i];
    if (token === undefined) continue;
    add(`1:${token}`, 1);
    const next = tokens[i + 1];
    // Bigrams are prefixed into their own key space so "zone c" can never
    // collide with a unigram that happens to hash the same way.
    if (next !== undefined) add(`2:${token} ${next}`, 1);
  }

  return vector;
}

/**
 * Build one. Deterministic, synchronous underneath, no network.
 *
 * The `kind` hint is recorded but does not change the vector: a fake that
 * embedded queries into a different space would decide the ranking by fiat,
 * which is exactly what this file exists to avoid. Asymmetry is a real
 * provider's behaviour, and it is tested where it is real — against the
 * injected `fetch` in `voyageEmbedder`.
 */
export function createFakeEmbedder(): FakeEmbedder {
  const calls: FakeEmbedderCall[] = [];
  return {
    calls,
    async embed(texts: string[], kind: EmbeddingKind): Promise<number[][]> {
      calls.push({ texts: [...texts], kind });
      return texts.map(fakeVector);
    },
  };
}
