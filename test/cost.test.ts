/**
 * cost.test.ts — the pricing constants and the arithmetic over them, offline.
 *
 * Nothing here touches a network, a clock, or the environment: `src/cost.ts`
 * is a pure module and these tests are the proof. The prices themselves cannot
 * be tested — they are facts about two vendors' pricing pages on a stated date
 * — so what is tested instead is everything around them: that the arithmetic
 * is the arithmetic, that the formatter never renders a real cost as zero,
 * that the constants name the same models the CLI defaults to, and that the
 * as-of dates are dates.
 */

import { describe, expect, it } from "vitest";

import {
  EMBEDDING_PRICE_AS_OF,
  EMBEDDING_PRICED_MODEL,
  EMBEDDING_USD_PER_MILLION_TOKENS,
  GENERATION_INPUT_USD_PER_MILLION_TOKENS,
  GENERATION_OUTPUT_USD_PER_MILLION_TOKENS,
  GENERATION_PRICE_AS_OF,
  GENERATION_PRICED_MODEL,
  PRICING,
  computeEmbeddingCost,
  computeGenerationCost,
  formatUsd,
} from "../src/cost.js";
import { DEFAULT_EMBEDDING_MODEL, DEFAULT_GENERATION_MODEL } from "../src/eval-cli.js";

describe("cost arithmetic", () => {
  it("prices a million embedding tokens at exactly the committed constant", () => {
    expect(computeEmbeddingCost(1_000_000)).toBeCloseTo(EMBEDDING_USD_PER_MILLION_TOKENS, 12);
  });

  it("prices zero tokens at exactly zero, not a rounding artefact", () => {
    expect(computeEmbeddingCost(0)).toBe(0);
    expect(computeGenerationCost(0, 0)).toBe(0);
  });

  it("scales linearly, so per-question costs sum to the run total", () => {
    const a = computeEmbeddingCost(1_234);
    const b = computeEmbeddingCost(5_678);
    expect(a + b).toBeCloseTo(computeEmbeddingCost(1_234 + 5_678), 12);
  });

  it("prices input and output generation tokens at their own rates", () => {
    expect(computeGenerationCost(1_000_000, 0)).toBeCloseTo(
      GENERATION_INPUT_USD_PER_MILLION_TOKENS,
      12,
    );
    expect(computeGenerationCost(0, 1_000_000)).toBeCloseTo(
      GENERATION_OUTPUT_USD_PER_MILLION_TOKENS,
      12,
    );
    // A worked example a human can re-derive on paper: 200 in and 500 out at
    // $5/$25 per million is $0.001 + $0.0125.
    expect(computeGenerationCost(200, 500)).toBeCloseTo(
      (200 * GENERATION_INPUT_USD_PER_MILLION_TOKENS +
        500 * GENERATION_OUTPUT_USD_PER_MILLION_TOKENS) /
        1_000_000,
      12,
    );
  });
});

describe("the formatter", () => {
  it("renders at least six decimal places, never a bare $0.00", () => {
    // A single query embedding is a few dozen tokens — fractions of a cent.
    // Rounding that to $0.00 would be instrumentation that lies.
    const tiny = computeEmbeddingCost(150); // $0.000009
    expect(formatUsd(tiny)).toBe("$0.000009");
    expect(formatUsd(tiny)).not.toBe("$0.00");
  });

  it("renders zero as zero, at full width", () => {
    expect(formatUsd(0)).toBe("$0.000000");
  });

  it("renders ordinary amounts with the same six decimals", () => {
    expect(formatUsd(1.25)).toBe("$1.250000");
    expect(formatUsd(0.0135)).toBe("$0.013500");
  });

  it("falls back to exponential rather than rounding a real cost to zero", () => {
    // One embedding token costs $0.00000006 — beyond six decimals, but not
    // nothing. The formatter must say so rather than print $0.000000.
    const oneToken = computeEmbeddingCost(1);
    expect(oneToken).toBeGreaterThan(0);
    expect(formatUsd(oneToken)).not.toBe("$0.000000");
    expect(formatUsd(oneToken)).toContain("e-");
  });
});

describe("the constants", () => {
  it("price exactly the models eval-cli defaults to", () => {
    // The prices are per-model facts. If someone changes a default model in
    // eval-cli.ts without repricing this file, every cost figure goes quietly
    // wrong; this assertion is what makes that a red build instead.
    expect(EMBEDDING_PRICED_MODEL).toBe(DEFAULT_EMBEDDING_MODEL);
    expect(GENERATION_PRICED_MODEL).toBe(DEFAULT_GENERATION_MODEL);
  });

  it("carry as-of dates that are actual ISO dates", () => {
    expect(EMBEDDING_PRICE_AS_OF).toMatch(/^\d{4}-\d{2}-\d{2}$/);
    expect(GENERATION_PRICE_AS_OF).toMatch(/^\d{4}-\d{2}-\d{2}$/);
  });

  it("are mirrored faithfully into the snapshot the artifact stores", () => {
    expect(PRICING.embedding.model).toBe(EMBEDDING_PRICED_MODEL);
    expect(PRICING.embedding.usdPerMillionTokens).toBe(EMBEDDING_USD_PER_MILLION_TOKENS);
    expect(PRICING.embedding.asOf).toBe(EMBEDDING_PRICE_AS_OF);
    expect(PRICING.generation.model).toBe(GENERATION_PRICED_MODEL);
    expect(PRICING.generation.inputUsdPerMillionTokens).toBe(
      GENERATION_INPUT_USD_PER_MILLION_TOKENS,
    );
    expect(PRICING.generation.outputUsdPerMillionTokens).toBe(
      GENERATION_OUTPUT_USD_PER_MILLION_TOKENS,
    );
    expect(PRICING.generation.asOf).toBe(GENERATION_PRICE_AS_OF);
    // The sources are pinned pages, not a search away: a reader re-checking a
    // stale price should land where the constant was read.
    expect(PRICING.embedding.source).toMatch(/^https:\/\//);
    expect(PRICING.generation.source).toMatch(/^https:\/\//);
  });
});
