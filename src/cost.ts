/**
 * cost.ts — committed pricing constants and the arithmetic over them.
 *
 * PURE ON PURPOSE. No `process.env`, no `fetch`, no import of anything but
 * types. A module that phoned a pricing endpoint would produce a dollar figure
 * nobody could reproduce from the repository, and a module that read the price
 * from the environment would let a CI variable silently change what a stored
 * artifact appears to have cost. The prices live here, in source, under review,
 * with the date they were read — and every report that uses them prints that
 * date, because the one failure mode this file cannot prevent is going stale.
 *
 * THE CONSTANTS NAME THE MODELS THEY PRICE. These are the prices for the two
 * models committed as defaults in `eval-cli.ts` (`DEFAULT_EMBEDDING_MODEL`,
 * `DEFAULT_GENERATION_MODEL`). A run that overrides `VOYAGE_MODEL` or
 * `ANTHROPIC_MODEL` is costed at THESE prices anyway — the artifact records
 * both the model that ran and the model that was priced, so the mismatch is
 * visible in the file rather than silently wrong.
 */

/* ===========================================================================
 * Embedding: Voyage AI.
 * ======================================================================== */

/** The model these embedding prices are for. Must match `DEFAULT_EMBEDDING_MODEL`. */
export const EMBEDDING_PRICED_MODEL = "voyage-4";

/**
 * USD per one million input tokens for `voyage-3`.
 *
 * $0.06 / 1M tokens, AS OF 2026-08-27, read from the public pricing page
 * https://docs.voyageai.com/docs/pricing (Text Embeddings table).
 *
 * VOYAGE GRANTS 200 MILLION FREE TOKENS PER ACCOUNT FOR THIS MODEL, so a run
 * of this size may bill nothing at all. The per-token rate above is still the
 * applicable price and is what this module computes with: a free-tier credit
 * is an account-level fact that changes what you are invoiced, not what the
 * work costs, and a report that silently zeroed the figure would understate
 * the cost of running this pipeline for anyone past their allowance.
 *
 * PRICES CHANGE AND THIS FILE DOES NOT PHONE HOME. A stale constant produces a
 * wrong dollar figure silently — the arithmetic still runs, the report still
 * prints, and nothing anywhere throws. That is why the as-of date above is a
 * committed constant too, and why every report that renders a cost prints it:
 * the reader, not this module, is the one who can notice the date is old.
 */
export const EMBEDDING_USD_PER_MILLION_TOKENS = 0.06;

/** The date the embedding price above was read. Printed in every report. */
export const EMBEDDING_PRICE_AS_OF = "2026-08-27";

/** Where the embedding price was read from. */
export const EMBEDDING_PRICE_SOURCE = "https://docs.voyageai.com/docs/pricing";

/* ===========================================================================
 * Generation: Anthropic.
 * ======================================================================== */

/** The model these generation prices are for. Must match `DEFAULT_GENERATION_MODEL`. */
export const GENERATION_PRICED_MODEL = "claude-opus-5";

/**
 * USD per one million input tokens for `claude-opus-5`.
 *
 * $5.00 / 1M input tokens, AS OF 2026-08-27, read from the public pricing page
 * https://platform.claude.com/docs/en/about-claude/pricing ("Base Input
 * Tokens" column; this pipeline uses no prompt caching and no batching, so the
 * base rate is the applicable one).
 *
 * PRICES CHANGE AND THIS FILE DOES NOT PHONE HOME — same warning as the
 * embedding constant, for the same reason: a stale number here is a wrong
 * dollar figure with nothing loud about it, so the as-of date travels with
 * every figure computed from it.
 */
export const GENERATION_INPUT_USD_PER_MILLION_TOKENS = 5.0;

/**
 * USD per one million output tokens for `claude-opus-5`.
 *
 * $25.00 / 1M output tokens, AS OF 2026-08-27, read from the same page as the
 * input price: https://platform.claude.com/docs/en/about-claude/pricing.
 * The same staleness warning applies verbatim.
 */
export const GENERATION_OUTPUT_USD_PER_MILLION_TOKENS = 25.0;

/** The date the generation prices above were read. Printed in every report. */
export const GENERATION_PRICE_AS_OF = "2026-08-27";

/** Where the generation prices were read from. */
export const GENERATION_PRICE_SOURCE =
  "https://platform.claude.com/docs/en/about-claude/pricing";

/* ===========================================================================
 * The snapshot that travels with a run.
 * ======================================================================== */

/**
 * Every pricing constant, as one value, so an artifact can carry the prices it
 * was costed under. SAME PRINCIPLE AS STORING THE POLICY BESIDE THE VERDICT:
 * a stored dollar figure without the per-token prices it was computed from is
 * not a record, it is a rumour — six months from now these constants will have
 * moved, and a cost read from an old artifact would be silently reinterpreted
 * against prices that were not in force when it was measured.
 */
export interface PricingSnapshot {
  readonly embedding: {
    readonly model: string;
    readonly usdPerMillionTokens: number;
    readonly asOf: string;
    readonly source: string;
  };
  readonly generation: {
    readonly model: string;
    readonly inputUsdPerMillionTokens: number;
    readonly outputUsdPerMillionTokens: number;
    readonly asOf: string;
    readonly source: string;
  };
}

/** The committed constants above, assembled for the artifact. */
export const PRICING: PricingSnapshot = {
  embedding: {
    model: EMBEDDING_PRICED_MODEL,
    usdPerMillionTokens: EMBEDDING_USD_PER_MILLION_TOKENS,
    asOf: EMBEDDING_PRICE_AS_OF,
    source: EMBEDDING_PRICE_SOURCE,
  },
  generation: {
    model: GENERATION_PRICED_MODEL,
    inputUsdPerMillionTokens: GENERATION_INPUT_USD_PER_MILLION_TOKENS,
    outputUsdPerMillionTokens: GENERATION_OUTPUT_USD_PER_MILLION_TOKENS,
    asOf: GENERATION_PRICE_AS_OF,
    source: GENERATION_PRICE_SOURCE,
  },
};

/* ===========================================================================
 * Arithmetic.
 * ======================================================================== */

const ONE_MILLION = 1_000_000;

/** USD cost of embedding `tokens` input tokens at the committed price. */
export function computeEmbeddingCost(tokens: number): number {
  return (tokens / ONE_MILLION) * EMBEDDING_USD_PER_MILLION_TOKENS;
}

/** USD cost of one or more generation calls at the committed prices. */
export function computeGenerationCost(inputTokens: number, outputTokens: number): number {
  return (
    (inputTokens / ONE_MILLION) * GENERATION_INPUT_USD_PER_MILLION_TOKENS +
    (outputTokens / ONE_MILLION) * GENERATION_OUTPUT_USD_PER_MILLION_TOKENS
  );
}

/**
 * Render a USD amount for a report.
 *
 * SIX DECIMAL PLACES, NOT TWO. A per-question cost here is a fraction of a
 * cent — a single query embedding is on the order of $0.000001 — and a
 * formatter that rendered it as `$0.00` would be instrumentation that lies:
 * the measurement happened, the number exists, and the report would say zero.
 *
 * A nonzero amount too small even for six decimals is rendered in exponential
 * notation rather than rounded to `$0.000000`, for exactly the same reason.
 */
export function formatUsd(amount: number): string {
  if (amount === 0) return "$0.000000";
  const fixed = amount.toFixed(6);
  if (Number(fixed) !== 0) return `$${fixed}`;
  return `$${amount.toExponential(2)}`;
}
