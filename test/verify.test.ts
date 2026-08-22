import { fileURLToPath } from "node:url";
import { describe, expect, it } from "vitest";

import { type Chunk, loadCorpus } from "../src/chunk.js";
import type { SearchResult } from "../src/retrieve.js";
import {
  type Claim,
  type ClaimStatus,
  type RejectedClaim,
  type RejectionKind,
  type SupportingQuote,
  type VerifiedAnswer,
  retrievedChunks,
  summarise,
  verifyAnswer,
} from "../src/verify.js";

const CORPUS_DIR = fileURLToPath(new URL("../corpus", import.meta.url));

/* ===========================================================================
 * Fixtures come out of the corpus on disk, never out of this file.
 * ===========================================================================
 * Every quote below is sliced from text `loadCorpus` read off disk, so it is
 * byte-exact by construction. Hand-typing chunk text into a test would test the
 * typing: a stray smart quote or collapsed double space would make a "valid
 * claim passes" test fail, or — far worse — a hand-typed *near* match would let
 * a containment test pass for the wrong reason and quietly certify a verifier
 * that had gone fuzzy.
 *
 * The one thing this file does author is the invented text: paraphrases,
 * fabricated ids. That is the point — that half is supposed to be absent from
 * the corpus, and each test asserts its absence before relying on it.
 * ======================================================================== */

/** Real chunks, all from one document so the near-miss cases are plausible. */
const ZONE_C = "delivery-zones-and-schedules#zone-c-freight-surcharge";
const REDELIVERY = "delivery-zones-and-schedules#redelivery-after-a-failed-attempt";
const DEPOT = "delivery-zones-and-schedules#depot-collection";

/**
 * A fabricated id shaped exactly like a real one: real document stem, real
 * naming convention, a section that does not exist. This is what an invented
 * citation actually looks like — nothing about it reads as wrong.
 */
const INVENTED = "delivery-zones-and-schedules#zone-d-freight-surcharge";

async function corpus(): Promise<Chunk[]> {
  return loadCorpus(CORPUS_DIR);
}

/** Look a chunk up, failing with a useful message if the corpus moved. */
function chunkById(chunks: readonly Chunk[], chunkId: string): Chunk {
  const chunk = chunks.find((candidate) => candidate.chunkId === chunkId);
  expect(chunk, `the corpus no longer contains "${chunkId}"`).toBeDefined();
  return chunk as Chunk;
}

/**
 * The sentence of `chunk` containing `marker`, sliced out of the chunk's own
 * text — so the returned string is a byte-exact substring by construction, not
 * by transcription. Asserts as much before returning.
 */
function sentenceWith(chunk: Chunk, marker: string): string {
  const at = chunk.text.indexOf(marker);
  expect(at, `"${marker}" is no longer in ${chunk.chunkId}`).toBeGreaterThanOrEqual(0);

  const previousStop = chunk.text.lastIndexOf(". ", at);
  const start = previousStop === -1 ? 0 : previousStop + 2;
  const stop = chunk.text.indexOf(".", at);
  const end = stop === -1 ? chunk.text.length : stop + 1;

  const sentence = chunk.text.slice(start, end);
  expect(sentence.length).toBeGreaterThan(0);
  expect(chunk.text).toContain(sentence);
  return sentence;
}

/**
 * Change exact words in a real sentence, asserting each one was actually there
 * and appears once. A silent no-op replacement would turn a paraphrase test
 * into a test that a verbatim quote verifies — it would pass, for the opposite
 * of the intended reason.
 */
function withWordsChanged(
  sentence: string,
  swaps: readonly (readonly [string, string])[],
): string {
  let changed = sentence;
  for (const [from, to] of swaps) {
    expect(changed.split(from), `"${from}" must appear exactly once`).toHaveLength(2);
    changed = changed.replace(from, to);
  }
  expect(changed).not.toBe(sentence);
  return changed;
}

/** Default claim text; irrelevant to every rule except the empty-text one. */
const CLAIM_TEXT = "Every Zone C drop carries a flat freight surcharge of $45.";

function makeClaim(
  status: ClaimStatus,
  citations: readonly string[],
  supportingQuotes: readonly SupportingQuote[],
  text: string = CLAIM_TEXT,
): Claim {
  return { text, status, citations, supportingQuotes };
}

/** Assert exactly one claim was rejected, for exactly `kind`, and return it. */
function expectSoleRejection(answer: VerifiedAnswer, kind: RejectionKind): RejectedClaim {
  expect(
    answer.rejected.map((entry) => entry.reason.kind),
    `expected a single rejection of kind "${kind}"`,
  ).toEqual([kind]);
  expect(answer.verified).toHaveLength(0);
  return answer.rejected[0] as RejectedClaim;
}

/* ===========================================================================
 * Shape.
 * ======================================================================== */

describe("shape rules, checked before anything looks at the retrieved set", () => {
  it("rejects a stated claim citing two chunks", async () => {
    const chunks = await corpus();
    const zoneC = chunkById(chunks, ZONE_C);
    const redelivery = chunkById(chunks, REDELIVERY);
    const surcharge = sentenceWith(zoneC, "freight surcharge of $45");
    const fee = sentenceWith(redelivery, "$32 per drop");

    // Both quotes are real and both chunks were retrieved: nothing here fails a
    // content check. The claim is rejected purely for what its label promises.
    const answer = verifyAnswer(
      [
        makeClaim(
          "stated",
          [ZONE_C, REDELIVERY],
          [
            { chunkId: ZONE_C, quote: surcharge },
            { chunkId: REDELIVERY, quote: fee },
          ],
        ),
      ],
      [zoneC, redelivery],
    );

    const rejection = expectSoleRejection(answer, "stated-claim-multiple-citations");
    expect(rejection.reason.message).toContain("stated");
    expect(rejection.reason.message).toContain(CLAIM_TEXT);
  });

  it("rejects an inferred claim citing one chunk", async () => {
    const chunks = await corpus();
    const zoneC = chunkById(chunks, ZONE_C);
    const surcharge = sentenceWith(zoneC, "freight surcharge of $45");

    const answer = verifyAnswer(
      [makeClaim("inferred", [ZONE_C], [{ chunkId: ZONE_C, quote: surcharge }])],
      [zoneC],
    );

    const rejection = expectSoleRejection(answer, "inferred-claim-single-citation");
    expect(rejection.reason.message).toContain("inferred");
  });

  it("rejects a single-parent inference dressed up as two citations", async () => {
    // ["a", "a"] would satisfy "two or more" on a technicality, and the quote
    // and chunk are both real, so nothing downstream would object.
    const chunks = await corpus();
    const zoneC = chunkById(chunks, ZONE_C);
    const surcharge = sentenceWith(zoneC, "freight surcharge of $45");

    const answer = verifyAnswer(
      [makeClaim("inferred", [ZONE_C, ZONE_C], [{ chunkId: ZONE_C, quote: surcharge }])],
      [zoneC],
    );

    expectSoleRejection(answer, "duplicate-citation");
  });

  it("rejects an empty claim text", async () => {
    const chunks = await corpus();
    const zoneC = chunkById(chunks, ZONE_C);
    const surcharge = sentenceWith(zoneC, "freight surcharge of $45");

    const answer = verifyAnswer(
      [makeClaim("stated", [ZONE_C], [{ chunkId: ZONE_C, quote: surcharge }], "   ")],
      [zoneC],
    );

    expectSoleRejection(answer, "empty-claim-text");
  });

  it("rejects an empty citation array", async () => {
    const chunks = await corpus();
    const answer = verifyAnswer([makeClaim("stated", [], [])], [chunkById(chunks, ZONE_C)]);
    expectSoleRejection(answer, "empty-citations");
  });

  it("rejects an empty quote, and does not let whitespace stand in for one", async () => {
    const chunks = await corpus();
    const zoneC = chunkById(chunks, ZONE_C);

    // A single space is a substring of essentially every chunk in the corpus,
    // so containment alone would certify this.
    expect(zoneC.text).toContain(" ");

    const answer = verifyAnswer(
      [makeClaim("stated", [ZONE_C], [{ chunkId: ZONE_C, quote: " " }])],
      [zoneC],
    );

    expectSoleRejection(answer, "empty-quote");
  });

  it("rejects a quote attributed to a chunk the claim does not cite", async () => {
    const chunks = await corpus();
    const zoneC = chunkById(chunks, ZONE_C);
    const redelivery = chunkById(chunks, REDELIVERY);
    const fee = sentenceWith(redelivery, "$32 per drop");
    const surcharge = sentenceWith(zoneC, "freight surcharge of $45");

    const answer = verifyAnswer(
      [
        makeClaim(
          "stated",
          [ZONE_C],
          [
            { chunkId: ZONE_C, quote: surcharge },
            { chunkId: REDELIVERY, quote: fee },
          ],
        ),
      ],
      [zoneC, redelivery],
    );

    const rejection = expectSoleRejection(answer, "quote-for-uncited-chunk");
    expect(rejection.reason.chunkId).toBe(REDELIVERY);
  });

  it("reports the shape failure when a claim is wrong in shape and in content both", async () => {
    const chunks = await corpus();
    const zoneC = chunkById(chunks, ZONE_C);

    // Mislabelled shape, an unretrieved citation, and an invented quote at once.
    const answer = verifyAnswer(
      [
        makeClaim(
          "stated",
          [ZONE_C, INVENTED],
          [
            { chunkId: ZONE_C, quote: "words that are nowhere in the corpus" },
            { chunkId: INVENTED, quote: "nor are these" },
          ],
        ),
      ],
      [zoneC],
    );

    expectSoleRejection(answer, "stated-claim-multiple-citations");
  });
});

/* ===========================================================================
 * Existence.
 * ======================================================================== */

describe("existence: a citation is checked against what the model was shown", () => {
  it("rejects a citation to a chunk that exists in the corpus but was not retrieved", async () => {
    const chunks = await corpus();
    const zoneC = chunkById(chunks, ZONE_C);
    const depot = chunkById(chunks, DEPOT);
    const collection = sentenceWith(depot, "removes freight charges");

    // Everything about this claim is true. The chunk is real, the quote is
    // byte-exact, and the fact is correct — so containment would pass. The one
    // thing wrong with it is that the model never saw the chunk, which makes
    // the citation a guess that landed rather than a report of something read.
    expect(depot.text).toContain(collection);
    const retrieved = [zoneC];
    expect(retrieved.map((chunk) => chunk.chunkId)).not.toContain(DEPOT);

    const answer = verifyAnswer(
      [
        makeClaim(
          "stated",
          [DEPOT],
          [{ chunkId: DEPOT, quote: collection }],
          "Depot collection removes freight charges, including the Zone C surcharge.",
        ),
      ],
      retrieved,
    );

    const rejection = expectSoleRejection(answer, "chunk-not-retrieved");
    expect(rejection.reason.chunkId).toBe(DEPOT);
  });

  it("verifies that same claim once the chunk is actually retrieved", async () => {
    // The control for the test above: nothing about the claim changes, only
    // what was retrieved. That is what pins the rejection to the retrieved set
    // rather than to some other defect in the fixture.
    const chunks = await corpus();
    const depot = chunkById(chunks, DEPOT);
    const collection = sentenceWith(depot, "removes freight charges");

    const answer = verifyAnswer(
      [
        makeClaim(
          "stated",
          [DEPOT],
          [{ chunkId: DEPOT, quote: collection }],
          "Depot collection removes freight charges, including the Zone C surcharge.",
        ),
      ],
      [chunkById(chunks, ZONE_C), depot],
    );

    expect(answer.rejected).toHaveLength(0);
    expect(answer.verified).toHaveLength(1);
  });

  it("rejects an invented chunk id that looks exactly like a real one", async () => {
    const chunks = await corpus();
    expect(chunks.map((chunk) => chunk.chunkId)).not.toContain(INVENTED);

    const zoneC = chunkById(chunks, ZONE_C);
    const answer = verifyAnswer(
      [makeClaim("stated", [INVENTED], [{ chunkId: INVENTED, quote: "Zone D drops are free." }])],
      [zoneC],
    );

    expectSoleRejection(answer, "chunk-not-retrieved");
  });
});

/* ===========================================================================
 * Quote containment.
 * ======================================================================== */

describe("quote containment: exact, by indexOf, with no tolerance at all", () => {
  it("rejects a quote that is not in its cited chunk", async () => {
    const chunks = await corpus();
    const zoneC = chunkById(chunks, ZONE_C);
    const redelivery = chunkById(chunks, REDELIVERY);
    const fee = sentenceWith(redelivery, "$32 per drop");

    // A real id on a real quote from the wrong chunk — both were retrieved, so
    // existence passes and only the bytes catch it.
    expect(zoneC.text).not.toContain(fee);

    const answer = verifyAnswer(
      [makeClaim("stated", [ZONE_C], [{ chunkId: ZONE_C, quote: fee }])],
      [zoneC, redelivery],
    );

    const rejection = expectSoleRejection(answer, "quote-not-in-chunk");
    expect(rejection.reason.chunkId).toBe(ZONE_C);
  });

  it("rejects a paraphrase that is semantically identical but textually different", async () => {
    const chunks = await corpus();
    const zoneC = chunkById(chunks, ZONE_C);

    const original = sentenceWith(zoneC, "freight surcharge of $45");
    // Two words, both exact synonyms in context. Nothing about the meaning
    // moves; a human reviewer would not blink, and an embedding-similarity or
    // edit-distance check would score this a match. It is still not what the
    // document says, and this module's answer is the same as for any other
    // invention.
    const paraphrase = withWordsChanged(original, [
      ["carries", "incurs"],
      ["flat", "fixed"],
    ]);

    expect(zoneC.text).toContain(original);
    expect(zoneC.text).not.toContain(paraphrase);

    const answer = verifyAnswer(
      [makeClaim("stated", [ZONE_C], [{ chunkId: ZONE_C, quote: paraphrase }])],
      [zoneC],
    );

    expectSoleRejection(answer, "quote-not-in-chunk");

    // And the unmodified sentence passes, so the rejection is about those two
    // words and nothing else about the fixture.
    const control = verifyAnswer(
      [makeClaim("stated", [ZONE_C], [{ chunkId: ZONE_C, quote: original }])],
      [zoneC],
    );
    expect(control.verified).toHaveLength(1);
  });

  it("rejects a quote that differs only in case", async () => {
    const chunks = await corpus();
    const zoneC = chunkById(chunks, ZONE_C);
    const original = sentenceWith(zoneC, "freight surcharge of $45");
    const shouted = original.toUpperCase();
    expect(zoneC.text).not.toContain(shouted);

    const answer = verifyAnswer(
      [makeClaim("stated", [ZONE_C], [{ chunkId: ZONE_C, quote: shouted }])],
      [zoneC],
    );

    expectSoleRejection(answer, "quote-not-in-chunk");
  });

  it("rejects a quote that differs only in whitespace", async () => {
    const chunks = await corpus();
    const zoneC = chunkById(chunks, ZONE_C);
    const original = sentenceWith(zoneC, "freight surcharge of $45");
    const respaced = original.replace(" ", "  ");
    expect(respaced).not.toBe(original);
    expect(zoneC.text).not.toContain(respaced);

    const answer = verifyAnswer(
      [makeClaim("stated", [ZONE_C], [{ chunkId: ZONE_C, quote: respaced }])],
      [zoneC],
    );

    // Deliberate: this is the false rejection the exactness rule costs, and it
    // is cheaper than the false acceptance normalising it away would buy.
    expectSoleRejection(answer, "quote-not-in-chunk");
  });
});

/* ===========================================================================
 * Coverage.
 * ======================================================================== */

describe("quote coverage: every citation carries a passage", () => {
  it("rejects a citation with no supporting quote", async () => {
    const chunks = await corpus();
    const zoneC = chunkById(chunks, ZONE_C);
    const redelivery = chunkById(chunks, REDELIVERY);
    const surcharge = sentenceWith(zoneC, "freight surcharge of $45");

    // Two citations, one quote. The second id is an assertion of relevance
    // with nothing behind it.
    const answer = verifyAnswer(
      [
        makeClaim(
          "inferred",
          [ZONE_C, REDELIVERY],
          [{ chunkId: ZONE_C, quote: surcharge }],
          "A failed Zone C delivery costs both the surcharge and the redelivery fee.",
        ),
      ],
      [zoneC, redelivery],
    );

    const rejection = expectSoleRejection(answer, "citation-without-quote");
    expect(rejection.reason.chunkId).toBe(REDELIVERY);
  });
});

/* ===========================================================================
 * The happy path.
 * ======================================================================== */

describe("grounded claims", () => {
  it("verifies a stated claim built from real corpus text", async () => {
    const chunks = await corpus();
    const zoneC = chunkById(chunks, ZONE_C);
    const surcharge = sentenceWith(zoneC, "freight surcharge of $45");

    const claim = makeClaim("stated", [ZONE_C], [{ chunkId: ZONE_C, quote: surcharge }]);
    const answer = verifyAnswer([claim], [zoneC]);

    expect(answer.rejected).toEqual([]);
    expect(answer.verified).toHaveLength(1);
    // Returned as the same object: verification never rewrites what it passes.
    expect(answer.verified[0]).toBe(claim);
  });

  it("verifies an inferred claim resting on two retrieved chunks", async () => {
    const chunks = await corpus();
    const zoneC = chunkById(chunks, ZONE_C);
    const redelivery = chunkById(chunks, REDELIVERY);

    const claim = makeClaim(
      "inferred",
      [ZONE_C, REDELIVERY],
      [
        { chunkId: ZONE_C, quote: sentenceWith(zoneC, "freight surcharge of $45") },
        { chunkId: REDELIVERY, quote: sentenceWith(redelivery, "$32 per drop") },
      ],
      "A failed Zone C delivery costs $45 in freight plus $32 to redeliver.",
    );

    const answer = verifyAnswer([claim], [zoneC, redelivery]);
    expect(answer.rejected).toEqual([]);
    expect(answer.verified).toEqual([claim]);
  });

  it("accepts a quote that is a fragment of a sentence, not just a whole one", async () => {
    const chunks = await corpus();
    const zoneC = chunkById(chunks, ZONE_C);
    const fragment = sentenceWith(zoneC, "freight surcharge of $45").slice(6, 40);
    expect(fragment.length).toBeGreaterThan(0);
    expect(zoneC.text).toContain(fragment);

    const answer = verifyAnswer(
      [makeClaim("stated", [ZONE_C], [{ chunkId: ZONE_C, quote: fragment }])],
      [zoneC],
    );

    expect(answer.verified).toHaveLength(1);
  });
});

/* ===========================================================================
 * Whole answers.
 * ======================================================================== */

describe("a whole answer", () => {
  /** Two grounded claims and two laundered ones, interleaved. */
  async function mixedAnswer(): Promise<{ claims: Claim[]; retrieved: Chunk[] }> {
    const chunks = await corpus();
    const zoneC = chunkById(chunks, ZONE_C);
    const redelivery = chunkById(chunks, REDELIVERY);
    const depot = chunkById(chunks, DEPOT);

    const surcharge = sentenceWith(zoneC, "freight surcharge of $45");
    const fee = sentenceWith(redelivery, "$32 per drop");
    const collection = sentenceWith(depot, "removes freight charges");
    const paraphrase = withWordsChanged(surcharge, [
      ["carries", "incurs"],
      ["flat", "fixed"],
    ]);

    return {
      claims: [
        // grounded
        makeClaim("stated", [ZONE_C], [{ chunkId: ZONE_C, quote: surcharge }], "Zone C: $45."),
        // laundered — paraphrase in quotation marks
        makeClaim(
          "stated",
          [REDELIVERY],
          [{ chunkId: REDELIVERY, quote: paraphrase }],
          "Redelivery costs $45.",
        ),
        // grounded
        makeClaim(
          "inferred",
          [ZONE_C, REDELIVERY],
          [
            { chunkId: ZONE_C, quote: surcharge },
            { chunkId: REDELIVERY, quote: fee },
          ],
          "A failed Zone C delivery costs $77 in total.",
        ),
        // laundered — real chunk, real quote, never retrieved
        makeClaim(
          "stated",
          [DEPOT],
          [{ chunkId: DEPOT, quote: collection }],
          "Collecting from the depot removes freight charges.",
        ),
      ],
      retrieved: [zoneC, redelivery],
    };
  }

  it("returns some verified and some rejected rather than failing wholesale", async () => {
    const { claims, retrieved } = await mixedAnswer();
    const answer = verifyAnswer(claims, retrieved);

    expect(answer.verified).toHaveLength(2);
    expect(answer.rejected).toHaveLength(2);

    // Input order is preserved on both sides.
    expect(answer.verified.map((claim) => claim.text)).toEqual([
      "Zone C: $45.",
      "A failed Zone C delivery costs $77 in total.",
    ]);
    expect(answer.rejected.map((entry) => entry.reason.kind)).toEqual([
      "quote-not-in-chunk",
      "chunk-not-retrieved",
    ]);

    // Every claim lands on exactly one side.
    expect(answer.verified.length + answer.rejected.length).toBe(claims.length);
  });

  it("never throws on malformed model output, however malformed", async () => {
    const chunks = await corpus();
    const zoneC = chunkById(chunks, ZONE_C);

    // What `JSON.parse` of a model's answer can actually hand a caller. Each
    // one would throw a TypeError out of a naive implementation.
    const junk = [
      {},
      { text: "a claim", status: "stated" },
      { text: "a claim", status: "stated", citations: [ZONE_C] },
      { text: "a claim", status: "guessed", citations: [ZONE_C], supportingQuotes: [] },
      { text: 42, status: "stated", citations: [ZONE_C], supportingQuotes: [] },
      { text: "a claim", status: "stated", citations: ZONE_C, supportingQuotes: [] },
      { text: "a claim", status: "stated", citations: [7], supportingQuotes: [] },
      { text: "a claim", status: "stated", citations: [ZONE_C], supportingQuotes: [null] },
      { text: "a claim", status: "stated", citations: [ZONE_C], supportingQuotes: [{ quote: "x" }] },
      null,
    ] as unknown as Claim[];

    const answer = verifyAnswer(junk, [zoneC]);

    expect(answer.verified).toHaveLength(0);
    expect(answer.rejected).toHaveLength(junk.length);
    for (const entry of answer.rejected) {
      expect(entry.reason.kind).toBe("malformed-claim");
    }
  });

  it("rejects whole claims, keeping the original untouched for inspection", async () => {
    const chunks = await corpus();
    const zoneC = chunkById(chunks, ZONE_C);
    const original = sentenceWith(zoneC, "freight surcharge of $45");
    const paraphrase = withWordsChanged(original, [
      ["carries", "incurs"],
      ["flat", "fixed"],
    ]);

    // One good quote and one bad one. A repairing verifier would drop the bad
    // quote and publish the claim; this one rejects the claim entire.
    const claim = makeClaim(
      "inferred",
      [ZONE_C, REDELIVERY],
      [
        { chunkId: ZONE_C, quote: original },
        { chunkId: REDELIVERY, quote: paraphrase },
      ],
    );

    const answer = verifyAnswer([claim], [zoneC, chunkById(chunks, REDELIVERY)]);
    const rejection = expectSoleRejection(answer, "quote-not-in-chunk");

    expect(rejection.claim).toBe(claim);
    expect(rejection.claim.supportingQuotes).toHaveLength(2);
    expect(rejection.claim.supportingQuotes[1]?.quote).toBe(paraphrase);
  });
});

/* ===========================================================================
 * Purity.
 * ======================================================================== */

describe("verification is a pure function", () => {
  it("returns identical output for the same inputs twice and mutates neither", async () => {
    const chunks = await corpus();
    const zoneC = chunkById(chunks, ZONE_C);
    const redelivery = chunkById(chunks, REDELIVERY);
    const surcharge = sentenceWith(zoneC, "freight surcharge of $45");

    // One of each: verified, content-rejected, shape-rejected.
    const claims: Claim[] = [
      makeClaim("stated", [ZONE_C], [{ chunkId: ZONE_C, quote: surcharge }]),
      makeClaim("stated", [DEPOT], [{ chunkId: DEPOT, quote: "invented" }]),
      makeClaim("inferred", [ZONE_C], [{ chunkId: ZONE_C, quote: surcharge }]),
    ];
    const retrieved: Chunk[] = [zoneC, redelivery];

    const claimsBefore = structuredClone(claims);
    const retrievedBefore = structuredClone(retrieved);

    const first = verifyAnswer(claims, retrieved);
    const second = verifyAnswer(claims, retrieved);

    // Same output, deeply.
    expect(second).toEqual(first);
    expect(JSON.stringify(second)).toBe(JSON.stringify(first));
    expect(summarise(second)).toEqual(summarise(first));

    // Neither argument was touched — not the array, not a claim, not a chunk,
    // not a nested citations or supportingQuotes array.
    expect(claims).toEqual(claimsBefore);
    expect(retrieved).toEqual(retrievedBefore);

    // The verdict is a fresh pair of arrays, not the caller's array handed back.
    expect(first.verified).not.toBe(claims);
    expect(first.verified).not.toBe(second.verified);

    // Each claim landed on exactly one side, with the reasons pinned.
    expect(first.verified).toEqual([claims[0]]);
    expect(first.rejected.map((entry) => entry.reason.kind)).toEqual([
      "chunk-not-retrieved",
      "inferred-claim-single-citation",
    ]);
  });

  it("returns an empty verdict for no claims without touching the retrieved set", async () => {
    const chunks = await corpus();
    const answer = verifyAnswer([], chunks);
    expect(answer.verified).toEqual([]);
    expect(answer.rejected).toEqual([]);
    expect(summarise(answer).total).toBe(0);
  });

  it("rejects everything when nothing was retrieved", async () => {
    const chunks = await corpus();
    const zoneC = chunkById(chunks, ZONE_C);
    const surcharge = sentenceWith(zoneC, "freight surcharge of $45");

    const answer = verifyAnswer(
      [makeClaim("stated", [ZONE_C], [{ chunkId: ZONE_C, quote: surcharge }])],
      [],
    );

    expectSoleRejection(answer, "chunk-not-retrieved");
  });
});

/* ===========================================================================
 * Reporting.
 * ======================================================================== */

describe("summarise", () => {
  it("counts by reason, reporting zeroes rather than omitting them", async () => {
    const chunks = await corpus();
    const zoneC = chunkById(chunks, ZONE_C);
    const redelivery = chunkById(chunks, REDELIVERY);
    const surcharge = sentenceWith(zoneC, "freight surcharge of $45");

    const answer = verifyAnswer(
      [
        makeClaim("stated", [ZONE_C], [{ chunkId: ZONE_C, quote: surcharge }]),
        makeClaim("stated", [ZONE_C], [{ chunkId: ZONE_C, quote: "not in there" }]),
        makeClaim("stated", [REDELIVERY], [{ chunkId: REDELIVERY, quote: "nor this" }]),
        makeClaim("stated", [DEPOT], [{ chunkId: DEPOT, quote: surcharge }]),
      ],
      [zoneC, redelivery],
    );

    const summary = summarise(answer);
    expect(summary.total).toBe(4);
    expect(summary.verified).toBe(1);
    expect(summary.rejected).toBe(3);
    expect(summary.byReason["quote-not-in-chunk"]).toBe(2);
    expect(summary.byReason["chunk-not-retrieved"]).toBe(1);
    // Measured and none, not absent.
    expect(summary.byReason["citation-without-quote"]).toBe(0);
    expect(summary.byReason["empty-claim-text"]).toBe(0);
    expect(Object.values(summary.byReason).reduce((a, b) => a + b, 0)).toBe(summary.rejected);
  });
});

/* ===========================================================================
 * The retrieval adapter.
 * ======================================================================== */

describe("retrievedChunks", () => {
  it("unwraps search results into the chunks the model was shown, in rank order", async () => {
    const chunks = await corpus();
    const zoneC = chunkById(chunks, ZONE_C);
    const redelivery = chunkById(chunks, REDELIVERY);

    const results: SearchResult[] = [
      { chunk: zoneC, score: 0.81 },
      { chunk: redelivery, score: 0.62 },
    ];

    expect(retrievedChunks(results)).toEqual([zoneC, redelivery]);

    const surcharge = sentenceWith(zoneC, "freight surcharge of $45");
    const answer = verifyAnswer(
      [makeClaim("stated", [ZONE_C], [{ chunkId: ZONE_C, quote: surcharge }])],
      retrievedChunks(results),
    );
    expect(answer.verified).toHaveLength(1);
  });
});
