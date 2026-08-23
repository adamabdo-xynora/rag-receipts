/**
 * demo.ts — the pipeline, run end to end, with every layer labelled.
 *
 * ===========================================================================
 * THE HONESTY CONTRACT. READ THIS BEFORE READING THE OUTPUT.
 * ===========================================================================
 *
 * This demo runs offline. There is no API key, no network call, and no
 * language model anywhere in this process. A demo that implied otherwise —
 * that printed a generated answer and let the reader assume a model had just
 * produced it — would be exactly the dishonesty this repository exists to
 * argue against. The whole point of `src/verify.ts` is that a confident,
 * well-formatted output is not evidence of anything; a demo that asks to be
 * believed on presentation alone is making the same mistake one level up.
 *
 * So every step prints its layer, and there are exactly three:
 *
 *   LIVE      Real math on real data, computed in this process, right now.
 *             Corpus loading and chunking (`src/chunk.ts`), cosine similarity
 *             and threshold partitioning (`src/retrieve.ts`), every claim
 *             verification check (`src/verify.ts`), contradiction detection
 *             and the refusal decision (`src/answer.ts`), and eval scoring and
 *             the gate (`src/eval.ts`). Nothing here is staged. If the corpus
 *             changes under it, the numbers change, and the assertions fail.
 *
 *   REPLAYED  Generator output, read from a fixture under `demo/fixtures/`
 *             and handed to the real pipeline as if a model had returned it.
 *             THE MODEL IS NOT BEING CALLED. It is not called because there is
 *             no key and no network in CI, and because a demo whose output
 *             changes run to run cannot be asserted on — this file doubles as
 *             a smoke test and exits nonzero on any miss.
 *
 *             AND THE FIXTURES ARE HAND-WRITTEN, NOT RECORDED. No live run has
 *             happened yet, so nothing has been recorded from one. A human
 *             typed those claims against the corpus text on disk. They are
 *             what a well-behaved generator SHOULD return; they are not
 *             evidence that any generator did. Each fixture says so in a
 *             `$provenance` block, and this demo prints that block rather than
 *             summarising it. When a live run happens they can be replaced
 *             with real recordings and the label changed to RECORDED.
 *
 *             What is NOT faked: the fixtures go through the real
 *             `parseClaims`, the real `verifyAnswer`, and the real
 *             `detectContradiction`. Every quote in them is checked
 *             byte-for-byte against the corpus at run time. A hand-written
 *             fixture cannot smuggle an unverifiable receipt through.
 *
 *   SCRIPTED  Adversarial payloads, hand-written to fail. A fabricated chunk
 *             id, a paraphrase wearing quotation marks, a quote lifted from a
 *             chunk that was never retrieved. These exist to make the checker
 *             say no in front of the reader, with a typed reason.
 *
 * ONE MORE THING THE READER IS OWED. The embedder here is the deterministic
 * lexical stand-in from `test/fake-embedder.ts` — bag-of-words hashing, no
 * semantics. It is imported rather than copied so it cannot drift from the
 * one the tests use. It is not the production embedder, and its consequences
 * are visible in this output rather than hidden: the demo questions are worded
 * to share vocabulary with the corpus, because a lexical embedder scores
 * paraphrase at nearly zero, and step 9 shows a real question genuinely
 * over-refused because of it. `MIN_SIMILARITY` is never lowered to make a step
 * look better. `src/retrieve.ts` says the floor is a property of the embedder
 * and that 0.35 is a guess; that is a live limitation of this repository, and
 * a demo that tuned around it would be lying about the state of the work.
 */

import { readFile } from "node:fs/promises";
import { fileURLToPath } from "node:url";

import {
  type AnsweredResult,
  type AnswerResult,
  DEFAULT_K,
  type Generator,
  answerQuestion,
} from "../src/answer.js";
import { type Chunk, loadCorpus } from "../src/chunk.js";
import {
  type EvalQuestion,
  applyGate,
  formatGateReport,
  loadQuestionSet,
  scoreQuestion,
} from "../src/eval.js";
import {
  MIN_SIMILARITY,
  type SearchResult,
  type VectorIndex,
  buildIndex,
  partitionByThreshold,
  search,
} from "../src/retrieve.js";
import {
  type Claim,
  type RejectedClaim,
  retrievedChunks,
  summarise,
  verifyAnswer,
} from "../src/verify.js";
import { createFakeEmbedder } from "../test/fake-embedder.js";

/* ===========================================================================
 * Layer labels. Printed verbatim under every step header.
 * ======================================================================== */

const LIVE = "LIVE — real math on real data, computed now.";
const REPLAYED_AND_LIVE =
  "REPLAYED generator output (HAND-WRITTEN fixture, no model called) + LIVE verifier.";
const SCRIPTED_AND_LIVE =
  "SCRIPTED adversarial payload (hand-written to fail) + LIVE verifier.";
const LIVE_SCORING_REPLAYED_ANSWERS =
  "LIVE gate scoring over REPLAYED and SCRIPTED outcomes. Not a run against a model.";

/* ===========================================================================
 * Printing.
 * ======================================================================== */

const WIDTH = 84;
const RULE = "=".repeat(WIDTH);
const THIN = "-".repeat(WIDTH);

/** Wrap a paragraph to `WIDTH`, so narration reads on an 80-column terminal. */
function wrap(text: string, indent = ""): string {
  const words = text.split(/\s+/).filter((word) => word !== "");
  const lines: string[] = [];
  let line = indent;

  for (const word of words) {
    const candidate = line === indent ? indent + word : `${line} ${word}`;
    if (candidate.length > WIDTH && line !== indent) {
      lines.push(line);
      line = indent + word;
    } else {
      line = candidate;
    }
  }
  if (line !== indent) lines.push(line);
  return lines.join("\n");
}

/** Narration. One argument per paragraph, blank line between. */
function say(...paragraphs: string[]): void {
  for (const paragraph of paragraphs) {
    console.log(wrap(paragraph));
    console.log("");
  }
}

function step(number: number, title: string, layer: string): void {
  console.log("");
  console.log(RULE);
  console.log(`STEP ${number} — ${title}`);
  console.log(`LAYER: ${layer}`);
  console.log(RULE);
  console.log("");
}

function subheading(title: string): void {
  console.log(title);
  console.log(THIN);
}

/* ===========================================================================
 * Assertions. The demo is also a smoke test: any miss exits nonzero.
 * ======================================================================== */

let failures = 0;

function check(condition: boolean, description: string): void {
  if (condition) {
    console.log(`  [ok]   ${description}`);
    return;
  }
  failures += 1;
  console.log(`  [FAIL] ${description}`);
}

function checkEqual(actual: unknown, expected: unknown, description: string): void {
  const same = JSON.stringify(actual) === JSON.stringify(expected);
  if (same) {
    console.log(`  [ok]   ${description}`);
    return;
  }
  failures += 1;
  console.log(`  [FAIL] ${description}`);
  console.log(`         expected ${JSON.stringify(expected)}`);
  console.log(`         actual   ${JSON.stringify(actual)}`);
}

/* ===========================================================================
 * Paths. Resolved from this file, not from the working directory.
 * ======================================================================== */

function fromRoot(...parts: string[]): string {
  return fileURLToPath(new URL(`../${parts.join("/")}`, import.meta.url));
}

/* ===========================================================================
 * Fixtures.
 * ======================================================================== */

/** A replayed generator response, with the provenance block that labels it. */
interface Fixture {
  readonly name: string;
  readonly provenance: Readonly<Record<string, string>>;
  /** The wire bytes handed to the generator seam. */
  readonly response: string;
}

/**
 * Read a fixture, and refuse one that does not say where it came from.
 *
 * An unlabelled fixture is the exact failure this demo exists to avoid, so it
 * is an error rather than a default.
 */
async function loadFixture(name: string): Promise<Fixture> {
  const raw: unknown = JSON.parse(await readFile(fromRoot("demo", "fixtures", `${name}.json`), "utf8"));
  if (typeof raw !== "object" || raw === null || Array.isArray(raw)) {
    throw new Error(`fixture ${name}: not a JSON object`);
  }

  const record = raw as Record<string, unknown>;
  const provenance = record["$provenance"];
  const response = record["response"];

  if (typeof provenance !== "object" || provenance === null || Array.isArray(provenance)) {
    throw new Error(
      `fixture ${name}: no $provenance block. A fixture that does not say whether it was recorded or hand-written cannot be printed honestly, so it is not loaded at all.`,
    );
  }
  if (typeof response !== "object" || response === null || Array.isArray(response)) {
    throw new Error(`fixture ${name}: no "response" object`);
  }

  return {
    name,
    provenance: provenance as Readonly<Record<string, string>>,
    response: JSON.stringify(response),
  };
}

/** Print the provenance block whole. Summarising it would be editorialising. */
function printProvenance(fixture: Fixture): void {
  subheading(`FIXTURE PROVENANCE — demo/fixtures/${fixture.name}.json`);
  for (const [key, value] of Object.entries(fixture.provenance)) {
    console.log(`  ${key}:`);
    console.log(wrap(value, "    "));
  }
  console.log("");
}

/* ===========================================================================
 * Generators. Neither one talks to anything.
 * ======================================================================== */

/** Records every call, returns the fixture. Satisfies the `Generator` seam. */
interface RecordingGenerator extends Generator {
  readonly calls: string[];
}

function replayGenerator(response: string): RecordingGenerator {
  const calls: string[] = [];
  return {
    calls,
    async generate(_system: string, user: string): Promise<string> {
      calls.push(user);
      return response;
    },
  };
}

/**
 * A tripwire. If retrieval decides correctly, this is never called; if it is
 * called, the call is recorded and the demo fails on the assertion rather than
 * crashing, so the rest of the output still prints.
 */
function tripwireGenerator(): RecordingGenerator {
  const calls: string[] = [];
  return {
    calls,
    async generate(_system: string, user: string): Promise<string> {
      calls.push(user);
      return "the generator should never have been called here";
    },
  };
}

/* ===========================================================================
 * Small helpers over live retrieval.
 * ======================================================================== */

function printResults(results: readonly SearchResult[], indent = "  "): void {
  for (const [position, result] of results.entries()) {
    const mark = result.score >= MIN_SIMILARITY ? "ABOVE" : "below";
    console.log(
      `${indent}${String(position + 1).padStart(2)}. ${result.score.toFixed(4)}  ${mark}  ${result.chunk.chunkId}`,
    );
  }
}

function chunkById(chunks: readonly Chunk[], chunkId: string): Chunk | undefined {
  return chunks.find((chunk) => chunk.chunkId === chunkId);
}

/** A real cosine score for one named chunk against one question. Live math. */
async function scoreOf(
  index: VectorIndex,
  question: string,
  chunkId: string,
): Promise<SearchResult | undefined> {
  const all = await search(index, question, createFakeEmbedder(), index.entries.length);
  return all.find((result) => result.chunk.chunkId === chunkId);
}

function printClaim(claim: Claim, position: number): void {
  console.log(`  claim ${position + 1}  status: ${claim.status}`);
  console.log(wrap(`text: ${claim.text}`, "    "));
  console.log(`    citations: ${claim.citations.join(", ")}`);
  for (const quote of claim.supportingQuotes) {
    console.log(`    quote from ${quote.chunkId}:`);
    console.log(wrap(JSON.stringify(quote.quote), "      "));
  }
  console.log("");
}

function printRejection(rejected: RejectedClaim): void {
  console.log(`    REJECTED`);
  console.log(`      kind    : ${rejected.reason.kind}`);
  console.log(`      chunkId : ${rejected.reason.chunkId ?? "(not attributable to one chunk)"}`);
  console.log(wrap(`message : ${rejected.reason.message}`, "      "));
}

/* ===========================================================================
 * Fixed strings the demo asserts against.
 * ======================================================================== */

const ZONE_C_SURCHARGE = "delivery-zones-and-schedules#zone-c-freight-surcharge";
const ZONE_C_THRESHOLD = "wholesale-pricing-and-minimums#free-freight-thresholds-by-zone";
const REDELIVERY = "delivery-zones-and-schedules#redelivery-after-a-failed-attempt";
const OLD_WINDOW = "returns-and-credits-policy#return-window-and-condition-requirements";
const NEW_WINDOW = "customer-care-handbook#return-window-for-stocked-goods";
const ORDER_HELD = "wholesale-faq#why-was-my-order-held";

/**
 * Worded to share vocabulary with the corpus, because the offline embedder is
 * lexical. The eval's own wording for this question is printed in step 9
 * alongside this one, so the difference is visible rather than quietly
 * exploited.
 */
const ZONE_C_QUESTION =
  "What is the Zone C freight surcharge per drop, and at what pre-tax order value is freight waived for Zone C?";
const RETURN_WINDOW_QUESTION =
  "How long is the return window for stocked goods, measured from the delivery date?";
const SATURDAY_QUESTION = "Do you deliver on Saturday?";
const RESTOCKING_QUESTION = "Is there a restocking fee on an approved return, and how is it collected?";
const LATE_PAYMENT_QUESTION =
  "What late fee or interest does Larkspur charge on an overdue invoice, and how many days past due before it applies?";

/* ===========================================================================
 * The demo.
 * ======================================================================== */

async function main(): Promise<void> {
  console.log(RULE);
  console.log("RAG RECEIPTS — END-TO-END DEMO");
  console.log(RULE);
  say(
    "Three layers, labelled at every step. LIVE is real computation on real data in this process. REPLAYED is generator output read from a hand-written fixture — no model is called, here or anywhere in this file. SCRIPTED is an adversarial payload written by hand to make the checker say no.",
    "No API key is read. No network request is made. The similarity floor is never lowered to improve a result. Every expectation is asserted inline and the process exits nonzero on any miss.",
  );

  /* --- 1. INGEST ------------------------------------------------------- */

  step(1, "INGEST", LIVE);
  say(
    "src/chunk.ts reads every Markdown file in corpus/ and cuts it at its `##` headings. Chunk text is a byte-exact substring of the file — no normalisation, no whitespace collapsing — which is the precondition for quote verification meaning anything later.",
  );

  const chunks = await loadCorpus(fromRoot("corpus"));
  const docIds = new Set(chunks.map((chunk) => chunk.docId));

  subheading("COUNTS");
  console.log(`  documents : ${docIds.size}`);
  console.log(`  chunks    : ${chunks.length}`);
  console.log("");

  subheading("EXAMPLE CHUNK IDS — the format is <docId>#<section-slug>");
  for (const chunk of chunks.slice(0, 3)) {
    console.log(`  ${chunk.chunkId}`);
    console.log(`      from "${chunk.sectionHeading}" in ${chunk.docTitle} (v${chunk.version})`);
  }
  console.log("");
  say(
    "The id is derived from the heading, so a reader can trace a citation back to a line in a file by eye, without running anything. Slug collisions are a typed error rather than a `-2` suffix, precisely so that stays true.",
  );

  subheading("CHECKS");
  checkEqual(docIds.size, 14, "the corpus loads 14 documents");
  checkEqual(chunks.length, 71, "the corpus produces 71 chunks");
  check(
    chunks.every((chunk) => /^[a-z0-9-]+#[a-z0-9-]+$/.test(chunk.chunkId)),
    "every chunk id is <docId>#<section-slug>",
  );
  check(
    new Set(chunks.map((chunk) => chunk.chunkId)).size === chunks.length,
    "every chunk id is unique across the corpus",
  );

  /* --- 2. RETRIEVE ----------------------------------------------------- */

  step(2, "RETRIEVE", LIVE);
  say(
    "The index is built with the deterministic lexical embedder from test/fake-embedder.ts, and every score below is a cosine similarity computed here, now, by src/retrieve.ts. The floor is MIN_SIMILARITY as committed in that file, unmodified.",
    `Question: ${JSON.stringify(ZONE_C_QUESTION)}`,
  );

  const embedder = createFakeEmbedder();
  const index = await buildIndex(chunks, embedder);
  const results = await search(index, ZONE_C_QUESTION, embedder, DEFAULT_K);
  const partition = partitionByThreshold(results, MIN_SIMILARITY);

  subheading(`TOP 5 OF k=${DEFAULT_K} (k = DEFAULT_K, what the pipeline uses)`);
  printResults(results.slice(0, 5));
  console.log("");

  subheading(`THRESHOLD PARTITION AT MIN_SIMILARITY = ${MIN_SIMILARITY}`);
  console.log(`  cleared the floor (${partition.above.length}):`);
  printResults(partition.above, "    ");
  console.log(`  did not clear it (${partition.below.length}):`);
  printResults(partition.below, "    ");
  console.log("");

  say(
    "The two chunks the answer actually needs rank first and second, and they live in different documents: the delivery document states the $45 surcharge and refuses to name the threshold, the pricing document states the $900 threshold and refuses to name the surcharge. Neither half answers the question alone.",
    "The below-floor rows are kept, not discarded. `we searched and everything fell short` and `we never searched` are different facts, and an empty array cannot tell them apart from a broken index. Step 7 spends that distinction.",
  );

  subheading("CHECKS");
  checkEqual(partition.above.length + partition.below.length, DEFAULT_K, "the partition loses nothing");
  checkEqual(
    partition.above.slice(0, 2).map((result) => result.chunk.chunkId),
    [ZONE_C_SURCHARGE, ZONE_C_THRESHOLD],
    "both halves of the cross-document join clear the floor, ranked first and second",
  );
  check(
    partition.above.every((result) => result.score >= MIN_SIMILARITY),
    "every above-floor score is at or above the floor",
  );
  check(
    partition.below.every((result) => result.score < MIN_SIMILARITY),
    "every below-floor score is under it",
  );

  const shown = retrievedChunks(partition.above);

  /* --- 3. ANSWER WITH RECEIPTS ----------------------------------------- */

  step(3, "ANSWER WITH RECEIPTS", REPLAYED_AND_LIVE);

  const zoneCFixture = await loadFixture("zone-c-freight");
  printProvenance(zoneCFixture);

  say(
    "What follows is the fixture above, handed to answerQuestion through the injected generator seam exactly as a model's response would be. Retrieval, parsing, verification and the refusal decision are all live. The generator is not.",
  );

  const answerResult = await answerQuestion(ZONE_C_QUESTION, {
    index,
    embedder,
    generator: replayGenerator(zoneCFixture.response),
  });

  if (answerResult.kind !== "answered") {
    failures += 1;
    console.log(`  [FAIL] expected an answer, got a refusal: ${answerResult.reason}`);
    console.log(wrap(answerResult.detail, "         "));
  }
  const answered = answerResult as AnsweredResult;

  subheading("CLAIMS, EACH WITH ITS RECEIPTS");
  for (const [position, claim] of answered.claims.entries()) printClaim(claim, position);

  subheading("THE VERIFIER, RUN LIVE OVER THOSE CLAIMS AND THOSE CHUNKS");
  const verdict = verifyAnswer(answered.claims, shown);
  const summary = summarise(verdict);
  console.log(`  claims total : ${summary.total}`);
  console.log(`  verified     : ${summary.verified}`);
  console.log(`  rejected     : ${summary.rejected}`);
  console.log("  rejections by kind:");
  for (const [kind, count] of Object.entries(summary.byReason)) {
    console.log(`    ${kind.padEnd(32)} ${count}`);
  }
  console.log("");

  say(
    "Zeroes are printed rather than omitted, because an absent key reads as `not measured` and a zero reads as `measured, none`.",
    "Note what the verifier checked and what it did not. It checked that every cited id was in the set the model was shown, and that every quote appears byte-for-byte in the chunk it is attributed to. It did not check that $820 is less than $900, and it has no opinion on whether the claims are true. It is a provenance check, not a truth oracle — which is why steps 7 and 9 matter.",
  );

  subheading("CHECKS");
  checkEqual(answerResult.kind, "answered", "the pipeline answered");
  checkEqual(answered.claims.length, 3, "three claims survived verification");
  checkEqual(answered.rejectedClaims.length, 0, "nothing was stripped");
  checkEqual(summary.verified, 3, "the live verifier verifies all three");
  checkEqual(summary.rejected, 0, "the live verifier rejects none");
  check(
    answered.claims.every((claim) =>
      claim.supportingQuotes.every((quote) => {
        const chunk = chunkById(shown, quote.chunkId);
        return chunk !== undefined && chunk.text.includes(quote.quote);
      }),
    ),
    "independently: every quote is a literal substring of the chunk it cites",
  );
  const citedIds = new Set(answered.claims.flatMap((claim) => [...claim.citations]));
  check(citedIds.has(ZONE_C_SURCHARGE) && citedIds.has(ZONE_C_THRESHOLD), "both join halves are cited");

  /* --- 4. FABRICATED CITATION ------------------------------------------ */

  step(4, "THE CHECKER CATCHES A FABRICATED CITATION", SCRIPTED_AND_LIVE);

  const FABRICATED_ID = "delivery-zones-and-schedules#zone-c-freight-threshold";
  say(
    `The payload below cites ${JSON.stringify(FABRICATED_ID)}. That id looks entirely real: the document exists, the slug is well-formed, and the wording is what the section would plausibly be called. It is not in the corpus. The real section is ${JSON.stringify(ZONE_C_SURCHARGE)}.`,
  );

  const fabricated: Claim = {
    text: "The Zone C free-freight threshold is $900 per drop.",
    status: "stated",
    citations: [FABRICATED_ID],
    supportingQuotes: [
      { chunkId: FABRICATED_ID, quote: "The Zone C free-freight threshold is $900 per drop." },
    ],
  };

  const fabricatedVerdict = verifyAnswer([fabricated], shown);
  subheading("THE TYPED REJECTION");
  const fabricatedRejection = fabricatedVerdict.rejected[0];
  if (fabricatedRejection === undefined) {
    failures += 1;
    console.log("  [FAIL] the fabricated citation was not rejected");
  } else {
    printRejection(fabricatedRejection);
  }
  console.log("");

  say(
    "The kind is `chunk-not-retrieved`, not `chunk-does-not-exist`, and that is the stronger check rather than a looser one. The verifier never asks the corpus whether an id is real; it asks whether the id was in the set the model was actually shown. An id that does not exist anywhere fails that test as a special case of an id the model could not have read.",
  );

  subheading("CHECKS");
  check(chunkById(chunks, FABRICATED_ID) === undefined, "the fabricated id is absent from the whole corpus");
  check(chunkById(chunks, ZONE_C_SURCHARGE) !== undefined, "the id it imitates does exist");
  checkEqual(fabricatedVerdict.verified.length, 0, "nothing verified");
  checkEqual(fabricatedRejection?.reason.kind, "chunk-not-retrieved", "rejected as chunk-not-retrieved");
  checkEqual(fabricatedRejection?.reason.chunkId, FABRICATED_ID, "the reason names the offending id");

  /* --- 5. PARAPHRASE --------------------------------------------------- */

  step(5, "THE CHECKER CATCHES A PARAPHRASE", SCRIPTED_AND_LIVE);

  const surchargeChunk = chunkById(shown, ZONE_C_SURCHARGE);
  if (surchargeChunk === undefined) {
    throw new Error(`${ZONE_C_SURCHARGE} is not in the retrieved set; the demo cannot continue`);
  }

  const REAL_SENTENCE =
    "Every Zone C drop carries a flat freight surcharge of $45, applied per drop and not per line or per pallet.";
  const PARAPHRASE =
    "Every Zone C drop incurs a fixed freight surcharge of $45, applied per drop and not per line or per pallet.";

  say(
    "Two words changed, and nothing else: `carries` became `incurs`, `flat` became `fixed`. The number is right. The unit is right. The citation is the correct chunk. The meaning is unchanged. It is rejected anyway.",
  );

  subheading("THE TWO STRINGS");
  console.log("  IN THE DOCUMENT:");
  console.log(`    ${REAL_SENTENCE}`);
  console.log("  IN THE CLAIM:");
  console.log(`    ${PARAPHRASE}`);

  // Computed live, so the marker cannot drift from the strings above.
  const marker = Array.from({ length: Math.max(REAL_SENTENCE.length, PARAPHRASE.length) }, (_, at) =>
    REAL_SENTENCE[at] === PARAPHRASE[at] ? " " : "^",
  )
    .join("")
    .trimEnd();
  console.log("  DIFFERS AT:");
  console.log(`    ${marker}`);
  console.log("");

  const paraphrased: Claim = {
    text: "Zone C drops carry a flat $45 freight surcharge, charged per drop.",
    status: "stated",
    citations: [ZONE_C_SURCHARGE],
    supportingQuotes: [{ chunkId: ZONE_C_SURCHARGE, quote: PARAPHRASE }],
  };

  const paraphraseVerdict = verifyAnswer([paraphrased], shown);
  subheading("THE TYPED REJECTION");
  const paraphraseRejection = paraphraseVerdict.rejected[0];
  if (paraphraseRejection === undefined) {
    failures += 1;
    console.log("  [FAIL] the paraphrase was not rejected");
  } else {
    printRejection(paraphraseRejection);
  }
  console.log("");

  say(
    "This is the check the whole project turns on, and it is deliberately unforgiving. A quote is a claim that these exact words appear at this exact place. Once the checker accepts `close enough`, someone has to decide how close, and that judgement is the same act as writing the answer — performed with less information and no receipts.",
    "The rejection is also whole. The claim is not repaired by substituting the real sentence, and it is not partially kept. A repaired claim is a claim the reader can no longer trace to anything the generator actually said.",
  );

  subheading("CHECKS");
  check(surchargeChunk.text.includes(REAL_SENTENCE), "the real sentence is byte-exact in the chunk");
  check(!surchargeChunk.text.includes(PARAPHRASE), "the paraphrase is not in the chunk");
  checkEqual(REAL_SENTENCE.length, PARAPHRASE.length, "the two strings are even the same length");
  checkEqual(paraphraseVerdict.verified.length, 0, "nothing verified");
  checkEqual(paraphraseRejection?.reason.kind, "quote-not-in-chunk", "rejected as quote-not-in-chunk");
  checkEqual(paraphraseRejection?.claim.supportingQuotes[0]?.quote, PARAPHRASE, "the claim is kept whole, unedited");

  /* --- 6. UNRETRIEVED CITATION ----------------------------------------- */

  step(6, "THE CHECKER CATCHES AN UNRETRIEVED CITATION", SCRIPTED_AND_LIVE);

  const redeliveryChunk = chunkById(chunks, REDELIVERY);
  if (redeliveryChunk === undefined) {
    throw new Error(`${REDELIVERY} is missing from the corpus; the demo cannot continue`);
  }
  const REDELIVERY_QUOTE =
    "Redelivery is charged at $32 per drop and is scheduled on the next available run for that zone.";

  say(
    `Everything about this payload is true. ${JSON.stringify(REDELIVERY)} is a real chunk in the corpus. The quote is byte-exact in it. Redelivery really does cost $32 per drop. And it is rejected.`,
  );

  const redeliveryScore = await scoreOf(index, ZONE_C_QUESTION, REDELIVERY);
  console.log(`  its live score against this question : ${redeliveryScore?.score.toFixed(4) ?? "n/a"}`);
  console.log(`  the floor                            : ${MIN_SIMILARITY.toFixed(4)}`);
  console.log(`  so it was NOT among the ${shown.length} chunks shown to the generator.`);
  console.log("");

  const unretrieved: Claim = {
    text: "Redelivery after a failed attempt is charged at $32 per drop.",
    status: "stated",
    citations: [REDELIVERY],
    supportingQuotes: [{ chunkId: REDELIVERY, quote: REDELIVERY_QUOTE }],
  };

  const unretrievedVerdict = verifyAnswer([unretrieved], shown);
  subheading("THE TYPED REJECTION");
  const unretrievedRejection = unretrievedVerdict.rejected[0];
  if (unretrievedRejection === undefined) {
    failures += 1;
    console.log("  [FAIL] the unretrieved citation was not rejected");
  } else {
    printRejection(unretrievedRejection);
  }
  console.log("");

  say(
    "WHY A TRUE FACT WITH A REAL CITATION IS STILL REJECTED. A citation is not an assertion that a fact is true. It is an assertion about where the answer came from: `I read this here, in the material you gave me`. That assertion is false. The generator was shown three chunks and this was not one of them, so whatever produced this sentence, it was not the evidence in front of it.",
    "Accepting it would mean accepting model recall dressed as retrieval. The next such claim will be equally confident, equally well-cited, and wrong — and there would be no check left that could tell the two apart. Verifying against the retrieved set rather than against the corpus is what makes the difference visible at all.",
  );

  subheading("CHECKS");
  check(redeliveryChunk.text.includes(REDELIVERY_QUOTE), "the quote is byte-exact in that real chunk");
  check(chunkById(shown, REDELIVERY) === undefined, "the chunk was not in the retrieved set");
  check((redeliveryScore?.score ?? 1) < MIN_SIMILARITY, "and its live score is genuinely below the floor");
  checkEqual(unretrievedVerdict.verified.length, 0, "nothing verified");
  checkEqual(unretrievedRejection?.reason.kind, "chunk-not-retrieved", "rejected as chunk-not-retrieved");

  /* --- 7. REFUSAL ------------------------------------------------------ */

  step(7, "REFUSAL — NOTHING RETRIEVED", LIVE);
  say(
    `Question: ${JSON.stringify(SATURDAY_QUESTION)}`,
    "The corpus is dense with delivery detail — zones, cut-offs, nominated windows, redelivery, depot collection, public holiday closures — and fluent in per-drop dollar charges. It never says which days runs operate. A Saturday surcharge would fit the pattern perfectly, which is exactly why it is the designed trap in docs/CORPUS-DESIGN.md.",
    "The generator passed in below is a tripwire: it records any call it receives. Watch the call count.",
  );

  const tripwire = tripwireGenerator();
  const refusalResult = await answerQuestion(SATURDAY_QUESTION, {
    index,
    embedder,
    generator: tripwire,
  });

  subheading("THE TYPED REFUSAL");
  console.log(`  kind   : ${refusalResult.kind}`);
  if (refusalResult.kind === "refused") {
    console.log(`  reason : ${refusalResult.reason}`);
    console.log("  detail :");
    console.log(wrap(refusalResult.detail, "    "));
    console.log("");
    subheading("NEAR MISSES — kept, not discarded");
    printResults(refusalResult.nearMisses, "    ");
  }
  console.log("");
  console.log(`  GENERATOR CALLS: ${tripwire.calls.length}`);
  console.log("");

  say(
    "The generator was never called. Not called and then filtered — never called. Nothing cleared the floor, so there was nothing to ground an answer in, and src/answer.ts returns before the generator seam is touched.",
    "That ordering is the whole design. A model handed six irrelevant passages and a question does not reply `these do not address your question`; it answers, out of the passages plus everything it knows, because that is what the shape of the request asks for. By the time a downstream check sees that output the hallucination has already been paid for, and every cleanup step is one bug away from letting it through.",
    "The refusal names numbers — how many chunks were searched, what k was, what the floor is, what the best score was and which chunk earned it. `I don't have enough information` cannot distinguish a corpus that lacks the answer from a threshold set too high, an embedder misconfigured, or an index built over the wrong directory. Those have different fixes and identical symptoms unless the refusal is specific.",
  );

  subheading("CHECKS");
  checkEqual(refusalResult.kind, "refused", "the pipeline refused");
  checkEqual(
    refusalResult.kind === "refused" ? refusalResult.reason : null,
    "no-relevant-documents",
    "the refusal reason is no-relevant-documents",
  );
  checkEqual(tripwire.calls.length, 0, "THE GENERATOR WAS NEVER CALLED");
  checkEqual(
    refusalResult.kind === "refused" ? refusalResult.nearMisses.length : 0,
    DEFAULT_K,
    "all k near misses are reported rather than dropped",
  );
  check(
    refusalResult.kind === "refused" &&
      refusalResult.nearMisses.every((result) => result.score < MIN_SIMILARITY),
    "every near miss really is below the floor",
  );

  /* --- 8. CONTRADICTION ------------------------------------------------ */

  step(8, "CONTRADICTION", REPLAYED_AND_LIVE);

  const contradictionFixture = await loadFixture("return-window-contradiction");
  printProvenance(contradictionFixture);

  say(
    `Question: ${JSON.stringify(RETURN_WINDOW_QUESTION)}`,
    "returns-and-credits-policy (v1.2, effective 2023-04-03) says 14 calendar days. customer-care-handbook (v3.0, effective 2025-02-10) says 30 calendar days. Neither is marked superseded. Both are confidently worded. A request on day 22 is either allowed or declined depending on which one you happen to read.",
    "The two claims below are replayed from the hand-written fixture. Both are verified live against the chunks that were actually retrieved before anything else happens — an unverified quote could make two fabrications look like a disagreement — and then detectContradiction runs live over the survivors.",
  );

  const contradictionResult = await answerQuestion(RETURN_WINDOW_QUESTION, {
    index,
    embedder,
    generator: replayGenerator(contradictionFixture.response),
  });

  subheading("THE TYPED REFUSAL");
  console.log(`  kind   : ${contradictionResult.kind}`);
  if (contradictionResult.kind === "refused") {
    console.log(`  reason : ${contradictionResult.reason}`);
    console.log("  detail :");
    console.log(wrap(contradictionResult.detail, "    "));
  }
  console.log("");

  say(
    "Both positions are surfaced with their document titles, versions, effective dates, citing claims and byte-exact quotes. Neither is marked correct. The later effective date is reported and is not allowed to decide.",
    "Sorting by date and returning 30 would produce a clean, confident, single-number answer, and it would substitute this module's judgement for the reader's on exactly the question the reader needs to decide. The newer document might be a handbook that never had authority to amend the policy; the older one might be the contractual text the customer's signed terms incorporate by reference. Which is true is not in the corpus. Picking silently does not resolve the ambiguity, it hides it — behind the same confident presentation a genuinely unanimous answer would get.",
    "This detector is a lexical heuristic and src/answer.ts says so at length: numbers only, vocabulary rather than meaning, cross-document only, pairwise across claims. The corpus contains a contradiction with no number in it (repack versus do-not-repack) that this function cannot see. Naming that limit is part of the demo, not a footnote to it.",
  );

  subheading("CHECKS");
  checkEqual(contradictionResult.kind, "refused", "the pipeline refused rather than answering");
  checkEqual(
    contradictionResult.kind === "refused" ? contradictionResult.reason : null,
    "contradictory-sources",
    "the refusal reason is contradictory-sources",
  );
  const contradictionDetail = contradictionResult.kind === "refused" ? contradictionResult.detail : "";
  check(contradictionDetail.includes(OLD_WINDOW), "the 14-day position is named with its chunk id");
  check(contradictionDetail.includes(NEW_WINDOW), "the 30-day position is named with its chunk id");
  check(contradictionDetail.includes("14 calendar days"), "the 14-day value is quoted");
  check(contradictionDetail.includes("30 calendar days"), "the 30-day value is quoted");
  check(contradictionDetail.includes("2023-04-03") && contradictionDetail.includes("2025-02-10"), "both effective dates are shown");
  check(contradictionDetail.includes("1.2") && contradictionDetail.includes("3.0"), "both versions are shown");
  check(
    !/\b(correct|governs|therefore the answer is)\b/i.test(contradictionDetail) ||
      contradictionDetail.includes("question for the reader"),
    "no winner is declared; the decision is handed to the reader",
  );

  /* --- 9. THE EVAL GATE ------------------------------------------------ */

  step(9, "THE EVAL GATE", LIVE_SCORING_REPLAYED_ANSWERS);

  say(
    "THIS IS NOT A RUN AGAINST A MODEL. It is a demonstration of what src/eval.ts prints. The question set, the ground-truth outcomes, the scoring rules and the committed per-category thresholds are all real and are applied live. Four of the five outcomes below were produced by the real pipeline in this process; the fifth is scripted, and is marked as such.",
    "For the full run against a live embedder and a live generator, use `npm run eval`, which needs keys.",
  );

  const questionSet = await loadQuestionSet(fromRoot("eval", "questions.json"));
  const byId = new Map(questionSet.questions.map((question) => [question.id, question]));

  function questionOrThrow(id: string): EvalQuestion {
    const question = byId.get(id);
    if (question === undefined) throw new Error(`eval/questions.json has no question "${id}"`);
    return question;
  }

  // The over-refusal is not staged: at the committed floor, the lexical
  // stand-in embedder genuinely fails to surface the restocking chunk. Running
  // it live is more honest than scripting the refusal, and it puts the
  // stand-in's real weakness in the results table where it belongs.
  const restockingTripwire = tripwireGenerator();
  const restockingResult = await answerQuestion(RESTOCKING_QUESTION, {
    index,
    embedder,
    generator: restockingTripwire,
  });

  // SCRIPTED. Nothing produced this; it is written by hand to show the reader
  // what a headline failure looks like in the table. Note that the claim would
  // PASS the verifier — the citation is real and the quote is byte-exact. The
  // fabricated part is the conclusion, and no provenance check can see it.
  // What stops this in the real pipeline is the retrieval floor (step 7).
  const heldChunk = chunkById(chunks, ORDER_HELD);
  if (heldChunk === undefined) throw new Error(`${ORDER_HELD} is missing from the corpus`);
  const heldResult = await scoreOf(index, LATE_PAYMENT_QUESTION, ORDER_HELD);
  if (heldResult === undefined) throw new Error(`could not score ${ORDER_HELD}`);

  const hallucinated: AnsweredResult = {
    kind: "answered",
    claims: [
      {
        text: "Larkspur charges 1.5% interest per month on invoices more than 30 days past due, at which point the account also moves to credit hold.",
        status: "stated",
        citations: [ORDER_HELD],
        supportingQuotes: [
          {
            chunkId: ORDER_HELD,
            quote:
              "Orders are held for three reasons: the order sits below the minimum order value, the account is inside its first three orders and prepayment has not cleared, or the account is on credit hold.",
          },
        ],
      },
    ],
    rejectedClaims: [],
    retrieved: [heldResult],
  };

  const cases: readonly { readonly id: string; readonly result: AnswerResult; readonly provenance: string }[] = [
    {
      id: "join-zone-c-freight-820",
      result: answerResult,
      provenance: "LIVE retrieval + REPLAYED fixture + LIVE verify (step 3)",
    },
    {
      id: "return-window-contradiction",
      result: contradictionResult,
      provenance: "LIVE retrieval + REPLAYED fixture + LIVE detector (step 8)",
    },
    {
      id: "saturday-delivery-surcharge",
      result: refusalResult,
      provenance: "LIVE, generator never called (step 7)",
    },
    {
      id: "restocking-fee",
      result: restockingResult,
      provenance: "LIVE, generator never called — a real over-refusal by the lexical stand-in",
    },
    {
      id: "late-payment-fee",
      result: hallucinated,
      provenance: "SCRIPTED — hand-written to show what a headline failure looks like",
    },
  ];

  subheading("WHERE EACH SCORED OUTCOME CAME FROM");
  for (const entry of cases) {
    console.log(`  ${entry.id.padEnd(28)} ${entry.provenance}`);
  }
  console.log("");

  say(
    `A wording note, because it would otherwise be invisible. Step 3 asked ${JSON.stringify(ZONE_C_QUESTION)}, whereas eval/questions.json asks ${JSON.stringify(questionOrThrow("join-zone-c-freight-820").question)}. The ground truth being scored against is the file's; the wording is the demo's, chosen because the offline embedder is lexical rather than semantic. The eval's own wording retrieves nothing above the floor with this stand-in. That is a real limitation and it is stated rather than smoothed over.`,
  );

  const scores = cases.map((entry) => scoreQuestion(questionOrThrow(entry.id), entry.result));
  const report = applyGate(scores);

  console.log(formatGateReport(report));
  console.log("");

  say(
    "A CORRECT REFUSAL IS A PASS. Not partial credit, not a neutral non-answer — for saturday-delivery-surcharge the corpus does not contain the answer, so declining IS the correct output, scored exactly as highly as a correct answer elsewhere.",
    "AND OVER-REFUSAL FAILS. restocking-fee is answerable and the system declined it, so it is a failure in the answerable category. That is deliberate: a system that refuses everything would score 100% on both must-refuse categories, and the answerable bar is what stops refusing everything from being a way through.",
    "THERE IS NO BLENDED SCORE. `3 of 5` would let the hallucination hide behind the passes. The per-category bars keep a soundness violation from being cancelled out by good answers, and the two headline kinds are printed above the table because one instance of either falsifies the claim this project makes.",
  );

  subheading("CHECKS");
  checkEqual(report.totalQuestions, 5, "five questions scored");
  checkEqual(report.passed, false, "the gate fails, as designed for this demonstration");
  checkEqual(report.headlineViolations.length, 1, "exactly one headline failure");
  checkEqual(
    report.headlineViolations[0]?.kind,
    "hallucinated-answer",
    "the headline failure is a hallucinated answer",
  );
  checkEqual(
    report.headlineViolations[0]?.questionId,
    "late-payment-fee",
    "and it is the scripted one",
  );
  const scoreById = new Map(scores.map((score) => [score.id, score]));
  check(scoreById.get("saturday-delivery-surcharge")?.passed === true, "a correct refusal scores as a PASS");
  check(scoreById.get("return-window-contradiction")?.passed === true, "surfacing both sides scores as a PASS");
  check(scoreById.get("join-zone-c-freight-820")?.passed === true, "the cited cross-document answer scores as a PASS");
  checkEqual(
    scoreById.get("restocking-fee")?.failure?.kind,
    "over-refusal",
    "the over-refusal is named as one, not silently forgiven",
  );
  checkEqual(scoreById.get("restocking-fee")?.failure?.headline, false, "over-refusal is not a headline failure");
  checkEqual(restockingTripwire.calls.length, 0, "the generator was not called for the over-refusal either");

  /* --- The five rules -------------------------------------------------- */

  console.log("");
  console.log(RULE);
  console.log("THE FIVE RULES THIS DEMO DEMONSTRATED");
  console.log(RULE);
  console.log("");
  say(
    "ONE. RETRIEVAL DECIDES WHETHER TO SPEAK, AND IT DECIDES FIRST. The similarity floor is a committed constant compared against a number, not a behaviour prompted for. When nothing clears it the generator is never called, so there is no fluent wrong answer to catch, discard, or leak. Refusing before the call is not an optimisation over checking afterwards; it is the difference between a system that cannot fabricate here and one that fabricates and then tries to notice.",
    "TWO. A CITATION IS A CLAIM ABOUT PROVENANCE, AND IT IS CHECKED AGAINST WHAT THE MODEL WAS SHOWN. Not against the corpus. A real chunk, a byte-exact quote, and a true fact are all rejected together if that chunk was not in the retrieved set, because the assertion being made — I read this here, in the material you gave me — is false. That is the check that separates retrieval from recall wearing retrieval's clothes.",
    "THREE. QUOTES ARE CHECKED BYTE FOR BYTE, AND CLOSE ENOUGH IS NOT CLOSE ENOUGH. Two synonyms swapped in a hundred-character sentence, same length, same meaning, correct number, correct citation: rejected. The moment the checker accepts approximate quotes, somebody has to decide how approximate, and that judgement call is the same act as writing the answer, made with less information and no receipts.",
    "FOUR. A FAILED CLAIM IS REJECTED WHOLE, WITH A TYPED REASON, AND REPORTED RATHER THAN HIDDEN. Never repaired, never partially kept. `chunk-not-retrieved` and `quote-not-in-chunk` point at different defects with different fixes, and six rejections all of one kind names a specific problem in a specific prompt in a way that `six rejected` never could. Stripped claims travel with the answer, because a partial answer whose discarded half is invisible reads as a complete one.",
    "FIVE. WHEN THE DOCUMENTS DISAGREE, BOTH SIDES ARE SURFACED AND NEITHER IS CROWNED — AND A CORRECT REFUSAL IS A PASS, WHILE OVER-REFUSAL IS A FAILURE. Recency is reported and is not allowed to decide, because the newer document is not automatically the operative one and the corpus does not say which governs. The gate scores a correct refusal exactly as highly as a correct answer, and fails a refusal of an answerable question, so that refusing everything is not a way through.",
  );

  console.log(RULE);
  if (failures > 0) {
    console.log(`DEMO FAILED — ${failures} assertion${failures === 1 ? "" : "s"} did not hold.`);
    console.log(RULE);
    process.exit(1);
  }
  console.log("DEMO PASSED — every assertion held. Exit 0.");
  console.log(RULE);
}

main().catch((error: unknown) => {
  console.error("");
  console.error("DEMO CRASHED");
  console.error(error instanceof Error ? (error.stack ?? error.message) : String(error));
  process.exit(1);
});
