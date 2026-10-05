# rag-receipts

RAG demos are everywhere. RAG you can audit is not.

Most retrieval systems answer confidently and cite something. This one proves where every claim came from — and refuses when it can't. The provenance rules are deterministic checks in code, not instructions in a prompt, because instructions are requests and a check is a check.

    git clone https://github.com/adamabdo-xynora/rag-receipts
    cd rag-receipts && npm install && npm run demo

Thirty seconds, no API key, no network. You'll watch a two-word paraphrase get rejected, a true fact with a real citation get rejected for not having been retrieved, and a question the corpus can't answer get declined without the generator ever being called.

## Where this comes from

In the quoting engine I build for a commercial photography studio, every parsed field carries provenance — stated, inferred, default, or missing, each with its evidence — because the client audits those numbers professionally and a plausible wrong number destroys trust faster than no tool at all. In the CRM agent I run against a live contact database, the write path is fail-closed: it cannot execute without an authorization object the guard mints, and never-write states refuse at the transport layer rather than the decision layer.

This repo applies that same standard to retrieval. Every claim shows its receipt, the receipts are verified in code, and when the receipts aren't there the system says so instead of writing fiction.

One honest distinction: retrieval-over-generation for judgment calls is something I've shipped — the licensing layer in that quoting engine surfaces the closest past grants with explainable match reasoning and requires a human decision before pricing anything. This repo is the fuller RAG pattern, built to the same trust bar, on a fictional corpus.

## The threat model: citation laundering

A model asked to cite its sources will produce citations. Some will be real. The rest come in three shapes:

1. **An invented chunk id.** The id format is short, regular, and visible in the prompt, so a model can synthesise one that looks exactly right for a section that doesn't exist. Or a real id for a chunk that was never retrieved — the model couldn't have read it, so citing it is fabricated provenance even when the fact happens to be true.
2. **A real id on the wrong quote.** The chunk exists, it was retrieved, and the passage quoted simply isn't in it.
3. **A paraphrase in quotation marks.** The model read the passage, restated it, and wrapped the restatement in quotes. Two words moved, a number rounded, a hedge dropped. This is the dangerous one, because it's usually *nearly* right — and "nearly right" about a surcharge or a cut-off time is a wrong answer that survives review.

Each produces an answer that *looks* grounded: it carries ids, it carries quotation marks, and a reader who doesn't open the source documents cannot tell it from the real thing. The citation machinery launders an ungrounded assertion into an apparently sourced one. That's worse than an uncited hallucination, which at least looks like what it is.

Prompt instructions don't prevent it. "Only cite chunks you were given" and "quote exactly" raise compliance and never reach certainty; nothing enforces them, and their failures are silent and indistinguishable from success. **A system whose grounding guarantee is a sentence in a system prompt has no grounding guarantee. It has a hope.**

## The five pillars

**1. Chunking with citations a human can check.** One chunk per `##` section, id `<docId>#<section-slug>` derived from the heading — so `delivery-zones-and-schedules#zone-c-freight-surcharge` sends a reader to one file, one heading, one short paragraph. Chunk text is a **byte-exact substring** of the source file; the only transformation is trimming the two ends. That invariant is what makes quote verification mean anything, and it's enforced by a test that reads every corpus file raw and asserts each chunk's text via `indexOf`. Slug collisions and preamble text are typed errors, not silent repairs. The size-versus-coherence tradeoff and the rejection of fixed-token windows are argued in `docs/ADR-001-chunking.md`.

**2. Retrieval with a committed floor.** An injected `Embedder` interface, a hand-rolled in-memory cosine index (~six lines, zero-magnitude guarded), and `MIN_SIMILARITY = 0.4364` as an exported constant with its derivation written next to it — measured, not guessed: a live `voyage-4` calibration run scored every chunk against every question and put the floor at the midpoint between the best score any must-refuse question earns (0.3622) and the worst correct retrieval (0.5106). The run is recorded in `docs/EVAL-RUN-2026-08-28.md`. A floor exists because every vector has *some* similarity to every query: without one, "no results" never happens and the system always has something to hallucinate from. Results are **partitioned**, not filtered — "we searched and everything fell short" and "we never searched" are different facts, and the refusal path needs both.

**3. Per-claim provenance, verified deterministically (the centerpiece).** Answers are structured claims, each carrying its chunk ids, a verbatim supporting quote, and a `stated`/`inferred` status. Then `src/verify.ts` — a pure module importing two types and nothing else — checks that every cited id was in the set **the model was actually shown**, and that every quote appears in its cited chunk by exact `indexOf` containment. No fuzzy matching: a threshold has no defensible value, because a model can round $45 to $50 or drop a "not" and still clear 0.95. The edits that matter most to a reader are the smallest ones. A failed claim is **rejected whole**, never repaired — repairs compose, and three lenient checks make an answer that passes verification while being wrong in three places.

**4. Typed refusal, never a hedge.** The result is a discriminated union: `answered` with its claims and its visible rejections, or `refused` with a typed reason — `no-relevant-documents`, `claims-failed-verification`, `contradictory-sources`, `no-claims-produced` — and a detail that names numbers. When nothing clears the floor the **generator is never called**, not called and filtered. A system that generates first and checks after has already paid for the hallucination. And when two live documents disagree, both positions are surfaced with their versions and effective dates and **neither is declared the winner** — the newer document is not automatically operative, and picking silently doesn't resolve the ambiguity, it hides it behind the same confident presentation a unanimous answer would get.

**5. An eval gate that scores refusal as success.** Fourteen questions with deterministic ground truth, per-category thresholds as committed constants, exit 1 on failure. A correct refusal is a **pass**, scored exactly as highly as a correct answer. A fluent answer to the question the corpus cannot answer is the **headline failure**, printed above the table, because one instance falsifies the claim this project makes. And over-refusal fails too — the answerable bar is what stops "refuse everything" from being a way through.

## The corpus is a designed test battery

Fourteen fictional documents for Larkspur Supply Co., a wholesale supplier — the same fictional company whose CRM appears in [mcp-capability-guard](https://github.com/adamabdo-xynora/mcp-capability-guard). Every property is load-bearing and documented in `docs/CORPUS-DESIGN.md`:

- **A contradiction pair.** The return window is 14 calendar days in one live policy document and 30 in another. Neither says "superseded." The surrounding condition language is near-identical, so the disagreement is isolated to one number, and a day-22 request is allowed or declined depending on which chunk you read.
- **A single-source fact.** The cocoa tin's 540-day shelf life appears in exactly one section — verified by grep — so one chunk is right and all seventy others are wrong.
- **Two questions the corpus cannot answer,** dense with adjacent detail. Saturday delivery: six documents discuss delivery operations in dollars-and-cut-offs detail and none ever states which days runs operate, because schedules are expressed in terms of a "service day" that's used constantly and defined nowhere. Late payment: three documents state payment terms in exactly the sentence skeleton a late-fee rule would sit beside, and one names "credit hold" as a consequence without any of its causes or costs. Both absences are proven by a grep recorded next to the claim.
- **Two cross-document joins.** The delivery document states the $45 Zone C surcharge and explicitly defers the threshold to the pricing document; the pricing document states the $900 threshold and refuses to restate the surcharge. Neither half answers alone.
- **Five near-duplicate tempters,** including two collection procedures that open and close with the same sentences and give **opposite** instructions — returned goods must be palletised, rejected claim goods must not be, because repacking destroys the evidence the claim rests on.

The corpus is entirely invented: `@example.com` addresses, 555-01xx phone numbers, no real brands or people. Design notes live in `docs/`, deliberately outside `corpus/`, so they can never be retrieved and accidentally answer the questions they describe.

## Running it

    npm install
    npm run demo        # offline, no key: the narrated walkthrough
    npm test            # 240 offline tests
    npm run typecheck   # strict TypeScript
    npm run eval        # live gate: needs ANTHROPIC_API_KEY and VOYAGE_API_KEY

Exit codes on the eval: 0 = gate PASS, 1 = gate FAIL, 2 = setup error. Each run writes an artifact to `results/` recording the policy alongside the verdict, so a stored FAIL stays interpretable next to the thresholds it was judged against.

**In Docker.** The image preserves the same split. The `test` stage carries the full toolchain and runs the whole offline story — `npm test`, `npm run typecheck`, `npm run demo` all work in it, no keys. The default build is a lean runtime carrying only the compiled eval CLI, the corpus, and the question set; keys reach it through `docker run -e` at run time and nothing else — `.dockerignore` keeps `.env` out of the build context, so a key cannot end up in a layer.

    docker build --target test -t rag-receipts:test .
    docker run --rm rag-receipts:test npm test    # the same 240 offline tests
    docker build -t rag-receipts .
    docker run --rm -e ANTHROPIC_API_KEY -e VOYAGE_API_KEY rag-receipts

### Container image

The runtime image is published to GHCR on every version tag, by a workflow whose gate runs the 240
tests, the typecheck and the demo inside the test image first — the push step is unreachable unless
all three pass.

    docker pull ghcr.io/adamabdo-xynora/rag-receipts:0.1.1
    docker run --rm -e ANTHROPIC_API_KEY -e VOYAGE_API_KEY \
      ghcr.io/adamabdo-xynora/rag-receipts:0.1.1

It runs the compiled eval CLI — the same entry point, the same exit codes, and the corpus and
question set alongside it. `--help` prints the usage and exits 0; without keys it exits 2. Eval
artifacts land in the container's `results/`, which is not mounted anywhere by default. `0.1.1` is
published for `linux/amd64` and `linux/arm64`, so no `--platform` flag is needed on either.

The image carries signed build provenance, so you can check that these bytes came from this
repository's CI rather than from someone with push access to the registry:

    gh attestation verify oci://ghcr.io/adamabdo-xynora/rag-receipts:0.1.1 --owner adamabdo-xynora

`0.1.0` remains published, `linux/amd64` only and without an attestation. Its digest has not changed
and will not: a version that alters its bytes is not a version.

**What is live and what is not.** The demo is honest about this at every step, and so is this README. Corpus loading, chunking, cosine similarity, threshold partitioning, and every verification check are real computation, and the retrieval floor has been calibrated against live `voyage-4` embeddings (`docs/EVAL-RUN-2026-08-28.md`). Generation has not: the responses replayed in the demo are **hand-written fixtures, not recordings** — no model has answered a question in this repo — and they're labelled as such in the files and in the output. The adversarial payloads are hand-written to fail. The live path exists and is one command away; it just isn't what runs in CI.

**Stack.** TypeScript throughout, Vitest, two runtime dependencies: the Anthropic SDK for generation, and nothing else — Voyage embeddings travel over raw `fetch` with an injected transport. Anthropic publishes no embedding model and names [Voyage](https://platform.claude.com/docs/en/build-with-claude/embeddings) as its recommended provider, which is why the stack pairs the two. The vector index is in-memory by design: the enforcement story lives entirely above the store, and a database dependency would obscure rather than demonstrate it. **Adapting to pgvector** means replacing `buildIndex` and `search` in `src/retrieve.ts` — the `Embedder` interface, the threshold partition, and everything downstream are unchanged.

**One env read in the whole repository.** `src/eval-cli.ts` reads `process.env` once, at the bottom of the file, and passes the environment as a parameter to every helper. Every other module takes its key as an argument. That's why 240 tests run offline with fakes, and why `grep -rn "process.env" src/` is a short and boring list.

## Limitations, stated plainly

- **n = 14.** A question set this size demonstrates a method; it does not benchmark anything. The categories that matter most — must-refuse, must-surface-contradiction — have two and one members respectively.
- **The corpus is fiction, written by the same person who wrote the questions.** Real documentation is messier, and real questions are worse-posed.
- **The retrieval floor is measured; generation is still fixture-replayed.** A live `voyage-4` calibration run (2026-08-28, `npm run eval -- --calibrate`) replaced the original guessed floor with `MIN_SIMILARITY = 0.4364` — the midpoint of the measured band between the best score any chunk achieves against a must-refuse question (0.3622) and the worst correct retrieval (0.5106); the run's numbers are quoted in `src/retrieve.ts`, the committed record is `docs/EVAL-RUN-2026-08-28.md`, and the JSON artifact lives in gitignored `results/`. No live *generation* run has happened yet: the demo fixtures remain hand-written, and that is stated wherever it matters rather than smoothed over.
- **The offline embedder is lexical, and its ceiling is visible in the demo.** Step 9 of `npm run demo` shows the eval's own `restocking-fee` question — plainly answerable, the fact is right there in `returns-and-credits-policy#restocking-fee` — being **over-refused** by the stand-in embedder at 0.3044 against a 0.4364 floor. That's a real failure printed as a failure rather than swapped out for a scripted pass, and it's precisely the evidence that the live eval isn't decoration.
- **The contradiction detector is a lexical heuristic and says so.** It fires on differing numbers sharing a unit word across documents. A contradiction with no number in it is invisible to it — and the corpus contains one on purpose (repack versus do-not-repack), which the detector does not see. That limit is documented in the source and tested as a known blind spot.
- **Verification is a provenance check, not a truth oracle.** It confirms that a quote is really in the chunk it cites and that the chunk was really retrieved. It has no opinion on whether the resulting claim is correct.

## Adapting it

Replace `corpus/` with your documents, rewrite `eval/questions.json` for your domain — and derive the outcomes by *reading* your corpus, recording the grep next to each must-refuse claim. Keep the floor as a committed constant, keep quote verification exact, and keep the generator behind the retrieval decision rather than in front of it. The verifier imports two types and nothing else; it lifts out whole.

MIT license.
