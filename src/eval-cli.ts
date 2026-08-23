/**
 * eval-cli.ts — the only file in this repository permitted to read
 * `process.env`, and the only one that opens a network connection on purpose.
 *
 * ---------------------------------------------------------------------------
 * WHY THE BOUNDARY IS A WHOLE FILE
 *
 * Every other module takes its credentials as a parameter. `voyageEmbedder`
 * wants an `apiKey`, `anthropicGenerator` wants an `apiKey`, and neither will
 * pick one up on its own. That rule is worth something only if there is exactly
 * one place that breaks it, and this is that place: a reader who wants to know
 * how a key gets into this system reads one file, and a grep for `process.env`
 * across `src/` returns this file or it returns a bug.
 *
 * The alternative — each module falling back to an environment variable when
 * its parameter is absent — works in production and in tests for different
 * reasons. It passes the tests because the variable is unset and the fake is
 * injected; it works in production because the variable is set and the fallback
 * fires. The two paths are never exercised together, so the day the fallback
 * reads the wrong variable, every test still passes.
 *
 * ---------------------------------------------------------------------------
 * KEYS ARE NEVER ECHOED
 *
 * No function below interpolates a key value into a message, an error, a log
 * line, or an artifact. Not truncated, not masked, not "first four characters
 * for debugging" — a masked key in a CI log is still a key in a CI log, and the
 * mask is one refactor away from being dropped. Errors name the VARIABLE, never
 * the VALUE, and `test/eval.test.ts` puts a decoy key in the environment and
 * greps every produced string for it.
 *
 * ---------------------------------------------------------------------------
 * EXIT CODES
 *
 *   0  the gate passed
 *   1  the gate failed — the eval ran to completion and the corpus results did
 *      not meet the committed thresholds
 *   2  setup error — a missing key, an unreadable question set, a corpus that
 *      would not load, a provider that would not answer
 *
 * 1 AND 2 ARE DIFFERENT ON PURPOSE. A build that goes red on a missing
 * `VOYAGE_API_KEY` and a build that goes red because the system fabricated an
 * answer to a question the corpus cannot answer demand entirely different
 * responses, and a shared exit code makes the second look like the first — the
 * cheapest possible way to ignore the finding this repository exists to
 * produce. Nothing here converts an infrastructure failure into a verdict, and
 * no artifact is written for an incomplete run.
 */

import { fileURLToPath, pathToFileURL } from "node:url";
import { readFile } from "node:fs/promises";

import { anthropicGenerator } from "./answer.js";
import { loadCorpus } from "./chunk.js";
import {
  type EvalQuestion,
  type RunMetadata,
  DEFAULT_ARTIFACT_DIR,
  applyGate,
  buildArtifact,
  calibrateThreshold,
  formatCalibrationReport,
  formatGateReport,
  loadQuestionSet,
  runEvaluation,
  writeArtifact,
} from "./eval.js";
import { MIN_SIMILARITY, buildIndex, voyageEmbedder } from "./retrieve.js";
import { DEFAULT_K } from "./answer.js";

/* ===========================================================================
 * Exit codes.
 * ======================================================================== */

export const EXIT_PASS = 0;
export const EXIT_GATE_FAILED = 1;
export const EXIT_SETUP_ERROR = 2;

/* ===========================================================================
 * The environment, as a value.
 * ======================================================================== */

/**
 * An environment. Shaped like `process.env` — values may be undefined — so the
 * real one can be passed straight in and a test can pass a literal.
 *
 * EVERY HELPER BELOW TAKES ONE OF THESE. None of them reads the ambient
 * `process.env`; `main` reads it once, at the very bottom of this file, and
 * hands it down. That is what makes the resolution logic testable without
 * mutating global state — and a test that has to mutate `process.env` to run is
 * a test that interferes with every other test in the file.
 */
export type Environment = Readonly<Record<string, string | undefined>>;

/* ===========================================================================
 * .env parsing.
 * ======================================================================== */

/**
 * Parse a `.env` file.
 *
 * DELIBERATELY SMALL. `KEY=value`, one per line, `#` comments, blank lines
 * ignored, an optional `export ` prefix tolerated because people paste those in,
 * and one layer of matching surrounding quotes removed. That is the whole
 * grammar.
 *
 * It does NOT strip trailing `#` comments from unquoted values, and that is the
 * one decision here worth defending: `#` is a legal character in a credential,
 * so a parser that treats it as a comment marker silently truncates a key and
 * produces a 401 that looks like a wrong key rather than a mangled one. The
 * failure mode of not stripping is a value with a stray comment on it, which
 * also 401s but is visible in the file. Quote the value if it contains a `#`.
 *
 * It does NOT expand `${VAR}` references, and does not read `.env.local` or any
 * other cascade. Layered env files are how a key ends up coming from a file
 * nobody remembers is on disk.
 */
export function parseDotEnv(source: string): Environment {
  const values: Record<string, string> = {};

  for (const rawLine of source.split(/\r?\n/)) {
    const line = rawLine.trim();
    if (line === "" || line.startsWith("#")) continue;

    const withoutExport = line.startsWith("export ") ? line.slice("export ".length).trim() : line;
    const separator = withoutExport.indexOf("=");
    if (separator <= 0) continue;

    const name = withoutExport.slice(0, separator).trim();
    if (name === "") continue;

    let value = withoutExport.slice(separator + 1).trim();
    const first = value[0];
    if (
      value.length >= 2 &&
      (first === '"' || first === "'") &&
      value[value.length - 1] === first
    ) {
      value = value.slice(1, -1);
    }

    values[name] = value;
  }

  return values;
}

/**
 * Read a `.env` file if there is one.
 *
 * A MISSING FILE IS NOT AN ERROR. In CI the keys come from the environment and
 * there is no `.env` at all; requiring one would make the normal deployment the
 * exceptional path. Any other read failure — a directory where a file should
 * be, a permissions problem — propagates, because that is someone's mistake
 * rather than a configuration choice.
 */
export async function readDotEnv(filePath: string): Promise<Environment> {
  try {
    return parseDotEnv(await readFile(filePath, "utf8"));
  } catch (error) {
    if (isMissingFile(error)) return {};
    throw error;
  }
}

function isMissingFile(error: unknown): boolean {
  return (
    typeof error === "object" &&
    error !== null &&
    (error as { code?: unknown }).code === "ENOENT"
  );
}

/**
 * Overlay a `.env` onto the process environment.
 *
 * THE PROCESS ENVIRONMENT WINS. A key exported in the shell, or injected by a
 * CI secret store, is the more deliberate act — someone typed it for this run —
 * whereas `.env` is a file that has been sitting on the disk since whenever. The
 * other precedence produces the worst debugging session available: an
 * explicitly exported key silently ignored in favour of a stale one in a file
 * the developer has forgotten exists.
 *
 * An empty or whitespace-only value in the process environment does NOT win. It
 * is what an unset CI variable expands to, and letting it shadow a real `.env`
 * value would mean `KEY=""` and no `KEY` at all behave differently for no
 * reason a user could predict.
 */
export function mergeEnvironment(processEnv: Environment, dotEnv: Environment): Environment {
  const merged: Record<string, string | undefined> = { ...dotEnv };
  for (const [name, value] of Object.entries(processEnv)) {
    if (value !== undefined && value.trim() !== "") merged[name] = value;
  }
  return merged;
}

/* ===========================================================================
 * Key resolution.
 * ======================================================================== */

/** Where a resolved value came from. The VALUE never appears in a report. */
export type KeyResolution =
  | { readonly ok: true; readonly name: string; readonly value: string }
  | { readonly ok: false; readonly name: string; readonly message: string };

/**
 * Resolve one required variable.
 *
 * The failure message names the variable, says where it can be set, and says
 * nothing whatsoever about any value — including whether a value was present
 * but empty, beyond the fact of it. "Set but blank" is worth distinguishing
 * from "absent" because the fixes differ (a broken CI secret versus a forgotten
 * one), and neither message needs a single character of the credential to make
 * that distinction.
 */
export function resolveKey(name: string, env: Environment): KeyResolution {
  const raw = env[name];

  if (raw === undefined) {
    return {
      ok: false,
      name,
      message: `${name} is not set. Export it, or put it in a .env file at the repository root as ${name}=...`,
    };
  }

  if (raw.trim() === "") {
    return {
      ok: false,
      name,
      message: `${name} is set but empty. That usually means a CI secret did not expand; check the secret exists in the runner, not the value in this log.`,
    };
  }

  return { ok: true, name, value: raw.trim() };
}

/**
 * Resolve several at once and report EVERY failure, not the first.
 *
 * Failing on the first missing key means someone sets it, re-runs, waits, and
 * finds out about the second one. Two round trips to learn two facts that were
 * both known before the first message was printed.
 */
export function resolveKeys(
  names: readonly string[],
  env: Environment,
): { readonly ok: true; readonly values: Readonly<Record<string, string>> } | {
  readonly ok: false;
  readonly messages: readonly string[];
} {
  const values: Record<string, string> = {};
  const messages: string[] = [];

  for (const name of names) {
    const resolved = resolveKey(name, env);
    if (resolved.ok) values[name] = resolved.value;
    else messages.push(resolved.message);
  }

  return messages.length > 0 ? { ok: false, messages } : { ok: true, values };
}

/* ===========================================================================
 * Model selection.
 * ======================================================================== */

/**
 * Default models, committed rather than defaulted inside the library.
 *
 * `voyageEmbedder` and `anthropicGenerator` both REQUIRE a model id and neither
 * has a default, on the grounds that `MIN_SIMILARITY` is only meaningful for
 * one embedding model — changing the model invalidates the constant outright
 * rather than shifting it. The default belongs here, at the edge, where it is
 * visible next to the environment variable that overrides it and next to the
 * artifact field that records which one actually ran.
 */
export const DEFAULT_EMBEDDING_MODEL = "voyage-3";
export const DEFAULT_GENERATION_MODEL = "claude-opus-5";

/** Read a model override, or fall back. Never throws; never reads a key. */
export function resolveModel(name: string, env: Environment, fallback: string): string {
  const raw = env[name];
  return raw === undefined || raw.trim() === "" ? fallback : raw.trim();
}

/* ===========================================================================
 * Arguments.
 * ======================================================================== */

/** What the CLI was asked to do. */
export interface CliOptions {
  /** Report the score distribution instead of running the gate. */
  readonly calibrate: boolean;
  /** Print usage and exit 0. */
  readonly help: boolean;
  /** Unrecognised flags, reported rather than ignored. */
  readonly unknown: readonly string[];
}

/**
 * Parse argv.
 *
 * UNKNOWN FLAGS ARE A SETUP ERROR, not a silent no-op. A typo'd `--calibrate`
 * that is quietly ignored runs the full gate against a live provider and bills
 * for it, and the operator reads a results table wondering where their
 * distribution went.
 */
export function parseArgs(argv: readonly string[]): CliOptions {
  let calibrate = false;
  let help = false;
  const unknown: string[] = [];

  for (const argument of argv) {
    switch (argument) {
      case "--calibrate":
        calibrate = true;
        break;
      case "--help":
      case "-h":
        help = true;
        break;
      default:
        unknown.push(argument);
    }
  }

  return { calibrate, help, unknown };
}

export const USAGE = [
  `Usage: npm run eval [-- --calibrate]`,
  ``,
  `  (no flags)    Run the committed question set through the pipeline and apply the gate.`,
  `                Requires VOYAGE_API_KEY and ANTHROPIC_API_KEY.`,
  `                Exit 0 if the gate passes, 1 if it fails, 2 on a setup error.`,
  ``,
  `  --calibrate   Score every chunk against every question and report the distribution of`,
  `                correct retrievals against wrong ones, so MIN_SIMILARITY can be set from`,
  `                data instead of from reasoning. Requires VOYAGE_API_KEY only: no answer is`,
  `                generated, so no generation key is needed and none is asked for.`,
  `                This prints a recommendation and changes nothing.`,
  ``,
  `Environment (process environment first, then .env at the repository root):`,
  `  VOYAGE_API_KEY       required`,
  `  ANTHROPIC_API_KEY    required unless --calibrate`,
  `  VOYAGE_MODEL         optional, defaults to ${DEFAULT_EMBEDDING_MODEL}`,
  `  ANTHROPIC_MODEL      optional, defaults to ${DEFAULT_GENERATION_MODEL}`,
].join("\n");

/* ===========================================================================
 * Wiring.
 * ======================================================================== */

/** Somewhere to print. Injected so `main` can be run in a test with no console. */
export interface ConsoleLike {
  log(message: string): void;
  error(message: string): void;
}

/** Paths this CLI reads and writes, resolved from the module's own location. */
export interface CliPaths {
  readonly questionSet: string;
  readonly corpusDir: string;
  readonly dotEnv: string;
  readonly resultsDir: string;
}

/** Repository-root-relative paths, resolved from `import.meta.url`. */
export function defaultPaths(): CliPaths {
  const root = new URL("../", import.meta.url);
  return {
    questionSet: fileURLToPath(new URL("eval/questions.json", root)),
    corpusDir: fileURLToPath(new URL("corpus/", root)),
    dotEnv: fileURLToPath(new URL(".env", root)),
    resultsDir: fileURLToPath(new URL(`${DEFAULT_ARTIFACT_DIR}/`, root)),
  };
}

/** Everything `main` needs from the outside world. */
export interface MainOptions {
  readonly argv: readonly string[];
  readonly env: Environment;
  readonly paths: CliPaths;
  readonly out: ConsoleLike;
  /** ISO 8601, injected so the artifact's timestamp is not read from a clock here. */
  readonly now: string;
}

/** Describe a thrown value without letting a provider's body into the log. */
function describeError(error: unknown): string {
  if (error instanceof Error) return `${error.name}: ${error.message}`;
  return String(error);
}

/**
 * Run the CLI. Returns an exit code; never calls `process.exit` itself.
 *
 * Returning the code instead of exiting is what lets a test drive the whole
 * program — argument parsing, key resolution, ordering of printed sections —
 * without a subprocess or a stubbed `process.exit`. The single call to
 * `process.exit` lives at the very bottom of this file and does nothing but
 * forward what this returns.
 */
export async function main(options: MainOptions): Promise<number> {
  const { argv, env, paths, out, now } = options;
  const args = parseArgs(argv);

  if (args.help) {
    out.log(USAGE);
    return EXIT_PASS;
  }

  if (args.unknown.length > 0) {
    out.error(`Unrecognised ${args.unknown.length === 1 ? "argument" : "arguments"}: ${args.unknown.join(", ")}`);
    out.error(USAGE);
    return EXIT_SETUP_ERROR;
  }

  // ---- credentials -------------------------------------------------------

  let merged: Environment;
  try {
    merged = mergeEnvironment(env, await readDotEnv(paths.dotEnv));
  } catch (error) {
    out.error(`Could not read ${paths.dotEnv}: ${describeError(error)}`);
    return EXIT_SETUP_ERROR;
  }

  // Calibration never generates an answer, so it never needs a generation key,
  // so it never asks for one. Demanding a credential a run cannot use is how
  // people end up pasting keys into places that do not need them.
  const required = args.calibrate
    ? ["VOYAGE_API_KEY"]
    : ["VOYAGE_API_KEY", "ANTHROPIC_API_KEY"];

  const keys = resolveKeys(required, merged);
  if (!keys.ok) {
    out.error(`Cannot run the eval: ${keys.messages.length} configuration ${keys.messages.length === 1 ? "problem" : "problems"}.`);
    for (const message of keys.messages) out.error(`  - ${message}`);
    return EXIT_SETUP_ERROR;
  }

  const embeddingModel = resolveModel("VOYAGE_MODEL", merged, DEFAULT_EMBEDDING_MODEL);
  const generationModel = resolveModel("ANTHROPIC_MODEL", merged, DEFAULT_GENERATION_MODEL);

  // ---- inputs ------------------------------------------------------------

  let questions: readonly EvalQuestion[];
  let corpusDir: string;
  try {
    const questionSet = await loadQuestionSet(paths.questionSet);
    questions = questionSet.questions;
    corpusDir = paths.corpusDir;
  } catch (error) {
    out.error(`Could not load the question set at ${paths.questionSet}: ${describeError(error)}`);
    return EXIT_SETUP_ERROR;
  }

  const voyageKey = keys.values["VOYAGE_API_KEY"];
  if (voyageKey === undefined) {
    // Unreachable: `resolveKeys` returned ok, so every required name is present.
    // Written out rather than asserted so the type is narrowed honestly.
    out.error(`VOYAGE_API_KEY resolved to nothing after a successful resolution. This is a bug in eval-cli.ts.`);
    return EXIT_SETUP_ERROR;
  }

  const embedder = voyageEmbedder({ apiKey: voyageKey, model: embeddingModel });

  let chunkCount: number;
  let index;
  try {
    const chunks = await loadCorpus(corpusDir);
    chunkCount = chunks.length;
    out.log(`Indexing ${chunkCount} chunks from ${corpusDir} with ${embeddingModel}...`);
    index = await buildIndex(chunks, embedder);
  } catch (error) {
    out.error(`Could not build the index: ${describeError(error)}`);
    return EXIT_SETUP_ERROR;
  }

  // ---- calibration -------------------------------------------------------

  if (args.calibrate) {
    try {
      const report = await calibrateThreshold(questions, index, embedder);
      out.log(formatCalibrationReport(report));
      return EXIT_PASS;
    } catch (error) {
      out.error(`Calibration could not complete: ${describeError(error)}`);
      return EXIT_SETUP_ERROR;
    }
  }

  // ---- the gate ----------------------------------------------------------

  const anthropicKey = keys.values["ANTHROPIC_API_KEY"];
  if (anthropicKey === undefined) {
    out.error(`ANTHROPIC_API_KEY resolved to nothing after a successful resolution. This is a bug in eval-cli.ts.`);
    return EXIT_SETUP_ERROR;
  }

  const generator = anthropicGenerator({ apiKey: anthropicKey, model: generationModel });

  let report;
  try {
    out.log(`Running ${questions.length} questions against ${generationModel}...`);
    report = applyGate(await runEvaluation(questions, { index, embedder, generator }));
  } catch (error) {
    // An outage is not a verdict. Nothing is written and the exit code says
    // "setup", not "fail" — see the exit-code note at the top of this file.
    out.error(`The eval did not run to completion, so there is no verdict: ${describeError(error)}`);
    out.error(`No artifact was written. A partial run cannot be scored against a threshold.`);
    return EXIT_SETUP_ERROR;
  }

  out.log("");
  out.log(formatGateReport(report));

  const metadata: RunMetadata = {
    generatedAt: now,
    embedder: embeddingModel,
    generator: generationModel,
    corpusDir,
    chunkCount,
    k: DEFAULT_K,
    threshold: MIN_SIMILARITY,
  };

  try {
    const written = await writeArtifact(paths.resultsDir, buildArtifact(report, metadata));
    out.log("");
    for (const file of written) out.log(`artifact: ${file}`);
  } catch (error) {
    // The verdict already printed. Losing the file is bad, and it does not
    // change what the eval found, so it does not change the exit code either.
    out.error(`The verdict above stands, but the artifact could not be written: ${describeError(error)}`);
  }

  return report.passed ? EXIT_PASS : EXIT_GATE_FAILED;
}

/* ===========================================================================
 * Entry point.
 * ======================================================================== */

/**
 * THE ONE READ OF `process.env` IN THIS REPOSITORY.
 *
 * Guarded so that importing this module from a test does not run the eval.
 */
if (process.argv[1] !== undefined && import.meta.url === pathToFileURL(process.argv[1]).href) {
  const code = await main({
    argv: process.argv.slice(2),
    env: process.env,
    paths: defaultPaths(),
    out: {
      log: (message: string) => console.log(message),
      error: (message: string) => console.error(message),
    },
    now: new Date().toISOString(),
  });
  process.exit(code);
}
