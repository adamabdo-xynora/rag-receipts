/**
 * chunk.ts — corpus documents in, citable chunks out.
 *
 * PURE MODULE. It imports `node:fs/promises` and `node:path` and nothing else:
 * no SDK, no third-party libraries, no `process.env`. Every value it produces is
 * a deterministic function of bytes on disk, so its tests need no network, no
 * secrets, and no fixtures beyond the corpus itself.
 *
 * FRONTMATTER IS PARSED BY HAND, ON PURPOSE. The block is a fixed five-key list
 * of flat strings (id, title, docType, effectiveDate, version). A full YAML
 * parser is a large dependency — and a large bug and supply-chain surface — for
 * five flat string keys, and its permissiveness works against us here: YAML
 * would happily coerce `version: 5.1` to the number 5.1, accept keys we never
 * meant to allow, and fill in defaults where we want a loud failure. This
 * parser understands nothing but those five keys, keeps every value a string,
 * and refuses everything else by name.
 *
 * FAIL LOUDLY, NEVER PARTIALLY. Every failure is a `ChunkError` carrying a
 * machine-readable `kind`, the offending file, and a message that names the
 * problem. There are no silent defaults, no dropped sections, and no partial
 * documents: a malformed file fails its own load, and a malformed file anywhere
 * in the corpus fails the whole corpus load.
 */

/* ===========================================================================
 * INVARIANT — VERBATIM PRESERVATION.
 * ===========================================================================
 * `chunk.text` is a byte-exact substring of the section body in the source
 * file on disk. The one and only permitted transformation is trimming
 * whitespace off the two ends. Nothing else: interior whitespace is not
 * collapsed, smart quotes and em dashes are not normalised, lines are not
 * reflowed, blank lines are not dropped, line endings are not rewritten.
 *
 * WHY IT MATTERS. Downstream, a model-supplied quote is verified by plain
 * string containment against `chunk.text`. That containment check is this
 * project's entire receipt: if a quote is not literally present in the chunk
 * it claims to come from, the answer is not grounded and we say so.
 *
 * So normalisation here does not merely lose fidelity — it makes an *honest*
 * quote fail verification. And the failure does not present as a chunking bug.
 * It presents as the model having invented a quote. Whoever debugs it goes
 * hunting for a hallucination that never happened, in the model layer, in a
 * prompt that is fine, while the actual cause sits in this file. Nothing
 * crashes and nothing is logged; the receipts just quietly stop meaning
 * anything, and the investigation is aimed at the wrong bug from the first
 * minute.
 *
 * A caller that wants normalised text should normalise its own copy and leave
 * `chunk.text` as the thing quotes are checked against.
 *
 * ENFORCED BY `test/chunk.test.ts`, describe block "verbatim text preservation
 * against the real corpus":
 *   - "finds every chunk's text verbatim in its own source file" reads each
 *     corpus file raw off disk and asserts containment via `indexOf`.
 *   - "trims only the two ends of a section body" pins the single permitted
 *     transformation.
 *   - "finds each section in source order, so no interior text was altered"
 *     rules out interior edits that containment alone could miss.
 * If you are about to add a normalisation step here, one of those three will
 * stop you. Leave it stopping you.
 * ===========================================================================
 */

import { readFile, readdir } from "node:fs/promises";
import { basename, join } from "node:path";

/** The five frontmatter keys. Exactly these, no more and no fewer. */
const FRONTMATTER_KEYS = ["id", "title", "docType", "effectiveDate", "version"] as const;

type FrontmatterKey = (typeof FRONTMATTER_KEYS)[number];

/** A parsed frontmatter block. Every value stays a string — no type coercion. */
export type Frontmatter = { readonly [K in FrontmatterKey]: string };

/** The delimiter line that opens and closes a frontmatter block. */
const DELIMITER = "---";

/**
 * One `##` section of one document, with everything a citation needs to be
 * checked by a human: which document, which section, which version of it.
 */
export interface Chunk {
  /** `<docId>#<section-slug>` — deterministic, and readable at a glance. */
  readonly chunkId: string;
  readonly docId: string;
  readonly docTitle: string;
  readonly docType: string;
  readonly effectiveDate: string;
  readonly version: string;
  /** The `##` heading text, as written. */
  readonly sectionHeading: string;
  /** Byte-exact substring of the source file. See the header comment. */
  readonly text: string;
}

/** Every way a document can be rejected. Each one is covered by a test. */
export type ChunkErrorKind =
  /** File does not open with a `---` delimiter line. */
  | "missing-frontmatter"
  /** Opening `---` is never closed by a second `---`. */
  | "unterminated-frontmatter"
  /** A frontmatter line is not a `key: value` pair with a non-empty value. */
  | "malformed-frontmatter-line"
  /** A frontmatter key appears twice. */
  | "duplicate-key"
  /** A frontmatter key outside the fixed five. */
  | "unknown-key"
  /** One of the fixed five is absent. */
  | "missing-key"
  /** Frontmatter `id` disagrees with the filename stem. */
  | "id-filename-mismatch"
  /** Text sits between the frontmatter and the first `##` heading. */
  | "preamble-text"
  /** The document has no `##` sections at all. */
  | "no-sections"
  /** A `##` section has an empty body. */
  | "empty-section"
  /** A heading slugifies to the empty string. */
  | "empty-slug"
  /** Two headings in one document slugify identically. */
  | "slug-collision"
  /** The corpus directory holds something that is not a Markdown file. */
  | "non-markdown-entry";

/** Thrown for every rejection. Carries the file and a machine-readable kind. */
export class ChunkError extends Error {
  readonly kind: ChunkErrorKind;
  /** The file the problem is in, as a bare filename. */
  readonly file: string;

  constructor(kind: ChunkErrorKind, file: string, problem: string) {
    super(`${file}: ${problem}`);
    this.name = "ChunkError";
    this.kind = kind;
    this.file = file;
  }
}

/**
 * Build a section slug: lowercased, every run of non-alphanumerics collapsed to
 * a single hyphen, hyphens trimmed off both ends.
 *
 * The point is human-checkability. `delivery-zones-and-schedules#zone-c-freight-surcharge`
 * can be traced back to a heading in a file in about two seconds, without
 * running anything. That is why collisions are an error rather than something
 * we paper over with a `-2` suffix (see `chunkDocument`): a suffixed id is no
 * longer derivable from the heading, and the promise breaks.
 */
export function slugify(heading: string): string {
  return heading
    .toLowerCase()
    .replace(/[^a-z0-9]+/g, "-")
    .replace(/^-+|-+$/g, "");
}

/** A source line plus its offsets, so bodies can be sliced from raw source. */
interface SourceLine {
  /** Line content without its terminator (and without a trailing `\r`). */
  readonly text: string;
  /** Offset of the line's first character in the source. */
  readonly start: number;
  /** Offset just past the line's terminator. */
  readonly end: number;
}

/**
 * Split source into lines while keeping offsets. Offsets — not the split text —
 * are what section bodies get sliced with, so a CRLF file keeps its CRLFs in
 * `chunk.text` even though this strips `\r` for the purpose of matching
 * delimiters and headings.
 */
function scanLines(source: string): SourceLine[] {
  const lines: SourceLine[] = [];
  let start = 0;

  for (;;) {
    const newline = source.indexOf("\n", start);
    const stop = newline === -1 ? source.length : newline;
    const raw = source.slice(start, stop);
    lines.push({
      text: raw.endsWith("\r") ? raw.slice(0, -1) : raw,
      start,
      end: newline === -1 ? source.length : newline + 1,
    });
    if (newline === -1) return lines;
    start = newline + 1;
  }
}

/** A line is a section heading iff it opens with exactly `## `. */
function isHeading(text: string): boolean {
  return text.startsWith("## ");
}

/** Keep error messages readable when the offending line is long. */
function quoteSnippet(text: string): string {
  const trimmed = text.length > 60 ? `${text.slice(0, 60)}…` : text;
  return JSON.stringify(trimmed);
}

function isFrontmatterKey(key: string): key is FrontmatterKey {
  return (FRONTMATTER_KEYS as readonly string[]).includes(key);
}

/**
 * Parse the frontmatter block and report where the body starts.
 * Rejects anything it does not fully understand.
 */
function parseFrontmatter(
  file: string,
  lines: readonly SourceLine[],
): { frontmatter: Frontmatter; bodyLineIndex: number } {
  const first = lines[0];
  if (first === undefined || first.text !== DELIMITER) {
    throw new ChunkError(
      "missing-frontmatter",
      file,
      `expected the first line to be the frontmatter delimiter "---", found ${quoteSnippet(first?.text ?? "")}`,
    );
  }

  let closing = -1;
  for (let i = 1; i < lines.length; i += 1) {
    const line = lines[i];
    if (line !== undefined && line.text === DELIMITER) {
      closing = i;
      break;
    }
  }
  if (closing === -1) {
    throw new ChunkError(
      "unterminated-frontmatter",
      file,
      'the opening "---" is never closed by a second "---" line',
    );
  }

  const values = new Map<FrontmatterKey, string>();
  for (let i = 1; i < closing; i += 1) {
    const line = lines[i];
    if (line === undefined) continue;

    const match = /^([A-Za-z][A-Za-z0-9_-]*):[ \t]*(\S.*)$/.exec(line.text);
    if (match === null) {
      throw new ChunkError(
        "malformed-frontmatter-line",
        file,
        `frontmatter line ${i + 1} is not a "key: value" pair with a non-empty value: ${quoteSnippet(line.text)}`,
      );
    }

    const key = match[1] ?? "";
    const value = (match[2] ?? "").trimEnd();

    if (!isFrontmatterKey(key)) {
      throw new ChunkError(
        "unknown-key",
        file,
        `unknown frontmatter key "${key}" on line ${i + 1}; the block accepts exactly ${FRONTMATTER_KEYS.join(", ")}`,
      );
    }
    if (values.has(key)) {
      throw new ChunkError(
        "duplicate-key",
        file,
        `frontmatter key "${key}" appears more than once`,
      );
    }
    values.set(key, value);
  }

  const missing = FRONTMATTER_KEYS.filter((key) => !values.has(key));
  if (missing.length > 0) {
    throw new ChunkError(
      "missing-key",
      file,
      `frontmatter is missing required ${missing.length === 1 ? "key" : "keys"}: ${missing.join(", ")}`,
    );
  }

  const frontmatter: Frontmatter = {
    id: values.get("id") ?? "",
    title: values.get("title") ?? "",
    docType: values.get("docType") ?? "",
    effectiveDate: values.get("effectiveDate") ?? "",
    version: values.get("version") ?? "",
  };

  return { frontmatter, bodyLineIndex: closing + 1 };
}

/**
 * Turn one document into its chunks — one chunk per `##` section.
 *
 * `filePath` is used for the filename-stem check and to name the file in errors;
 * `source` is the file's exact contents. Splitting the read from the parse keeps
 * this half of the module synchronous and trivially testable against strings.
 */
export function chunkDocument(filePath: string, source: string): Chunk[] {
  const file = basename(filePath);
  const stem = file.endsWith(".md") ? file.slice(0, -".md".length) : file;
  const lines = scanLines(source);

  const { frontmatter, bodyLineIndex } = parseFrontmatter(file, lines);

  if (frontmatter.id !== stem) {
    throw new ChunkError(
      "id-filename-mismatch",
      file,
      `frontmatter id "${frontmatter.id}" does not match the filename stem "${stem}"; every chunk id begins with the doc id, so the two must agree for a citation to be traceable back to a file`,
    );
  }

  // Everything between the frontmatter and the first heading belongs to no
  // section. Dropping it would make it uncitable and unretrievable, and nobody
  // would notice, so it is an error.
  let firstHeadingIndex = -1;
  for (let i = bodyLineIndex; i < lines.length; i += 1) {
    const line = lines[i];
    if (line === undefined) continue;
    if (isHeading(line.text)) {
      firstHeadingIndex = i;
      break;
    }
    if (line.text.trim() !== "") {
      throw new ChunkError(
        "preamble-text",
        file,
        `line ${i + 1} carries text before the first "## " heading: ${quoteSnippet(line.text)}. Preamble text belongs to no section, so it would be silently unretrievable and uncitable — move it under a heading`,
      );
    }
  }

  if (firstHeadingIndex === -1) {
    throw new ChunkError(
      "no-sections",
      file,
      'no "## " section headings found; the document would produce no citable chunks',
    );
  }

  const headingIndices: number[] = [];
  for (let i = firstHeadingIndex; i < lines.length; i += 1) {
    const line = lines[i];
    if (line !== undefined && isHeading(line.text)) headingIndices.push(i);
  }

  const chunks: Chunk[] = [];
  const slugOwner = new Map<string, string>();

  for (let n = 0; n < headingIndices.length; n += 1) {
    const headingIndex = headingIndices[n];
    if (headingIndex === undefined) continue;
    const headingLine = lines[headingIndex];
    if (headingLine === undefined) continue;

    const sectionHeading = headingLine.text.slice("## ".length).trim();
    const slug = slugify(sectionHeading);
    if (slug === "") {
      throw new ChunkError(
        "empty-slug",
        file,
        `the heading on line ${headingIndex + 1} (${quoteSnippet(headingLine.text)}) has no alphanumeric characters, so it cannot produce a chunk id`,
      );
    }

    const owner = slugOwner.get(slug);
    if (owner !== undefined) {
      throw new ChunkError(
        "slug-collision",
        file,
        `headings "${owner}" and "${sectionHeading}" both slugify to "${slug}". Deduping with a numeric suffix would make the chunk id underivable from the heading and break the promise that a citation is checkable by eye — rename one of the headings instead`,
      );
    }
    slugOwner.set(slug, sectionHeading);

    // The one and only route to chunk text: slice the raw source between the
    // end of this heading line and the start of the next heading (or EOF), then
    // trim the two ends. No other transformation. See the header comment.
    const nextHeadingIndex = headingIndices[n + 1];
    const bodyStart = headingLine.end;
    const bodyEnd =
      nextHeadingIndex === undefined ? source.length : (lines[nextHeadingIndex]?.start ?? source.length);
    const text = source.slice(bodyStart, bodyEnd).trim();

    if (text === "") {
      throw new ChunkError(
        "empty-section",
        file,
        `section "${sectionHeading}" has an empty body; an empty chunk can be retrieved but never quoted`,
      );
    }

    chunks.push({
      chunkId: `${frontmatter.id}#${slug}`,
      docId: frontmatter.id,
      docTitle: frontmatter.title,
      docType: frontmatter.docType,
      effectiveDate: frontmatter.effectiveDate,
      version: frontmatter.version,
      sectionHeading,
      text,
    });
  }

  return chunks;
}

/** Read one document off disk and chunk it. */
export async function loadDocument(filePath: string): Promise<Chunk[]> {
  const source = await readFile(filePath, "utf8");
  return chunkDocument(filePath, source);
}

/**
 * Read every Markdown file in `dir`, in filename order, and return all chunks.
 *
 * No filtering. The corpus directory holds corpus documents by design, so this
 * never skips a file for looking unusual — an entry that is not a `.md` file is
 * an error, because the alternative is silently dropping a document and serving
 * a corpus that is quietly smaller than the one on disk. Likewise, one malformed
 * document fails the entire load: a half-loaded corpus produces answers that
 * look grounded and are missing their evidence.
 *
 * Filename order (codepoint order, not locale collation) makes the returned
 * array stable across machines, which keeps chunk indices reproducible.
 */
export async function loadCorpus(dir: string): Promise<Chunk[]> {
  const entries = await readdir(dir, { withFileTypes: true });
  entries.sort((a, b) => (a.name < b.name ? -1 : a.name > b.name ? 1 : 0));

  const chunks: Chunk[] = [];
  for (const entry of entries) {
    if (!entry.isFile() || !entry.name.endsWith(".md")) {
      throw new ChunkError(
        "non-markdown-entry",
        entry.name,
        `the corpus directory ${dir} contains an entry that is not a Markdown file; corpus loading refuses to skip entries, because a silently skipped document is a silently missing answer`,
      );
    }
    chunks.push(...(await loadDocument(join(dir, entry.name))));
  }

  return chunks;
}
