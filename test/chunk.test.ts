import { mkdtemp, readFile, readdir, rm, writeFile } from "node:fs/promises";
import { tmpdir } from "node:os";
import { join } from "node:path";
import { fileURLToPath } from "node:url";
import { afterAll, describe, expect, it } from "vitest";

import {
  ChunkError,
  type ChunkErrorKind,
  chunkDocument,
  loadCorpus,
  loadDocument,
  slugify,
} from "../src/chunk.js";

const CORPUS_DIR = fileURLToPath(new URL("../corpus", import.meta.url));

/** The corpus is a fixed, designed battery; these numbers are load-bearing. */
const EXPECTED_DOCUMENT_COUNT = 14;
const EXPECTED_CHUNK_COUNT = 71;

/**
 * Assert that `fn` throws a ChunkError of exactly `kind`, and hand the error
 * back so the caller can check that the message names the real problem.
 */
function expectChunkError(fn: () => unknown, kind: ChunkErrorKind): ChunkError {
  let thrown: unknown;
  try {
    fn();
  } catch (error) {
    thrown = error;
  }
  expect(thrown, `expected a ChunkError of kind "${kind}"`).toBeInstanceOf(ChunkError);
  const error = thrown as ChunkError;
  expect(error.kind).toBe(kind);
  return error;
}

/** A minimal well-formed document, so each error fixture varies one thing. */
function validSource(): string {
  return [
    "---",
    "id: sample-doc",
    "title: A Sample Document",
    "docType: policy",
    "effectiveDate: 2025-02-24",
    "version: 1.0",
    "---",
    "",
    "## First Section",
    "",
    "Body of the first section.",
    "",
    "## Second Section",
    "",
    "Body of the second section.",
    "",
  ].join("\n");
}

describe("slugify", () => {
  it("lowercases, collapses non-alphanumeric runs, and trims hyphens", () => {
    expect(slugify("Zone C Freight Surcharge")).toBe("zone-c-freight-surcharge");
    // An em dash surrounded by spaces is one run of non-alphanumerics, not three.
    expect(slugify("Cocoa Powder — Storage and Shelf Life")).toBe(
      "cocoa-powder-storage-and-shelf-life",
    );
    expect(slugify("Why Was My Order Held?")).toBe("why-was-my-order-held");
    expect(slugify("  Leading and trailing  ")).toBe("leading-and-trailing");
    expect(slugify("$45 / drop (per-drop)")).toBe("45-drop-per-drop");
    expect(slugify("Already-Hyphenated")).toBe("already-hyphenated");
    expect(slugify("Digits 2025 Stay")).toBe("digits-2025-stay");
    expect(slugify("---")).toBe("");
  });

  it("is deterministic", () => {
    const heading = "Redelivery After a Failed Attempt";
    expect(slugify(heading)).toBe(slugify(heading));
    expect(slugify(heading)).toBe("redelivery-after-a-failed-attempt");
  });
});

describe("chunkDocument", () => {
  it("produces one chunk per ## section, carrying the full document metadata", () => {
    const chunks = chunkDocument("sample-doc.md", validSource());

    expect(chunks).toHaveLength(2);
    expect(chunks[0]).toEqual({
      chunkId: "sample-doc#first-section",
      docId: "sample-doc",
      docTitle: "A Sample Document",
      docType: "policy",
      effectiveDate: "2025-02-24",
      version: "1.0",
      sectionHeading: "First Section",
      text: "Body of the first section.",
    });
    expect(chunks[1]?.chunkId).toBe("sample-doc#second-section");
    expect(chunks[1]?.text).toBe("Body of the second section.");
  });

  it("keeps every frontmatter value a string, with no type coercion", () => {
    const chunks = chunkDocument("sample-doc.md", validSource());
    // A YAML parser would hand back the number 1.0 and lose the trailing zero.
    expect(chunks[0]?.version).toBe("1.0");
    expect(chunks[0]?.effectiveDate).toBe("2025-02-24");
  });

  it("generates identical output for identical input", () => {
    const source = validSource();
    expect(chunkDocument("sample-doc.md", source)).toEqual(
      chunkDocument("sample-doc.md", source),
    );
  });

  it("preserves interior blank lines and whitespace inside a section body", () => {
    const source = [
      "---",
      "id: sample-doc",
      "title: A Sample Document",
      "docType: policy",
      "effectiveDate: 2025-02-24",
      "version: 1.0",
      "---",
      "",
      "## Only Section",
      "",
      "First paragraph.",
      "",
      "  Indented line with  two  interior  spaces.",
      "",
      "Last paragraph.",
      "",
    ].join("\n");

    const text = chunkDocument("sample-doc.md", source)[0]?.text;
    expect(text).toBe(
      "First paragraph.\n\n  Indented line with  two  interior  spaces.\n\nLast paragraph.",
    );
    // The whole point: it is still a literal substring of the file.
    expect(source.indexOf(text ?? "")).toBeGreaterThan(-1);
  });

  it("does not rewrite CRLF line endings inside a section body", () => {
    const source = validSource().replace(/\n/g, "\r\n");
    const text = chunkDocument("sample-doc.md", source)[0]?.text ?? "";

    expect(text).toBe("Body of the first section.");
    expect(source.indexOf(text)).toBeGreaterThan(-1);
  });

  it("treats ### subheadings as body text, not as new chunks", () => {
    const source = [
      "---",
      "id: sample-doc",
      "title: A Sample Document",
      "docType: policy",
      "effectiveDate: 2025-02-24",
      "version: 1.0",
      "---",
      "",
      "## Only Section",
      "",
      "### A subheading",
      "",
      "Body text.",
      "",
    ].join("\n");

    const chunks = chunkDocument("sample-doc.md", source);
    expect(chunks).toHaveLength(1);
    expect(chunks[0]?.text).toBe("### A subheading\n\nBody text.");
  });
});

describe("chunkDocument error paths", () => {
  it("rejects a file with no opening frontmatter delimiter", () => {
    const error = expectChunkError(
      () => chunkDocument("sample-doc.md", "## First Section\n\nBody.\n"),
      "missing-frontmatter",
    );
    expect(error.file).toBe("sample-doc.md");
    expect(error.message).toContain("sample-doc.md");
    expect(error.message).toContain("---");
  });

  it("rejects a frontmatter block that is never closed", () => {
    const source = validSource().replace("version: 1.0\n---", "version: 1.0");
    expectChunkError(() => chunkDocument("sample-doc.md", source), "unterminated-frontmatter");
  });

  it("rejects a frontmatter line that is not a key: value pair", () => {
    const source = validSource().replace("docType: policy", "docType policy");
    const error = expectChunkError(
      () => chunkDocument("sample-doc.md", source),
      "malformed-frontmatter-line",
    );
    expect(error.message).toContain("docType policy");
  });

  it("rejects a frontmatter key with an empty value instead of defaulting it", () => {
    const source = validSource().replace("version: 1.0", "version:");
    expectChunkError(() => chunkDocument("sample-doc.md", source), "malformed-frontmatter-line");
  });

  it("rejects a repeated frontmatter key", () => {
    const source = validSource().replace("version: 1.0", "version: 1.0\nversion: 2.0");
    const error = expectChunkError(() => chunkDocument("sample-doc.md", source), "duplicate-key");
    expect(error.message).toContain("version");
  });

  it("rejects an unknown frontmatter key", () => {
    const source = validSource().replace("version: 1.0", "version: 1.0\nauthor: Someone");
    const error = expectChunkError(() => chunkDocument("sample-doc.md", source), "unknown-key");
    expect(error.message).toContain("author");
  });

  it("rejects a missing frontmatter key and names it", () => {
    const source = validSource().replace("effectiveDate: 2025-02-24\n", "");
    const error = expectChunkError(() => chunkDocument("sample-doc.md", source), "missing-key");
    expect(error.message).toContain("effectiveDate");
  });

  it("names every missing key at once", () => {
    const source = ["---", "id: sample-doc", "---", "", "## S", "", "Body.", ""].join("\n");
    const error = expectChunkError(() => chunkDocument("sample-doc.md", source), "missing-key");
    expect(error.message).toContain("title");
    expect(error.message).toContain("docType");
    expect(error.message).toContain("effectiveDate");
    expect(error.message).toContain("version");
  });

  it("rejects an id that disagrees with the filename stem", () => {
    const error = expectChunkError(
      () => chunkDocument("a-different-name.md", validSource()),
      "id-filename-mismatch",
    );
    expect(error.message).toContain("sample-doc");
    expect(error.message).toContain("a-different-name");
  });

  it("rejects text before the first ## heading rather than dropping it", () => {
    const source = validSource().replace(
      "## First Section",
      "This sentence belongs to no section.\n\n## First Section",
    );
    const error = expectChunkError(() => chunkDocument("sample-doc.md", source), "preamble-text");
    expect(error.message).toContain("This sentence belongs to no section.");
  });

  it("rejects a document with no sections", () => {
    const source = validSource().split("## First Section")[0] ?? "";
    expectChunkError(() => chunkDocument("sample-doc.md", source), "no-sections");
  });

  it("rejects a section with an empty body", () => {
    const source = validSource().replace("Body of the second section.\n", "");
    const error = expectChunkError(() => chunkDocument("sample-doc.md", source), "empty-section");
    expect(error.message).toContain("Second Section");
  });

  it("rejects a heading that slugifies to nothing", () => {
    const source = validSource().replace("## Second Section", "## ???");
    expectChunkError(() => chunkDocument("sample-doc.md", source), "empty-slug");
  });

  it("rejects two headings that slugify identically instead of suffixing them", () => {
    // Distinct headings, identical slug — exactly the case a `-2` suffix would hide.
    const source = validSource()
      .replace("## First Section", "## Zone C: Freight Surcharge")
      .replace("## Second Section", "## Zone C — Freight Surcharge");

    const error = expectChunkError(() => chunkDocument("sample-doc.md", source), "slug-collision");
    expect(error.message).toContain("zone-c-freight-surcharge");
    expect(error.message).toContain("Zone C: Freight Surcharge");
    expect(error.message).toContain("Zone C — Freight Surcharge");
  });
});

describe("verbatim text preservation against the real corpus", () => {
  it("finds every chunk's text verbatim in its own source file", async () => {
    const files = (await readdir(CORPUS_DIR)).sort();
    expect(files).toHaveLength(EXPECTED_DOCUMENT_COUNT);

    for (const file of files) {
      // Read the file raw, independently of the module under test.
      const raw = await readFile(join(CORPUS_DIR, file), "utf8");
      const chunks = await loadDocument(join(CORPUS_DIR, file));
      expect(chunks.length).toBeGreaterThan(0);

      for (const chunk of chunks) {
        expect(
          raw.indexOf(chunk.text),
          `${chunk.chunkId} is not a verbatim substring of ${file}`,
        ).toBeGreaterThan(-1);
      }
    }
  });

  it("trims only the two ends of a section body", async () => {
    // Trimming the ends is the one permitted transformation, so no chunk starts
    // or ends with whitespace — and no chunk is empty.
    const chunks = await loadCorpus(CORPUS_DIR);
    for (const chunk of chunks) {
      expect(chunk.text).toBe(chunk.text.trim());
      expect(chunk.text.length).toBeGreaterThan(0);
    }
  });

  it("finds each section in source order, so no interior text was altered", async () => {
    // A stronger statement than containment: nothing between the headings was
    // altered, so a quote spanning any part of a section still verifies.
    const file = "delivery-zones-and-schedules.md";
    const raw = await readFile(join(CORPUS_DIR, file), "utf8");
    const chunks = await loadDocument(join(CORPUS_DIR, file));

    let cursor = 0;
    for (const chunk of chunks) {
      const found = raw.indexOf(chunk.text, cursor);
      expect(found, `${chunk.chunkId} out of order or altered`).toBeGreaterThan(-1);
      cursor = found + chunk.text.length;
    }
  });
});

describe("loadCorpus", () => {
  it("loads the whole corpus with the expected document and chunk counts", async () => {
    const chunks = await loadCorpus(CORPUS_DIR);

    expect(chunks).toHaveLength(EXPECTED_CHUNK_COUNT);
    expect(new Set(chunks.map((chunk) => chunk.docId)).size).toBe(EXPECTED_DOCUMENT_COUNT);
  });

  it("contains the known chunk ids, spelled exactly as a human would derive them", async () => {
    const chunks = await loadCorpus(CORPUS_DIR);
    const byId = new Map(chunks.map((chunk) => [chunk.chunkId, chunk]));

    const freight = byId.get("delivery-zones-and-schedules#zone-c-freight-surcharge");
    expect(freight).toBeDefined();
    expect(freight?.sectionHeading).toBe("Zone C Freight Surcharge");
    expect(freight?.docTitle).toBe("Delivery Zones, Schedules, and Freight Surcharges");
    expect(freight?.docType).toBe("process");
    expect(freight?.effectiveDate).toBe("2025-02-24");
    expect(freight?.version).toBe("5.1");
    expect(freight?.text).toContain("flat freight surcharge of $45");

    const cocoa = byId.get("dry-goods-product-specs#cocoa-powder-storage-and-shelf-life");
    expect(cocoa).toBeDefined();
    expect(cocoa?.sectionHeading).toBe("Cocoa Powder — Storage and Shelf Life");
    expect(cocoa?.docType).toBe("product");
    expect(cocoa?.version).toBe("3.1");
    expect(cocoa?.text).toContain("540-day shelf life");
  });

  it("gives every chunk a unique id across the whole corpus", async () => {
    const chunks = await loadCorpus(CORPUS_DIR);
    const ids = chunks.map((chunk) => chunk.chunkId);

    expect(new Set(ids).size).toBe(ids.length);
    for (const chunk of chunks) {
      expect(chunk.chunkId).toBe(`${chunk.docId}#${slugify(chunk.sectionHeading)}`);
    }
  });

  it("returns chunks in filename order, then document order, deterministically", async () => {
    const first = await loadCorpus(CORPUS_DIR);
    const second = await loadCorpus(CORPUS_DIR);

    expect(first).toEqual(second);

    const docOrder: string[] = [];
    for (const chunk of first) {
      if (docOrder[docOrder.length - 1] !== chunk.docId) docOrder.push(chunk.docId);
    }
    expect(docOrder).toEqual([...docOrder].sort());
    // Each document appears as one contiguous run, not interleaved.
    expect(new Set(docOrder).size).toBe(docOrder.length);
  });
});

describe("loadCorpus failure modes", () => {
  const scratchDirs: string[] = [];

  async function scratchCorpus(files: Record<string, string>): Promise<string> {
    const dir = await mkdtemp(join(tmpdir(), "rag-receipts-corpus-"));
    scratchDirs.push(dir);
    for (const [name, contents] of Object.entries(files)) {
      await writeFile(join(dir, name), contents, "utf8");
    }
    return dir;
  }

  afterAll(async () => {
    for (const dir of scratchDirs) await rm(dir, { recursive: true, force: true });
  });

  it("fails the whole load when any single document is malformed", async () => {
    const dir = await scratchCorpus({
      "sample-doc.md": validSource(),
      "zz-broken-doc.md": "## No frontmatter here\n\nBody.\n",
    });

    await expect(loadCorpus(dir)).rejects.toThrow(ChunkError);
    await expect(loadCorpus(dir)).rejects.toMatchObject({
      kind: "missing-frontmatter",
      file: "zz-broken-doc.md",
    });
  });

  it("refuses to silently skip an entry that is not a Markdown file", async () => {
    const dir = await scratchCorpus({
      "sample-doc.md": validSource(),
      "notes.txt": "not a corpus document",
    });

    await expect(loadCorpus(dir)).rejects.toMatchObject({
      kind: "non-markdown-entry",
      file: "notes.txt",
    });
  });
});
