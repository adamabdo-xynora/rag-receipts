import { readFileSync } from "node:fs";
import { createRequire } from "node:module";
import { dirname, join } from "node:path";
import { describe, expect, it } from "vitest";

import Anthropic, { type ClientOptions } from "@anthropic-ai/sdk";

/**
 * Toolchain smoke tests. These assert that the pinned Anthropic SDK is wired up
 * correctly under ESM + NodeNext + strict typechecking. They make no network
 * calls and read no environment variables — the API key below is a literal
 * placeholder, and process.env is off limits everywhere in this project except
 * the single designated CLI entry point.
 */

const FAKE_API_KEY = "sk-ant-fake-key-for-tests";

/** Compile-time assertion: fails `tsc --noEmit` if the argument isn't type T. */
const expectType = <T>(_value: T): void => {};

describe("toolchain", () => {
  it("resolves the Anthropic SDK entry point at runtime", () => {
    const require = createRequire(import.meta.url);
    const entryPoint = require.resolve("@anthropic-ai/sdk");

    expect(entryPoint).toContain("@anthropic-ai/sdk");
    // The ESM default export is the client constructor, not a namespace object.
    expect(typeof Anthropic).toBe("function");
    expect(Anthropic.name).toBe("Anthropic");

    // The package does not expose ./package.json through "exports", so read it
    // off disk next to the resolved entry point to confirm the exact pin.
    const manifest: unknown = JSON.parse(
      readFileSync(join(dirname(entryPoint), "package.json"), "utf8"),
    );
    expect(manifest).toMatchObject({
      name: "@anthropic-ai/sdk",
      version: "0.120.0",
    });
  });

  it("resolves SDK types under strict NodeNext typechecking", () => {
    // Each annotation below is load-bearing: it only compiles if the SDK's
    // NodeNext type entry point resolves and exports these names.
    const model: Anthropic.Model = "claude-opus-5";
    const message: Anthropic.MessageParam = { role: "user", content: "ping" };

    const params = {
      model,
      max_tokens: 16000,
      messages: [message],
    } satisfies Anthropic.MessageCreateParamsNonStreaming;

    expectType<Anthropic.Model>(params.model);
    expectType<readonly Anthropic.MessageParam[]>(params.messages);

    // noUncheckedIndexedAccess is on, so indexing widens to `| undefined`.
    const first: Anthropic.MessageParam | undefined = params.messages[0];
    expect(first?.role).toBe("user");

    // exactOptionalPropertyTypes is on: `undefined` is not assignable to an
    // optional property, so options must omit keys rather than blank them out.
    const options: ClientOptions = { apiKey: FAKE_API_KEY };
    expect(options.apiKey).toBe(FAKE_API_KEY);
  });

  it("constructs a client from an injected API key", () => {
    const client = new Anthropic({ apiKey: FAKE_API_KEY });

    expect(client).toBeInstanceOf(Anthropic);
    expect(client.apiKey).toBe(FAKE_API_KEY);
    expect(typeof client.messages.create).toBe("function");
  });
});
