import { describe, it, expect, vi } from "vitest";
import type { Context, Model, SimpleStreamOptions, StreamFn } from "../../llm/types.js";
import { AssistantMessageEventStream } from "../llm/utils/event-stream.js";
import { wrapContextHashStreamFn, fingerprintSafeOptions } from "./context-hash-stream-wrapper.js";
import { sha256, canonicalizeValue } from "./context-hash-utils.js";

// ── Test helpers ────────────────────────────────────────────────────────────

const testModel: Model = {
  id: "DeepSeek-V4-Flash",
  name: "DeepSeek V4 Flash",
  api: "openai-completions" as const,
  provider: "exo",
  baseUrl: "http://localhost:8080",
  reasoning: false,
  input: ["text" as const],
  cost: { input: 0, output: 0, cacheRead: 0, cacheWrite: 0 } as const,
  maxTokens: 4096,
} satisfies Model;

const testContext: Context = {
  systemPrompt: "Be helpful.",
  tools: [{ name: "read", description: "Read", parameters: { type: "object" } }],
  messages: [{ role: "user" as const, content: "hello", timestamp: 1 }],
};

const testOptions: SimpleStreamOptions = { temperature: 0.7 };

const defaultMeta = { runtimeChannel: "telegram", runId: "run-test" };

function makeStreamFn(): { streamFn: StreamFn; sentinel: AssistantMessageEventStream } {
  const sentinel = new AssistantMessageEventStream();
  const streamFn: StreamFn = () => sentinel;
  return { streamFn, sentinel };
}

// ── fingerprintSafeOptions ──────────────────────────────────────────────────

describe("fingerprintSafeOptions", () => {
  it("excludes apiKey, headers, signal, callbacks, metadata", () => {
    const result = fingerprintSafeOptions({
      temperature: 0.7,
      maxTokens: 4096,
      apiKey: "sk-secret",
      headers: { Authorization: "***" },
      signal: new AbortController().signal,
      onPayload: () => undefined,
      onResponse: () => undefined,
      onActiveResponse: (() => undefined) as any,
      metadata: { user_id: "abc" },
    });
    const canon = canonicalizeValue({ temperature: 0.7, maxTokens: 4096 });
    expect(result.optionsH).toBe(sha256(canon));
    expect(result.optionsKeys).toBe(2);
  });

  it("redacts nested secret-bearing keys in provider extras", () => {
    const result = fingerprintSafeOptions({
      temperature: 0.7,
      extra_body: { apiKey: "nested-secret", safe_field: "hello" },
    });
    const canon = canonicalizeValue({
      temperature: 0.7,
      extra_body: { apiKey: "[redacted]", safe_field: "hello" },
    });
    expect(result.optionsH).toBe(sha256(canon));
  });

  it("omits function values entirely", () => {
    const result = fingerprintSafeOptions({
      temperature: 0.7,
      customCallback: (() => "secret") as any,
    });
    const canon = canonicalizeValue({ temperature: 0.7 });
    expect(result.optionsH).toBe(sha256(canon));
    expect(result.optionsKeys).toBe(1);
  });

  it("omits AbortSignal-like values entirely", () => {
    const result = fingerprintSafeOptions({
      temperature: 0.7,
      signal: new AbortController().signal,
    });
    const canon = canonicalizeValue({ temperature: 0.7 });
    expect(result.optionsH).toBe(sha256(canon));
    expect(result.optionsKeys).toBe(1);
  });

  it("omits nested callback/signal entries (key removed)", () => {
    const result = fingerprintSafeOptions({
      temperature: 0.7,
      extra_body: { myCallback: (() => "x") as any, innerSignal: new AbortController().signal },
    });
    const canon = canonicalizeValue({ temperature: 0.7 });
    expect(result.optionsH).toBe(sha256(canon));
  });

  it("detects prompt-affecting option changes", () => {
    expect(fingerprintSafeOptions({ temperature: 0.7 }).optionsH).not.toBe(
      fingerprintSafeOptions({ temperature: 0.9 }).optionsH,
    );
  });

  it("retains sessionId and promptCacheKey", () => {
    const result = fingerprintSafeOptions({
      temperature: 0.7,
      sessionId: "sess-1",
      promptCacheKey: "cache-key-1",
    });
    const canon = canonicalizeValue({
      temperature: 0.7,
      sessionId: "sess-1",
      promptCacheKey: "cache-key-1",
    });
    expect(result.optionsH).toBe(sha256(canon));
    expect(result.optionsKeys).toBe(3);
  });

  it("handles undefined options", () => {
    const result = fingerprintSafeOptions(undefined);
    expect(result.optionsH).toHaveLength(16);
    expect(result.optionsKeys).toBe(0);
  });
});

// ── wrapContextHashStreamFn ─────────────────────────────────────────────────

describe("wrapContextHashStreamFn", () => {
  it("returns original streamFn when disabled", () => {
    const { streamFn } = makeStreamFn();
    const logSpy = vi.fn();
    const wrapped = wrapContextHashStreamFn({
      streamFn,
      enabled: false,
      ...defaultMeta,
      log: logSpy,
    });
    expect(wrapped).toBe(streamFn);
    expect(logSpy).not.toHaveBeenCalled();
  });

  it("logs one line and preserves exact arguments and stream when enabled", () => {
    const { streamFn, sentinel } = makeStreamFn();
    let capturedModel: Model | undefined;
    let capturedContext: Context | undefined;
    let capturedOptions: SimpleStreamOptions | undefined;
    const capturingStreamFn: StreamFn = (m, c, o) => {
      capturedModel = m;
      capturedContext = c;
      capturedOptions = o;
      return sentinel;
    };

    const logSpy = vi.fn();
    const wrapped = wrapContextHashStreamFn({
      streamFn: capturingStreamFn,
      enabled: true,
      ...defaultMeta,
      log: logSpy,
    });

    const result = wrapped(testModel, testContext, testOptions);

    // Exact argument identity preserved
    expect(capturedModel).toBe(testModel);
    expect(capturedContext).toBe(testContext);
    expect(capturedOptions).toBe(testOptions);
    // Exact stream identity preserved
    expect(result).toBe(sentinel);

    // One log line with all required fields
    expect(logSpy).toHaveBeenCalledTimes(1);
    const line = logSpy.mock.calls[0][0] as string;
    expect(line).toContain("[context-diag-hash]");
    expect(line).toContain("runId=run-test");
    expect(line).toContain("runtimeChannel=telegram");
    expect(line).toContain("provider=exo");
    expect(line).toContain("model=DeepSeek-V4-Flash");
    expect(line).toContain("api=openai-completions");
    expect(line).toContain("syspH=");
    expect(line).toContain("roleH=");
    expect(line).toContain("msgH=");
  });

  it("inner wrapper hashes reflect transformed context values", () => {
    const { streamFn } = makeStreamFn();
    const logSpy = vi.fn();

    const innerWrapped = wrapContextHashStreamFn({
      streamFn,
      enabled: true,
      ...defaultMeta,
      log: logSpy,
    });

    const outerTransform: StreamFn = (_model, _context, options) => {
      const transformedContext: Context = {
        systemPrompt: "Be VERY helpful.",
        messages: [{ role: "user" as const, content: "transformed query", timestamp: 2 }],
      };
      return innerWrapped(testModel, transformedContext, options);
    };

    outerTransform(testModel, testContext, testOptions);

    const line = logSpy.mock.calls[0][0] as string;
    const match = line.match(/syspH=([0-9a-f]+)/);
    expect(match).not.toBeNull();
    expect(match![1]).toBe(sha256("Be VERY helpful."));
    expect(line).toContain("msgN=1");
  });

  it("sensitive option changes do not affect optionsH", () => {
    const { streamFn: sf1 } = makeStreamFn();
    const { streamFn: sf2 } = makeStreamFn();
    const log1 = vi.fn();
    const log2 = vi.fn();

    const w1 = wrapContextHashStreamFn({ streamFn: sf1, enabled: true, ...defaultMeta, log: log1 });
    const w2 = wrapContextHashStreamFn({ streamFn: sf2, enabled: true, ...defaultMeta, log: log2 });

    w1(testModel, testContext, { temperature: 0.7, apiKey: "sk-1" });
    w2(testModel, testContext, { temperature: 0.7, apiKey: "sk-2" });

    const m1 = log1.mock.calls[0][0].match(/optionsH=([0-9a-f]+)/);
    const m2 = log2.mock.calls[0][0].match(/optionsH=([0-9a-f]+)/);
    expect(m1![1]).toBe(m2![1]);
  });

  it("prompt-affecting option change changes optionsH", () => {
    const { streamFn: sf1 } = makeStreamFn();
    const { streamFn: sf2 } = makeStreamFn();
    const log1 = vi.fn();
    const log2 = vi.fn();

    const w1 = wrapContextHashStreamFn({ streamFn: sf1, enabled: true, ...defaultMeta, log: log1 });
    const w2 = wrapContextHashStreamFn({ streamFn: sf2, enabled: true, ...defaultMeta, log: log2 });

    w1(testModel, testContext, { temperature: 0.7 });
    w2(testModel, testContext, { temperature: 0.9 });

    const m1 = log1.mock.calls[0][0].match(/optionsH=([0-9a-f]+)/);
    const m2 = log2.mock.calls[0][0].match(/optionsH=([0-9a-f]+)/);
    expect(m1![1]).not.toBe(m2![1]);
  });

  it("fail-open: logger throws, original StreamFn still returns exact sentinel", () => {
    const { streamFn, sentinel } = makeStreamFn();
    const throwingLog = vi.fn().mockImplementation(() => {
      throw new Error("log failure");
    });

    const wrapped = wrapContextHashStreamFn({
      streamFn,
      enabled: true,
      ...defaultMeta,
      log: throwingLog,
    });

    const result = wrapped(testModel, testContext, testOptions);
    expect(result).toBe(sentinel);
    expect(throwingLog).toHaveBeenCalled();
  });

  it("fail-open: non-Error thrown value contained (string throw)", () => {
    const { streamFn, sentinel } = makeStreamFn();
    const logSpy = vi.fn();

    // Use a Proxy that throws a string from messages to exercise non-Error formatting
    const throwingContext = new Proxy(testContext, {
      get(target, prop) {
        if (prop === "messages") throw "messages exploded";
        return Reflect.get(target, prop);
      },
    }) as Context;

    const wrapped = wrapContextHashStreamFn({
      streamFn,
      enabled: true,
      ...defaultMeta,
      log: logSpy,
    });

    const result = wrapped(testModel, throwingContext, testOptions);
    expect(result).toBe(sentinel);

    // Should have logged an error line with the thrown string
    const errorCalls = logSpy.mock.calls.filter(
      (c) => typeof c[0] === "string" && c[0].includes("[context-diag-hash-error]"),
    );
    expect(errorCalls.length).toBeGreaterThanOrEqual(1);
    const errorLine = errorCalls[0][0] as string;
    expect(errorLine).toContain("messages exploded");
    expect(errorLine).toContain("runId=run-test");
  });

  it("omits unknown callback key from optionsH", () => {
    const { streamFn } = makeStreamFn();
    const logSpy = vi.fn();
    const wrapped = wrapContextHashStreamFn({
      streamFn,
      enabled: true,
      ...defaultMeta,
      log: logSpy,
    });

    wrapped(testModel, testContext, {
      temperature: 0.7,
      unknownCallback: (() => "x") as any,
    });

    const line = logSpy.mock.calls[0][0] as string;
    const match = line.match(/optionsH=([0-9a-f]+)/);
    const expected = sha256(canonicalizeValue({ temperature: 0.7 }));
    expect(match![1]).toBe(expected);
  });

  it("excludes metadata from optionsH", () => {
    const { streamFn } = makeStreamFn();
    const logSpy = vi.fn();
    const wrapped = wrapContextHashStreamFn({
      streamFn,
      enabled: true,
      ...defaultMeta,
      log: logSpy,
    });

    wrapped(testModel, testContext, {
      temperature: 0.7,
      metadata: { user_id: "abc" },
    });

    const line = logSpy.mock.calls[0][0] as string;
    const match = line.match(/optionsH=([0-9a-f]+)/);
    const expected = sha256(canonicalizeValue({ temperature: 0.7 }));
    expect(match![1]).toBe(expected);
  });
});
