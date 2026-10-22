import { describe, it, expect } from "vitest";
import type { Tool, Message, Context } from "../../llm/types.js";
import {
  sha256,
  canonicalizeValue,
  canonicalizeTools,
  canonicalizeToolsSorted,
  computeRollingMsgHashes,
  computeContextHashes,
  formatContextHashLog,
} from "./context-hash-utils.js";

// ── sha256 ──────────────────────────────────────────────────────────────────

describe("sha256", () => {
  it("returns a deterministic 16-char hex digest by default", () => {
    const a = sha256("hello");
    const b = sha256("hello");
    expect(a).toBe(b);
    expect(a).toHaveLength(16);
    expect(a).toMatch(/^[0-9a-f]+$/);
  });

  it("returns different hashes for different inputs", () => {
    expect(sha256("hello")).not.toBe(sha256("world"));
  });

  it("accepts custom truncation length", () => {
    expect(sha256("hello", 8)).toHaveLength(8);
    expect(sha256("hello", 64)).toHaveLength(64);
  });
});

// ── canonicalizeValue ───────────────────────────────────────────────────────

describe("canonicalizeValue", () => {
  it("handles primitives", () => {
    expect(canonicalizeValue(null)).toBe("null");
    expect(canonicalizeValue(true)).toBe("true");
    expect(canonicalizeValue(42)).toBe("42");
    expect(canonicalizeValue("hello")).toBe('"hello"');
  });

  it("sorts object keys", () => {
    expect(canonicalizeValue({ b: 1, a: 2 })).toBe('{"a":2,"b":1}');
  });

  it("respects max depth", () => {
    const deep = { a: { b: { c: { d: { e: { f: { g: "deep" } } } } } } };
    expect(canonicalizeValue(deep, { depth: 21 })).toContain("[max-depth]");
  });
});

// ── canonicalizeTools (real Tool type, recursive canonicalization) ──────────

describe("canonicalizeTools", () => {
  it("includes nested schema properties (no JSON.stringify replacer bug)", () => {
    const tools: Tool[] = [
      {
        name: "edit",
        description: "Edit files",
        parameters: {
          type: "object",
          properties: { path: { type: "string" }, content: { type: "string" } },
          required: ["path", "content"],
        },
      },
    ];
    const result = canonicalizeTools(tools);
    expect(result).toContain("properties");
    expect(result).toContain("path");
    expect(result).toContain("required");
  });

  it("detects nested schema-only mutation", () => {
    const base: Tool[] = [
      {
        name: "edit",
        description: "Edit",
        parameters: {
          type: "object",
          properties: { path: { type: "string" } },
        },
      },
    ];
    const modified: Tool[] = [
      {
        name: "edit",
        description: "Edit",
        parameters: {
          type: "object",
          properties: { path: { type: "integer" } },
        },
      },
    ];
    // Nested schema change should change both toolH and toolSetH
    expect(canonicalizeTools(base)).not.toBe(canonicalizeTools(modified));
    expect(canonicalizeToolsSorted(base)).not.toBe(canonicalizeToolsSorted(modified));
  });

  it("is deterministic for sorted variant", () => {
    const a: Tool[] = [
      { name: "z", parameters: {} } as Tool,
      { name: "a", parameters: {} } as Tool,
    ];
    const b: Tool[] = [
      { name: "a", parameters: {} } as Tool,
      { name: "z", parameters: {} } as Tool,
    ];
    expect(canonicalizeToolsSorted(a)).toBe(canonicalizeToolsSorted(b));
  });
});

// ── Message helpers (valid llm-core literals, no `as any`) ──────────────────

function userMsg(content: string | readonly unknown[], timestamp = 1): Message {
  return { role: "user", content, timestamp } satisfies Message;
}

function assistantMsg(content: readonly unknown[], timestamp = 2): Message {
  return {
    role: "assistant",
    content,
    api: "test" as const,
    provider: "test" as const,
    model: "test",
    usage: {
      input: 0,
      output: 0,
      cacheRead: 0,
      cacheWrite: 0,
      totalTokens: 0,
      cost: { input: 0, output: 0, cacheRead: 0, cacheWrite: 0 },
    },
    stopReason: "endTurn" as const,
    timestamp,
  } satisfies Message;
}

function toolResultMsg(
  toolCallId: string,
  toolName: string,
  text: string,
  isError = false,
  timestamp = 3,
): Message {
  return {
    role: "toolResult",
    toolCallId,
    toolName,
    content: [{ type: "text" as const, text }],
    isError,
    timestamp,
  } satisfies Message;
}

// ── computeRollingMsgHashes ─────────────────────────────────────────────────

describe("computeRollingMsgHashes", () => {
  it("produces pre for checkpoints below threshold", () => {
    const r = computeRollingMsgHashes([userMsg("hi")]);
    expect(r.role64H).toBe("pre");
    expect(r.msg64H).toBe("pre");
    expect(r.roleH).toHaveLength(16);
    expect(r.msgH).toHaveLength(16);
    expect(r.msgN).toBe(1);
  });

  it("computes checkpoints at exact 64/128/256/512", () => {
    const msgs: Message[] = [];
    for (let i = 0; i < 512; i++) msgs.push(userMsg(`m${i}`, i));
    const r = computeRollingMsgHashes(msgs);
    expect(r.role64H).not.toBe("pre");
    expect(r.role128H).not.toBe("pre");
    expect(r.role256H).not.toBe("pre");
    expect(r.role512H).not.toBe("pre");
    expect(r.msg64H).not.toBe("pre");
    expect(r.msg128H).not.toBe("pre");
    expect(r.msg256H).not.toBe("pre");
    expect(r.msg512H).not.toBe("pre");
    expect(r.msgN).toBe(512);
  });

  it("appending 512→562 keeps checkpoints stable while final hashes change", () => {
    function build(n: number): Message[] {
      const msgs: Message[] = [];
      for (let i = 0; i < n; i++) msgs.push(userMsg(`m${i}`, i));
      return msgs;
    }
    const base512 = computeRollingMsgHashes(build(512));
    const extended562 = computeRollingMsgHashes(build(562));

    // Checkpoints at 64/128/256/512 identical
    expect(base512.role64H).toBe(extended562.role64H);
    expect(base512.role128H).toBe(extended562.role128H);
    expect(base512.role256H).toBe(extended562.role256H);
    expect(base512.role512H).toBe(extended562.role512H);
    expect(base512.msg64H).toBe(extended562.msg64H);
    expect(base512.msg128H).toBe(extended562.msg128H);
    expect(base512.msg256H).toBe(extended562.msg256H);
    expect(base512.msg512H).toBe(extended562.msg512H);

    // Final hashes change (more messages)
    expect(base512.roleH).not.toBe(extended562.roleH);
    expect(base512.msgH).not.toBe(extended562.msgH);
    expect(extended562.msgN).toBe(562);
  });

  it("mutation at message 490 changes msg512H but not earlier checkpoints", () => {
    function build(mutateAt490: boolean): Message[] {
      const msgs: Message[] = [];
      for (let i = 0; i < 512; i++) {
        msgs.push(mutateAt490 && i === 490 ? userMsg("DIFFERENT", i) : userMsg(`m${i}`, i));
      }
      return msgs;
    }
    const base = computeRollingMsgHashes(build(false));
    const mod = computeRollingMsgHashes(build(true));

    // Earlier checkpoints unchanged
    expect(base.role64H).toBe(mod.role64H);
    expect(base.role128H).toBe(mod.role128H);
    expect(base.role256H).toBe(mod.role256H);
    // role512H same (both "user")
    expect(base.role512H).toBe(mod.role512H);
    // msg512H changes (mutation is before 512)
    expect(base.msg512H).not.toBe(mod.msg512H);
  });

  it("is deterministic", () => {
    const msgs = [userMsg("hello"), assistantMsg([])];
    expect(computeRollingMsgHashes(msgs)).toEqual(computeRollingMsgHashes(msgs));
  });

  it("role-prefix stays equal while content changes", () => {
    const base = [userMsg("hello"), userMsg("world")];
    const mod = [userMsg("hello"), userMsg("different")];
    expect(computeRollingMsgHashes(base).roleH).toBe(computeRollingMsgHashes(mod).roleH);
    expect(computeRollingMsgHashes(base).msgH).not.toBe(computeRollingMsgHashes(mod).msgH);
  });

  it("detects ToolCall argument mutation", () => {
    function build(args: Record<string, unknown>): Message[] {
      return [
        userMsg("find"),
        assistantMsg([
          { type: "text" as const, text: "Checking..." },
          {
            type: "toolCall" as const,
            id: "c1",
            name: "read",
            arguments: args,
          },
        ]),
      ];
    }
    const base = build({ path: "/safe" });
    const mod = build({ path: "/different" });
    expect(computeRollingMsgHashes(base).roleH).toBe(computeRollingMsgHashes(mod).roleH);
    expect(computeRollingMsgHashes(base).msgH).not.toBe(computeRollingMsgHashes(mod).msgH);
  });

  it("detects Thinking text mutation", () => {
    function build(thinking: string): Message[] {
      return [
        userMsg("q"),
        assistantMsg([
          { type: "thinking" as const, thinking },
          { type: "text" as const, text: "Answer." },
        ]),
      ];
    }
    const base = build("reason A");
    const mod = build("reason B");
    expect(computeRollingMsgHashes(base).roleH).toBe(computeRollingMsgHashes(mod).roleH);
    expect(computeRollingMsgHashes(base).msgH).not.toBe(computeRollingMsgHashes(mod).msgH);
  });

  it("same-size different image data diverges", () => {
    const imgA: Message = userMsg([
      {
        type: "image" as const,
        data: "abcdefghijklmnopqrstuvwxyz0123456789",
        mimeType: "image/png",
      },
    ]);
    const imgB: Message = userMsg([
      {
        type: "image" as const,
        data: "zyxwvutsrqponmlkjihgfedcba9876543210",
        mimeType: "image/png",
      },
    ]);
    // Same mimeType, same data length (36) — must diverge on content
    expect(computeRollingMsgHashes([imgA]).msgH).not.toBe(computeRollingMsgHashes([imgB]).msgH);
  });

  it("detects textSignature mutation", () => {
    const base: Message = userMsg([
      { type: "text" as const, text: "hello", textSignature: "sig-v1" },
    ]);
    const mod: Message = userMsg([
      { type: "text" as const, text: "hello", textSignature: "sig-v2" },
    ]);
    expect(computeRollingMsgHashes([base]).roleH).toBe(computeRollingMsgHashes([mod]).roleH);
    expect(computeRollingMsgHashes([base]).msgH).not.toBe(computeRollingMsgHashes([mod]).msgH);
  });

  it("detects thinkingSignature mutation", () => {
    const base: Message = assistantMsg([
      {
        type: "thinking" as const,
        thinking: "hmm",
        thinkingSignature: "ts-v1",
      },
      { type: "text" as const, text: "answer" },
    ]);
    const mod: Message = assistantMsg([
      {
        type: "thinking" as const,
        thinking: "hmm",
        thinkingSignature: "ts-v2",
      },
      { type: "text" as const, text: "answer" },
    ]);
    expect(computeRollingMsgHashes([base]).msgH).not.toBe(computeRollingMsgHashes([mod]).msgH);
  });

  it("delimiter collision immunity: unequal structures that naive concatenation equates", () => {
    // Single text "a|b" vs two texts ["a","b"] produce identical naive `role|content` = "user|a|b"
    // Length-framing encodes block boundaries, so they diverge
    const single: Message = userMsg([{ type: "text" as const, text: "a|b" }]);
    const split: Message = userMsg([
      { type: "text" as const, text: "a" },
      { type: "text" as const, text: "b" },
    ]);
    expect(computeRollingMsgHashes([single]).roleH).toBe(computeRollingMsgHashes([split]).roleH);
    expect(computeRollingMsgHashes([single]).msgH).not.toBe(computeRollingMsgHashes([split]).msgH);
  });

  it("handles empty list", () => {
    const r = computeRollingMsgHashes([]);
    expect(r.msgN).toBe(0);
    expect(r.roleH).toHaveLength(16);
    expect(r.msgH).toHaveLength(16);
  });
});

// ── computeContextHashes (uses Context wrapper, no extraParams) ─────────────

describe("computeContextHashes", () => {
  it("produces all hash fields", () => {
    const result = computeContextHashes({
      context: {
        systemPrompt: "You are helpful.",
        tools: [{ name: "read", description: "Read", parameters: { type: "object" } }],
        messages: [userMsg("hello"), assistantMsg([])],
      },
      runtimeChannel: "telegram",
      runId: "run-1",
    });
    expect(result.syspH).toHaveLength(16);
    expect(result.toolH).toHaveLength(16);
    expect(result.roleH).toHaveLength(16);
    expect(result.msgH).toHaveLength(16);
    expect(result.toolN).toBe(1);
    expect(result.msgN).toBe(2);
    expect(result.runtimeChannel).toBe("telegram");
    expect(result.runId).toBe("run-1");
  });

  it("handles undefined systemPrompt and tools", () => {
    const result = computeContextHashes({
      context: { messages: [userMsg("hi")] },
      runtimeChannel: "test",
      runId: "run-2",
    });
    expect(result.syspH).toHaveLength(16);
    expect(result.toolN).toBe(0);
    expect(result.msgN).toBe(1);
  });

  it("is deterministic", () => {
    const input = {
      context: {
        systemPrompt: "Be helpful.",
        tools: [] as Tool[],
        messages: [userMsg("hi")],
      },
      runtimeChannel: "test",
      runId: "run-3",
    };
    expect(computeContextHashes(input)).toEqual(computeContextHashes(input));
  });
});

// ── formatContextHashLog ────────────────────────────────────────────────────

describe("formatContextHashLog", () => {
  it("produces a structured log line without extra fields", () => {
    const line = formatContextHashLog("[context-hash]", {
      syspH: "a",
      toolH: "b",
      toolSetH: "c",
      toolN: 3,
      role64H: "pre",
      role128H: "pre",
      role256H: "pre",
      role512H: "pre",
      roleH: "d",
      msg64H: "pre",
      msg128H: "pre",
      msg256H: "pre",
      msg512H: "pre",
      msgH: "e",
      msgN: 10,
      runtimeChannel: "telegram",
      runId: "run-42",
    });
    expect(line).toContain("runId=run-42");
    expect(line).toContain("runtimeChannel=telegram");
    expect(line).toContain("roleH=d");
    expect(line).toContain("msgH=e");
    expect(line).toContain("msgN=10");
    // No extra fields
    expect(line).not.toContain("extraOverrideH");
    expect(line).not.toContain("extraKeys");
  });

  it("includes extras when provided", () => {
    const line = formatContextHashLog(
      "[test]",
      {
        syspH: "a",
        toolH: "b",
        toolSetH: "c",
        toolN: 0,
        role64H: "pre",
        role128H: "pre",
        role256H: "pre",
        role512H: "pre",
        roleH: "d",
        msg64H: "pre",
        msg128H: "pre",
        msg256H: "pre",
        msg512H: "pre",
        msgH: "e",
        msgN: 0,
        runtimeChannel: "test",
        runId: "r1",
      },
      { trigger: "user_message" },
    );
    expect(line).toContain("trigger=user_message");
  });
});
