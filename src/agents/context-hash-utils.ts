/**
 * Context diagnostic hashing — deterministic SHA-256 fingerprints for
 * detecting cache-key divergence across ingress paths without logging
 * raw prompt content.
 *
 * Fingerprints the final llm-core Context at the StreamFn boundary:
 * - Uses real Context, Message, Tool types
 * - Two streaming hash streams, both length-framed for delimiter immunity:
 *   1. role*H — structural: role + toolName per message
 *   2. msg*H — full prompt-bearing payload: role, exact content blocks,
 *      toolCallId, toolName, isError
 * - Both streams use Hash.copy() checkpoints at 64/128/256/512
 * - Tools: recursive canonicalization (no JSON.stringify replacer bug)
 * - Images: data replaced with {dataHash: sha256(data), dataLength} before canonicalization
 * - No tool-call double counting (tool calls are already in content)
 * - No extraParams — those are fingerprinted separately in the StreamFn wrapper
 */
import crypto from "node:crypto";
import type { Context, Message, Tool } from "../../llm/types.js";

// ── SHA-256 helper ──────────────────────────────────────────────────────────

/** SHA-256 hex digest, truncated to first `len` hex chars (default 16). */
export function sha256(input: string, len = 16): string {
  return crypto.createHash("sha256").update(input).digest("hex").slice(0, len);
}

// ── Recursive canonicalization ──────────────────────────────────────────────

/** Recursively canonicalize a value into a deterministic string. */
export function canonicalizeValue(value: unknown, opts?: { depth?: number }): string {
  const depth = opts?.depth ?? 0;
  if (depth > 20) return "[max-depth]";

  if (value === null || value === undefined) return "null";
  if (typeof value === "boolean") return value ? "true" : "false";
  if (typeof value === "number")
    return Number.isFinite(value) ? String(value) : `"${String(value)}"`;
  if (typeof value === "string") return JSON.stringify(value);
  if (typeof value === "bigint") return value.toString();

  if (Array.isArray(value)) {
    return "[" + value.map((v) => canonicalizeValue(v, { depth: depth + 1 })).join(",") + "]";
  }

  if (typeof value === "object") {
    const keys = Object.keys(value as Record<string, unknown>).sort();
    const parts = keys.map((k) => {
      const v = (value as Record<string, unknown>)[k];
      return JSON.stringify(k) + ":" + canonicalizeValue(v, { depth: depth + 1 });
    });
    return "{" + parts.join(",") + "}";
  }

  return String(value);
}

// ── Length-framed canonical helpers ─────────────────────────────────────────

/**
 * Length-frame a canonical string for delimiter immunity.
 * Format: `<byteLength>:<canonical>`.
 */
function lengthFrame(canonical: string): string {
  return `${Buffer.byteLength(canonical, "utf8")}:${canonical}`;
}

// ── Tool canonicalization (real llm-core Tool type) ─────────────────────────

/**
 * Canonicalize a single Tool using recursive canonicalization.
 * NOT JSON.stringify with replacer — that strips nested schema keys.
 */
function canonicalizeTool(t: Tool): string {
  return lengthFrame(
    canonicalizeValue({ name: t.name, description: t.description, parameters: t.parameters }),
  );
}

/** Canonicalize tool definitions (order-sensitive, preserves insertion order). */
export function canonicalizeTools(tools: Tool[]): string {
  return tools.map(canonicalizeTool).join("");
}

/** Canonicalize tool definitions (order-insensitive, sorted by name). */
export function canonicalizeToolsSorted(tools: Tool[]): string {
  const sorted = [...tools].sort((a, b) => a.name.localeCompare(b.name));
  return canonicalizeTools(sorted);
}

// ── Image content normalization ─────────────────────────────────────────────

/**
 * Normalize an image content block for deterministic hashing.
 * Replaces raw data with { dataHash, dataLength } to avoid giant
 * strings while preserving exact identity.
 */
function normalizeImageBlock(block: { type: "image"; data: string; mimeType: string }): {
  type: "image";
  mimeType: string;
  dataHash: string;
  dataLength: number;
} {
  return {
    type: "image",
    mimeType: block.mimeType,
    dataHash: sha256(block.data, 64),
    dataLength: block.data.length,
  };
}

// ── Message canonicalization (real llm-core Message types) ──────────────────

/** Normalize content blocks: image data → {dataHash, dataLength}. */
function normalizeContent(content: string | readonly unknown[]): string | unknown[] {
  if (typeof content === "string") return content;
  if (Array.isArray(content)) {
    return content.map((block) => {
      if (
        block &&
        typeof block === "object" &&
        (block as Record<string, unknown>).type === "image"
      ) {
        return normalizeImageBlock(block as { type: "image"; data: string; mimeType: string });
      }
      return block;
    });
  }
  return content;
}

/**
 * Build the prompt-bearing projection for a Message.
 * Uses typed objects with canonicalizeValue. Image data is normalized.
 */
function messageProjection(m: Message): Record<string, unknown> {
  switch (m.role) {
    case "user":
      return { role: "user", content: normalizeContent(m.content) };
    case "assistant":
      return { role: "assistant", content: normalizeContent(m.content) };
    case "toolResult":
      return {
        role: "toolResult",
        toolCallId: m.toolCallId,
        toolName: m.toolName,
        content: normalizeContent(m.content),
        isError: m.isError,
      };
  }
}

/** Canonicalize role prefix (structural stream): role + toolName for toolResult. */
function canonicalizeRolePrefix(m: Message): string {
  if (m.role === "toolResult") return `toolResult:${m.toolName}`;
  return m.role;
}

/** Canonicalize the full prompt-bearing payload for one Message. */
function canonicalizeMessagePayload(m: Message): string {
  return lengthFrame(canonicalizeValue(messageProjection(m)));
}

// ── Rolling hash computation ───────────────────────────────────────────────

export interface RollingMsgHashes {
  // Stream 1: Structural (role + toolName only)
  role64H: string;
  role128H: string;
  role256H: string;
  role512H: string;
  roleH: string;
  // Stream 2: Full prompt-bearing payload
  msg64H: string;
  msg128H: string;
  msg256H: string;
  msg512H: string;
  msgH: string;
  msgN: number;
}

/**
 * Compute incremental rolling hashes for two streams using Node's Hash.copy()
 * with length-framed inputs for delimiter immunity.
 */
export function computeRollingMsgHashes(messages: Message[]): RollingMsgHashes {
  const roleHash = crypto.createHash("sha256");
  const msgHash = crypto.createHash("sha256");

  let role64: string | undefined;
  let role128: string | undefined;
  let role256: string | undefined;
  let role512: string | undefined;
  let msg64: string | undefined;
  let msg128: string | undefined;
  let msg256: string | undefined;
  let msg512: string | undefined;

  for (let i = 0; i < messages.length; i++) {
    const m = messages[i];

    // Stream 1: structural
    roleHash.update(lengthFrame(canonicalizeRolePrefix(m)));

    // Stream 2: full payload
    msgHash.update(canonicalizeMessagePayload(m));

    const count = i + 1;
    if (count === 64) {
      role64 = roleHash.copy().digest("hex").slice(0, 16);
      msg64 = msgHash.copy().digest("hex").slice(0, 16);
    }
    if (count === 128) {
      role128 = roleHash.copy().digest("hex").slice(0, 16);
      msg128 = msgHash.copy().digest("hex").slice(0, 16);
    }
    if (count === 256) {
      role256 = roleHash.copy().digest("hex").slice(0, 16);
      msg256 = msgHash.copy().digest("hex").slice(0, 16);
    }
    if (count === 512) {
      role512 = roleHash.copy().digest("hex").slice(0, 16);
      msg512 = msgHash.copy().digest("hex").slice(0, 16);
    }
  }

  return {
    role64H: messages.length >= 64 && role64 ? role64 : "pre",
    role128H: messages.length >= 128 && role128 ? role128 : "pre",
    role256H: messages.length >= 256 && role256 ? role256 : "pre",
    role512H: messages.length >= 512 && role512 ? role512 : "pre",
    roleH: roleHash.digest("hex").slice(0, 16),
    msg64H: messages.length >= 64 && msg64 ? msg64 : "pre",
    msg128H: messages.length >= 128 && msg128 ? msg128 : "pre",
    msg256H: messages.length >= 256 && msg256 ? msg256 : "pre",
    msg512H: messages.length >= 512 && msg512 ? msg512 : "pre",
    msgH: msgHash.digest("hex").slice(0, 16),
    msgN: messages.length,
  };
}

// ── Aggregate context hash (llm-core Context types) ─────────────────────────

export interface ContextHashInput {
  /** The final llm-core Context at the StreamFn boundary. */
  context: Pick<Context, "systemPrompt" | "tools" | "messages">;
  runtimeChannel: string;
  runId: string;
}

export interface ContextHashes {
  syspH: string;
  toolH: string;
  toolSetH: string;
  toolN: number;
  // Stream 1: Structural (role + toolName)
  role64H: string;
  role128H: string;
  role256H: string;
  role512H: string;
  roleH: string;
  // Stream 2: Full prompt-bearing payload
  msg64H: string;
  msg128H: string;
  msg256H: string;
  msg512H: string;
  msgH: string;
  msgN: number;
  runtimeChannel: string;
  runId: string;
}

/** Compute all context hashes from an llm-core Context at the StreamFn boundary. */
export function computeContextHashes(input: ContextHashInput): ContextHashes {
  const tools = input.context.tools ?? [];
  const toolCanon = canonicalizeTools(tools);
  const toolSortedCanon = canonicalizeToolsSorted(tools);
  const rolling = computeRollingMsgHashes(input.context.messages);

  return {
    syspH: sha256(input.context.systemPrompt ?? ""),
    toolH: sha256(toolCanon),
    toolSetH: sha256(toolSortedCanon),
    toolN: tools.length,
    role64H: rolling.role64H,
    role128H: rolling.role128H,
    role256H: rolling.role256H,
    role512H: rolling.role512H,
    roleH: rolling.roleH,
    msg64H: rolling.msg64H,
    msg128H: rolling.msg128H,
    msg256H: rolling.msg256H,
    msg512H: rolling.msg512H,
    msgH: rolling.msgH,
    msgN: rolling.msgN,
    runtimeChannel: input.runtimeChannel,
    runId: input.runId,
  };
}

/** Format context hashes into a single debug log line. */
export function formatContextHashLog(
  prefix: string,
  hashes: ContextHashes,
  extras?: Record<string, string>,
): string {
  const parts = [
    `${prefix} runId=${hashes.runId} runtimeChannel=${hashes.runtimeChannel}`,
    `syspH=${hashes.syspH}`,
    `toolH=${hashes.toolH} toolSetH=${hashes.toolSetH} toolN=${hashes.toolN}`,
    `role64H=${hashes.role64H} role128H=${hashes.role128H} role256H=${hashes.role256H} role512H=${hashes.role512H} roleH=${hashes.roleH}`,
    `msg64H=${hashes.msg64H} msg128H=${hashes.msg128H} msg256H=${hashes.msg256H} msg512H=${hashes.msg512H} msgH=${hashes.msgH} msgN=${hashes.msgN}`,
  ];
  if (extras) {
    for (const [k, v] of Object.entries(extras)) {
      parts.push(`${k}=${v}`);
    }
  }
  return parts.join(" ");
}
