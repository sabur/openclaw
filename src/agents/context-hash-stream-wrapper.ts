import type { Context, Model, SimpleStreamOptions, StreamFn } from "../llm/types.js";
/**
 * Thin StreamFn diagnostic wrapper — fingerprints the final llm-core Context
 * at the provider boundary.
 *
 * Placement: wrap the selected provider/base stream first, so every
 * OpenClaw/plugin context/options-transforming wrapper wraps around it.
 * The inner wrapper sees exactly what reaches the provider.
 *
 * No behavior mutation, no payload modification, no raw content logged.
 * Gated by caller — normal runs pay zero hashing cost.
 * Diagnostics fail open: hashing/log errors emit a warning but never
 * block the original streamFn.
 */
import {
  sha256,
  canonicalizeValue,
  computeContextHashes,
  formatContextHashLog,
} from "./context-hash-utils.js";

// ── Safe logger wrapper ─────────────────────────────────────────────────────

/**
 * Call a logger safely — if the logger throws, the error is swallowed.
 * A broken diagnostic logger must never escape into the caller.
 */
function safeLog(log: (line: string) => void, line: string): void {
  try {
    log(line);
  } catch {
    // Logger failure is contained — diagnostic must never break inference.
  }
}

// ── Deep option sanitization ────────────────────────────────────────────────

/**
 * Key patterns that carry secrets — matched at every nesting level.
 */
const SECRET_KEY_RE =
  /^(?:access[-_]?token|auth[-_]?token|hook[-_]?token|refresh[-_]?token|id[-_]?token|api[-_]?key|apikey|client[-_]?secret|app[-_]?secret|private[-_]?key|privateKey|credential|authorization|authorization[-_]?header|token|secret|password|pass|passwd|auth|jwt|session|signature|cookie|apiKey|bearer)$/i;

/**
 * Recursively sanitize a value for safe options hashing.
 * - Functions → omitted (key removed)
 * - AbortSignal-like objects → omitted (key removed)
 * - Objects with secret-bearing keys → value redacted to "[redacted]"
 * - Primitives and safe objects pass through
 * Returns a new value or the special OMIT symbol to signal key removal.
 */
const OMIT = Symbol("omit");

function sanitizeOptionValue(value: unknown): unknown | typeof OMIT {
  if (typeof value === "function") return OMIT;
  if (
    (typeof AbortSignal !== "undefined" && value instanceof AbortSignal) ||
    (value &&
      typeof value === "object" &&
      "aborted" in (value as Record<string, unknown>) &&
      typeof (value as Record<string, unknown>).aborted === "boolean")
  ) {
    return OMIT;
  }
  if (value === null || value === undefined) return value;
  if (
    typeof value === "boolean" ||
    typeof value === "number" ||
    typeof value === "string" ||
    typeof value === "bigint"
  ) {
    return value;
  }
  if (Array.isArray(value)) {
    const mapped = value.map(sanitizeOptionValue).filter((v) => v !== OMIT);
    return mapped.length > 0 ? mapped : OMIT;
  }
  if (typeof value === "object") {
    const result: Record<string, unknown> = {};
    for (const [k, v] of Object.entries(value as Record<string, unknown>)) {
      if (SECRET_KEY_RE.test(k)) {
        result[k] = "[redacted]";
      } else {
        const sanitized = sanitizeOptionValue(v);
        if (sanitized !== OMIT) {
          result[k] = sanitized;
        }
      }
    }
    return Object.keys(result).length > 0 ? result : OMIT;
  }
  return value;
}

/** Keys excluded entirely from options fingerprinting. */
const EXCLUDED_OPTION_KEYS = new Set([
  "apiKey",
  "headers",
  "signal",
  "onPayload",
  "onResponse",
  "onActiveResponse",
  "metadata", // non-prompt metadata; excluded from hashing
]);

/**
 * Canonicalize prompt-affecting options.
 * - Excluded keys are removed entirely
 * - Functions and AbortSignal-like values are omitted (not replaced with markers)
 * - Remaining values are deeply sanitized (nested secrets redacted)
 * - Includes sessionId, promptCacheKey, temperature, maxTokens, responseFormat, stop, etc.
 */
function buildSafeOptions(options?: SimpleStreamOptions): Record<string, unknown> {
  const safe: Record<string, unknown> = {};
  if (!options) return safe;
  for (const [k, v] of Object.entries(options)) {
    if (EXCLUDED_OPTION_KEYS.has(k)) continue;
    const sanitized = sanitizeOptionValue(v);
    if (sanitized !== OMIT) {
      safe[k] = sanitized;
    }
  }
  return safe;
}

export interface SanitizedOptionsFingerprint {
  optionsH: string;
  /** Number of retained top-level keys after sanitization. */
  optionsKeys: number;
}

/** Compute a sanitized options fingerprint. */
export function fingerprintSafeOptions(options?: SimpleStreamOptions): SanitizedOptionsFingerprint {
  const safe = buildSafeOptions(options);
  return {
    optionsH: sha256(canonicalizeValue(safe)),
    optionsKeys: Object.keys(safe).length,
  };
}

// ── StreamFn diagnostic wrapper ─────────────────────────────────────────────

export interface ContextHashStreamWrapperOptions {
  /** The underlying stream function to wrap. */
  readonly streamFn: StreamFn;
  /** Whether context-hash diagnostics are enabled. */
  readonly enabled: boolean;
  /** Runtime channel identifier. */
  readonly runtimeChannel: string;
  /** Run identifier for trace correlation. */
  readonly runId: string;
  /** Logger function for the diagnostic line. Required. */
  readonly log: (line: string) => void;
}

/**
 * Wrap a StreamFn with context-hash diagnostics.
 *
 * When enabled, fingerprints the final Context (system prompt, tools, messages)
 * and options before the provider sees them, logging one diagnostic line.
 * The underlying streamFn is called with unchanged arguments.
 *
 * When disabled, returns the original streamFn unchanged (zero cost).
 * Diagnostics fail open: hashing/log errors emit a warning but never
 * block the original streamFn.
 */
export function wrapContextHashStreamFn(options: ContextHashStreamWrapperOptions): StreamFn {
  const { streamFn, enabled, runtimeChannel, runId, log } = options;

  if (!enabled) {
    return streamFn;
  }

  return (model: Model, context: Context, streamOptions?: SimpleStreamOptions) => {
    try {
      const hashes = computeContextHashes({
        context: {
          systemPrompt: context.systemPrompt,
          tools: context.tools,
          messages: context.messages,
        },
        runtimeChannel,
        runId,
      });

      const safeOptions = fingerprintSafeOptions(streamOptions);

      safeLog(
        log,
        formatContextHashLog("[context-diag-hash]", hashes, {
          optionsH: safeOptions.optionsH,
          optionsKeys: String(safeOptions.optionsKeys),
          provider: model.provider,
          model: model.id,
          api: model.api,
        }),
      );
    } catch (err) {
      const message = err instanceof Error ? err.message : String(err);
      safeLog(log, `[context-diag-hash-error] runId=${runId} failed: ${message}`);
    }

    // Always pass through unchanged, even on error
    return streamFn(model, context, streamOptions);
  };
}
