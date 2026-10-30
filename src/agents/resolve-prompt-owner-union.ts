/**
 * Deterministic prompt-only owner union for default `agent:main:main` sessions.
 *
 * Resolves `cfg.commands.ownerAllowFrom` into a stable, ingress-independent
 * array for system-prompt rendering. This is a *prompt-only* resolver — it
 * never feeds into runtime authorization checks.
 *
 * Separated from `command-auth.ts` to keep authorization scoped and prevent
 * accidental auth widening.
 *
 * Scope: `agent:main:main` (the default single-agent session). Other session
 * types (subagents, ACP, named agents) continue using per-request owner context.
 */
import { normalizeStringEntries } from "@openclaw/normalization-core/string-normalization";

/**
 * Resolve prompt owner numbers for a given session.
 *
 * For default `agent:main:main` sessions, uses the deterministic configured
 * owner union. Falls back to `fallbackOwnerNumbers` (the per-request resolved
 * list) when the config has no explicit owners.
 *
 * For all other sessions, returns `fallbackOwnerNumbers` unchanged.
 */
export function resolvePromptOwnerNumbers(params: {
  sessionKey: string | undefined | null;
  config: { commands?: { ownerAllowFrom?: Array<string | number> } } | undefined | null;
  /** Per-request owner numbers from authorization resolution. */
  fallbackOwnerNumbers: string[] | undefined;
}): string[] | undefined {
  if (isDefaultMainSession(params.sessionKey)) {
    const configured = resolvePromptOwnerUnion(params.config);
    if (configured) {
      return configured;
    }
    // No configured union — fall back to per-request owners
    return normalizedFallbackOwnerNumbers(params.fallbackOwnerNumbers);
  }
  // Non-main sessions: use per-request owners as-is
  return normalizedFallbackOwnerNumbers(params.fallbackOwnerNumbers);
}

function normalizedFallbackOwnerNumbers(fallback: string[] | undefined): string[] | undefined {
  if (!fallback || fallback.length === 0) return undefined;
  const normalized = normalizeStringEntries(fallback);
  return normalized.length > 0 ? normalized : undefined;
}

/**
 * Resolve the deterministic prompt-only owner union from raw config.
 *
 * Rules:
 * - Normalize entries as trimmed non-empty strings
 * - Strip wildcard (`*`) and empty values
 * - Preserve canonical channel prefixes (e.g. `telegram:123456789`)
 * - Deduplicate
 * - Stable-sort for deterministic output
 *
 * Returns `undefined` when no explicit owners are configured.
 */
export function resolvePromptOwnerUnion(
  cfg: { commands?: { ownerAllowFrom?: Array<string | number> } } | undefined | null,
): string[] | undefined {
  const raw = cfg?.commands?.ownerAllowFrom;
  if (!Array.isArray(raw) || raw.length === 0) {
    return undefined;
  }

  const normalized = normalizeStringEntries(raw as (string | number)[]);
  if (normalized.length === 0) {
    return undefined;
  }

  // Filter wildcards and empty values
  const filtered = normalized.filter((id) => id.length > 0 && id !== "*");

  if (filtered.length === 0) {
    return undefined;
  }

  // Deduplicate (first occurrence wins, then stable-sort)
  const seen = new Set<string>();
  const deduped: string[] = [];
  for (const id of filtered) {
    if (!seen.has(id)) {
      seen.add(id);
      deduped.push(id);
    }
  }

  // Stable sort for deterministic output across all ingresses
  deduped.sort();

  return deduped;
}

/**
 * Check whether a session key represents the default `agent:main:main` session.
 *
 * This is intentionally scoped to `agent:main:main` only — the default
 * single-agent session where all ingresses (Telegram, CLI, pack-relay)
 * converge on the same session. Named/other agent main sessions may need
 * their own resolution and are not covered here.
 *
 * Returns false for subagent, ACP, named-agent, or null/undefined keys.
 */
export function isDefaultMainSession(sessionKey: string | undefined | null): boolean {
  return sessionKey === "agent:main:main";
}
