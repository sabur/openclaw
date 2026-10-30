import { describe, it, expect } from "vitest";
import {
  resolvePromptOwnerUnion,
  isDefaultMainSession,
  resolvePromptOwnerNumbers,
} from "./resolve-prompt-owner-union.js";

const configWithOwner = {
  commands: { ownerAllowFrom: ["telegram:123456789"] },
};
const configWithoutOwner = {};
const configWithWildcard = {
  commands: { ownerAllowFrom: ["*"] },
};

describe("resolvePromptOwnerUnion", () => {
  it("returns undefined for null config", () => {
    expect(resolvePromptOwnerUnion(null)).toBeUndefined();
  });

  it("returns undefined for undefined config", () => {
    expect(resolvePromptOwnerUnion(undefined)).toBeUndefined();
  });

  it("returns undefined when ownerAllowFrom is not configured", () => {
    expect(resolvePromptOwnerUnion({})).toBeUndefined();
  });

  it("returns undefined when ownerAllowFrom is empty", () => {
    expect(resolvePromptOwnerUnion({ commands: { ownerAllowFrom: [] } })).toBeUndefined();
  });

  it("returns undefined when ownerAllowFrom contains only wildcard", () => {
    expect(resolvePromptOwnerUnion({ commands: { ownerAllowFrom: ["*"] } })).toBeUndefined();
  });

  it("filters out wildcards alongside real IDs", () => {
    const result = resolvePromptOwnerUnion({
      commands: { ownerAllowFrom: ["*", "telegram:123456789"] },
    });
    expect(result).toEqual(["telegram:123456789"]);
  });

  it("preserves canonical channel prefixes", () => {
    const result = resolvePromptOwnerUnion({
      commands: { ownerAllowFrom: ["telegram:123456789"] },
    });
    expect(result).toEqual(["telegram:123456789"]);
  });

  it("deduplicates entries", () => {
    const result = resolvePromptOwnerUnion({
      commands: { ownerAllowFrom: ["telegram:123456789", "telegram:123456789"] },
    });
    expect(result).toEqual(["telegram:123456789"]);
  });

  it("stable-sorts entries", () => {
    const result = resolvePromptOwnerUnion({
      commands: { ownerAllowFrom: ["discord:user:123", "telegram:123456789"] },
    });
    expect(result).toEqual(["discord:user:123", "telegram:123456789"]);
  });

  it("handles mixed string and number entries", () => {
    const result = resolvePromptOwnerUnion({
      commands: { ownerAllowFrom: ["telegram:123456789", 99999] },
    });
    expect(result).toContain("telegram:123456789");
    expect(result).toContain("99999");
  });

  it("returns deterministic sorted order for same input", () => {
    const input = { commands: { ownerAllowFrom: ["z:999", "a:111", "m:555"] } };
    const r1 = resolvePromptOwnerUnion(input);
    const r2 = resolvePromptOwnerUnion(input);
    expect(r1).toEqual(r2);
    expect(r1).toEqual(["a:111", "m:555", "z:999"]);
  });

  it("filters out empty strings after normalization", () => {
    const result = resolvePromptOwnerUnion({
      commands: { ownerAllowFrom: ["", "telegram:123456789"] },
    });
    expect(result).toEqual(["telegram:123456789"]);
  });
});

describe("isDefaultMainSession", () => {
  it("returns true for agent:main:main", () => {
    expect(isDefaultMainSession("agent:main:main")).toBe(true);
  });

  it("returns false for subagent sessions", () => {
    expect(isDefaultMainSession("agent:main:sub:abc")).toBe(false);
  });

  it("returns false for null/undefined", () => {
    expect(isDefaultMainSession(null)).toBe(false);
    expect(isDefaultMainSession(undefined)).toBe(false);
  });

  it("returns false for empty string", () => {
    expect(isDefaultMainSession("")).toBe(false);
  });

  it("returns false for other main-like keys (intentionally scoped to agent:main:main)", () => {
    expect(isDefaultMainSession("agent:other:main")).toBe(false);
  });
});

describe("resolvePromptOwnerNumbers — central resolver", () => {
  it("returns configured union for default main when config has owners", () => {
    const result = resolvePromptOwnerNumbers({
      sessionKey: "agent:main:main",
      config: configWithOwner,
      fallbackOwnerNumbers: ["per-request:id"],
    });
    expect(result).toEqual(["telegram:123456789"]);
  });

  it("falls back to per-request owners for default main when config has no explicit owners", () => {
    const result = resolvePromptOwnerNumbers({
      sessionKey: "agent:main:main",
      config: configWithoutOwner,
      fallbackOwnerNumbers: ["telegram:123456789"],
    });
    // No configured union — uses per-request fallback
    expect(result).toEqual(["telegram:123456789"]);
  });

  it("falls back to per-request owners for default main when config has only wildcard", () => {
    const result = resolvePromptOwnerNumbers({
      sessionKey: "agent:main:main",
      config: configWithWildcard,
      fallbackOwnerNumbers: ["telegram:123456789"],
    });
    // Wildcard excluded from union — uses per-request fallback
    expect(result).toEqual(["telegram:123456789"]);
  });

  it("returns undefined for default main when neither config nor fallback has owners", () => {
    const result = resolvePromptOwnerNumbers({
      sessionKey: "agent:main:main",
      config: configWithoutOwner,
      fallbackOwnerNumbers: undefined,
    });
    expect(result).toBeUndefined();
  });

  it("returns per-request owners for non-main sessions", () => {
    const result = resolvePromptOwnerNumbers({
      sessionKey: "agent:main:sub:abc",
      config: configWithOwner,
      fallbackOwnerNumbers: ["per-request:id"],
    });
    expect(result).toEqual(["per-request:id"]);
  });

  it("returns per-request owners for non-default main sessions", () => {
    const result = resolvePromptOwnerNumbers({
      sessionKey: "agent:other:main",
      config: configWithOwner,
      fallbackOwnerNumbers: ["cli-owner"],
    });
    expect(result).toEqual(["cli-owner"]);
  });

  it("returns undefined for non-main sessions with empty fallback", () => {
    const result = resolvePromptOwnerNumbers({
      sessionKey: "agent:other:main",
      config: configWithOwner,
      fallbackOwnerNumbers: [],
    });
    expect(result).toBeUndefined();
  });

  it("handles null/undefined sessionKey", () => {
    expect(
      resolvePromptOwnerNumbers({
        sessionKey: null,
        config: configWithOwner,
        fallbackOwnerNumbers: ["telegram:123456789"],
      }),
    ).toEqual(["telegram:123456789"]);

    expect(
      resolvePromptOwnerNumbers({
        sessionKey: undefined,
        config: configWithOwner,
        fallbackOwnerNumbers: ["telegram:123456789"],
      }),
    ).toEqual(["telegram:123456789"]);
  });
});
