/** Builds fallback-model handoff messages that preserve active subagent state. */
import type { AgentMessage } from "../agents/runtime/index.js";

/** Summary of the prior model state used to brief the fallback model. */
export interface HandoffSnapshot {
  summary: string;
  activeSubagents: Array<{
    sessionId: string;
    role?: string;
    lastStatus?: string;
  }>;
}

/**
 * Builds the recovery briefing injected as the first user-side turn after a
 * model failover. The user role is used (not assistant) so the new model
 * treats the content as input rather than its own prior output.
 */
export function buildHierarchyReinforcementMessage(snapshot: HandoffSnapshot): AgentMessage {
  const subagentReport = snapshot.activeSubagents
    .map((s) => `- Subagent ${s.sessionId} (${s.role ?? "leaf"}): ${s.lastStatus ?? "running"}`)
    .join("\n");

  const content = [
    "[RUNTIME SUBSTRATE HANDOFF] The active model changed while the same companion session and task continue.",
    "Continue coordinating the existing task. Do not repeat work already delegated to bounded workers.",
    "",
    "ACTIVE SUBORDINATE UNITS:",
    subagentReport || "None active.",
    "",
    "CURRENT STATE SUMMARY:",
    snapshot.summary,
    "",
    "INSTRUCTIONS:",
    "1. Review the state and subordinate reports.",
    "2. Provide strategic guidance and commands to subordinates.",
    "3. Do not repeat work already performed by subordinates.",
  ].join("\n");

  return {
    role: "user",
    content,
    timestamp: Date.now(),
  };
}
