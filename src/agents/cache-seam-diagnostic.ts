/**
 * Cache-seam diagnostic — logs system prompt section variations across ingress routes.
 *
 * Temporary helper. Wired at every entry point that builds the system prompt
 * for agent:main:main. Removed once the fix is verified.
 */
import { createSubsystemLogger } from "../logging/subsystem.js";

const log = createSubsystemLogger("cache-seam");

const SECTIONS = {
  authorizedSenders: "## Authorized Senders",
  inboundContext: "## Inbound Context",
  directChat: "direct conversation",
  cacheBoundary: "OPENCLAW_CACHE_BOUNDARY",
} as const;

export type CacheSeamDiagnosticInputs = {
  source: string;
  sessionKey?: string | null;
  ownerNumbers?: string[] | null;
  extraSystemPrompt?: string | null;
  extraSystemPromptStatic?: string | null;
  systemPrompt?: string | null;
  messageChannel?: string | null;
  messageProvider?: string | null;
  agentAccountId?: string | null;
};

export function logCacheSeamDiagnostic(params: CacheSeamDiagnosticInputs): void {
  const {
    source,
    systemPrompt,
    ownerNumbers,
    extraSystemPrompt,
    extraSystemPromptStatic,
    sessionKey,
    messageChannel,
    messageProvider,
    agentAccountId,
  } = params;

  const isMainSession = sessionKey === "agent:main:main";

  // Section presence analysis
  const sections = systemPrompt
    ? {
        hasAuthSenders: systemPrompt.includes(SECTIONS.authorizedSenders),
        hasInboundMeta: systemPrompt.includes(SECTIONS.inboundContext),
        hasDirectChat: systemPrompt.includes(SECTIONS.directChat),
        hasCacheBoundary: systemPrompt.includes(SECTIONS.cacheBoundary),
      }
    : null;

  // Extract actual values from extraSystemPrompt to see what's being injected
  const extraPromptLines = extraSystemPrompt ? extraSystemPrompt.split("\n") : [];
  const channelField = extraPromptLines
    .filter((l) => l.includes('"channel":'))
    .map((l) => l.split('"channel":')[1]?.trim().replace(/[",]/g, ""))
    .filter(Boolean)[0];
  const providerField = extraPromptLines
    .filter((l) => l.includes('"provider":'))
    .map((l) => l.split('"provider":')[1]?.trim().replace(/[",]/g, ""))
    .filter(Boolean)[0];
  const surfaceField = extraPromptLines
    .filter((l) => l.includes('"surface":'))
    .map((l) => l.split('"surface":')[1]?.trim().replace(/[",]/g, ""))
    .filter(Boolean)[0];
  const directChatChannel = extraPromptLines
    .filter((l) => l.includes("direct conversation"))
    .map((l) => l.match(/in a (\w+) direct/)?.[1])
    .filter(Boolean)[0];

  // Owner numbers
  const ownerValues = ownerNumbers?.filter(Boolean) ?? [];
  const ownerPreview = ownerValues.length > 0 ? ownerValues.join(",") : "none";

  // Build message
  const parts: string[] = [`[${source}]`, `session=${sessionKey ?? "?"}`, `main=${isMainSession}`];

  if (systemPrompt) {
    parts.push(
      `len=${systemPrompt.length}`,
      `auth=${sections!.hasAuthSenders}`,
      `inbound=${sections!.hasInboundMeta}`,
      `chat=${sections!.hasDirectChat}`,
    );
  }

  parts.push(
    `owners=${ownerPreview}`,
    `epLen=${extraSystemPrompt?.length ?? 0}`,
    `epStatic=${extraSystemPromptStatic?.length ?? 0}`,
    `epChan=${channelField ?? "?"}`,
    `epProv=${providerField ?? "?"}`,
    `epSurf=${surfaceField ?? "?"}`,
    `chatChan=${directChatChannel ?? "?"}`,
  );

  if (messageChannel) parts.push(`mChan=${messageChannel}`);
  if (messageProvider) parts.push(`mProv=${messageProvider}`);
  if (agentAccountId) parts.push(`acct=${agentAccountId}`);

  log.debug(parts.join(" "));
}
