export type PromptShape = {
  examplesPerType: number;
  typeCount?: number;
  instructionTokens?: number;
  tokensPerType?: number;
  tokensPerExample?: number;
  messageTokens?: number;
};

/** Rough size of one classification prompt, in tokens. The defaults fit the Banking77 task. */
export function promptTokens({
  examplesPerType,
  typeCount = 77,
  instructionTokens = 40,
  tokensPerType = 7,
  tokensPerExample = 22,
  messageTokens = 20,
}: PromptShape): number {
  return instructionTokens + typeCount * tokensPerType + messageTokens + examplesPerType * typeCount * tokensPerExample;
}

/** Cost of labelling `messages` messages, with prices per million tokens. */
export function labellingCost(
  inputTokens: number,
  outputTokens: number,
  messages: number,
  usdPerMTokIn: number,
  usdPerMTokOut: number,
): number {
  return (messages * (inputTokens * usdPerMTokIn + outputTokens * usdPerMTokOut)) / 1_000_000;
}
