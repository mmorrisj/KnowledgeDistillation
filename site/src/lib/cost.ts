/** Cost of 1,000 API requests, from tokens per request and prices per million tokens. */
export function apiCostPer1k(
  inTokens: number,
  outTokens: number,
  usdPerMTokIn: number,
  usdPerMTokOut: number,
): number {
  return ((inTokens * usdPerMTokIn + outTokens * usdPerMTokOut) / 1_000_000) * 1000;
}

/** Cost of one machine left on for a full day. */
export function selfHostedDailyCost(instanceUsdPerHour: number): number {
  return instanceUsdPerHour * 24;
}

/** Requests per day at which an always-on machine costs the same as the API. */
export function breakEvenRequestsPerDay(apiPer1k: number, instanceUsdPerHour: number): number {
  return Math.round(selfHostedDailyCost(instanceUsdPerHour) / (apiPer1k / 1000));
}

export function isValidInput(...values: number[]): boolean {
  return values.every((value) => Number.isFinite(value) && value > 0);
}
