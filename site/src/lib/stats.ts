const Z95 = 1.96;
const clamp = (value: number) => Math.min(1, Math.max(0, value));

/** 95% interval for an accuracy measured on n test messages (normal approximation). */
export function accuracyInterval(accuracy: number, n: number): { low: number; high: number } {
  const margin = Z95 * Math.sqrt((accuracy * (1 - accuracy)) / n);
  return { low: clamp(accuracy - margin), high: clamp(accuracy + margin) };
}

export type DifferenceResult = {
  difference: number;
  low: number;
  high: number;
  verdict: "real" | "unclear";
};

/**
 * 95% interval for the difference between two accuracies, each measured on n messages.
 * Treats the two measurements as independent, which is a little pessimistic when both
 * models were scored on the same messages.
 */
export function differenceInterval(a: number, b: number, n: number): DifferenceResult {
  const difference = b - a;
  const margin = Z95 * Math.sqrt((a * (1 - a)) / n + (b * (1 - b)) / n);
  const low = difference - margin;
  const high = difference + margin;
  return { difference, low, high, verdict: low > 0 || high < 0 ? "real" : "unclear" };
}

/** Share of the baseline-to-teacher distance the distilled student covered. */
export function gapClosed(baseline: number, distilled: number, teacher: number): number | null {
  if (teacher <= baseline) return null;
  return (distilled - baseline) / (teacher - baseline);
}

/** Scores of a model that always answers with the common class, on two-class data. */
export function imbalanceScores(commonShare: number): { accuracy: number; macroF1: number } {
  const commonF1 = (2 * commonShare) / (1 + commonShare);
  return { accuracy: commonShare, macroF1: commonF1 / 2 };
}
