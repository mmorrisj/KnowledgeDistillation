/** Scores to probabilities. A temperature above 1 flattens the result. */
export function softmax(scores: number[], temperature = 1): number[] {
  const scaled = scores.map((s) => s / temperature);
  const max = Math.max(...scaled);
  const exps = scaled.map((s) => Math.exp(s - max));
  const sum = exps.reduce((a, b) => a + b, 0);
  return exps.map((e) => e / sum);
}

/** How far distribution q is from distribution p. Zero when they match. */
export function klDivergence(p: number[], q: number[]): number {
  return p.reduce((total, pi, i) => (pi > 0 ? total + pi * Math.log(pi / q[i]) : total), 0);
}

export function crossEntropy(probabilities: number[], label: number): number {
  return -Math.log(probabilities[label]);
}

export type KdLoss = { teacherTerm: number; labelTerm: number; total: number };

/** The distillation loss for one example: alpha * teacher term + (1 - alpha) * label term. */
export function kdLoss(
  studentScores: number[],
  teacherScores: number[],
  label: number,
  temperature: number,
  alpha: number,
): KdLoss {
  const teacherTerm =
    temperature ** 2 * klDivergence(softmax(teacherScores, temperature), softmax(studentScores, temperature));
  const labelTerm = crossEntropy(softmax(studentScores), label);
  return { teacherTerm, labelTerm, total: alpha * teacherTerm + (1 - alpha) * labelTerm };
}
