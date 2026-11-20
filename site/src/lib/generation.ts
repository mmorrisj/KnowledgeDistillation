const GB = 1024 ** 3;

/**
 * Disk space for a teacher's next-token targets.
 * Full vocabulary: 2 bytes per probability. Top-k: a 4-byte token id and a 2-byte probability each.
 */
export function storageGb(positions: number, keep: { k: number } | { vocabulary: number }): number {
  const perPosition = "vocabulary" in keep ? keep.vocabulary * 2 : keep.k * 6;
  return (positions * perPosition) / GB;
}

// A one-dimensional stand-in for "all possible outputs": 101 points from -5 to 5.
export const XS = Array.from({ length: 101 }, (_, i) => -5 + i * 0.1);

/** A bell curve over XS, scaled to add up to 1. */
export function gaussianBins(mean: number, width: number): number[] {
  const raw = XS.map((x) => Math.exp(-0.5 * ((x - mean) / width) ** 2));
  const total = raw.reduce((a, b) => a + b, 0);
  return raw.map((v) => v / total);
}

/** A teacher that considers two different kinds of answer about equally good. */
export function twoPeakTeacher(): number[] {
  const left = gaussianBins(-2, 0.6);
  const right = gaussianBins(2, 0.6);
  return left.map((v, i) => 0.5 * v + 0.5 * right[i]);
}

const FLOOR = 1e-12;
function kl(p: number[], q: number[]): number {
  return p.reduce((total, pi, i) => (pi > FLOOR ? total + pi * Math.log(pi / Math.max(q[i], FLOOR)) : total), 0);
}

/** Forward KL: large wherever the teacher has weight and the student does not. */
export function forwardKl(teacher: number[], student: number[]): number {
  return kl(teacher, student);
}

/** Reverse KL: large wherever the student has weight and the teacher does not. */
export function reverseKl(teacher: number[], student: number[]): number {
  return kl(student, teacher);
}
