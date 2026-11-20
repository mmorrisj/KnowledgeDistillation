/** A small deterministic random number generator, so examples never change between visits. */
function generator(seed: number): () => number {
  let state = seed >>> 0;
  return () => {
    state = (Math.imul(state, 1664525) + 1013904223) >>> 0;
    return state / 4294967296;
  };
}

// Twenty-four illustrative weights, bunched around zero the way real weights are.
export const SAMPLE_WEIGHTS: number[] = (() => {
  const next = generator(7);
  return Array.from({ length: 24 }, () => {
    const bell = next() + next() + next() - 1.5; // roughly bell-shaped, about -1.5 to 1.5
    return Math.round((bell / 1.5) * 1000) / 1000;
  });
})();

/** Round each value to the nearest of 2^bits evenly spaced values between the smallest and largest. */
export function quantize(values: number[], bits: number): number[] {
  const low = Math.min(...values);
  const high = Math.max(...values);
  const steps = 2 ** bits - 1;
  const size = (high - low) / steps;
  return values.map((v) => low + Math.round((v - low) / size) * size);
}

export function meanAbsError(original: number[], rounded: number[]): number {
  return original.reduce((total, v, i) => total + Math.abs(v - rounded[i]), 0) / original.length;
}

export type ExampleMessage = { confidence: number; studentRight: boolean; teacherRight: boolean };

/**
 * A thousand made-up messages. The student is well calibrated (when it says 80% it is right
 * about 80% of the time) and the teacher is right about 92% of the time.
 */
export function exampleMessages(): ExampleMessage[] {
  const next = generator(42);
  return Array.from({ length: 1000 }, () => {
    const confidence = 0.35 + 0.65 * Math.sqrt(next()); // most messages are confident ones
    return { confidence, studentRight: next() < confidence, teacherRight: next() < 0.92 };
  });
}

export type CascadeResult = { sentToTeacher: number; accuracy: number; relativeCost: number };

/**
 * Messages the student is less than `threshold` sure of go to the teacher.
 * Cost is relative to sending everything to the teacher; the student costs `studentCost`
 * of a teacher call and runs on every message.
 */
export function cascade(messages: ExampleMessage[], threshold: number, studentCost = 0.02): CascadeResult {
  let sent = 0;
  let right = 0;
  for (const m of messages) {
    const toTeacher = m.confidence < threshold;
    if (toTeacher) sent += 1;
    if (toTeacher ? m.teacherRight : m.studentRight) right += 1;
  }
  const sentToTeacher = sent / messages.length;
  return { sentToTeacher, accuracy: right / messages.length, relativeCost: studentCost + sentToTeacher };
}
