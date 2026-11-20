import { SAMPLE_WEIGHTS, cascade, exampleMessages, meanAbsError, quantize } from "./serving";

describe("quantize", () => {
  it("keeps the smallest and largest values exactly", () => {
    const q = quantize(SAMPLE_WEIGHTS, 4);
    expect(Math.min(...q)).toBeCloseTo(Math.min(...SAMPLE_WEIGHTS), 10);
    expect(Math.max(...q)).toBeCloseTo(Math.max(...SAMPLE_WEIGHTS), 10);
  });
  it("uses at most 2^bits different values", () => {
    expect(new Set(quantize(SAMPLE_WEIGHTS, 2).map((v) => v.toFixed(8))).size).toBeLessThanOrEqual(4);
  });
  it("has less error with more bits", () => {
    const error = (bits: number) => meanAbsError(SAMPLE_WEIGHTS, quantize(SAMPLE_WEIGHTS, bits));
    expect(error(8)).toBeLessThan(error(4));
    expect(error(4)).toBeLessThan(error(2));
    expect(error(16)).toBeLessThan(1e-4);
  });
});

describe("cascade", () => {
  const messages = exampleMessages();
  it("with threshold 0 nothing goes to the teacher", () => {
    const r = cascade(messages, 0);
    expect(r.sentToTeacher).toBe(0);
    expect(r.accuracy).toBeCloseTo(messages.filter((m) => m.studentRight).length / messages.length, 10);
  });
  it("with a threshold above 1 everything goes to the teacher", () => {
    const r = cascade(messages, 1.01);
    expect(r.sentToTeacher).toBe(1);
    expect(r.accuracy).toBeCloseTo(messages.filter((m) => m.teacherRight).length / messages.length, 10);
  });
  it("sends more to the teacher as the threshold rises", () => {
    expect(cascade(messages, 0.9).sentToTeacher).toBeGreaterThan(cascade(messages, 0.6).sentToTeacher);
  });
  it("a middle threshold gets most of the teacher's accuracy for a fraction of its cost", () => {
    const studentOnly = cascade(messages, 0);
    const teacherOnly = cascade(messages, 1.01);
    const mixed = cascade(messages, 0.7);
    expect(mixed.accuracy).toBeGreaterThan(studentOnly.accuracy);
    expect(mixed.relativeCost).toBeLessThan(teacherOnly.relativeCost * 0.6);
  });
  it("the example data is the same every time", () => {
    expect(exampleMessages()).toEqual(messages);
  });
});
