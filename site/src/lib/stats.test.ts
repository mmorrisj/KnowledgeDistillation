import { accuracyInterval, differenceInterval, gapClosed, imbalanceScores } from "./stats";

describe("accuracyInterval", () => {
  it("gives the normal-approximation 95% interval", () => {
    const { low, high } = accuracyInterval(0.8, 400);
    expect(low).toBeCloseTo(0.8 - 1.96 * Math.sqrt((0.8 * 0.2) / 400), 6);
    expect(high).toBeCloseTo(0.8 + 1.96 * Math.sqrt((0.8 * 0.2) / 400), 6);
  });
  it("narrows as the test set grows", () => {
    const small = accuracyInterval(0.8, 100);
    const large = accuracyInterval(0.8, 10000);
    expect(large.high - large.low).toBeLessThan(small.high - small.low);
  });
  it("stays inside 0 and 1", () => {
    expect(accuracyInterval(0.99, 20).high).toBeLessThanOrEqual(1);
    expect(accuracyInterval(0.01, 20).low).toBeGreaterThanOrEqual(0);
  });
});

describe("differenceInterval", () => {
  it("calls a large gap on a large test set real", () => {
    const result = differenceInterval(0.7, 0.8, 2000);
    expect(result.difference).toBeCloseTo(0.1, 6);
    expect(result.low).toBeGreaterThan(0);
    expect(result.verdict).toBe("real");
  });
  it("calls a small gap on a small test set unclear", () => {
    const result = differenceInterval(0.7, 0.72, 200);
    expect(result.low).toBeLessThan(0);
    expect(result.high).toBeGreaterThan(0);
    expect(result.verdict).toBe("unclear");
  });
});

describe("gapClosed", () => {
  it("is the share of the baseline-to-teacher distance the student covered", () => {
    expect(gapClosed(0.6, 0.72, 0.8)).toBeCloseTo(0.6, 6);
  });
  it("is null when the teacher is no better than the baseline", () => {
    expect(gapClosed(0.8, 0.82, 0.8)).toBeNull();
    expect(gapClosed(0.8, 0.82, 0.7)).toBeNull();
  });
});

describe("imbalanceScores", () => {
  it("scores a model that always answers with the common class", () => {
    const { accuracy, macroF1 } = imbalanceScores(0.95);
    expect(accuracy).toBeCloseTo(0.95, 6);
    // common class: precision .95, recall 1 -> F1 = 1.9/1.95 ; rare class: 0
    expect(macroF1).toBeCloseTo(1.9 / 1.95 / 2, 6);
  });
});
