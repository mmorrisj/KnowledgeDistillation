import { crossEntropy, kdLoss, klDivergence, softmax } from "./distill";

describe("softmax", () => {
  it("turns scores into probabilities that add up to 1", () => {
    const p = softmax([2, 1, 0]);
    expect(p.reduce((a, b) => a + b, 0)).toBeCloseTo(1, 10);
    expect(p[0]).toBeCloseTo(0.665, 3);
  });
  it("flattens as temperature rises and keeps the order", () => {
    const cold = softmax([2, 1, 0], 1);
    const hot = softmax([2, 1, 0], 5);
    expect(hot[0]).toBeLessThan(cold[0]);
    expect(hot[2]).toBeGreaterThan(cold[2]);
    expect(hot[0]).toBeGreaterThan(hot[1]);
    expect(hot[1]).toBeGreaterThan(hot[2]);
  });
  it("does not overflow on large scores", () => {
    expect(softmax([1000, 999])[0]).toBeCloseTo(0.731, 3);
  });
});

describe("losses", () => {
  it("KL divergence is zero for identical distributions", () => {
    expect(klDivergence([0.7, 0.2, 0.1], [0.7, 0.2, 0.1])).toBeCloseTo(0, 10);
  });
  it("KL divergence is positive when they differ", () => {
    expect(klDivergence([0.7, 0.2, 0.1], [0.1, 0.2, 0.7])).toBeGreaterThan(0);
  });
  it("cross-entropy is minus the log of the probability given to the right answer", () => {
    expect(crossEntropy([0.25, 0.75], 1)).toBeCloseTo(-Math.log(0.75), 10);
  });
});

describe("kdLoss", () => {
  const teacher = [3, 1.5, -1];
  const student = [1, 2, 0];
  it("with alpha 0 is the cross-entropy with the human label", () => {
    const { total, labelTerm } = kdLoss(student, teacher, 0, 2, 0);
    expect(total).toBeCloseTo(crossEntropy(softmax(student), 0), 10);
    expect(labelTerm).toBeCloseTo(total, 10);
  });
  it("with alpha 1 is T squared times the KL to the softened teacher", () => {
    const { total } = kdLoss(student, teacher, 0, 2, 1);
    expect(total).toBeCloseTo(4 * klDivergence(softmax(teacher, 2), softmax(student, 2)), 10);
  });
  it("mixes the two terms by alpha", () => {
    const { total, teacherTerm, labelTerm } = kdLoss(student, teacher, 0, 2, 0.3);
    expect(total).toBeCloseTo(0.3 * teacherTerm + 0.7 * labelTerm, 10);
  });
});
