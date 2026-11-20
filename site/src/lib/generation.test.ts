import { forwardKl, gaussianBins, reverseKl, storageGb, twoPeakTeacher } from "./generation";

describe("storageGb", () => {
  it("full vocabulary: 2 bytes per token in the vocabulary, for every position", () => {
    // 1,000,000 positions x 150,000 tokens x 2 bytes = 300 GB (decimal)
    expect(storageGb(1_000_000, { vocabulary: 150_000 })).toBeCloseTo(279.4, 1);
  });
  it("top-k: 6 bytes per kept token", () => {
    expect(storageGb(1_000_000, { k: 20 })).toBeCloseTo(0.1118, 3);
  });
});

describe("distributions", () => {
  it("gaussianBins adds up to 1", () => {
    const p = gaussianBins(0, 1);
    expect(p.reduce((a, b) => a + b, 0)).toBeCloseTo(1, 8);
  });
  it("the teacher has two peaks, one on each side", () => {
    const p = twoPeakTeacher();
    const middle = p[Math.floor(p.length / 2)];
    expect(Math.max(...p.slice(0, p.length / 2))).toBeGreaterThan(middle * 3);
    expect(Math.max(...p.slice(p.length / 2))).toBeGreaterThan(middle * 3);
  });
});

describe("KL direction", () => {
  const teacher = twoPeakTeacher();
  it("both are zero when student equals teacher", () => {
    expect(forwardKl(teacher, teacher)).toBeCloseTo(0, 8);
    expect(reverseKl(teacher, teacher)).toBeCloseTo(0, 8);
  });
  it("forward KL prefers a wide student that covers both peaks", () => {
    const wide = gaussianBins(0, 2.4);
    const narrowOnOne = gaussianBins(-2, 0.7);
    expect(forwardKl(teacher, wide)).toBeLessThan(forwardKl(teacher, narrowOnOne));
  });
  it("reverse KL prefers a narrow student sitting on one peak", () => {
    const wide = gaussianBins(0, 2.4);
    const narrowOnOne = gaussianBins(-2, 0.7);
    expect(reverseKl(teacher, narrowOnOne)).toBeLessThan(reverseKl(teacher, wide));
  });
});
