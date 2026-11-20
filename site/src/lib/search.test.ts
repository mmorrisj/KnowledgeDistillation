import { overlapAtK, vectorStorageGb } from "./search";

describe("vectorStorageGb", () => {
  it("is count x dimensions x 4 bytes", () => {
    // 1,000,000 vectors of 1,024 numbers at 4 bytes each = 4.096 GB (decimal) = 3.81 GiB
    expect(vectorStorageGb(1_000_000, 1024)).toBeCloseTo(3.815, 2);
  });
  it("scales with dimensions", () => {
    expect(vectorStorageGb(1_000_000, 384) / vectorStorageGb(1_000_000, 1024)).toBeCloseTo(0.375, 6);
  });
});

describe("overlapAtK", () => {
  it("is the share of the first k results two lists have in common", () => {
    expect(overlapAtK(["a", "b", "c", "d"], ["b", "a", "x", "c"], 2)).toBe(1);
    expect(overlapAtK(["a", "b", "c", "d"], ["b", "a", "x", "c"], 3)).toBeCloseTo(2 / 3, 10);
    expect(overlapAtK(["a", "b"], ["x", "y"], 2)).toBe(0);
  });
});
