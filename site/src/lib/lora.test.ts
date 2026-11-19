import { fullParams, loraParams, loraShare, trainingMemoryGb } from "./lora";

describe("LoRA sizes", () => {
  it("counts a full weight matrix", () => {
    expect(fullParams(896, 896)).toBe(802_816);
  });
  it("counts the two LoRA matrices", () => {
    expect(loraParams(896, 896, 16)).toBe(16 * (896 + 896));
  });
  it("gives the share of the full matrix that LoRA trains", () => {
    expect(loraShare(896, 896, 16)).toBeCloseTo(28_672 / 802_816, 10);
  });
  it("grows in a straight line with rank", () => {
    expect(loraParams(1000, 1000, 8) * 2).toBe(loraParams(1000, 1000, 16));
  });
});

describe("trainingMemoryGb", () => {
  it("full fine-tuning needs about 16 bytes per weight", () => {
    expect(trainingMemoryGb(1.5, "full")).toBeCloseTo(24, 1);
  });
  it("LoRA needs about 2 bytes per weight for the frozen model", () => {
    expect(trainingMemoryGb(7.6, "lora")).toBeCloseTo(15.2, 1);
  });
  it("QLoRA needs about 0.7 bytes per weight", () => {
    expect(trainingMemoryGb(7.6, "qlora")).toBeCloseTo(5.3, 1);
  });
});
