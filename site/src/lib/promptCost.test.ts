import { labellingCost, promptTokens } from "./promptCost";

describe("promptTokens", () => {
  it("counts instructions, the type list and the message when there are no examples", () => {
    expect(promptTokens({ examplesPerType: 0 })).toBe(40 + 77 * 7 + 20);
  });
  it("adds tokens for every example of every type", () => {
    expect(promptTokens({ examplesPerType: 2 })).toBe(40 + 77 * 7 + 20 + 2 * 77 * 22);
  });
});

describe("labellingCost", () => {
  it("multiplies tokens by messages and prices per million", () => {
    // 1,000 input tokens and 10 output tokens per message, 10,000 messages
    expect(labellingCost(1000, 10, 10_000, 3, 15)).toBeCloseTo(30 + 1.5, 6);
  });
});
