import { apiCostPer1k, breakEvenRequestsPerDay, isValidInput, selfHostedDailyCost } from "./cost";

describe("cost", () => {
  it("prices 1,000 API requests from token counts and per-million prices", () => {
    expect(apiCostPer1k(500, 100, 3, 15)).toBeCloseTo(3.0, 6);
  });
  it("prices a machine left on all day", () => {
    expect(selfHostedDailyCost(1.6)).toBeCloseTo(38.4, 6);
  });
  it("finds the daily volume where an always-on machine costs the same as the API", () => {
    expect(breakEvenRequestsPerDay(3.0, 1.6)).toBe(12800);
  });
  it("accepts only finite numbers above zero", () => {
    expect(isValidInput(1, 2.5)).toBe(true);
    expect(isValidInput(1, 0)).toBe(false);
    expect(isValidInput(1, Number.NaN)).toBe(false);
    expect(isValidInput(1, -2)).toBe(false);
    expect(isValidInput(Number.POSITIVE_INFINITY)).toBe(false);
  });
});
