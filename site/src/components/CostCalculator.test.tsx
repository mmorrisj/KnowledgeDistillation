import { render, screen } from "@testing-library/react";
import userEvent from "@testing-library/user-event";
import CostCalculator from "./CostCalculator";

async function setField(label: RegExp, value: string) {
  const input = screen.getByLabelText(label);
  await userEvent.clear(input);
  if (value) await userEvent.type(input, value);
}

describe("CostCalculator", () => {
  it("labels the starting prices as examples", () => {
    render(<CostCalculator />);
    expect(screen.getByText(/example values/i)).toBeVisible();
  });

  it("shows the daily cost of both options and names the cheaper one", async () => {
    render(<CostCalculator defaults={{ inTokens: 500, outTokens: 100, usdPerMTokIn: 3, usdPerMTokOut: 15, instanceUsdPerHour: 1.6, requestsPerDay: 100000 }} />);
    const results = screen.getByRole("status");
    expect(results).toHaveTextContent("$300.00");
    expect(results).toHaveTextContent("$38.40");
    expect(results).toHaveTextContent(/self-hosted student is cheaper/i);
    expect(results).toHaveTextContent("12,800");
  });

  it("names the API as cheaper at low volume", async () => {
    render(<CostCalculator defaults={{ inTokens: 500, outTokens: 100, usdPerMTokIn: 3, usdPerMTokOut: 15, instanceUsdPerHour: 1.6, requestsPerDay: 100000 }} />);
    await setField(/requests per day/i, "1000");
    expect(screen.getByRole("status")).toHaveTextContent(/API is cheaper/i);
  });

  it.each(["", "0", "-5"])("asks for a fix and shows no numbers when a field is '%s'", async (value) => {
    render(<CostCalculator />);
    await setField(/requests per day/i, value);
    const results = screen.getByRole("status");
    expect(results).toHaveTextContent(/enter a number above zero/i);
    expect(results).not.toHaveTextContent(/NaN|Infinity|\$/);
  });
});
