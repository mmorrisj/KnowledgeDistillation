import { fireEvent, render, screen } from "@testing-library/react";
import PromptCostEstimator from "./PromptCostEstimator";

describe("PromptCostEstimator", () => {
  it("shows prompt size and total cost for the starting values", () => {
    render(<PromptCostEstimator />);
    const status = screen.getByRole("status");
    expect(status).toHaveTextContent("599 tokens");
    expect(status).toHaveTextContent(/\$\d/);
  });

  it("grows the prompt and the cost when examples are added", () => {
    render(<PromptCostEstimator />);
    fireEvent.change(screen.getByRole("slider", { name: /examples per type/i }), { target: { value: "2" } });
    expect(screen.getByRole("status")).toHaveTextContent("3,987 tokens");
  });

  it("labels its prices as examples", () => {
    render(<PromptCostEstimator />);
    expect(screen.getByText(/example prices/i)).toBeVisible();
  });

  it("asks for a valid price instead of showing a broken number", () => {
    render(<PromptCostEstimator />);
    fireEvent.change(screen.getByLabelText(/price per million input tokens/i), { target: { value: "" } });
    expect(screen.getByRole("status")).toHaveTextContent(/enter a price/i);
    expect(screen.getByRole("status")).not.toHaveTextContent(/NaN/);
  });
});
