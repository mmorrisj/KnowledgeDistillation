import { fireEvent, render, screen } from "@testing-library/react";
import CascadeExplorer from "./CascadeExplorer";
import QuantizationExplorer from "./QuantizationExplorer";

describe("QuantizationExplorer", () => {
  it("reports the number of allowed values and the size", () => {
    render(<QuantizationExplorer />);
    const status = screen.getByRole("status");
    expect(status).toHaveTextContent("256 allowed values");
    expect(status).toHaveTextContent("25%");
  });
  it("updates when bits change", () => {
    render(<QuantizationExplorer />);
    fireEvent.change(screen.getByRole("slider", { name: /bits/i }), { target: { value: "4" } });
    const status = screen.getByRole("status");
    expect(status).toHaveTextContent("16 allowed values");
    expect(status).toHaveTextContent("13%");
  });
});

describe("CascadeExplorer", () => {
  const share = () => Number(screen.getByTestId("sent").textContent?.replace("%", ""));
  it("sends nothing to the teacher at threshold 0", () => {
    render(<CascadeExplorer />);
    fireEvent.change(screen.getByRole("slider", { name: /threshold/i }), { target: { value: "0" } });
    expect(share()).toBe(0);
  });
  it("sends more as the threshold rises", () => {
    render(<CascadeExplorer />);
    fireEvent.change(screen.getByRole("slider", { name: /threshold/i }), { target: { value: "0.5" } });
    const low = share();
    fireEvent.change(screen.getByRole("slider", { name: /threshold/i }), { target: { value: "0.95" } });
    expect(share()).toBeGreaterThan(low);
  });
});
