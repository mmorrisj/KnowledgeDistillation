import { fireEvent, render, screen, within } from "@testing-library/react";
import KdLossExplorer from "./KdLossExplorer";
import TemperatureExplorer from "./TemperatureExplorer";

const percent = (row: HTMLElement) => Number(within(row).getByTestId("value").textContent?.replace("%", ""));

describe("TemperatureExplorer", () => {
  it("starts at temperature 1 with the top type well ahead", () => {
    render(<TemperatureExplorer />);
    const rows = screen.getAllByRole("listitem");
    expect(percent(rows[0])).toBeGreaterThan(50);
  });

  it("flattens the distribution as temperature rises without changing the order", () => {
    render(<TemperatureExplorer />);
    const before = screen.getAllByRole("listitem").map(percent);
    fireEvent.change(screen.getByRole("slider", { name: /temperature/i }), { target: { value: "6" } });
    const after = screen.getAllByRole("listitem").map(percent);
    expect(after[0]).toBeLessThan(before[0]);
    expect(after[after.length - 1]).toBeGreaterThan(before[before.length - 1]);
    expect([...after].sort((a, b) => b - a)).toEqual(after);
  });

  it("describes the current state in words", () => {
    render(<TemperatureExplorer />);
    fireEvent.change(screen.getByRole("slider", { name: /temperature/i }), { target: { value: "6" } });
    expect(screen.getByRole("status")).toHaveTextContent(/flatter/i);
  });
});

describe("KdLossExplorer", () => {
  it("shows the teacher term, the label term and the total", () => {
    render(<KdLossExplorer />);
    const status = screen.getByRole("status");
    expect(status).toHaveTextContent(/teacher term/i);
    expect(status).toHaveTextContent(/label term/i);
    expect(status).toHaveTextContent(/total/i);
  });

  it("says the teacher is ignored when alpha is 0", () => {
    render(<KdLossExplorer />);
    fireEvent.change(screen.getByRole("slider", { name: /alpha/i }), { target: { value: "0" } });
    expect(screen.getByRole("status")).toHaveTextContent(/ordinary fine-tuning/i);
  });

  it("says the human label is ignored when alpha is 1", () => {
    render(<KdLossExplorer />);
    fireEvent.change(screen.getByRole("slider", { name: /alpha/i }), { target: { value: "1" } });
    expect(screen.getByRole("status")).toHaveTextContent(/teacher alone/i);
  });
});
