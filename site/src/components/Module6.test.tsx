import { fireEvent, render, screen } from "@testing-library/react";
import KlDirectionExplorer from "./KlDirectionExplorer";
import TopKExplorer from "./TopKExplorer";

describe("TopKExplorer", () => {
  it("shows the full-vocabulary size and the top-k size", () => {
    render(<TopKExplorer />);
    const status = screen.getByRole("status");
    expect(status).toHaveTextContent(/GB/);
    expect(status).toHaveTextContent(/times smaller/i);
  });

  it("shrinks the top-k size when k is lowered", () => {
    render(<TopKExplorer />);
    const read = () => Number(screen.getByTestId("topk-size").textContent?.replace(/[^\d.]/g, ""));
    const before = read();
    fireEvent.change(screen.getByRole("slider", { name: /tokens kept/i }), { target: { value: "5" } });
    expect(read()).toBeLessThan(before);
  });
});

describe("KlDirectionExplorer", () => {
  const set = (name: RegExp, value: string) =>
    fireEvent.change(screen.getByRole("slider", { name }), { target: { value } });
  const value = (id: string) => Number(screen.getByTestId(id).textContent);

  it("shows both divergences", () => {
    render(<KlDirectionExplorer />);
    expect(value("forward")).toBeGreaterThan(0);
    expect(value("reverse")).toBeGreaterThan(0);
  });

  it("a narrow student on one peak has low reverse KL and high forward KL", () => {
    render(<KlDirectionExplorer />);
    set(/position/i, "-2");
    set(/width/i, "0.7");
    const narrowForward = value("forward");
    const narrowReverse = value("reverse");
    set(/position/i, "0");
    set(/width/i, "2.4");
    expect(value("forward")).toBeLessThan(narrowForward);
    expect(value("reverse")).toBeGreaterThan(narrowReverse);
  });

  it("describes what the student is doing in words", () => {
    render(<KlDirectionExplorer />);
    set(/position/i, "-2");
    set(/width/i, "0.7");
    expect(screen.getByRole("status")).toHaveTextContent(/one peak/i);
  });
});
