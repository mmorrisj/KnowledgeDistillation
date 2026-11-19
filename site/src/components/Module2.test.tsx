import { fireEvent, render, screen } from "@testing-library/react";
import userEvent from "@testing-library/user-event";
import GapClosedCalculator from "./GapClosedCalculator";
import ImbalanceExplorer from "./ImbalanceExplorer";
import IntervalExplorer from "./IntervalExplorer";

async function setField(label: RegExp, value: string) {
  const input = screen.getByLabelText(label);
  await userEvent.clear(input);
  if (value) await userEvent.type(input, value);
}

describe("GapClosedCalculator", () => {
  it("shows the gap closed for the starting example", () => {
    render(<GapClosedCalculator defaults={{ baseline: 60, distilled: 72, teacher: 80 }} />);
    expect(screen.getByRole("status")).toHaveTextContent("60% of the gap");
  });
  it("explains when there is no gap to close", async () => {
    render(<GapClosedCalculator defaults={{ baseline: 60, distilled: 72, teacher: 80 }} />);
    await setField(/Teacher/, "55");
    expect(screen.getByRole("status")).toHaveTextContent(/no gap to close/i);
  });
  it("says so when the student beats the teacher", async () => {
    render(<GapClosedCalculator defaults={{ baseline: 60, distilled: 72, teacher: 80 }} />);
    await setField(/Distilled student/, "85");
    expect(screen.getByRole("status")).toHaveTextContent(/beat the teacher/i);
  });
  it("says so when distillation made things worse", async () => {
    render(<GapClosedCalculator defaults={{ baseline: 60, distilled: 72, teacher: 80 }} />);
    await setField(/Distilled student/, "50");
    expect(screen.getByRole("status")).toHaveTextContent(/worse than the baseline/i);
  });
  it.each(["", "120", "-3"])("asks for a valid score when a field is '%s'", async (value) => {
    render(<GapClosedCalculator />);
    await setField(/Teacher/, value);
    expect(screen.getByRole("status")).toHaveTextContent(/between 0 and 100/i);
    expect(screen.getByRole("status")).not.toHaveTextContent(/NaN|Infinity/);
  });
});

describe("ImbalanceExplorer", () => {
  it("shows high accuracy and low macro-F1 for a lazy model on lopsided data", () => {
    render(<ImbalanceExplorer />);
    fireEvent.change(screen.getByLabelText(/common type/i), { target: { value: "95" } });
    const results = screen.getByRole("status");
    expect(results).toHaveTextContent("95%");
    expect(results).toHaveTextContent("49%");
  });
});

describe("IntervalExplorer", () => {
  it("calls a 2-point gap on 200 test messages unclear", () => {
    render(<IntervalExplorer defaults={{ a: 70, b: 72, n: 200 }} />);
    expect(screen.getByRole("status")).toHaveTextContent(/cannot tell/i);
  });
  it("calls the same gap real once the test set is large enough", () => {
    render(<IntervalExplorer defaults={{ a: 70, b: 72, n: 200 }} />);
    fireEvent.change(screen.getByLabelText(/test messages/i), { target: { value: "20000" } });
    expect(screen.getByRole("status")).toHaveTextContent(/real difference/i);
  });
});
