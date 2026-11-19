import { fireEvent, render, screen } from "@testing-library/react";
import userEvent from "@testing-library/user-event";
import LoraSizeExplorer from "./LoraSizeExplorer";
import MemoryExplorer from "./MemoryExplorer";

describe("LoraSizeExplorer", () => {
  it("shows the full and LoRA weight counts for the starting rank", () => {
    render(<LoraSizeExplorer />);
    const status = screen.getByRole("status");
    expect(status).toHaveTextContent("802,816");
    expect(status).toHaveTextContent("28,672");
    expect(status).toHaveTextContent("3.6%");
  });

  it("updates when the rank changes", () => {
    render(<LoraSizeExplorer />);
    fireEvent.change(screen.getByRole("slider", { name: /rank/i }), { target: { value: "4" } });
    expect(screen.getByRole("status")).toHaveTextContent("7,168");
  });

  it("updates when a larger model is chosen", async () => {
    render(<LoraSizeExplorer />);
    await userEvent.click(screen.getByRole("button", { name: /7B/ }));
    expect(screen.getByRole("status")).toHaveTextContent("12,845,056");
  });
});

describe("MemoryExplorer", () => {
  it("says a 0.5B full fine-tune fits on a 24 GB card", () => {
    render(<MemoryExplorer />);
    expect(screen.getByRole("status")).toHaveTextContent(/fits/i);
  });

  it("says a 7B full fine-tune does not fit", async () => {
    render(<MemoryExplorer />);
    await userEvent.click(screen.getByRole("button", { name: "7.6B" }));
    expect(screen.getByRole("status")).toHaveTextContent(/does not fit/i);
  });

  it("says a 7B model fits with QLoRA", async () => {
    render(<MemoryExplorer />);
    await userEvent.click(screen.getByRole("button", { name: "7.6B" }));
    await userEvent.click(screen.getByRole("button", { name: "QLoRA" }));
    expect(screen.getByRole("status")).toHaveTextContent(/fits/i);
    expect(screen.getByRole("status")).not.toHaveTextContent(/does not fit/i);
  });
});
