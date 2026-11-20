import { render, screen } from "@testing-library/react";
import userEvent from "@testing-library/user-event";
import PatchExplorer from "./PatchExplorer";

describe("PatchExplorer", () => {
  it("starts at 16-pixel patches with 196 of them", () => {
    render(<PatchExplorer />);
    expect(screen.getByRole("status")).toHaveTextContent("196 patches");
    expect(screen.getByRole("button", { name: "16 px" })).toHaveAttribute("aria-pressed", "true");
  });
  it("shows more patches when the patch size shrinks", async () => {
    render(<PatchExplorer />);
    await userEvent.click(screen.getByRole("button", { name: "8 px" }));
    expect(screen.getByRole("status")).toHaveTextContent("784 patches");
  });
  it("draws one cell per patch", async () => {
    render(<PatchExplorer />);
    await userEvent.click(screen.getByRole("button", { name: "32 px" }));
    expect(screen.getAllByTestId("patch")).toHaveLength(49);
  });
});
