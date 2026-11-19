import { render, screen } from "@testing-library/react";
import userEvent from "@testing-library/user-event";
import ReviewCard from "./ReviewCard";

describe("ReviewCard", () => {
  it("toggles the answer and reports its state", async () => {
    render(
      <ReviewCard fromModule="Baselines" question="What three scores do you report?">
        <p>Teacher, baseline, distilled.</p>
      </ReviewCard>,
    );
    const button = screen.getByRole("button", { name: "Show answer" });
    expect(button).toHaveAttribute("aria-expanded", "false");
    expect(screen.queryByText(/Teacher, baseline/)).not.toBeVisible();
    await userEvent.click(button);
    expect(screen.getByRole("button", { name: "Hide answer" })).toHaveAttribute("aria-expanded", "true");
    expect(screen.getByText(/Teacher, baseline/)).toBeVisible();
  });
});
