import { render, screen } from "@testing-library/react";
import userEvent from "@testing-library/user-event";
import PredictReveal from "./PredictReveal";

describe("PredictReveal", () => {
  it("hides the explanation until an option is chosen, then keeps the choice marked", async () => {
    render(
      <PredictReveal id="p1" prompt="What happens to banana?" options={["Stays at 0%", "Rises"]}>
        <p>It rises, but the order never changes.</p>
      </PredictReveal>,
    );
    expect(screen.queryByText(/order never changes/)).not.toBeVisible();
    await userEvent.click(screen.getByRole("button", { name: "Rises" }));
    expect(screen.getByText(/order never changes/)).toBeVisible();
    expect(screen.getByRole("button", { name: "Rises" })).toHaveAttribute("aria-pressed", "true");
    expect(screen.getByRole("button", { name: "Stays at 0%" })).toHaveAttribute("aria-pressed", "false");
  });
});
