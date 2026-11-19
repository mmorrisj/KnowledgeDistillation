import { render, screen } from "@testing-library/react";
import userEvent from "@testing-library/user-event";
import TrainingLoopExplorer from "./TrainingLoopExplorer";

const changed = () =>
  screen
    .getAllByRole("listitem")
    .filter((li) => li.getAttribute("data-changed") === "true")
    .map((li) => li.getAttribute("data-stage"));

describe("TrainingLoopExplorer", () => {
  it("shows five stages with the first mode selected at rest", () => {
    render(<TrainingLoopExplorer />);
    expect(screen.getAllByRole("listitem")).toHaveLength(5);
    expect(screen.getByRole("button", { name: "Training from scratch" })).toHaveAttribute("aria-pressed", "true");
    expect(screen.getByRole("status")).not.toBeEmptyDOMElement();
  });

  it("marks only the target stage for distillation", async () => {
    render(<TrainingLoopExplorer />);
    await userEvent.click(screen.getByRole("button", { name: "Distillation" }));
    expect(changed()).toEqual(["target"]);
  });

  it("marks only the update stage for LoRA", async () => {
    render(<TrainingLoopExplorer />);
    await userEvent.click(screen.getByRole("button", { name: "LoRA" }));
    expect(changed()).toEqual(["update"]);
  });

  it("marks the data and the starting weights for fine-tuning", async () => {
    render(<TrainingLoopExplorer />);
    await userEvent.click(screen.getByRole("button", { name: "Fine-tuning" }));
    expect(changed()).toEqual(["input", "weights"]);
  });

  it("says in words which stage changes, not only by color", async () => {
    render(<TrainingLoopExplorer />);
    await userEvent.click(screen.getByRole("button", { name: "Distillation" }));
    expect(screen.getAllByText("Changes here")).toHaveLength(1);
  });

  it("gives every mode an explanation", async () => {
    render(<TrainingLoopExplorer />);
    for (const name of ["Training from scratch", "Pretraining", "Fine-tuning", "Distillation", "LoRA"]) {
      await userEvent.click(screen.getByRole("button", { name }));
      expect(screen.getByRole("status").textContent?.length).toBeGreaterThan(40);
    }
  });
});
