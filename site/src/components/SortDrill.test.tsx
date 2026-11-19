import { render, screen, waitFor, within } from "@testing-library/react";
import userEvent from "@testing-library/user-event";
import { LocalProgressStore, setProgressStore } from "../lib/progress";
import SortDrill from "./SortDrill";

const drill = (
  <SortDrill
    lessonId="m/a"
    id="d1"
    categories={["Fine-tuning", "Distillation", "LoRA"]}
    items={[
      { text: "The targets come from a larger model.", answer: "Distillation", why: "Distillation is about where targets come from." },
      { text: "Only small added matrices are trained.", answer: "LoRA", why: "LoRA is about which weights change." },
    ]}
  />
);

let store: LocalProgressStore;
beforeEach(() => {
  store = new LocalProgressStore();
  setProgressStore(store);
});

const item = (n: number) => within(screen.getAllByRole("listitem")[n]);

describe("SortDrill", () => {
  it("starts with nothing sorted", () => {
    render(drill);
    expect(screen.getByText("0 of 2 sorted correctly")).toBeVisible();
  });

  it("shows the right category and the explanation after a wrong choice", async () => {
    render(drill);
    await userEvent.click(item(0).getByRole("button", { name: "LoRA" }));
    expect(item(0).getByText(/Not quite\. It is Distillation\./)).toBeVisible();
    expect(item(0).getByText(/where targets come from/)).toBeVisible();
    expect(screen.getByText("0 of 2 sorted correctly")).toBeVisible();
  });

  it("counts an item once when answered wrong and then right", async () => {
    render(drill);
    await userEvent.click(item(0).getByRole("button", { name: "LoRA" }));
    await userEvent.click(item(0).getByRole("button", { name: "Distillation" }));
    expect(item(0).getByText(/Correct\./)).toBeVisible();
    expect(item(0).getByText(/where targets come from/)).toBeVisible();
    expect(screen.getByText("1 of 2 sorted correctly")).toBeVisible();
  });

  it("records each item under its own quiz id", async () => {
    render(drill);
    await userEvent.click(item(1).getByRole("button", { name: "LoRA" }));
    await waitFor(async () =>
      expect((await store.load()).quizzes["m/a#d1:1"]).toEqual({ attempts: 1, correct: true, lastAnswer: "LoRA" }),
    );
  });
});
