import { render, screen, waitFor } from "@testing-library/react";
import userEvent from "@testing-library/user-event";
import { LocalProgressStore, setProgressStore } from "../lib/progress";
import { setJupyterUrl } from "../lib/settings";
import CompleteLesson from "./CompleteLesson";
import ConceptGrid from "./ConceptGrid";
import LabCard from "./LabCard";
import ProgressBar from "./ProgressBar";

let store: LocalProgressStore;
beforeEach(() => {
  store = new LocalProgressStore();
  setProgressStore(store);
});

const cell = (row: number, col: number, title: string) => ({
  row,
  col,
  title,
  summary: `${title} summary`,
  example: "ex",
  bestFor: "best",
  taughtIn: "Module 3",
});

describe("ConceptGrid", () => {
  const grid = (
    <ConceptGrid
      rows={["Black-box", "White-box"]}
      cols={["Full fine-tune", "LoRA"]}
      cells={[cell(0, 0, "Synthetic SFT"), cell(0, 1, "SFT + LoRA"), cell(1, 0, "Logit KD"), cell(1, 1, "KD + LoRA")]}
    />
  );
  it("shows the first cell's detail at rest", () => {
    render(grid);
    expect(screen.getByText("Synthetic SFT summary")).toBeVisible();
    expect(screen.getByRole("button", { name: /Synthetic SFT/ })).toHaveAttribute("aria-pressed", "true");
  });
  it("switches detail on click", async () => {
    render(grid);
    await userEvent.click(screen.getByRole("button", { name: /Logit KD/ }));
    expect(screen.getByText("Logit KD summary")).toBeVisible();
    expect(screen.queryByText("Synthetic SFT summary")).not.toBeInTheDocument();
    expect(screen.getByRole("button", { name: /Synthetic SFT/ })).toHaveAttribute("aria-pressed", "false");
  });
});

describe("LabCard", () => {
  const card = <LabCard title="Check your GPU" path="labs/00_sample/check_gpu.ipynb" minutes={5} />;
  it("shows the notebook path and points to settings when no address is saved", async () => {
    render(card);
    expect(screen.getByText("labs/00_sample/check_gpu.ipynb")).toBeVisible();
    expect(await screen.findByRole("link", { name: /Set your Jupyter address/ })).toHaveAttribute(
      "href",
      "/settings/",
    );
    expect(screen.queryByRole("link", { name: /Open in Jupyter/ })).not.toBeInTheDocument();
  });
  it("links straight to the notebook when an address is saved", async () => {
    setJupyterUrl("http://localhost:8888");
    render(card);
    const link = await screen.findByRole("link", { name: /Open in Jupyter/ });
    expect(link).toHaveAttribute("href", "http://localhost:8888/lab/tree/labs/00_sample/check_gpu.ipynb");
    expect(link).toHaveAttribute("rel", "noopener noreferrer");
    expect(link).toHaveAttribute("target", "_blank");
  });
});

describe("CompleteLesson", () => {
  it("records completion and then reads Completed", async () => {
    render(<CompleteLesson lessonId="m/a" />);
    await userEvent.click(screen.getByRole("button", { name: "Mark lesson complete" }));
    expect(await screen.findByText("Completed")).toBeVisible();
    expect((await store.load()).lessons["m/a"]).toBeDefined();
  });
  it("reads Completed on mount when the lesson was already done", async () => {
    await store.completeLesson("m/a");
    render(<CompleteLesson lessonId="m/a" />);
    expect(await screen.findByText("Completed")).toBeVisible();
  });
});

describe("ProgressBar", () => {
  it("starts at 0% and updates when the store changes", async () => {
    render(<ProgressBar lessonIds={["m/a", "m/b"]} />);
    expect(screen.getByRole("progressbar")).toHaveAttribute("aria-valuenow", "0");
    await store.completeLesson("m/a");
    await waitFor(() => expect(screen.getByRole("progressbar")).toHaveAttribute("aria-valuenow", "50"));
    expect(screen.getByText("1 of 2 lessons complete")).toBeVisible();
  });
});
