import { render, screen, waitFor } from "@testing-library/react";
import userEvent from "@testing-library/user-event";
import { LocalProgressStore, setProgressStore } from "../lib/progress";
import Quiz from "./Quiz";

const options = [
  { text: "Logit distillation", feedback: "A text-only API does not expose probabilities." },
  { text: "Train on generated labels", correct: true, feedback: "You train on the teacher's answers." },
];
const quiz = <Quiz lessonId="m/a" id="q1" question="Which method works?" options={options} />;

let store: LocalProgressStore;
beforeEach(() => {
  store = new LocalProgressStore();
  setProgressStore(store);
});

describe("Quiz", () => {
  it("shows no feedback before an answer", () => {
    render(quiz);
    expect(screen.getByRole("status")).toBeEmptyDOMElement();
  });

  it("marks a right answer in words and shows its feedback", async () => {
    render(quiz);
    await userEvent.click(screen.getByRole("button", { name: /Train on generated labels/ }));
    expect(screen.getByRole("status")).toHaveTextContent("Correct. You train on the teacher's answers.");
  });

  it("marks a wrong answer in words and shows its feedback", async () => {
    render(quiz);
    await userEvent.click(screen.getByRole("button", { name: /Logit distillation/ }));
    expect(screen.getByRole("status")).toHaveTextContent(
      "Not quite. A text-only API does not expose probabilities.",
    );
  });

  it("records the attempt in the progress store", async () => {
    render(quiz);
    await userEvent.click(screen.getByRole("button", { name: /Logit distillation/ }));
    await waitFor(async () =>
      expect((await store.load()).quizzes["m/a#q1"]).toEqual({
        attempts: 1,
        correct: false,
        lastAnswer: "Logit distillation",
      }),
    );
  });

  it("restores the last answer on mount", async () => {
    await store.recordQuiz("m/a", "q1", "Train on generated labels", true);
    render(quiz);
    await waitFor(() => expect(screen.getByRole("status")).toHaveTextContent("Correct."));
    expect(screen.getByRole("button", { name: /Train on generated labels/ })).toHaveAttribute(
      "aria-pressed",
      "true",
    );
  });

  it("can be answered from the keyboard", async () => {
    render(quiz);
    await userEvent.tab();
    await userEvent.tab();
    await userEvent.keyboard("{Enter}");
    expect(screen.getByRole("status")).toHaveTextContent("Correct.");
  });
});
