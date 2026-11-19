import { useState } from "react";
import "./components.css";

type StageKey = "input" | "weights" | "prediction" | "target" | "update";

const STAGES: { key: StageKey; title: string }[] = [
  { key: "input", title: "Training data" },
  { key: "weights", title: "Model weights" },
  { key: "prediction", title: "Prediction" },
  { key: "target", title: "Target" },
  { key: "update", title: "Weight update" },
];

type Mode = {
  name: string;
  comparedWith: string | null;
  changed: StageKey[];
  stages: Record<StageKey, string>;
  explanation: string;
};

const MODES: Mode[] = [
  {
    name: "Training from scratch",
    comparedWith: null,
    changed: [],
    stages: {
      input: "Labelled examples for one task",
      weights: "Start random",
      prediction: "A probability for each class",
      target: "The human label",
      update: "Every weight moves",
    },
    explanation:
      "This is how most classical models learn. The model starts knowing nothing, and everything it learns comes from one labelled dataset for one task. The loop is always the same: predict, compare with the target, nudge the weights to shrink the gap.",
  },
  {
    name: "Pretraining",
    comparedWith: "training from scratch",
    changed: ["input", "target"],
    stages: {
      input: "Billions of words of unlabelled text",
      weights: "Start random",
      prediction: "A probability for every possible next token",
      target: "The token that actually came next",
      update: "Every weight moves",
    },
    explanation:
      "The text supplies its own answers: hide the next token and ask the model to predict it. Nobody has to label anything, so the model can learn from a huge amount of text. What comes out is general knowledge of language stored in the weights.",
  },
  {
    name: "Fine-tuning",
    comparedWith: "pretraining",
    changed: ["input", "weights"],
    stages: {
      input: "Your labelled examples for one task",
      weights: "Start from the pretrained weights",
      prediction: "A probability for each class or next token",
      target: "Your human labels",
      update: "Every weight moves",
    },
    explanation:
      "The same loop again, but it starts from pretrained weights and uses your task data. Because the model already knows language, a few thousand examples can be enough where training from scratch would need far more.",
  },
  {
    name: "Distillation",
    comparedWith: "fine-tuning",
    changed: ["target"],
    stages: {
      input: "Your examples for one task",
      weights: "Start from a small pretrained student",
      prediction: "A probability for each class or next token",
      target: "What a larger teacher model says",
      update: "Every weight moves",
    },
    explanation:
      "Only the target changes. The student is pulled toward a teacher model's answers or probabilities, in place of human labels or alongside them. Distillation is a choice about where the targets come from.",
  },
  {
    name: "LoRA",
    comparedWith: "fine-tuning",
    changed: ["update"],
    stages: {
      input: "Your examples for one task",
      weights: "Pretrained weights, frozen",
      prediction: "A probability for each class or next token",
      target: "Human labels or a teacher's outputs",
      update: "Only small added matrices move",
    },
    explanation:
      "Only the update changes. The original weights stay frozen and a small set of added matrices is trained, which needs far less GPU memory. LoRA is a choice about which weights move, so it works with human labels or with a teacher.",
  },
];

/** The training loop as five stages, showing what each training method changes. */
export default function TrainingLoopExplorer() {
  const [active, setActive] = useState(0);
  const mode = MODES[active];

  return (
    <section className="card loop" aria-label="Training loop explorer">
      <p className="card-kicker">Training loop explorer</p>
      <div className="choices" role="group" aria-label="Training method">
        {MODES.map((m, index) => (
          <button
            key={m.name}
            type="button"
            className="btn choice"
            aria-pressed={index === active}
            onClick={() => setActive(index)}
          >
            {m.name}
          </button>
        ))}
      </div>
      <ol className="loop-stages">
        {STAGES.map((stage) => {
          const changed = mode.changed.includes(stage.key);
          return (
            <li key={stage.key} data-stage={stage.key} data-changed={changed}>
              <span className="loop-title">{stage.title}</span>
              <span className="loop-detail">{mode.stages[stage.key]}</span>
              {changed && <span className="loop-tag">Changes here</span>}
            </li>
          );
        })}
      </ol>
      <p className="loop-explain" role="status">
        {mode.comparedWith && <strong>Compared with {mode.comparedWith}: </strong>}
        {mode.explanation}
      </p>
    </section>
  );
}
