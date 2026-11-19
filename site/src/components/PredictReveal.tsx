import { useState, type ReactNode } from "react";
import "./components.css";

interface Props {
  id: string;
  prompt: string;
  options: string[];
  children: ReactNode;
}

/** Ask for a prediction first; the explanation appears once the student commits. */
export default function PredictReveal({ id, prompt, options, children }: Props) {
  const [choice, setChoice] = useState<string | null>(null);
  const revealId = `predict-${id}`;

  return (
    <section className="card predict" aria-label="Predict first">
      <p className="card-kicker">Predict first</p>
      <p className="card-question">{prompt}</p>
      <div className="choices">
        {options.map((option) => (
          <button
            key={option}
            type="button"
            className="btn choice"
            aria-pressed={option === choice}
            aria-controls={revealId}
            onClick={() => setChoice(option)}
          >
            {option}
          </button>
        ))}
      </div>
      {choice === null && <p className="feedback">Pick an answer to see what happens.</p>}
      <div id={revealId} className="reveal" hidden={choice === null}>
        {children}
      </div>
    </section>
  );
}
