import { useId, useState } from "react";
import { cascade, exampleMessages } from "../lib/serving";
import "./components.css";

const MESSAGES = exampleMessages();
const percent = (value: number) => `${Math.round(value * 100)}%`;

/** Student first, teacher for the uncertain ones: what a confidence threshold buys. */
export default function CascadeExplorer() {
  const [threshold, setThreshold] = useState(0.7);
  const id = useId();
  const result = cascade(MESSAGES, threshold);

  return (
    <section className="card" aria-label="Cascade explorer">
      <p className="card-kicker">Student first, teacher when unsure</p>
      <p className="card-question">
        A thousand made-up messages. Any message the student is less sure of than the threshold is sent on to the
        teacher. The numbers are illustrative.
      </p>
      <div className="slider-row">
        <label htmlFor={id}>Confidence threshold</label>
        <input
          id={id}
          type="range"
          min={0}
          max={1}
          step={0.01}
          value={threshold}
          onChange={(e) => setThreshold(Number(e.target.value))}
        />
        <span className="slider-value">{threshold.toFixed(2)}</span>
      </div>
      <div className="calc-results" role="status">
        <div className="calc-row">
          <span>Sent to the teacher</span>
          <div className="calc-bar"><div style={{ width: percent(result.sentToTeacher) }} /></div>
          <strong data-testid="sent">{percent(result.sentToTeacher)}</strong>
        </div>
        <div className="calc-row">
          <span>Accuracy overall</span>
          <div className="calc-bar"><div className="own" style={{ width: percent(result.accuracy) }} /></div>
          <strong>{percent(result.accuracy)}</strong>
        </div>
        <div className="calc-row">
          <span>Cost against teacher-only</span>
          <div className="calc-bar"><div style={{ width: percent(Math.min(1, result.relativeCost)) }} /></div>
          <strong>{percent(result.relativeCost)}</strong>
        </div>
        <p className="feedback">
          At 0 the student answers everything. At 1 the teacher answers everything, and you still pay for the student.
        </p>
      </div>
    </section>
  );
}
