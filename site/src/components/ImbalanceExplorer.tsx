import { useId, useState } from "react";
import { imbalanceScores } from "../lib/stats";
import "./components.css";

const pct = (value: number) => `${Math.round(value * 100)}%`;

/** A model that always gives the common answer: accuracy looks fine, macro-F1 does not. */
export default function ImbalanceExplorer() {
  const [share, setShare] = useState(80);
  const id = useId();
  const { accuracy, macroF1 } = imbalanceScores(share / 100);

  return (
    <section className="card" aria-label="Imbalance explorer">
      <p className="card-kicker">A lazy model on lopsided data</p>
      <p className="card-question">
        Imagine only two types of message, and a model that always answers with the common one. Drag the slider to
        change how lopsided the data is.
      </p>
      <div className="slider-row">
        <label htmlFor={id}>Share of messages that are the common type</label>
        <input id={id} type="range" min={50} max={99} value={share} onChange={(e) => setShare(Number(e.target.value))} />
        <span className="slider-value">{share}%</span>
      </div>
      <div className="calc-results" role="status">
        <div className="calc-row">
          <span>Accuracy</span>
          <div className="calc-bar"><div style={{ width: pct(accuracy) }} /></div>
          <strong>{pct(accuracy)}</strong>
        </div>
        <div className="calc-row">
          <span>Macro-F1</span>
          <div className="calc-bar"><div className="own" style={{ width: pct(macroF1) }} /></div>
          <strong>{pct(macroF1)}</strong>
        </div>
        <p className="feedback">
          The model never finds a single message of the rare type. Accuracy rewards it anyway; macro-F1 stays below
          50% however lopsided the data gets.
        </p>
      </div>
    </section>
  );
}
