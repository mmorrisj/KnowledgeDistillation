import { useId, useState } from "react";
import { storageGb } from "../lib/generation";
import "./components.css";

const VOCABULARY = 150_000;
const whole = new Intl.NumberFormat("en-US");

/** Why a teacher's next-token probabilities are stored as a short list per position. */
export default function TopKExplorer() {
  const [k, setK] = useState(20);
  const [examples, setExamples] = useState(10_000);
  const [length, setLength] = useState(50);
  const id = useId();

  const positions = examples * length;
  const full = storageGb(positions, { vocabulary: VOCABULARY });
  const kept = storageGb(positions, { k });
  const ratio = full / kept;

  return (
    <section className="card" aria-label="Top-k storage explorer">
      <p className="card-kicker">How much disk do the teacher's probabilities need?</p>
      <p className="feedback">Assumes a vocabulary of about {whole.format(VOCABULARY)} tokens.</p>
      <div className="slider-row">
        <label htmlFor={`${id}-k`}>Tokens kept per position</label>
        <input id={`${id}-k`} type="range" min={1} max={100} value={k} onChange={(e) => setK(Number(e.target.value))} />
        <span className="slider-value">{k}</span>
      </div>
      <div className="slider-row">
        <label htmlFor={`${id}-n`}>Training examples</label>
        <input
          id={`${id}-n`}
          type="range"
          min={1000}
          max={100000}
          step={1000}
          value={examples}
          onChange={(e) => setExamples(Number(e.target.value))}
        />
        <span className="slider-value">{whole.format(examples)}</span>
      </div>
      <div className="slider-row">
        <label htmlFor={`${id}-len`}>Tokens per answer</label>
        <input
          id={`${id}-len`}
          type="range"
          min={10}
          max={500}
          step={10}
          value={length}
          onChange={(e) => setLength(Number(e.target.value))}
        />
        <span className="slider-value">{length}</span>
      </div>
      <div className="calc-results" role="status">
        <div className="calc-row">
          <span>Every token</span>
          <div className="calc-bar"><div style={{ width: "100%" }} /></div>
          <strong>{whole.format(Math.round(full))} GB</strong>
        </div>
        <div className="calc-row">
          <span>Top {k} only</span>
          <div className="calc-bar"><div className="own" style={{ width: `${Math.max(0.5, (kept / full) * 100)}%` }} /></div>
          <strong data-testid="topk-size">{(kept * 1024).toFixed(1)} MB</strong>
        </div>
        <p>
          Keeping the top {k} is about <strong>{whole.format(Math.round(ratio))} times smaller</strong>.
        </p>
      </div>
    </section>
  );
}
