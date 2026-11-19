import { useState } from "react";
import { trainingMemoryGb, type Method } from "../lib/lora";
import "./components.css";

const SIZES = [0.5, 1.5, 3.1, 7.6];
const METHODS: { key: Method; name: string }[] = [
  { key: "full", name: "Full fine-tune" },
  { key: "lora", name: "LoRA" },
  { key: "qlora", name: "QLoRA" },
];
const CARD_GB = 24;
const WORKING_GB = 3;
const SCALE_GB = 48;

/** Does this model fit on one 24 GB card with this training method? */
export default function MemoryExplorer() {
  const [size, setSize] = useState(0.5);
  const [method, setMethod] = useState<Method>("full");
  const needed = trainingMemoryGb(size, method);
  const fits = needed + WORKING_GB <= CARD_GB;

  return (
    <section className="card" aria-label="Memory explorer">
      <p className="card-kicker">Will it fit on one 24 GB card?</p>
      <div className="choices" role="group" aria-label="Model size in billions of weights">
        {SIZES.map((s) => (
          <button key={s} type="button" className="btn choice" aria-pressed={s === size} onClick={() => setSize(s)}>
            {s}B
          </button>
        ))}
      </div>
      <div className="choices" role="group" aria-label="Training method">
        {METHODS.map((m) => (
          <button key={m.key} type="button" className="btn choice" aria-pressed={m.key === method} onClick={() => setMethod(m.key)}>
            {m.name}
          </button>
        ))}
      </div>
      <div className="memory-track" aria-hidden="true">
        <div className={`memory-fill ${fits ? "" : "over"}`} style={{ width: `${Math.min(100, (needed / SCALE_GB) * 100)}%` }} />
        <div className="memory-limit" style={{ left: `${(CARD_GB / SCALE_GB) * 100}%` }}>
          <span>24 GB</span>
        </div>
      </div>
      <p role="status">
        About <strong>{needed.toFixed(1)} GB</strong> for the model while training.{" "}
        {fits ? (
          <strong>It fits,</strong>
        ) : (
          <strong>It does not fit,</strong>
        )}{" "}
        allowing about {WORKING_GB} GB of working memory for the batch.
      </p>
      <p className="feedback">
        These are rough estimates. Working memory grows with batch size and message length, so real runs vary.
      </p>
    </section>
  );
}
