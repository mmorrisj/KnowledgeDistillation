import { useId, useState } from "react";
import { fullParams, loraParams, loraShare } from "../lib/lora";
import "./components.css";

const MODELS = [
  { name: "0.5B student", width: 896 },
  { name: "1.5B student", width: 1536 },
  { name: "7B teacher", width: 3584 },
];
const whole = new Intl.NumberFormat("en-US");
const SIDE = 132;

/** One weight matrix and its LoRA update, drawn to scale. */
export default function LoraSizeExplorer() {
  const [model, setModel] = useState(0);
  const [rank, setRank] = useState(16);
  const id = useId();
  const width = MODELS[model].width;
  const thin = Math.max(3, (rank / width) * SIDE);

  return (
    <section className="card" aria-label="LoRA size explorer">
      <p className="card-kicker">How small is the update?</p>
      <p className="card-question">
        One attention layer's weight matrix, and the two thin matrices LoRA trains in its place. Sizes are drawn to
        scale.
      </p>
      <div className="choices" role="group" aria-label="Model">
        {MODELS.map((m, index) => (
          <button key={m.name} type="button" className="btn choice" aria-pressed={index === model} onClick={() => setModel(index)}>
            {m.name}
          </button>
        ))}
      </div>
      <div className="slider-row">
        <label htmlFor={id}>Rank</label>
        <input id={id} type="range" min={1} max={64} value={rank} onChange={(e) => setRank(Number(e.target.value))} />
        <span className="slider-value">{rank}</span>
      </div>
      <div className="lora-picture" aria-hidden="true">
        <figure>
          <div className="lora-box frozen" style={{ width: SIDE, height: SIDE }} />
          <figcaption>
            W, frozen
            <br />
            {width} × {width}
          </figcaption>
        </figure>
        <span className="lora-op">+</span>
        <figure>
          <div className="lora-box" style={{ width: thin, height: SIDE }} />
          <figcaption>
            B
            <br />
            {width} × {rank}
          </figcaption>
        </figure>
        <span className="lora-op">×</span>
        <figure>
          <div className="lora-box" style={{ width: SIDE, height: thin }} />
          <figcaption>
            A
            <br />
            {rank} × {width}
          </figcaption>
        </figure>
      </div>
      <p role="status">
        The full matrix has <strong>{whole.format(fullParams(width, width))}</strong> weights. LoRA trains{" "}
        <strong>{whole.format(loraParams(width, width, rank))}</strong>, which is{" "}
        <strong>{(loraShare(width, width, rank) * 100).toFixed(1)}%</strong> of that.
      </p>
    </section>
  );
}
