import { useId, useState } from "react";
import { SAMPLE_WEIGHTS, meanAbsError, quantize } from "../lib/serving";
import "./components.css";

const LOW = Math.min(...SAMPLE_WEIGHTS);
const HIGH = Math.max(...SAMPLE_WEIGHTS);
const place = (value: number) => `${((value - LOW) / (HIGH - LOW)) * 100}%`;
const whole = new Intl.NumberFormat("en-US");

/** Store each weight in fewer bits and see what rounding does. */
export default function QuantizationExplorer() {
  const [bits, setBits] = useState(8);
  const id = useId();
  const rounded = quantize(SAMPLE_WEIGHTS, bits);
  const levels = 2 ** bits;
  const error = meanAbsError(SAMPLE_WEIGHTS, rounded);

  return (
    <section className="card" aria-label="Quantization explorer">
      <p className="card-kicker">Fewer bits per weight</p>
      <p className="card-question">
        Twenty-four illustrative weights. The top row is where they really are. The bottom row is where they land when
        each is rounded to the nearest allowed value.
      </p>
      <div className="slider-row">
        <label htmlFor={id}>Bits per weight</label>
        <input id={id} type="range" min={2} max={16} value={bits} onChange={(e) => setBits(Number(e.target.value))} />
        <span className="slider-value">{bits}</span>
      </div>
      <div className="quant" aria-hidden="true">
        <div className="quant-row">
          <span>Original</span>
          <div className="quant-line">
            {SAMPLE_WEIGHTS.map((v, i) => (
              <i key={i} style={{ left: place(v) }} />
            ))}
          </div>
        </div>
        <div className="quant-row">
          <span>Rounded</span>
          <div className="quant-line">
            {rounded.map((v, i) => (
              <i key={i} className="own" style={{ left: place(v) }} />
            ))}
          </div>
        </div>
      </div>
      <p role="status">
        <strong>{whole.format(levels)} allowed values.</strong> Average rounding error {error.toFixed(4)}. The weights
        take <strong>{Math.round((bits / 32) * 100)}%</strong> of the space they would at 32 bits.
      </p>
    </section>
  );
}
