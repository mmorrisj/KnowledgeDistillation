import { useId, useState } from "react";
import { softmax } from "../lib/distill";
import "./components.css";

// Illustrative teacher scores for one message. Not measured from a real model.
const TYPES = [
  { name: "card_arrival", score: 5.5 },
  { name: "card_delivery_estimate", score: 3.9 },
  { name: "lost_or_stolen_card", score: 2.6 },
  { name: "order_physical_card", score: 1.8 },
  { name: "activate_my_card", score: 0.2 },
  { name: "exchange_rate", score: -3.0 },
];
const SCORES = TYPES.map((t) => t.score);
const BASE = softmax(SCORES, 1);

const show = (p: number) => (p * 100 >= 0.1 ? (p * 100).toFixed(1) : (p * 100).toFixed(3));

/** Drag the temperature and watch a teacher's distribution flatten or sharpen. */
export default function TemperatureExplorer() {
  const [temperature, setTemperature] = useState(1);
  const id = useId();
  const probabilities = softmax(SCORES, temperature);

  let summary: string;
  if (temperature === 1) summary = "Temperature 1: this is the teacher's own distribution.";
  else if (temperature > 1)
    summary = `Temperature ${temperature.toFixed(1)}: flatter. The teacher's second and third choices are easier for a student to see. The order has not changed.`;
  else summary = `Temperature ${temperature.toFixed(1)}: sharper. Almost everything goes to the top choice, close to a plain label.`;

  return (
    <section className="card" aria-label="Temperature explorer">
      <p className="card-kicker">Temperature explorer</p>
      <p className="card-question">
        Message: <em>“I still haven't got my new card.”</em> The bars show an illustrative teacher's probability for
        six of the 77 types.
      </p>
      <div className="slider-row">
        <label htmlFor={id}>Temperature</label>
        <input
          id={id}
          type="range"
          min={0.5}
          max={8}
          step={0.1}
          value={temperature}
          onChange={(e) => setTemperature(Number(e.target.value))}
        />
        <span className="slider-value">{temperature.toFixed(1)}</span>
      </div>
      <ul className="temp-bars">
        {TYPES.map((type, index) => (
          <li key={type.name}>
            <code>{type.name}</code>
            <span className="temp-track">
              <span className="temp-fill" style={{ width: `${probabilities[index] * 100}%` }} />
              <span className="temp-ref" style={{ left: `${BASE[index] * 100}%` }} />
            </span>
            <span className="temp-value" data-testid="value">
              {show(probabilities[index])}%
            </span>
          </li>
        ))}
      </ul>
      <p className="feedback">The thin line on each bar marks its value at temperature 1.</p>
      <p role="status">{summary}</p>
    </section>
  );
}
