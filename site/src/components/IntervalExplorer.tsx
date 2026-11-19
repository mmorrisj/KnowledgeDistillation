import { useId, useState } from "react";
import { accuracyInterval, differenceInterval } from "../lib/stats";
import "./components.css";

interface Props {
  defaults?: Partial<{ a: number; b: number; n: number }>;
}

const one = (value: number) => (value * 100).toFixed(1);

/** Two measured accuracies and a test-set size: is the difference real? */
export default function IntervalExplorer({ defaults }: Props) {
  const start = { a: 70, b: 73, n: 500, ...defaults };
  const [a, setA] = useState(start.a);
  const [b, setB] = useState(start.b);
  const [n, setN] = useState(start.n);
  const id = useId();

  const intervalA = accuracyInterval(a / 100, n);
  const intervalB = accuracyInterval(b / 100, n);
  const diff = differenceInterval(a / 100, b / 100, n);

  // Zoom the picture to the region that matters.
  const lowEdge = Math.max(0, Math.min(intervalA.low, intervalB.low) - 0.03);
  const highEdge = Math.min(1, Math.max(intervalA.high, intervalB.high) + 0.03);
  const place = (value: number) => `${((value - lowEdge) / (highEdge - lowEdge)) * 100}%`;
  const width = (interval: { low: number; high: number }) =>
    `${((interval.high - interval.low) / (highEdge - lowEdge)) * 100}%`;

  return (
    <section className="card" aria-label="Interval explorer">
      <p className="card-kicker">Is the difference real?</p>
      <div className="slider-row">
        <label htmlFor={`${id}-a`}>Model A measured accuracy</label>
        <input id={`${id}-a`} type="range" min={40} max={99} value={a} onChange={(e) => setA(Number(e.target.value))} />
        <span className="slider-value">{a}%</span>
      </div>
      <div className="slider-row">
        <label htmlFor={`${id}-b`}>Model B measured accuracy</label>
        <input id={`${id}-b`} type="range" min={40} max={99} value={b} onChange={(e) => setB(Number(e.target.value))} />
        <span className="slider-value">{b}%</span>
      </div>
      <div className="slider-row">
        <label htmlFor={`${id}-n`}>Number of test messages</label>
        <input
          id={`${id}-n`}
          type="range"
          min={50}
          max={20000}
          step={50}
          value={n}
          onChange={(e) => setN(Number(e.target.value))}
        />
        <span className="slider-value">{n.toLocaleString("en-US")}</span>
      </div>
      <div className="intervals" aria-hidden="true">
        {(
          [
            ["A", intervalA, a, ""],
            ["B", intervalB, b, "own"],
          ] as const
        ).map(([name, interval, point, tone]) => (
          <div className="interval-row" key={name}>
            <span>{name}</span>
            <div className="interval-track">
              <div className={`interval-band ${tone}`} style={{ left: place(interval.low), width: width(interval) }} />
              <div className="interval-point" style={{ left: place(point / 100) }} />
            </div>
          </div>
        ))}
      </div>
      <div className="calc-results" role="status">
        <p>
          A is likely between {one(intervalA.low)}% and {one(intervalA.high)}%. B is likely between{" "}
          {one(intervalB.low)}% and {one(intervalB.high)}%.
        </p>
        <p>
          {diff.verdict === "real" ? (
            <strong>This looks like a real difference.</strong>
          ) : (
            <strong>You cannot tell these two apart with this test set.</strong>
          )}{" "}
          The gap is {one(diff.difference)} points, and its likely range is {one(diff.low)} to {one(diff.high)} points
          {diff.verdict === "real" ? ", which does not include zero." : ", which includes zero."}
        </p>
      </div>
    </section>
  );
}
