import { useId, useState } from "react";
import { XS, forwardKl, gaussianBins, reverseKl, twoPeakTeacher } from "../lib/generation";
import "./components.css";

const TEACHER = twoPeakTeacher();
const W = 520;
const H = 150;
const PEAK = Math.max(...TEACHER) * 1.15;

function path(values: number[], close: boolean): string {
  const scale = Math.max(PEAK, Math.max(...values));
  const points = values.map((v, i) => `${(i / (XS.length - 1)) * W},${H - (v / scale) * H}`);
  return close ? `M0,${H} L${points.join(" L")} L${W},${H} Z` : `M${points.join(" L")}`;
}

/** A simple student trying to match a teacher with two peaks, scored both ways. */
export default function KlDirectionExplorer() {
  const [position, setPosition] = useState(0);
  const [width, setWidth] = useState(1.5);
  const id = useId();

  const student = gaussianBins(position, width);
  const forward = forwardKl(TEACHER, student);
  const reverse = reverseKl(TEACHER, student);

  const onPeak = Math.abs(Math.abs(position) - 2) < 0.6 && width <= 1;
  const spanning = Math.abs(position) < 1 && width >= 1.8;
  let note = "Move the student onto one peak, then stretch it across both, and compare the two scores.";
  if (onPeak)
    note =
      "The student sits on one peak and ignores the other. Reverse KL is low: everything the student produces is something the teacher likes. Forward KL is high: it never produces half of what the teacher would.";
  if (spanning)
    note =
      "The student spreads across both peaks, and across the gap between them where the teacher puts almost nothing. Forward KL is low: it covers everything the teacher does. Reverse KL is high: much of what it produces is something the teacher would not.";

  return (
    <section className="card" aria-label="KL direction explorer">
      <p className="card-kicker">Two ways to measure the same mismatch</p>
      <p className="card-question">
        The shaded shape is a teacher that likes two different kinds of answer. The line is a small student that can
        only make one bump.
      </p>
      <svg viewBox={`0 0 ${W} ${H}`} className="kl-plot" role="img" aria-label="Teacher and student distributions">
        <path d={path(TEACHER, true)} className="kl-teacher" />
        <path d={path(student, false)} className="kl-student" />
      </svg>
      <div className="slider-row">
        <label htmlFor={`${id}-pos`}>Student position</label>
        <input
          id={`${id}-pos`}
          type="range"
          min={-4}
          max={4}
          step={0.1}
          value={position}
          onChange={(e) => setPosition(Number(e.target.value))}
        />
        <span className="slider-value">{position.toFixed(1)}</span>
      </div>
      <div className="slider-row">
        <label htmlFor={`${id}-w`}>Student width</label>
        <input
          id={`${id}-w`}
          type="range"
          min={0.4}
          max={3}
          step={0.1}
          value={width}
          onChange={(e) => setWidth(Number(e.target.value))}
        />
        <span className="slider-value">{width.toFixed(1)}</span>
      </div>
      <div className="calc-results" role="status">
        <p>
          Forward KL <strong data-testid="forward">{forward.toFixed(2)}</strong> · Reverse KL{" "}
          <strong data-testid="reverse">{reverse.toFixed(2)}</strong>
        </p>
        <p className="feedback">{note}</p>
      </div>
    </section>
  );
}
