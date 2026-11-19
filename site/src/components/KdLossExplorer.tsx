import { useId, useState } from "react";
import { kdLoss, softmax } from "../lib/distill";
import "./components.css";

// One training example, with illustrative scores.
const TYPES = ["card_arrival", "card_delivery_estimate", "lost_or_stolen_card", "order_physical_card"];
const TEACHER = [5.5, 3.9, 2.6, 1.8];
const STUDENT = [1.2, 1.6, 0.3, 0.1];
const HUMAN_LABEL = 0;

/** One example, three distributions, and the loss that results from mixing two targets. */
export default function KdLossExplorer() {
  const [alpha, setAlpha] = useState(0.5);
  const [temperature, setTemperature] = useState(2);
  const id = useId();

  const teacher = softmax(TEACHER, temperature);
  const student = softmax(STUDENT, temperature);
  const loss = kdLoss(STUDENT, TEACHER, HUMAN_LABEL, temperature, alpha);

  let note = "The student is pulled toward both targets, weighted by alpha.";
  if (alpha === 0) note = "Alpha 0: the teacher is ignored. This is ordinary fine-tuning on the human label.";
  if (alpha === 1) note = "Alpha 1: the human label is ignored. The student learns from the teacher alone.";

  const groups: { title: string; values: number[]; tone: string }[] = [
    { title: "Human label", values: TYPES.map((_, i) => (i === HUMAN_LABEL ? 1 : 0)), tone: "plain" },
    { title: `Teacher at temperature ${temperature.toFixed(1)}`, values: teacher, tone: "own" },
    { title: `Student at temperature ${temperature.toFixed(1)}`, values: student, tone: "student" },
  ];

  return (
    <section className="card" aria-label="Distillation loss explorer">
      <p className="card-kicker">Distillation loss explorer</p>
      <p className="card-question">
        One training message, <em>“I still haven't got my new card.”</em> The student currently leans toward the wrong
        type. The numbers are illustrative.
      </p>
      <div className="slider-row">
        <label htmlFor={`${id}-alpha`}>Alpha (weight on the teacher)</label>
        <input
          id={`${id}-alpha`}
          type="range"
          min={0}
          max={1}
          step={0.05}
          value={alpha}
          onChange={(e) => setAlpha(Number(e.target.value))}
        />
        <span className="slider-value">{alpha.toFixed(2)}</span>
      </div>
      <div className="slider-row">
        <label htmlFor={`${id}-t`}>Temperature</label>
        <input
          id={`${id}-t`}
          type="range"
          min={1}
          max={6}
          step={0.5}
          value={temperature}
          onChange={(e) => setTemperature(Number(e.target.value))}
        />
        <span className="slider-value">{temperature.toFixed(1)}</span>
      </div>
      <div className="kd-groups" aria-hidden="true">
        {groups.map((group) => (
          <div className="kd-group" key={group.title}>
            <p>{group.title}</p>
            {group.values.map((value, index) => (
              <div className="kd-bar" key={TYPES[index]}>
                <span>{TYPES[index]}</span>
                <span className="temp-track">
                  <span className={`temp-fill ${group.tone}`} style={{ width: `${value * 100}%` }} />
                </span>
              </div>
            ))}
          </div>
        ))}
      </div>
      <div className="calc-results" role="status">
        <p>
          Teacher term {loss.teacherTerm.toFixed(2)} · Label term {loss.labelTerm.toFixed(2)} ·{" "}
          <strong>Total {loss.total.toFixed(2)}</strong>
        </p>
        <p className="feedback">{note}</p>
      </div>
    </section>
  );
}
