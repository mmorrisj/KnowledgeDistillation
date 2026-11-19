import { useId, useState } from "react";
import { gapClosed } from "../lib/stats";
import "./components.css";

type Scores = { baseline: number; distilled: number; teacher: number };
const LABELS: Record<keyof Scores, string> = {
  baseline: "Baseline student",
  distilled: "Distilled student",
  teacher: "Teacher",
};

interface Props {
  defaults?: Partial<Scores>;
}

/** Three scores in, one number out: how much of the gap to the teacher was closed. */
export default function GapClosedCalculator({ defaults }: Props) {
  const start = { baseline: 55, distilled: 68, teacher: 75, ...defaults };
  const [text, setText] = useState<Record<keyof Scores, string>>({
    baseline: String(start.baseline),
    distilled: String(start.distilled),
    teacher: String(start.teacher),
  });
  const baseId = useId();

  const value = (key: keyof Scores) => (text[key].trim() === "" ? Number.NaN : Number(text[key]));
  const scores: Scores = { baseline: value("baseline"), distilled: value("distilled"), teacher: value("teacher") };
  const valid = Object.values(scores).every((s) => Number.isFinite(s) && s >= 0 && s <= 100);
  const gap = valid ? gapClosed(scores.baseline, scores.distilled, scores.teacher) : null;

  let message: string;
  if (!valid) message = "Enter a score between 0 and 100 in every field.";
  else if (gap === null)
    message = "The teacher is no better than the baseline, so there is no gap to close. A teacher cannot help here.";
  else if (gap < 0) message = "The distilled student is worse than the baseline. Distillation hurt; look for a bug or a weak teacher signal.";
  else if (gap > 1)
    message = `The student closed ${Math.round(gap * 100)}% of the gap: it beat the teacher. This happens, usually because the student also learned from human labels.`;
  else message = `The student closed ${Math.round(gap * 100)}% of the gap between the baseline and the teacher.`;

  return (
    <section className="card gap" aria-label="Gap closed calculator">
      <p className="card-kicker">Gap closed</p>
      <div className="gap-fields">
        {(Object.keys(LABELS) as (keyof Scores)[]).map((key) => (
          <div className="field" key={key}>
            <label htmlFor={`${baseId}-${key}`}>{LABELS[key]} accuracy</label>
            <div className="calc-input">
              <input
                id={`${baseId}-${key}`}
                inputMode="decimal"
                value={text[key]}
                aria-invalid={!(Number.isFinite(scores[key]) && scores[key] >= 0 && scores[key] <= 100)}
                onChange={(e) => setText({ ...text, [key]: e.target.value })}
              />
              <span aria-hidden="true">%</span>
            </div>
          </div>
        ))}
      </div>
      {valid && (
        <div className="scale" aria-hidden="true">
          {(Object.keys(LABELS) as (keyof Scores)[]).map((key) => (
            <div key={key} className={`scale-mark scale-${key}`} style={{ left: `${scores[key]}%` }}>
              <span>{LABELS[key].split(" ")[0]}</span>
            </div>
          ))}
        </div>
      )}
      <p className="gap-result" role="status">
        {message}
      </p>
    </section>
  );
}
