import { useState } from "react";
import { getProgressStore } from "../lib/progress";
import "./components.css";

export type SortItem = { text: string; answer: string; why: string };

interface Props {
  lessonId: string;
  id: string;
  categories: string[];
  items: SortItem[];
}

/** Sort statements into categories, mixed together, with an explanation for each. */
export default function SortDrill({ lessonId, id, categories, items }: Props) {
  const [choices, setChoices] = useState<Record<number, string>>({});
  const [solved, setSolved] = useState<Record<number, boolean>>({});

  function choose(index: number, category: string) {
    const correct = category === items[index].answer;
    setChoices({ ...choices, [index]: category });
    if (correct) setSolved({ ...solved, [index]: true });
    void getProgressStore().recordQuiz(lessonId, `${id}:${index}`, category, correct);
  }

  const done = Object.values(solved).filter(Boolean).length;

  return (
    <section className="card drill" aria-label="Sorting drill">
      <p className="card-kicker">Which idea is it?</p>
      <ol className="drill-items">
        {items.map((item, index) => {
          const choice = choices[index];
          const correct = choice === item.answer;
          return (
            <li key={item.text}>
              <p className="drill-text">{item.text}</p>
              <div className="choices">
                {categories.map((category) => (
                  <button
                    key={category}
                    type="button"
                    className="btn choice"
                    aria-pressed={choice === category}
                    data-state={choice === category ? (correct ? "right" : "wrong") : undefined}
                    onClick={() => choose(index, category)}
                  >
                    {category}
                  </button>
                ))}
              </div>
              {choice && (
                <p className="feedback" data-state={correct ? "right" : "wrong"}>
                  <strong>{correct ? "Correct." : `Not quite. It is ${item.answer}.`}</strong> {item.why}
                </p>
              )}
            </li>
          );
        })}
      </ol>
      <p className="drill-score" role="status">
        {done} of {items.length} sorted correctly
      </p>
    </section>
  );
}
