import { useEffect, useState } from "react";
import { getProgressStore } from "../lib/progress";
import "./components.css";

export type QuizOption = { text: string; correct?: boolean; feedback: string };

interface Props {
  lessonId: string;
  id: string;
  question: string;
  options: QuizOption[];
}

/** One multiple-choice question with feedback that explains each answer. */
export default function Quiz({ lessonId, id, question, options }: Props) {
  const [answer, setAnswer] = useState<string | null>(null);

  useEffect(() => {
    let active = true;
    getProgressStore()
      .load()
      .then((progress) => {
        const last = progress.quizzes[`${lessonId}#${id}`]?.lastAnswer;
        if (active && last && options.some((option) => option.text === last)) {
          setAnswer((current) => current ?? last);
        }
      });
    return () => {
      active = false;
    };
  }, [lessonId, id, options]);

  const chosen = options.find((option) => option.text === answer);

  function choose(option: QuizOption) {
    setAnswer(option.text);
    void getProgressStore().recordQuiz(lessonId, id, option.text, Boolean(option.correct));
  }

  return (
    <section className="card quiz" aria-label="Check your understanding">
      <p className="card-kicker">Check your understanding</p>
      <p className="card-question">{question}</p>
      <div className="choices">
        {options.map((option) => {
          const selected = option.text === answer;
          const state = selected ? (option.correct ? "right" : "wrong") : undefined;
          return (
            <button
              key={option.text}
              type="button"
              className="btn choice"
              aria-pressed={selected}
              data-state={state}
              onClick={() => choose(option)}
            >
              {option.text}
            </button>
          );
        })}
      </div>
      <p className="feedback" role="status" data-state={chosen ? (chosen.correct ? "right" : "wrong") : undefined}>
        {chosen && (
          <>
            <strong>{chosen.correct ? "Correct." : "Not quite."}</strong> {chosen.feedback}
          </>
        )}
      </p>
    </section>
  );
}
