import { useId, useState, type ReactNode } from "react";
import "./components.css";

interface Props {
  fromModule: string;
  question: string;
  children: ReactNode;
}

/** A question from an earlier module, answered from memory before revealing. */
export default function ReviewCard({ fromModule, question, children }: Props) {
  const [open, setOpen] = useState(false);
  const answerId = useId();

  return (
    <section className="card review" aria-label={`Review from ${fromModule}`}>
      <p className="card-kicker">Review from {fromModule}</p>
      <p className="card-question">{question}</p>
      <div>
        <button
          type="button"
          className="btn"
          aria-expanded={open}
          aria-controls={answerId}
          onClick={() => setOpen(!open)}
        >
          {open ? "Hide answer" : "Show answer"}
        </button>
      </div>
      <div id={answerId} className="reveal" hidden={!open}>
        {children}
      </div>
    </section>
  );
}
