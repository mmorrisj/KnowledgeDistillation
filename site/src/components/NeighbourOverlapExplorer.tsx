import { useId, useState } from "react";
import { overlapAtK } from "../lib/search";
import "./components.css";

// Illustrative results for the query "my new card still hasn't arrived".
const TEACHER = [
  "Card ordered two weeks ago, not here yet",
  "When will my replacement card be delivered?",
  "Still waiting for my card in the post",
  "How long does card delivery take?",
  "My card never showed up",
  "Can I track my new card?",
  "Card was sent to my old address",
  "I need a new card urgently",
];
const STUDENT = [
  "Card ordered two weeks ago, not here yet",
  "Still waiting for my card in the post",
  "My card never showed up",
  "When will my replacement card be delivered?",
  "I lost my card yesterday",
  "How long does card delivery take?",
  "My card is not working at the till",
  "Can I track my new card?",
];

/** Two models' search results for one query, and how much they share. */
export default function NeighbourOverlapExplorer() {
  const [k, setK] = useState(5);
  const id = useId();
  const overlap = overlapAtK(TEACHER, STUDENT, k);
  const teacherTop = new Set(TEACHER.slice(0, k));
  const studentTop = new Set(STUDENT.slice(0, k));

  const column = (title: string, items: string[], other: Set<string>) => (
    <div className="overlap-column">
      <p>{title}</p>
      <ol>
        {items.slice(0, k).map((item) => (
          <li key={item} data-shared={other.has(item)}>
            {item}
            {other.has(item) && <span className="overlap-tag">in both</span>}
          </li>
        ))}
      </ol>
    </div>
  );

  return (
    <section className="card" aria-label="Neighbour overlap explorer">
      <p className="card-kicker">Do the two models find the same things?</p>
      <p className="card-question">
        Results for the query <em>“my new card still hasn't arrived”</em>. The lists are illustrative.
      </p>
      <div className="slider-row">
        <label htmlFor={id}>Results compared</label>
        <input id={id} type="range" min={1} max={8} value={k} onChange={(e) => setK(Number(e.target.value))} />
        <span className="slider-value">{k}</span>
      </div>
      <div className="overlap-columns">
        {column("Teacher", TEACHER, studentTop)}
        {column("Student", STUDENT, teacherTop)}
      </div>
      <p role="status">
        The two lists share <strong>{Math.round(overlap * 100)}%</strong> of their top {k}. Order within the list is
        not counted.
      </p>
    </section>
  );
}
