import { useState } from "react";
import "./components.css";

export type ConceptCell = {
  row: number;
  col: number;
  title: string;
  summary: string;
  example: string;
  bestFor: string;
  taughtIn: string;
};

interface Props {
  rows: string[];
  cols: string[];
  cells: ConceptCell[];
}

/** A small table of clickable cells with one detail panel below it. */
export default function ConceptGrid({ rows, cols, cells }: Props) {
  const [active, setActive] = useState(0);
  const current = cells[active];

  return (
    <section className="concept" aria-label="Concept map">
      <div className="concept-scroll">
        <table className="concept-table">
          <thead>
            <tr>
              <td />
              {cols.map((col) => (
                <th key={col} scope="col">
                  {col}
                </th>
              ))}
            </tr>
          </thead>
          <tbody>
            {rows.map((row, rowIndex) => (
              <tr key={row}>
                <th scope="row">{row}</th>
                {cols.map((col, colIndex) => {
                  const index = cells.findIndex((c) => c.row === rowIndex && c.col === colIndex);
                  if (index === -1) return <td key={col} />;
                  return (
                    <td key={col}>
                      <button
                        type="button"
                        className="btn concept-cell"
                        aria-pressed={index === active}
                        onClick={() => setActive(index)}
                      >
                        {cells[index].title}
                      </button>
                    </td>
                  );
                })}
              </tr>
            ))}
          </tbody>
        </table>
      </div>
      {current && (
        <div className="card concept-detail" aria-live="polite">
          <h3>{current.title}</h3>
          <p>{current.summary}</p>
          <dl>
            <dt>Example</dt>
            <dd>{current.example}</dd>
            <dt>Best for</dt>
            <dd>{current.bestFor}</dd>
            <dt>Taught in</dt>
            <dd>{current.taughtIn}</dd>
          </dl>
        </div>
      )}
    </section>
  );
}
