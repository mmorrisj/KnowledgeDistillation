import { useState } from "react";
import { vectorStorageGb } from "../lib/search";
import "./components.css";

const COUNTS = [
  { label: "100 thousand", value: 100_000 },
  { label: "1 million", value: 1_000_000 },
  { label: "10 million", value: 10_000_000 },
  { label: "100 million", value: 100_000_000 },
];
const DIMENSIONS = [384, 768, 1024, 3072];
const TOP = vectorStorageGb(100_000_000, 3072);

function show(gb: number): string {
  if (gb < 1) return `${Math.round(gb * 1024)} MB`;
  return gb < 10 ? `${gb.toFixed(1)} GB` : `${Math.round(gb)} GB`;
}

/** How much space stored embeddings take, by how many there are and how long each one is. */
export default function VectorSizeExplorer() {
  const [count, setCount] = useState(1_000_000);
  const [dimensions, setDimensions] = useState(1024);
  const gb = vectorStorageGb(count, dimensions);

  return (
    <section className="card" aria-label="Vector storage explorer">
      <p className="card-kicker">What do stored embeddings cost?</p>
      <div className="choices" role="group" aria-label="Number of texts">
        {COUNTS.map((c) => (
          <button key={c.value} type="button" className="btn choice" aria-pressed={c.value === count} onClick={() => setCount(c.value)}>
            {c.label}
          </button>
        ))}
      </div>
      <div className="choices" role="group" aria-label="Numbers per embedding">
        {DIMENSIONS.map((d) => (
          <button key={d} type="button" className="btn choice" aria-pressed={d === dimensions} onClick={() => setDimensions(d)}>
            {d}
          </button>
        ))}
      </div>
      <div className="calc-bar" aria-hidden="true">
        <div className="own" style={{ width: `${Math.max(0.5, (Math.log10(gb * 1024 + 1) / Math.log10(TOP * 1024 + 1)) * 100)}%` }} />
      </div>
      <p role="status">
        <strong>{show(gb)}</strong> for the embeddings alone, before any search index. The index adds more, and for
        fast search most of this has to fit in memory.
      </p>
      <p className="feedback">The bar uses a compressed scale so that small and large values both show.</p>
    </section>
  );
}
