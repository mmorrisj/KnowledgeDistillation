import { useState } from "react";
import { patchGrid } from "../lib/vision";
import "./components.css";

const IMAGE = 224;
const SIZES = [32, 16, 8];

/** An image becomes a grid of patches, and each patch becomes one token. */
export default function PatchExplorer() {
  const [size, setSize] = useState(16);
  const grid = patchGrid(IMAGE, size);

  return (
    <section className="card" aria-label="Patch explorer">
      <p className="card-kicker">An image as a row of tokens</p>
      <p className="card-question">
        A 224 by 224 pixel image is cut into square patches. Each patch is turned into one token, and the model reads
        the tokens the way a text model reads words.
      </p>
      <div className="choices" role="group" aria-label="Patch size">
        {SIZES.map((s) => (
          <button key={s} type="button" className="btn choice" aria-pressed={s === size} onClick={() => setSize(s)}>
            {s} px
          </button>
        ))}
      </div>
      <div className="patch-picture" aria-hidden="true">
        <div className="patch-grid" style={{ gridTemplateColumns: `repeat(${grid.perSide}, 1fr)` }}>
          {Array.from({ length: grid.patches }, (_, i) => (
            <i key={i} data-testid="patch" />
          ))}
        </div>
      </div>
      <p role="status">
        <strong>{grid.patches} patches</strong>, {grid.perSide} across and {grid.perSide} down, plus one extra token
        that stands for the whole image: {grid.tokens} tokens in all. Halving the patch size gives four times as many
        tokens, and the model's work grows faster than that.
      </p>
    </section>
  );
}
