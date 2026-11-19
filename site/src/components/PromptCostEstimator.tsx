import { useId, useState } from "react";
import { labellingCost, promptTokens } from "../lib/promptCost";
import "./components.css";

const OUTPUT_TOKENS = 8;
const usd = new Intl.NumberFormat("en-US", { style: "currency", currency: "USD" });
const whole = new Intl.NumberFormat("en-US");

/** How adding examples to a teacher's prompt changes what a labelling run costs. */
export default function PromptCostEstimator() {
  const [examples, setExamples] = useState(0);
  const [messages, setMessages] = useState(9000);
  const [priceIn, setPriceIn] = useState("3");
  const [priceOut, setPriceOut] = useState("15");
  const id = useId();

  const tokens = promptTokens({ examplesPerType: examples });
  const pIn = priceIn.trim() === "" ? Number.NaN : Number(priceIn);
  const pOut = priceOut.trim() === "" ? Number.NaN : Number(priceOut);
  const valid = Number.isFinite(pIn) && Number.isFinite(pOut) && pIn >= 0 && pOut >= 0;
  const cost = labellingCost(tokens, OUTPUT_TOKENS, messages, pIn, pOut);

  return (
    <section className="card" aria-label="Prompt cost estimator">
      <p className="card-kicker">What does a labelling run cost?</p>
      <p className="feedback">
        Token counts are rough estimates for the banking task. The starting figures are example prices, not current
        ones.
      </p>
      <div className="slider-row">
        <label htmlFor={`${id}-ex`}>Examples per type in the prompt</label>
        <input
          id={`${id}-ex`}
          type="range"
          min={0}
          max={5}
          value={examples}
          onChange={(e) => setExamples(Number(e.target.value))}
        />
        <span className="slider-value">{examples}</span>
      </div>
      <div className="slider-row">
        <label htmlFor={`${id}-n`}>Messages to label</label>
        <input
          id={`${id}-n`}
          type="range"
          min={500}
          max={50000}
          step={500}
          value={messages}
          onChange={(e) => setMessages(Number(e.target.value))}
        />
        <span className="slider-value">{whole.format(messages)}</span>
      </div>
      <div className="gap-fields">
        <div className="field">
          <label htmlFor={`${id}-in`}>Price per million input tokens</label>
          <div className="calc-input">
            <input id={`${id}-in`} inputMode="decimal" value={priceIn} onChange={(e) => setPriceIn(e.target.value)} />
            <span aria-hidden="true">USD</span>
          </div>
        </div>
        <div className="field">
          <label htmlFor={`${id}-out`}>Price per million output tokens</label>
          <div className="calc-input">
            <input id={`${id}-out`} inputMode="decimal" value={priceOut} onChange={(e) => setPriceOut(e.target.value)} />
            <span aria-hidden="true">USD</span>
          </div>
        </div>
      </div>
      <pre className="prompt-sketch" aria-hidden="true">{`Instructions                      ~40 tokens
List of 77 types                  ~${whole.format(77 * 7)} tokens
${examples > 0 ? `${examples * 77} example messages with types   ~${whole.format(examples * 77 * 22)} tokens\n` : ""}The message to label              ~20 tokens`}</pre>
      <div className="calc-results" role="status">
        {valid ? (
          <p>
            Each request sends about <strong>{whole.format(tokens)} tokens</strong>. Labelling {whole.format(messages)}{" "}
            messages costs about <strong>{usd.format(cost)}</strong>. Nearly all of that is the prompt, repeated for
            every message.
          </p>
        ) : (
          <p className="field-error">Enter a price of zero or more in both price fields.</p>
        )}
      </div>
    </section>
  );
}
