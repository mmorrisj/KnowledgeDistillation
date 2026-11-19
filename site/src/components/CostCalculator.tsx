import { useId, useState } from "react";
import { apiCostPer1k, breakEvenRequestsPerDay, isValidInput, selfHostedDailyCost } from "../lib/cost";
import "./components.css";

type Values = {
  requestsPerDay: number;
  inTokens: number;
  outTokens: number;
  usdPerMTokIn: number;
  usdPerMTokOut: number;
  instanceUsdPerHour: number;
};

const EXAMPLES: Values = {
  requestsPerDay: 50_000,
  inTokens: 800,
  outTokens: 150,
  usdPerMTokIn: 3,
  usdPerMTokOut: 15,
  instanceUsdPerHour: 1.62,
};

const FIELDS: { key: keyof Values; label: string; unit: string; group: "Workload" | "API teacher" | "Your own GPU" }[] = [
  { key: "requestsPerDay", label: "Requests per day", unit: "requests", group: "Workload" },
  { key: "inTokens", label: "Input tokens per request", unit: "tokens", group: "Workload" },
  { key: "outTokens", label: "Output tokens per request", unit: "tokens", group: "Workload" },
  { key: "usdPerMTokIn", label: "Price per million input tokens", unit: "USD", group: "API teacher" },
  { key: "usdPerMTokOut", label: "Price per million output tokens", unit: "USD", group: "API teacher" },
  { key: "instanceUsdPerHour", label: "GPU machine price per hour", unit: "USD", group: "Your own GPU" },
];

const usd = new Intl.NumberFormat("en-US", { style: "currency", currency: "USD" });
const whole = new Intl.NumberFormat("en-US");

interface Props {
  defaults?: Partial<Values>;
}

/** Compare the daily cost of calling an API teacher with running a small student on one GPU. */
export default function CostCalculator({ defaults }: Props) {
  const [text, setText] = useState<Record<keyof Values, string>>(() => {
    const start = { ...EXAMPLES, ...defaults };
    return Object.fromEntries(Object.entries(start).map(([k, v]) => [k, String(v)])) as Record<keyof Values, string>;
  });
  const baseId = useId();

  const v = Object.fromEntries(
    Object.entries(text).map(([k, raw]) => [k, raw.trim() === "" ? Number.NaN : Number(raw)]),
  ) as Values;
  const valid = isValidInput(...Object.values(v));

  const per1k = apiCostPer1k(v.inTokens, v.outTokens, v.usdPerMTokIn, v.usdPerMTokOut);
  const apiDaily = (per1k * v.requestsPerDay) / 1000;
  const ownDaily = selfHostedDailyCost(v.instanceUsdPerHour);
  const breakEven = breakEvenRequestsPerDay(per1k, v.instanceUsdPerHour);
  const max = Math.max(apiDaily, ownDaily);

  return (
    <section className="card calc" aria-label="Cost calculator">
      <p className="card-kicker">Cost calculator</p>
      <p className="feedback">
        The starting numbers are example values, not current prices. Replace them with your own
        workload and the prices you actually pay.
      </p>
      <div className="calc-fields">
        {(["Workload", "API teacher", "Your own GPU"] as const).map((group) => (
          <fieldset key={group} className="calc-group">
            <legend>{group}</legend>
            {FIELDS.filter((f) => f.group === group).map((f) => {
              const id = `${baseId}-${f.key}`;
              const bad = !isValidInput(v[f.key]);
              return (
                <div className="field" key={f.key}>
                  <label htmlFor={id}>{f.label}</label>
                  <div className="calc-input">
                    <input
                      id={id}
                      inputMode="decimal"
                      value={text[f.key]}
                      aria-invalid={bad}
                      onChange={(e) => setText({ ...text, [f.key]: e.target.value })}
                    />
                    <span aria-hidden="true">{f.unit}</span>
                  </div>
                </div>
              );
            })}
          </fieldset>
        ))}
      </div>
      <div className="calc-results" role="status">
        {valid ? (
          <>
            <div className="calc-row">
              <span>API teacher</span>
              <div className="calc-bar"><div style={{ width: `${(apiDaily / max) * 100}%` }} /></div>
              <strong>{usd.format(apiDaily)}</strong>
            </div>
            <div className="calc-row">
              <span>Self-hosted student</span>
              <div className="calc-bar"><div className="own" style={{ width: `${(ownDaily / max) * 100}%` }} /></div>
              <strong>{usd.format(ownDaily)}</strong>
            </div>
            <p>
              Cost per day. At this volume the{" "}
              <strong>{apiDaily > ownDaily ? "self-hosted student is cheaper" : "API is cheaper"}</strong>. The two cost
              the same at about {whole.format(breakEven)} requests per day.
            </p>
            <p className="feedback">
              This assumes one GPU machine left on all day can handle the volume, and it leaves out the one-time cost
              of building the student.
            </p>
          </>
        ) : (
          <p className="field-error">Enter a number above zero in every field to see the comparison.</p>
        )}
      </div>
    </section>
  );
}
