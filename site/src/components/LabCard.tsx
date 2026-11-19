import { useEffect, useState } from "react";
import { getJupyterUrl, labHref } from "../lib/settings";
import "./components.css";

interface Props {
  title: string;
  path: string;
  minutes: number;
}

/** Points the student at the notebook for this lesson. */
export default function LabCard({ title, path, minutes }: Props) {
  const [jupyterUrl, setUrl] = useState<string | null>(null);
  const [ready, setReady] = useState(false);
  const [copied, setCopied] = useState(false);

  useEffect(() => {
    setUrl(getJupyterUrl());
    setReady(true);
  }, []);

  async function copyPath() {
    try {
      await navigator.clipboard.writeText(path);
      setCopied(true);
    } catch {
      setCopied(false);
    }
  }

  return (
    <section className="card lab" aria-label="Lab">
      <p className="card-kicker">Lab · {title}</p>
      <p className="feedback">About {minutes} minutes on one A10G. Open this notebook on your GPU machine:</p>
      <div className="lab-path">
        <code>{path}</code>
        <button type="button" className="btn" onClick={copyPath}>
          {copied ? "Copied" : "Copy path"}
        </button>
      </div>
      {ready && (
        <div>
          {jupyterUrl ? (
            <a className="btn btn-primary" href={labHref(jupyterUrl, path)} target="_blank" rel="noopener noreferrer">
              Open in Jupyter
            </a>
          ) : (
            <a href="/settings/">Set your Jupyter address to open labs in one click</a>
          )}
        </div>
      )}
    </section>
  );
}
