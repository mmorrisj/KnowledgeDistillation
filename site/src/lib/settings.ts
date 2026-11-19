export const SETTINGS_KEY = "kd-course:settings:v1";

/**
 * Accept only http(s) addresses. Returns the address without trailing slashes,
 * query string or fragment (those often carry a login token), or null if invalid.
 */
export function normalizeJupyterUrl(input: string): string | null {
  const trimmed = input.trim();
  if (!trimmed) return null;
  let url: URL;
  try {
    url = new URL(trimmed);
  } catch {
    return null;
  }
  if (url.protocol !== "http:" && url.protocol !== "https:") return null;
  return `${url.origin}${url.pathname}`.replace(/\/+$/, "");
}

export function getJupyterUrl(): string | null {
  try {
    const data: unknown = JSON.parse(localStorage.getItem(SETTINGS_KEY) ?? "null");
    const stored = (data as { jupyterUrl?: unknown } | null)?.jupyterUrl;
    return typeof stored === "string" ? normalizeJupyterUrl(stored) : null;
  } catch {
    return null;
  }
}

/** Save the address. An empty string clears it. Returns false if the value is invalid. */
export function setJupyterUrl(input: string): boolean {
  const clearing = input.trim() === "";
  const normalized = normalizeJupyterUrl(input);
  if (!clearing && normalized === null) return false;
  try {
    localStorage.setItem(SETTINGS_KEY, JSON.stringify(clearing ? {} : { jupyterUrl: normalized }));
  } catch {
    return false;
  }
  return true;
}

/** Link that opens a notebook in JupyterLab. */
export function labHref(base: string, notebookPath: string): string {
  const encoded = notebookPath.split("/").map(encodeURIComponent).join("/");
  return `${base}/lab/tree/${encoded}`;
}
