export type Theme = "light" | "dark";

export const THEME_KEY = "kd-course:theme";

/** An explicit stored choice wins; otherwise follow the system setting. */
export function resolveTheme(stored: string | null, prefersDark: boolean): Theme {
  if (stored === "light" || stored === "dark") return stored;
  return prefersDark ? "dark" : "light";
}
