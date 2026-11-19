import { defineConfig } from "@playwright/test";

const PORT = 8123;

export default defineConfig({
  testDir: "e2e",
  use: { baseURL: `http://127.0.0.1:${PORT}`, trace: "off" },
  reporter: "list",
  webServer: {
    command: "uv run kd-course-serve",
    cwd: "..",
    env: { PORT: String(PORT), TRACKING_ENABLED: "false" },
    url: `http://127.0.0.1:${PORT}/api/health`,
    reuseExistingServer: false,
  },
});
