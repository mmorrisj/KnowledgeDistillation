import { defineConfig } from "astro/config";
import mdx from "@astrojs/mdx";
import react from "@astrojs/react";

export default defineConfig({
  integrations: [mdx(), react()],
  trailingSlash: "always",
  build: { format: "directory" },
  markdown: {
    // High-contrast themes keep code comments readable in both site themes.
    shikiConfig: {
      themes: { light: "github-light-high-contrast", dark: "github-dark-high-contrast" },
    },
  },
});
