import AxeBuilder from "@axe-core/playwright";
import { expect, test, type Page } from "@playwright/test";

const LESSON = "/modules/00-sample/01-components/";

/** Interactive parts are server-rendered first; wait until they respond to input. */
async function openLesson(page: Page) {
  await page.goto(LESSON);
  await expect(page.locator("astro-island[ssr]")).toHaveCount(0);
}

test("a student can answer a quiz and keep progress across a reload", async ({ page }) => {
  await openLesson(page);
  await expect(page.getByRole("heading", { level: 1, name: "Component sampler" })).toBeVisible();

  await page.getByRole("button", { name: "Train on teacher-generated labels" }).click();
  await expect(page.getByText("Correct.")).toBeVisible();

  await page.getByRole("button", { name: "Mark lesson complete" }).click();
  await expect(page.getByText("Completed", { exact: true })).toBeVisible();

  await page.reload();
  await expect(page.locator("astro-island[ssr]")).toHaveCount(0);
  await expect(page.getByText("Completed", { exact: true })).toBeVisible();
  await expect(page.getByText("Correct.")).toBeVisible();
  await expect(page.getByRole("progressbar")).toHaveAttribute("aria-valuenow", "100");
});

test("an unknown page shows the styled 404", async ({ page }) => {
  const response = await page.goto("/nope/");
  expect(response?.status()).toBe(404);
  await expect(page.getByRole("heading", { name: "Page not found" })).toBeVisible();
});

test("a saved Jupyter address turns the lab card into a direct link", async ({ page }) => {
  await page.goto("/settings/");
  await page.getByLabel("Jupyter address").fill("javascript:alert(1)");
  await page.getByRole("button", { name: "Save" }).click();
  await expect(page.getByText("Enter a full address")).toBeVisible();

  await page.getByLabel("Jupyter address").fill("http://localhost:8888/");
  await page.getByRole("button", { name: "Save" }).click();
  await expect(page.getByText("Saved.")).toBeVisible();

  await page.goto(LESSON);
  await expect(page.getByRole("link", { name: "Open in Jupyter" })).toHaveAttribute(
    "href",
    "http://localhost:8888/lab/tree/labs/00_sample/check_gpu.ipynb",
  );
});

for (const scheme of ["light", "dark"] as const) {
  test(`the lesson page has no serious accessibility violations (${scheme})`, async ({ page }) => {
    await page.emulateMedia({ colorScheme: scheme });
    await openLesson(page);
    await page.getByRole("button", { name: "Train on teacher-generated labels" }).click();
    const results = await new AxeBuilder({ page }).analyze();
    const serious = results.violations.filter((v) => v.impact === "serious" || v.impact === "critical");
    expect(serious.map((v) => `${v.id}: ${v.nodes.map((n) => n.target).join(", ")}`)).toEqual([]);
  });
}
