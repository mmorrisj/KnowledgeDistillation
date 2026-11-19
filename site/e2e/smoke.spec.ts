import AxeBuilder from "@axe-core/playwright";
import { expect, test, type Page } from "@playwright/test";

const LESSON = "/modules/00-sample/01-components/";

/**
 * Interactive parts are server-rendered first. Wait until React has attached to every
 * one of them, so a click or a slider change is not lost.
 */
async function hydrated(page: Page) {
  await expect(page.locator("astro-island[ssr]")).toHaveCount(0);
  await page.waitForFunction(() =>
    [...document.querySelectorAll("astro-island")].every((island) => {
      const first = island.firstElementChild;
      return !first || Object.keys(first).some((key) => key.startsWith("__react"));
    }),
  );
}

async function openLesson(page: Page) {
  await page.goto(LESSON);
  await hydrated(page);
}

test("a student can answer a quiz and keep progress across a reload", async ({ page }) => {
  await openLesson(page);
  await expect(page.getByRole("heading", { level: 1, name: "Component sampler" })).toBeVisible();

  await page.getByRole("button", { name: "Train on teacher-generated labels" }).click();
  await expect(page.getByText("Correct.")).toBeVisible();

  await page.getByRole("button", { name: "Mark lesson complete" }).click();
  await expect(page.getByText("Completed", { exact: true })).toBeVisible();

  await page.reload();
  await hydrated(page);
  await expect(page.getByText("Completed", { exact: true })).toBeVisible();
  await expect(page.getByText("Correct.")).toBeVisible();
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

const LESSONS = [
  ["01-foundations/01-why-distill", "Why distill"],
  ["01-foundations/02-how-transformers-learn", "How transformers learn"],
  ["01-foundations/03-the-map", "The map"],
  ["01-foundations/04-three-ideas", "Three ideas kept apart"],
  ["01-foundations/05-licenses-and-terms", "Licenses and terms"],
  ["01-foundations/06-lab-setup", "Lab: check your setup"],
  ["02-baselines/01-three-numbers", "Three numbers before you distill"],
  ["02-baselines/02-splits-and-leakage", "Splits and leakage"],
  ["02-baselines/03-choosing-a-metric", "Choosing a metric"],
  ["02-baselines/04-is-the-difference-real", "Is the difference real?"],
  ["02-baselines/05-lab-baselines", "Lab: baselines"],
  ["03-white-box/01-what-the-teacher-knows", "What the teacher knows"],
  ["03-white-box/02-reading-probabilities", "Reading a teacher's probabilities"],
  ["03-white-box/03-temperature", "Temperature"],
  ["03-white-box/04-the-distillation-loss", "The distillation loss"],
  ["03-white-box/05-lab-white-box", "Lab: white-box distillation"],
] as const;

for (const [slug, title] of LESSONS) {
  for (const scheme of ["light", "dark"] as const) {
    test(`lesson "${title}" renders with no serious accessibility violations (${scheme})`, async ({ page }) => {
      await page.emulateMedia({ colorScheme: scheme });
      await page.goto(`/modules/${slug}/`);
      await hydrated(page);
      await expect(page.getByRole("heading", { level: 1, name: title })).toBeVisible();
      const results = await new AxeBuilder({ page }).analyze();
      const serious = results.violations.filter((v) => v.impact === "serious" || v.impact === "critical");
      expect(serious.map((v) => `${v.id}: ${v.nodes.map((n) => n.target).join(", ")}`)).toEqual([]);
    });
  }
}

test("the home page starts at Why distill and does not list the component sampler", async ({ page }) => {
  await page.goto("/");
  await expect(page.getByRole("link", { name: /Start with: Why distill/ })).toBeVisible();
  await expect(page.getByText("Component sampler")).toHaveCount(0);
});

test("completing a lesson moves the course progress bar", async ({ page }) => {
  await page.goto("/modules/01-foundations/01-why-distill/");
  await hydrated(page);
  await page.getByRole("button", { name: "Mark lesson complete" }).click();
  await expect(page.getByText(/^1 of \d+ lessons complete$/)).toBeVisible();
});

test("the cost calculator updates when requests per day changes", async ({ page }) => {
  await page.goto("/modules/01-foundations/01-why-distill/");
  await hydrated(page);
  const results = page.getByRole("status").filter({ hasText: "Cost per day" });
  await expect(results).toContainText("self-hosted student is cheaper");
  await page.getByLabel("Requests per day").fill("1000");
  await expect(results).toContainText("API is cheaper");
});

test("the training loop explorer shows what distillation changes", async ({ page }) => {
  await page.goto("/modules/01-foundations/02-how-transformers-learn/");
  await hydrated(page);
  await page.getByRole("button", { name: "Distillation", exact: true }).click();
  await expect(page.locator('[data-stage="target"]')).toContainText("Changes here");
  await expect(page.getByText("Changes here")).toHaveCount(1);
});

test("the interval explorer changes its verdict as the test set grows", async ({ page }) => {
  await page.goto("/modules/02-baselines/04-is-the-difference-real/");
  await hydrated(page);
  const results = page.getByRole("status").filter({ hasText: "likely between" });
  await expect(results).toContainText("cannot tell");
  await page.getByLabel("Number of test messages").fill("20000");
  await expect(results).toContainText("real difference");
});

test("the gap-closed calculator reports the share of the gap", async ({ page }) => {
  await page.goto("/modules/02-baselines/01-three-numbers/");
  await hydrated(page);
  await page.getByLabel("Distilled student accuracy").fill("75");
  await expect(page.getByText("closed 100% of the gap")).toBeVisible();
});

test("the temperature explorer flattens the distribution", async ({ page }) => {
  await page.goto("/modules/03-white-box/03-temperature/");
  await hydrated(page);
  await page.getByRole("slider", { name: "Temperature" }).fill("6");
  await expect(page.getByRole("status").filter({ hasText: "Temperature 6.0" })).toContainText("flatter");
});

test("the loss explorer explains alpha at its extremes", async ({ page }) => {
  await page.goto("/modules/03-white-box/04-the-distillation-loss/");
  await hydrated(page);
  await page.getByRole("slider", { name: /Alpha/ }).fill("0");
  await expect(page.getByRole("status").filter({ hasText: "Teacher term" })).toContainText(
    "ordinary fine-tuning on the human label",
  );
});
