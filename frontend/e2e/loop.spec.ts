import { expect, test } from "@playwright/test";

test("generate a patient and run the closed loop to a terminal state", async ({ page }) => {
  await page.goto("/");

  // Disclaimer is always present.
  await expect(page.getByRole("alert")).toContainText(/NOT a medical device/i);

  // Generate a patient and confirm panels render.
  await page.getByRole("button", { name: "Generate patient" }).click();
  await expect(page.getByText("EEG band power (relative)")).toBeVisible();
  await expect(page.getByText(/Inferred patient state/)).toBeVisible();

  // Autonomously run the loop and wait for a terminal status.
  await page.getByRole("button", { name: /Auto-run/ }).click();
  await expect(page.locator(".status")).toContainText(
    /stabilized|exhausted|rejected/,
    { timeout: 60_000 },
  );

  // Timeline accumulated events.
  await expect(page.locator(".timeline li").first()).toBeVisible();
});

test("manual run surfaces candidate molecules and approval controls", async ({ page }) => {
  await page.goto("/");
  await page.getByRole("button", { name: "Generate patient" }).click();
  await page.getByRole("button", { name: "Start run (manual)" }).click();

  // Candidate molecule cards should appear with approve/reject controls.
  await expect(page.locator(".molecule-card").first()).toBeVisible({ timeout: 60_000 });
  await expect(page.getByRole("button", { name: /Approve/ })).toBeVisible();
});
