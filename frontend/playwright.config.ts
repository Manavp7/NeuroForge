import { defineConfig, devices } from "@playwright/test";

// End-to-end test of the full stack: spins up the backend API and the built frontend,
// then drives generate -> auto-run -> stabilized in a real browser.
export default defineConfig({
  testDir: "./e2e",
  timeout: 120_000,
  expect: { timeout: 30_000 },
  fullyParallel: false,
  retries: 0,
  use: {
    baseURL: "http://localhost:4173",
    trace: "off",
  },
  projects: [
    {
      name: "chromium",
      // Use the installed Google Chrome (works locally and after `npx playwright install chrome`).
      use: { ...devices["Desktop Chrome"], channel: "chrome" },
    },
  ],
  reporter: process.env.CI ? "line" : "list",
  webServer: [
    {
      command: "python3 -m uvicorn neuroforge.api.app:app --port 8000",
      cwd: "../backend",
      url: "http://localhost:8000/healthz",
      reuseExistingServer: !process.env.CI,
      timeout: 120_000,
    },
    {
      command: "npm run build && npm run preview -- --port 4173 --strictPort",
      url: "http://localhost:4173",
      reuseExistingServer: !process.env.CI,
      timeout: 120_000,
    },
  ],
});
