import { test, expect } from '@playwright/test';

test('has title and dashboard layout', async ({ page }) => {
  await page.goto('/');

  // Expect a title "to contain" MeetBot.
  await expect(page).toHaveTitle(/MeetBot/i);

  // Expect the main header to be present
  const header = page.locator('header');
  await expect(header).toBeVisible();
  await expect(header).toContainText('MeetBot');

  // Dashboard page layout
  await expect(page.locator('h1').first()).toContainText('Job Dashboard');
});
