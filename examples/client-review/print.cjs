// Optional local Chromium proof; an installed Playwright module is supplied explicitly.
const { chromium } = require(process.argv[2]);
const path = require('node:path');
const { pathToFileURL } = require('node:url');
(async () => {
  const root = path.resolve(process.argv[3]);
  const paper = process.argv[4] || 'A4';
  const browser = await chromium.launch({ headless: true });
  try {
    const page = await browser.newPage({ viewport: { width: 1280, height: 960 } });
    await page.goto(pathToFileURL(path.join(root, 'review.html')).href);
    await page.pdf({ path: path.join(root, 'review.pdf'), format: paper, printBackground: paper === 'A4' });
    await page.screenshot({ path: path.join(root, 'review-cover.png') });
    await page.setViewportSize({ width: 320, height: 844 });
    if (await page.evaluate(() => document.documentElement.scrollWidth > innerWidth)) {
      throw new Error('Client review overflows at 320 pixels');
    }
  } finally {
    await browser.close();
  }
})().catch(error => { console.error(error); process.exit(1); });
