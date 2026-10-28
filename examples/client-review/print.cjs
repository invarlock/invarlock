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
    const fs = require('node:fs');
    for (const document of ['review', 'brief']) {
      await page.setViewportSize({ width: 1280, height: 960 });
      await page.goto(pathToFileURL(path.join(root, document + '.html')).href);
      await page.pdf({ path: path.join(root, document + '.pdf'), format: paper, printBackground: paper === 'A4' });
      await page.screenshot({ path: path.join(root, document + '-cover.png') });
      await page.setViewportSize({ width: 320, height: 844 });
      if (await page.evaluate(() => document.documentElement.scrollWidth > innerWidth)) {
        throw new Error(document + ' overflows at 320 pixels');
      }
      if (document === 'brief') {
        for (const href of await page.locator('a').evaluateAll(links => links.map(link => link.getAttribute('href')))) {
          const [file, fragment] = href.split('#');
          if (!fs.existsSync(path.join(root, file))) throw new Error('Missing package link: ' + href);
          if (fragment) {
            const target = await browser.newPage();
            try {
              await target.goto(pathToFileURL(path.join(root, file)).href + '#' + fragment);
              const result = target.locator('[id="' + fragment + '"]');
              if (await result.count() !== 1 || !await result.isVisible()) throw new Error('Missing or hidden result: ' + href);
            } finally { await target.close(); }
          }
        }
      }
    }
  } finally {
    await browser.close();
  }
})().catch(error => { console.error(error); process.exit(1); });
