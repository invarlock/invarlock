const { chromium } = require(process.argv[2]);
const assert = require('node:assert/strict');

(async () => {
  const browser = await chromium.launch({ headless: true });
  try {
    const page = await browser.newPage({ viewport: { width: 1280, height: 900 } });
    const remote = [];
    page.on('request', request => {
      if (/^https?:/.test(request.url())) remote.push(request.url());
    });
    await page.goto(process.argv[3]);
    const tabs = page.getByRole('tab');
    assert.equal(await tabs.count(), 3);
    assert.equal(await tabs.nth(1).getAttribute('aria-selected'), 'true');
    await tabs.nth(1).focus();
    for (const [key, selected] of [['End', 2], ['ArrowRight', 0], ['ArrowLeft', 2], ['Home', 0]]) {
      await page.keyboard.press(key);
      assert.equal(await tabs.nth(selected).getAttribute('aria-selected'), 'true');
      assert(await tabs.nth(selected).evaluate(node => node === document.activeElement));
      assert.equal(await page.locator('.metric-group:visible').count(), 1);
    }
    // A finding must reveal and focus its result even when its tab was hidden.
    await page.locator('.findings a[href="#metric-result-3"]').focus();
    await page.keyboard.press('Enter');
    assert(await page.locator('#metric-result-3').isVisible());
    assert.equal(await page.evaluate(() => document.activeElement.id), 'metric-result-3');
    await page.emulateMedia({ media: 'print' });
    assert.equal(await page.locator('#metric-result-3').evaluate(node => getComputedStyle(node).outlineStyle), 'none', 'Keyboard focus must not become a printed callout');
    await page.emulateMedia({ media: 'screen' });
    await page.getByRole('button', { name: 'Show all metrics' }).focus();
    await page.keyboard.press('Enter');
    assert.equal(await page.locator('.metric-group:visible').count(), 3);
    assert.equal(await page.getByRole('tab').count(), 0);

    // Test computed colors, including status backgrounds, rather than screenshots.
    const failures = await page.evaluate(() => {
      const rgb = color => color.match(/[\d.]+/g).slice(0, 3).map(Number);
      const luminance = color => rgb(color).map(c => {
        c /= 255; return c <= .04045 ? c / 12.92 : ((c + .055) / 1.055) ** 2.4;
      }).reduce((total, c, i) => total + c * [.2126, .7152, .0722][i], 0);
      const ratio = (a, b) => (Math.max(luminance(a), luminance(b)) + .05) / (Math.min(luminance(a), luminance(b)) + .05);
      const background = node => {
        while (node) {
          const value = getComputedStyle(node).backgroundColor;
          if (value !== 'rgba(0, 0, 0, 0)' && value !== 'transparent') return value;
          node = node.parentElement;
        }
        return 'rgb(255, 255, 255)';
      };
      const issues = [];
      for (const node of document.querySelectorAll('.decision-summary,.scope,.badge,.finding-status,.interpretation-note,.review-action,.context-status li,.check-detail,.metric-navigation a,.metric-display-toggle')) {
        if (!node.checkVisibility()) continue;
        const contrast = ratio(getComputedStyle(node).color, background(node));
        if (contrast < 4.5) issues.push(`${node.className}: ${contrast}`);
      }
      for (const node of document.querySelectorAll('.range,.threshold,.neutral,.estimate')) {
        if (!node.checkVisibility()) continue;
        const contrast = ratio(getComputedStyle(node).stroke, 'rgb(237, 244, 237)');
        if (contrast < 3) issues.push(`${node.getAttribute('class')}: ${contrast}`);
      }
      return issues;
    });
    assert.deepEqual(failures, []);

    await page.setViewportSize({ width: 320, height: 844 });
    await page.addStyleTag({ content: '*{line-height:1.5!important;letter-spacing:.12em!important;word-spacing:.16em!important}p{margin-bottom:2em!important}' });
    assert(!(await page.evaluate(() => document.documentElement.scrollWidth > innerWidth)), '320px text-spacing overflow');
    const clipped = await page.locator('.review-action,.context-status li,.decision-summary,.metric-navigation a,.value,.interval-reading dd').evaluateAll(nodes =>
      nodes.filter(node => node.checkVisibility() && node.scrollWidth > node.clientWidth + 1).map(node => node.className));
    assert.deepEqual(clipped, [], 'Text must fit its reading containers');
    await page.screenshot({ path: process.argv[4], fullPage: true });
    assert.deepEqual(remote, [], 'A standalone report must not fetch remote assets');

    const offline = await browser.newContext({ javaScriptEnabled: false, offline: true });
    const plain = await offline.newPage();
    await plain.goto(process.argv[3]);
    assert.equal(await plain.locator('.metric-group:visible').count(), 3);
    assert.equal(await plain.locator('.metric-display-toggle:visible').count(), 0);
    assert.equal(await plain.locator('.interval svg[role="img"][aria-label]').count(), 3);
    await plain.emulateMedia({ forcedColors: 'active' });
    for (const label of ['Policy satisfied', 'Policy not met', 'More evidence needed']) {
      assert(await plain.locator('.metric .badge').filter({ hasText: label }).first().isVisible());
    }
    console.log('Keyboard navigation, hidden-result links, computed contrast, 320px spacing, offline/no-JS and explicit statuses passed.');
  } finally {
    await browser.close();
  }
})().catch(error => { console.error(error); process.exit(1); });
