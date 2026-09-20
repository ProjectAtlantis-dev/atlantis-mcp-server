// Native Brave inspection only: no GPU bypass flags. The scoped capability is
// read from stdin and never logged. PLAYWRIGHT_CORE_PATH selects test tooling.
const readline = require('node:readline');
const { chromium } = require(process.env.PLAYWRIGHT_CORE_PATH);
const input = readline.createInterface({ input: process.stdin });
input.once('line', async line => {
  input.close();
  let browser;
  try {
    const grant = JSON.parse(line);
    browser = await chromium.launch({ executablePath: '/Applications/Brave Browser.app/Contents/MacOS/Brave Browser', headless: true });
    const page = await browser.newPage({ viewport: { width: 1280, height: 900 } });
    const errors = [];
    page.on('pageerror', error => errors.push(error.message));
    await page.goto('http://127.0.0.1:5273/#' + grant.viewerFragment, { waitUntil: 'domcontentloaded' });
    await page.waitForFunction(() => window.takramDebug?.getVehicleRegistry, { timeout: 45000 });
    await page.waitForTimeout(8000);
    const state = await page.evaluate(() => ({
      vehicles: window.takramDebug.getVehicleRegistry(),
      authority: window.takramDebug.defense?.getAuthorityStatus?.(),
      fragmentCleared: location.hash === '',
    }));
    await page.screenshot({ path: '/private/tmp/atlantis-authoritative-viewer.png' });
    console.log(JSON.stringify({ state, errors, screenshot: '/private/tmp/atlantis-authoritative-viewer.png' }));
    if (errors.length || !state.fragmentCleared) process.exitCode = 1;
  } catch (error) { console.error(error.message); process.exitCode = 1; }
  finally { if (browser) await browser.close(); }
});
