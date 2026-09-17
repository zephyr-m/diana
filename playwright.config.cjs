const { defineConfig } = require('@playwright/test');
module.exports = defineConfig({
  testDir: './tests/browser', workers: 1, timeout: 120000,
  outputDir: '.local/browser-results',
  use: {
    baseURL: process.env.DIANA_TEST_URL || 'http://127.0.0.1:7864',
    viewport: { width: 1440, height: 980 }, permissions: ['microphone'],
    launchOptions: { executablePath: process.env.CHROME_PATH || '/run/current-system/sw/bin/google-chrome-stable',
      args: ['--use-fake-ui-for-media-stream','--use-fake-device-for-media-stream'] },
  },
});
