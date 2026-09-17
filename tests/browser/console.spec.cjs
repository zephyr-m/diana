const { test, expect } = require('@playwright/test');

test('verification toggle is off by default and acknowledged by server', async ({page}) => {
  await page.goto('/');
  await page.locator('#connect').click();
  await expect(page.locator('#connection')).toHaveText('Подключена', {timeout:60000});
  await page.locator('#mute').click();
  await page.locator('#settings-open').click();
  await expect(page.locator('#verify-owner')).not.toBeChecked();
  await page.locator('#verify-owner').check();
  await expect(page.locator('#verification-status')).toHaveText('Проверка владельца включена');
  await expect(page.locator('#verify-owner')).toBeEnabled();
  await page.locator('#verify-owner').uncheck();
  await expect(page.locator('#verification-status')).toContainText('принимается любая речь');
  await page.locator('[data-close="settings"]').click();
  await page.locator('#connect').click();
});

test('console connects, receives real reply and telemetry, and stops', async ({page}) => {
  const errors = [];
  page.on('pageerror', error => errors.push(error.message));
  await page.goto('/');
  await expect(page.getByRole('heading', {name:'Разговор', exact:true})).toBeVisible();
  await expect(page.locator('#send')).toBeDisabled();
  await page.locator('#settings-open').click();
  await expect(page.locator('#settings')).toBeVisible();
  await page.locator('[data-close="settings"]').click();
  await page.screenshot({path:'.local/console-desktop.png'});
  await page.locator('#connect').click();
  await expect(page.locator('#connection')).toHaveText('Подключена', {timeout:60000});
  await page.locator('#mute').click();
  await expect(page.locator('#mute')).toHaveText('Включить микрофон');
  await page.locator('#text-input').fill('Это проверка интерфейса. Ответь одним предложением: пульт работает.');
  await page.locator('#send').click();
  await expect(page.locator('.message.assistant')).toBeVisible({timeout:90000});
  await expect(page.locator('#total')).not.toHaveText('—', {timeout:30000});
  await expect(page.locator('#audio')).toHaveJSProperty('paused', false);
  await page.locator('#details-open').click();
  await expect(page.locator('#detail-content')).toContainText('Введено с клавиатуры');
  await expect(page.locator('#detail-content')).toContainText('Первое аудио');
  await page.locator('[data-close="details"]').click();
  await page.screenshot({path:'.local/console-connected.png'});
  await page.locator('#stop').click();
  await page.locator('#connect').click();
  await expect(page.locator('#connection')).toHaveText('Не подключена');
  expect(errors).toEqual([]);
});

test('mobile layout fits and legacy page remains available', async ({page,request}) => {
  await page.setViewportSize({width:390,height:844});
  await page.goto('/');
  expect(await page.evaluate(() => document.documentElement.scrollWidth)).toBeLessThanOrEqual(390);
  await page.screenshot({path:'.local/console-mobile.png',fullPage:true});
  expect((await request.get('/client/')).status()).toBe(200);
});
