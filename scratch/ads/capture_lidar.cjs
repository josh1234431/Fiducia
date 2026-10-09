/**
 * Capture the LiDAR step for the website tour, at the tour's 2000x1250.
 * A preload stands in for the desktop bridge (capture_preload.cjs).
 *
 *   npx electron scratch/ads/capture_lidar.cjs <cloud.laz> <output.png>
 *
 * Needs the renderer dev server on 5273 and an engine on 8731 with the demo
 * project open, its classified cloud passed as <cloud.laz>.
 */

const { app, BrowserWindow } = require('electron')
const fs = require('node:fs')
const path = require('node:path')

const [cloud, out] = process.argv.slice(-2)

// The tour's screenshots are the app at 1600x1000, drawn at 1.25 times.
app.commandLine.appendSwitch('force-device-scale-factor', '1.25')

const pause = (ms) => new Promise((r) => setTimeout(r, ms))

app.whenReady().then(async () => {
  const window = new BrowserWindow({
    width: 1600, height: 1000, show: false, useContentSize: true, enableLargerThanScreen: true,
    webPreferences: {
      offscreen: true,
      preload: path.join(__dirname, 'capture_preload.cjs'),
      additionalArguments: [`--cloud=${cloud}`],
    },
  })
  window.setContentSize(1600, 1000)
  window.webContents.setFrameRate(30)
  const run = (code) => window.webContents.executeJavaScript(code)

  await window.loadURL('http://127.0.0.1:5273/')
  await run(`localStorage.setItem('fiducia.introSeen', '1'); localStorage.setItem('fiducia.author', 'Demo'); 1`)
  await run(`fetch('http://127.0.0.1:8731/project', { method: 'PATCH', headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({ workflow: 'all' }) }).then(() => 1)`)
  await window.loadURL('http://127.0.0.1:5273/')
  await pause(3000)
  console.log('laid out at', (await run('[innerWidth, innerHeight, devicePixelRatio]')).join(' x '))

  // The project's folder is on this computer, under a user's name; it has no
  // place on a public page.
  await run(`document.head.insertAdjacentHTML('beforeend',
    '<style>.titlebar__path{visibility:hidden}</style>'); 1`)

  // Open the cloud through the same button a person would use.
  await run(`
    [...document.querySelectorAll('button')].find((b) => /^LiDAR/.test(b.textContent.trim()))?.click();
    1`)
  await pause(800)
  await run(`[...document.querySelectorAll('button')].find((b) => b.textContent.trim() === 'Open…').click(); 1`)
  for (let i = 0; i < 60; i += 1) {
    if (await run(`!!document.querySelector('.cloudview__plan img')`)) break
    await pause(500)
  }
  await run(`[...document.querySelectorAll('.cloudview button')].find((b) => b.textContent.trim() === 'Class').click(); 1`)
  await pause(2500)

  // Draw a section across the plan, through trees and a building.
  await run(`(async () => {
    const plan = document.querySelector('.cloudview__plan')
    const box = plan.getBoundingClientRect()
    const img = plan.querySelector('img')
    const scale = Math.min(box.width / img.naturalWidth, box.height / img.naturalHeight)
    const w = img.naturalWidth * scale, h = img.naturalHeight * scale
    const left = box.left + (box.width - w) / 2, top = box.top + (box.height - h) / 2
    const at = (fx, fy) => ({ clientX: left + fx * w, clientY: top + fy * h, bubbles: true, pointerId: 1 })
    plan.dispatchEvent(new PointerEvent('pointerdown', at(0.03, 0.8)))
    for (let i = 1; i <= 10; i += 1) {
      plan.dispatchEvent(new PointerEvent('pointermove', at(0.03 + 0.94 * i / 10, 0.8)))
      await new Promise((r) => setTimeout(r, 20))
    }
    plan.dispatchEvent(new PointerEvent('pointerup', at(0.97, 0.8)))
    return 1
  })()`)
  for (let i = 0; i < 40; i += 1) {
    if (await run(`!!document.querySelector('.cloudview__section canvas')`)) break
    await pause(250)
  }
  await pause(1500)

  // Choose a class to label with, and show the training result in the panel.
  await run(`
    [...document.querySelectorAll('.label-picker__item')].find((b) => b.textContent.startsWith('High vegetation'))?.click();
    (() => {
      // Scroll the panel alone, so the window itself stays put.
      const section = [...document.querySelectorAll('.section')].find((s) => s.textContent.includes('Classes from labels'))
      let pane = section.parentElement
      while (pane && !(pane.scrollHeight > pane.clientHeight && /auto|scroll/.test(getComputedStyle(pane).overflowY))) {
        pane = pane.parentElement
      }
      if (pane) pane.scrollTop += section.getBoundingClientRect().top - pane.getBoundingClientRect().top - 8
    })();
    1`)
  await pause(1500)

  const image = await window.webContents.capturePage()
  fs.writeFileSync(out, image.toPNG())
  const size = image.getSize()
  console.log(`${out}  ${size.width}x${size.height}`)
  app.quit()
})
