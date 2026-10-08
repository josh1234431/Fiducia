/**
 * Render the multi-monitor adverts to PNG with Electron's own renderer.
 *
 *   npx electron scratch/ads/render.cjs <output-folder>
 *
 * Needs the renderer dev server on 5273 and an engine on 8731 with the demo
 * block open (scratch/ads/make_scene.py sets that up).
 */

const { app, BrowserWindow } = require('electron')
const fs = require('node:fs')
const path = require('node:path')

const out = process.argv[process.argv.length - 1]
const ADS = [
  { id: '1', name: 'fiducia-three-screens', width: 1920, height: 1080 },
  { id: '2', name: 'fiducia-laptop-plus-one', width: 1920, height: 1080 },
  { id: '3', name: 'fiducia-spread-out', width: 1080, height: 1080 },
]

// One CSS pixel per image pixel: the adverts are laid out at their final size.
app.commandLine.appendSwitch('force-device-scale-factor', '1')

app.whenReady().then(async () => {
  fs.mkdirSync(out, { recursive: true })
  const only = process.env.ADS_ONLY ? process.env.ADS_ONLY.split(',') : null
  for (const ad of ADS.filter((a) => !only || only.includes(a.id))) {
    const window = new BrowserWindow({
      width: ad.width, height: ad.height, show: false, useContentSize: true,
      enableLargerThanScreen: true,
      webPreferences: { offscreen: true },
    })
    window.webContents.setFrameRate(30)
    const url = `http://127.0.0.1:5273/scratch/ads/ad.html?ad=${ad.id}`
    for (let attempt = 1; attempt <= 3; attempt++) {
      try {
        await window.loadURL(url)
        break
      } catch (error) {
        console.warn(`load attempt ${attempt} failed: ${error.message}`)
        await new Promise((r) => setTimeout(r, 1500))
      }
    }
    const viewport = await window.webContents.executeJavaScript('[innerWidth, innerHeight]')
    console.log(`${ad.name}: laid out at ${viewport.join('x')} CSS px`)

    const started = Date.now()
    while (Date.now() - started < 60000) {
      const done = await window.webContents.executeJavaScript('document.body.dataset.ready === "1"')
      if (done) break
      await new Promise((r) => setTimeout(r, 250))
    }

    const image = await window.webContents.capturePage()
    const file = path.join(out, `${ad.name}.png`)
    fs.writeFileSync(file, image.toPNG())
    const size = image.getSize()
    console.log(`${file}  ${size.width}x${size.height}`)
    window.destroy()
  }
  app.quit()
})
